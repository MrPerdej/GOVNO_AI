# dqn_agent.py
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os

class DQNAgent:
    
    def __init__(self, input_shape, action_space, model_path="dqn_model.keras"):
        self.state_shape = input_shape
        self.action_space = action_space
        self.action_size = int(np.prod(action_space.nvec))  # УБРАТЬ НЕЙРОСЕТЬ!!!
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model_path = model_path
        
        if os.path.exists(model_path):
            print("[INFO] Загрузка модели из файла")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        else:
            print("[INFO] Создание новой модели")
            self.model = self._build_model()
        
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
    
    def _build_model(self):
        model = tf.keras.Sequential([
            # обрабатываем по‑кадрово
            tf.keras.layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'),
                                            input_shape=(5, 100, 400, 3)),
            tf.keras.layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            tf.keras.layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
            tf.keras.layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            tf.keras.layers.TimeDistributed(layers.Flatten()),
            # теперь убираем ось time_steps
            tf.keras.layers.Flatten(),          # ← (batch, time_steps * features)
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [np.random.randint(n) for n in self.action_space.nvec]
        q_values = self.model.predict(state[np.newaxis], verbose=0)[0]
        best_index = np.argmax(q_values)
        action = np.unravel_index(best_index, self.action_space.nvec)
        return list(action)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # 1) выбор случайного батча переходов
        minibatch = random.sample(self.memory, batch_size)
        # распакуем в списки
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 2) превратим в массивы
        states      = np.stack(states)       # (batch, 5,100,400,3)
        next_states = np.stack(next_states)  # (batch, 5,100,400,3)

        # 3) предсказание Q и target-Q сразу для всего батча
        q_values         = self.model.predict(states, verbose=0)         # (batch, action_size)
        next_q_model     = self.model.predict(next_states, verbose=0)    # (batch, action_size)
        next_q_target    = self.target_model.predict(next_states, verbose=0)  # (batch, action_size)

        # 4) подготовим целевые вектора
        target_q = q_values.copy()  # shape (batch, action_size)

        for i in range(batch_size):
            # плоский индекс для конкретного действия в action_space.nvec
            flat_index = np.ravel_multi_index(actions[i], self.action_space.nvec)

            if dones[i]:
                target_q[i, flat_index] = rewards[i]
            else:
                # жадное действие из online‑сети
                best_next = np.argmax(next_q_model[i])
                # TD‑обновление по целевой сети
                target_q[i, flat_index] = rewards[i] + self.gamma * next_q_target[i, best_next]

        # 5) обучаем модель на всём батче сразу
        self.model.fit(states, target_q, epochs=1, verbose=0)

        # 6) и обновляем ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self):
        self.model.save("dqn_model.keras")
        print("[INFO] Модель сохранена в", self.model_path)

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path)
        print("[INFO] Модель загружена из", self.model_path)
