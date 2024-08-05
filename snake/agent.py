import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            tf.keras.Input(shape=(1, 6)),  # first layer - usuer input
            Flatten(),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(4, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state):
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + 0.95 * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
