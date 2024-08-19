import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
from game import FlappyBirdGame


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class DQAgent:
    def __init__(self, state_size, action_size, model_path="dq_agent_model.keras"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model_path = model_path
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.memory = []

        # Load existing model for continuous learning if it exists
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}...")
            self.model = load_model(self.model_path)
        else:
            print("Creating a new model...")
            self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_shape=(self.state_size,), activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state).reshape(-1)  # Ensure state is a flat array
        next_state = np.array(next_state).reshape(-1)  # Ensure next_state is a flat array
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        state = np.array(state).reshape(-1)  # Reshape state for consistency
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.array([state]))[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        for i in minibatch:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=1000):
        try:
            for episode in range(episodes):
                game = FlappyBirdGame()
                state = game.get_state()

                while not game.is_done():
                    action = self.choose_action(state)
                    next_state, reward = game.step(action)
                    done = game.is_done()
                    self.remember(state, action, reward, next_state, done)
                    state = next_state

                    if done:
                        game.reset()

                    self.replay()

        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving the model...")
            self.model.save(self.model_path)
            print("Model saved successfully!")

if __name__ == "__main__":
    agent = DQAgent(state_size=2, action_size=2)
    agent.train(episodes=1000)
    agent.model.save(agent.model_path)  # Save the model after training
