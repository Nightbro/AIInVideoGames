import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dropout  
from game import FlappyBirdGame





class DQAgent:
    def __init__(self, state_size, action_size, model_path="dq_agent_model"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model_path = model_path
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 1
        self.memory = []

        # Load existing model for continuous learning if it exists
        if os.path.exists(f"{self.model_path}.keras"):
            print(f"Loading existing model from {self.model_path}.keras ...")
            self.model = load_model(f"{self.model_path}.keras")
        else:
            print("Creating a new model...")
            self.model = self.build_model(input_size=state_size, output_size=action_size)

    def build_model(self,input_size, output_size):
        model = models.Sequential()
    
        # First hidden layer
        model.add(layers.Dense(8, input_shape=(input_size,), activation='relu'))  # Reduced to 8 neurons
        
        # Second hidden layer
        model.add(layers.Dense(16, activation='relu'))  # Reduced to 16 neurons
        
        # Output layer
        model.add(layers.Dense(output_size, activation='linear'))  # Linear activation for Q-values
        
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state), action, reward, np.array(next_state), done))

    def choose_action(self, state):
        state = np.array(state).reshape(-1)  # Ensure state is a flat array
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.array([state]))[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3] for i in minibatch])
        dones = np.array([self.memory[i][4] for i in minibatch])

        # Vectorized target calculation
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1) * (~dones)

        # Create target_f to train the model on
        target_f = self.model.predict(states)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]

        # Fit the model with the batch
        self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=1000,render_every=1):
        try:
            episode=0
            while True:
                game = FlappyBirdGame()
                state = game.get_state()

                while not game.is_done():
                    action = self.choose_action(state)
                    next_state, reward = game.step(action)
                    done = game.is_done()
                    self.remember(state, action, reward, next_state, done)
                    print(f"Episode: {episode + 1}, Score: {game.score}, Epsilon: {self.epsilon}, State: {state}")
                    state = next_state
                   
                    self.replay()

                    #game.render()

                if game.game_over:
                    episode+=1

                # Save model at specific intervals
                if episode + 1 in [1,2,3,10, 100, 1000,5000]:
                    save_path = f"{self.model_path}_ep{episode + 1}.keras"
                    self.model.save(save_path)
                    print(f"Model saved after {episode + 1} episodes at {save_path}")

                if (episode == episodes):
                    break

        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving the model...")
            self.model.save(f"{self.model_path}.keras")

if __name__ == "__main__":
    agent = DQAgent(state_size=4, action_size=2)
    agent.train(episodes=20000)
    agent.model.save(f"{agent.model_path}.keras")
