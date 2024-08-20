import multiprocessing as mp
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dropout  
import numpy as np
from game import FlappyBirdGame
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DQAgent:
    # Your DQAgent class implementation remains the same
    # Include a method to retrieve the score after the simulation

    def __init__(self, state_size, action_size, model_path="dq_agent_model"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model_path = model_path
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.memory = []
        self.last_score = 0  # Track the last score

        if os.path.exists(f"{self.model_path}.keras"):
            self.model = load_model(f"{self.model_path}.keras")
        else:
            self.model = self.build_model(self.state_size, self.action_size)

    def build_model(self, input_size, output_size):
        model = models.Sequential()
        model.add(layers.Dense(8, input_shape=(input_size,), activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state), action, reward, np.array(next_state), done))

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

    def train_single_episode(self):
        game = FlappyBirdGame()
        state = game.get_state()
        while not game.is_done():
            action = self.choose_action(state)
            next_state, reward = game.step(action)
            done = game.is_done()
            self.remember(state, action, reward, next_state, done)
            state = next_state
            self.replay()

        self.last_score = game.score  # Store the score at the end of the episode

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.array([state]))[0]
        return np.argmax(q_values)

def run_simulation(sim_id, model_path):
    agent = DQAgent(state_size=4, action_size=2, model_path=model_path)
    agent.train_single_episode()
    return sim_id, agent.last_score, model_path

def main():
    num_iterations = 500
    num_simulations = 20

    for iteration in range(1, num_iterations + 1):
        print(f"Iteration {iteration}:")

        # Determine the model to use for this iteration
        if iteration == 1:
            model_path = "dq_agent_model_initial"
        else:
            model_path = f"dq_agent_model_episode_{iteration - 1}"  # Load the model from the previous iteration

        # Run 20 parallel simulations
        pool = mp.Pool(num_simulations)
        simulation_results = []

        # Run simulations in parallel
        for sim_id in range(num_simulations):
            result = pool.apply_async(run_simulation, args=(sim_id, model_path))
            simulation_results.append(result)

        pool.close()
        pool.join()

        # Collect the results and pick the best model
        results = [res.get() for res in simulation_results]
        best_simulation = max(results, key=lambda x: x[1])  # Pick the simulation with the highest score
        best_sim_id, best_score, best_model_path = best_simulation

        print(f"Best score in iteration {iteration}: {best_score} by simulation {best_sim_id}")

        # Save the best model for the next iteration
        save_path = f"dq_agent_model_episode_{iteration}.keras"
        agent = DQAgent(state_size=4, action_size=2, model_path=best_model_path)
        agent.model.save(save_path)
        print(f"Model saved as {save_path}")

if __name__ == "__main__":
    main()