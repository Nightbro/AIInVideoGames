import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # needs to be set to 0 to avoid an error with the model saving

import numpy as np
from snake_game import SnakeGame
from agent import DQNAgent
import tensorflow as tf
import signal

stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True
    print("\nSaving model and stopping training...")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # ctrl+c

    game = SnakeGame()
    agent = DQNAgent()

    agent.model = agent.build_model()

    # Check if a model file exists
    model_path = 'dqn_model.keras'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        agent.model = tf.keras.models.load_model(model_path)
    else:
        print("No existing model found, starting from scratch.")
        agent.model = agent.build_model()  

    render_interval = 50
             
    try:
        for episode in range(1000):
            if stop_flag:
                break

            state = np.reshape(game.reset(), [1, 1, 6])
            total_reward = 0
            step_count = 0
            while not game.game_over:
                if stop_flag:
                    break

                action = agent.act(state)
                next_state, reward, game_over = game.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, 1, 6])
                agent.train(state, action, reward, next_state, game.game_over)
                state = next_state
                step_count += 1
                if step_count % render_interval == 0:
                    game.render()

            print(f'Episode: {episode}, Total reward: {total_reward}')

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        agent.model.save(model_path)
        print("Model saved.")
      

