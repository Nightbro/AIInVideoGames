import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Needs to be set to 0 to avoid an error with the model saving

import numpy as np
from snake_game import SnakeGame
from agent import DQNAgent
import tensorflow as tf
import signal
import json
import time
import logging

stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True
    print("\nSaving model and stopping training...")

def save_results(results, filename='results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('snake_game')

    file_handler = logging.FileHandler('game_logs.txt')
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d.%m.%Y %H:%M:%S')

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C

    logger = setup_logging()

    game = SnakeGame(render=False)
    agent = DQNAgent()

    model_path = 'dqn_model.keras'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        agent.model = tf.keras.models.load_model(model_path)
    else:
        print("No existing model found, starting from scratch.")
        agent.model = agent.build_model()

    algorithms = ['dqn', 'bfs', 'dfs', 'a_star']
    results = {alg: {'episodes': 0, 'total_reward': 0, 'start_time': time.time(), 'history': []} for alg in algorithms}

    try:
        for episode in range(1000):
            if stop_flag:
                break

            for alg in algorithms:
                state = np.reshape(game.reset(), [1, 1, 10])  # Update the state shape
                total_reward = 0
                steps = 0

                while not game.game_over:
                    if stop_flag:
                        break

                    if alg == 'bfs':
                        path = game.bfs()
                        if path:
                            next_node = path[0] if len(path) > 0 else None
                            action = agent.get_action_from_path(game.snake[0], next_node)
                        else:
                            action = agent.act(state)
                    elif alg == 'dfs':
                        path = game.dfs()
                        if path:
                            next_node = path[0] if len(path) > 0 else None
                            action = agent.get_action_from_path(game.snake[0], next_node)
                        else:
                            action = agent.act(state)
                    elif alg == 'a_star':
                        path = game.a_star()
                        if path:
                            next_node = path[0] if len(path) > 0 else None
                            action = agent.get_action_from_path(game.snake[0], next_node)
                        else:
                            action = agent.act(state)
                    else:
                        action = agent.act(state)

                    next_state, reward, game_over = game.step(action, player_type="bot", mechanism=alg, training=True)
                    total_reward += reward
                    next_state = np.reshape(next_state, [1, 1, 10])  # Update the state shape
                    if alg == 'dqn':
                        agent.train(state, action, reward, next_state, game.game_over)
                    state = next_state

                    steps += 1

                results[alg]['episodes'] += 1
                results[alg]['total_reward'] += total_reward
                results[alg]['history'].append({'episode': episode, 'reward': total_reward, 'steps': steps})

                # Log game details at the end of each episode
                score = len(game.snake) - 1
                log_type = "Train"
                logger.info(f"{log_type} - {alg} Bot player, Score: {score}, Total reward: {total_reward}, Steps: {steps}")

                print(f'Episode: {episode}, Algorithm: {alg}, Total reward: {total_reward}, Steps: {steps}')

                # Save checkpoints
                if episode == 1 or episode == 10 or episode == 100 or episode % 1000 == 0:
                    checkpoint_path = f'dqn_model_{alg}_episode_{episode}.keras'
                    agent.model.save(checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        agent.model.save(model_path)
        print("Model saved.")
        save_results(results)
        for alg, result in results.items():
            avg_reward = result['total_reward'] / result['episodes'] if result['episodes'] > 0 else 0
            elapsed_time = time.time() - result['start_time']
            print(f'Algorithm: {alg}, Episodes: {result["episodes"]}, Average reward: {avg_reward}, Elapsed time: {elapsed_time:.2f} seconds')
