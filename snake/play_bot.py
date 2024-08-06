import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Needs to be set to 0 to avoid an error with the model loading

import numpy as np
import tensorflow as tf
from snake_game import SnakeGame
from agent import DQNAgent

def play_with_bot(model_path='dqn_model.keras', mechanism='Regular'):
    game = SnakeGame(render=True)
    agent = DQNAgent()
    
    agent.model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    
    game_over = False
    state = np.reshape(game.reset(), [1, 1, 10])  # Update the state shape

    while not game_over:
        action = agent.act(state)
        next_state, _, game_over = game.step(action, player_type="bot", mechanism=mechanism, training=False)
        state = np.reshape(next_state, [1, 1, 10])  # Update the state shape
        game.render()
        game.clock.tick(10)

    game.close()

if __name__ == "__main__":
    play_with_bot()
