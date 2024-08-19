import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # needs to be set to 0 to avoid an error with the model loading



import numpy as np
from game import FlappyBirdGame
from tensorflow.keras.models import load_model



def play_dq_agent():
    model = load_model("dq_agent_model.keras")  # Load the saved model
    game = FlappyBirdGame()

    state = game.get_state()
    while not game.is_done():
        q_values = model.predict(np.array([state]))[0]
        action = np.argmax(q_values)
        next_state, _ = game.step(action)
        state = next_state
        game.render()

    game.close()

if __name__ == "__main__":
    play_dq_agent()
