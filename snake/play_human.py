import pygame
import sys
from snake_game import SnakeGame

def play_game():
    pygame.init()
    game = SnakeGame(render=True)  #  rendering is enabled

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                pygame.quit()
                sys.exit()
        
        keys = pygame.key.get_pressed()
        action = game.human_input(keys)
        state, reward, game_over = game.step(action)
        
        if game_over:
            game.reset()

        game.render()  
        game.clock.tick(10) 

if __name__ == "__main__":
    play_game()
