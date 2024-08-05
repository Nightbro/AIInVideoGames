import pygame
from snake_game import SnakeGame

def play_game():
    game = SnakeGame()
    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        keys = pygame.key.get_pressed()
        action = game.human_input(keys)
        if action != -1:
            _, _, game_over = game.step(action)
        
        game.render()
        game.clock.tick(10)  # Set the game speed to 10 frames per second

    game.close()

if __name__ == "__main__":
    play_game()
