import pygame
import sys
from snake_game import SnakeGame

def play_dfs_bot():
    pygame.init()
    game = SnakeGame(render=True)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                pygame.quit()
                sys.exit()

        path = game.dfs()

        if path:
            next_step = path[0]
            snake_head = game.snake[0]
            action = None

            if next_step[0] < snake_head[0]: 
                action = 0
            elif next_step[0] > snake_head[0]: 
                action = 1
            elif next_step[1] < snake_head[1]:
                action = 2
            elif next_step[1] > snake_head[1]: 
                action = 3

            state, reward, game_over = game.step(action, player_type="bot", mechanism="DFS", training=False)

            if game_over:
                game.reset()

        game.render()
        game.clock.tick(10)

if __name__ == "__main__":
    play_dfs_bot()