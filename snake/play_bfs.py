import pygame
import sys
from snake_game import SnakeGame

def play_bfs_bot(num_simulations=500):
    pygame.init()
    game = SnakeGame(render=True)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                pygame.quit()
                sys.exit()

        path = game.bfs()

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
            state, reward, game_over = game.step(action, player_type="bot", mechanism="BFS", training=False)

            if game_over:
                game.reset()
                num_simulations -= 1
        else:
            game.kill_snake("No path found", player_type="bot", mechanism="BFS", training=False)
            num_simulations -= 1

        if num_simulations<0:
            break


        game.render()
        game.clock.tick(10)

if __name__ == "__main__":
    play_bfs_bot(num_simulations=500)