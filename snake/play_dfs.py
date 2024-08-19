import pygame
import sys
from snake_game import SnakeGame

def play_dfs_bot(num_simulations=10):
    pygame.init()
    game = SnakeGame(render=True)

    turns_without_eating = 0  # 
    max_turns_without_eating = 625  # size of the board (500/25)^2
    

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                pygame.quit()
                sys.exit()

        path = game.dfs()
        REWARD_WHEN_ATE = 0

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

            if reward == REWARD_WHEN_ATE+1:
                turns_without_eating = 0
            else:
                turns_without_eating += 1
                REWARD_WHEN_ATE = reward

            if turns_without_eating >= max_turns_without_eating:
                game.kill_snake("Snake goes in circles", player_type="bot", mechanism="DFS", training=False)
                num_simulations -= 1
                continue 

            if game_over:
                game.reset()
                num_simulations -= 1

        else:
            game.kill_snake("No path found", player_type="bot", mechanism="DFS", training=False)
            num_simulations -= 1

        if num_simulations<0:
            break

        game.render()
        game.clock.tick(10)

if __name__ == "__main__":
    play_dfs_bot(num_simulations=500)
