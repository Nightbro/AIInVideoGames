import pygame
import random

WINDOW_SIZE = 500
CELL_SIZE = 20
NUM_CELLS = WINDOW_SIZE // CELL_SIZE
REWARD_EAT = 1
REWARD_STAY_ALIVE = 0
REWARD_DIE = -1
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(NUM_CELLS//2, NUM_CELLS//2)]
        self.direction = (0, 1)
        self.food = self.place_food()
        self.game_over = False
        return self.get_state()

    def place_food(self):
        while True:
            food = (random.randint(0, NUM_CELLS-1), random.randint(0, NUM_CELLS-1))
            if food not in self.snake:
                return food

    def step(self, action=None):
        if action is not None: # maintain direction if no action
            if action == 0 and self.direction != (1, 0):   # action K_UP
                self.direction = (-1, 0)
            elif action == 1 and self.direction != (-1, 0): # action K_DOWN
                self.direction = (1, 0)
            elif action == 2 and self.direction != (0, 1): # action K_LEFT
                self.direction = (0, -1)
            elif action == 3 and self.direction != (0, -1): # action K_RIGHT
                self.direction = (0, 1)
        
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        if new_head[0] < 0 or new_head[0] >= NUM_CELLS or new_head[1] < 0 or new_head[1] >= NUM_CELLS or new_head in self.snake:
            self.game_over = True
            return self.get_state(), REWARD_DIE, True

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self.place_food()
            reward = REWARD_EAT 
        else:
            self.snake.pop()
            reward = REWARD_STAY_ALIVE

        return self.get_state(), reward, False

    def render(self):
        self.screen.fill((0, 0, 0))
        for cell in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, (cell[1]*CELL_SIZE, cell[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, FOOD_COLOR, (self.food[1]*CELL_SIZE, self.food[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()

    def close(self):
        pygame.quit()

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        direction_x, direction_y = self.direction
        return [head_x, head_y, food_x, food_y, direction_x, direction_y]

    def human_input(self, keys):
        if keys[pygame.K_UP]:
            return 0
        elif keys[pygame.K_DOWN]:
            return 1
        elif keys[pygame.K_LEFT]:
            return 2
        elif keys[pygame.K_RIGHT]:
            return 3
        return None
