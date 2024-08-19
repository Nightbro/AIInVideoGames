import pygame
import random
from collections import deque
import heapq
import logging
import datetime

WINDOW_SIZE = 500
CELL_SIZE = 20
NUM_CELLS = WINDOW_SIZE // CELL_SIZE
REWARD_EAT = 1
REWARD_STAY_ALIVE = 0
REWARD_DIE = -1
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)




def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('snake_game')
    
    file_handler = logging.FileHandler('game_logs.txt')
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d.%m.%Y %H:%M:%S')
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class SnakeGame:
    def __init__(self, render=False):
        self.render_game = render
        if self.render_game:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        self.logger = setup_logging()
        self.reset()

    def reset(self):
        self.snake = [(NUM_CELLS // 2, NUM_CELLS // 2)]
        self.direction = (0, 1)
        self.food = self.place_food()
        self.game_over = False
        self.moves = 0  # Initialize move counter
        self.turns = 0  # Initialize turn counter
        return self.get_state()

    def place_food(self):
        while True:
            food = (random.randint(0, NUM_CELLS - 1), random.randint(0, NUM_CELLS - 1))
            if food not in self.snake:
                return food

    def step(self, action=None, player_type="bot", mechanism="", training=False):
        reason = ""
        if action is not None:
            # Detect if there is a turn (direction change)
            if (action == 0 and self.direction != (-1, 0)) or (action == 1 and self.direction != (1, 0)) or \
               (action == 2 and self.direction != (0, -1)) or (action == 3 and self.direction != (0, 1)):
                self.turns += 1  # Increment turn counter
            
            if action == 0 and self.direction != (1, 0):  # UP
                self.direction = (-1, 0)
            elif action == 1 and self.direction != (-1, 0):  # DOWN
                self.direction = (1, 0)
            elif action == 2 and self.direction != (0, 1):  # LEFT
                self.direction = (0, -1)
            elif action == 3 and self.direction != (0, -1):  # RIGHT
                self.direction = (0, 1)
        
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # Check for wall collision
        if new_head[0] < 0 or new_head[0] >= NUM_CELLS or new_head[1] < 0 or new_head[1] >= NUM_CELLS:
            self.game_over = True
            reason = "Hit the wall"
            self.log_game(player_type, mechanism, training, reason)
            return self.get_state(), REWARD_DIE, True

        # Check for self-collision
        if new_head in self.snake:
            self.game_over = True
            reason = "Hit itself"
            self.log_game(player_type, mechanism, training, reason)
            return self.get_state(), REWARD_DIE, True

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self.place_food()
            reward = REWARD_EAT
        else:
            self.snake.pop()
            reward = REWARD_STAY_ALIVE

        self.moves += 1  # Increment move counter
        return self.get_state(), reward, False
    
    def log_game(self, player_type, mechanism, training, reason):
        score = len(self.snake) - 1
        log_type = "Train" if training else "Play"
        snake_length = len(self.snake)
        moves = self.moves
        turns = self.turns

        if player_type == "human":
            self.logger.info(f"{log_type} - Human player, Score: {score}, Reason for death: {reason}, Moves: {moves}, Snake Length: {snake_length}, Turns: {turns}")
        else:
            self.logger.info(f"{log_type} - {mechanism} Bot player, Score: {score}, Reason for death: {reason}, Moves: {moves}, Snake Length: {snake_length}, Turns: {turns}")


    def render(self):
        if not self.render_game:
            return
        self.screen.fill((0, 0, 0))
        for cell in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, (cell[1] * CELL_SIZE, cell[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, FOOD_COLOR, (self.food[1] * CELL_SIZE, self.food[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()

    def close(self):
        if self.render_game:
            pygame.quit()

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        direction_x, direction_y = self.direction
        distance_to_wall_top = head_y
        distance_to_wall_bottom = NUM_CELLS - head_y - 1
        distance_to_wall_left = head_x
        distance_to_wall_right = NUM_CELLS - head_x - 1
        return [
            head_x, head_y, food_x, food_y, direction_x, direction_y,
            distance_to_wall_top, distance_to_wall_bottom, distance_to_wall_left, distance_to_wall_right
        ]

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

    def get_neighbors(self, node):
        neighbors = []
        x, y = node
        if x > 0:
            neighbors.append((x - 1, y))
        if x < NUM_CELLS - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < NUM_CELLS - 1:
            neighbors.append((x, y + 1))
        return neighbors
    
    def kill_snake(self, reason, player_type="bot", mechanism="", training=False):
        self.game_over = True
        self.log_game(player_type, mechanism, training, reason)
        self.reset()

    def bfs(self):
        start = self.snake[0]
        goal = self.food
        queue = deque([(start, [])])
        visited = set()

        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and neighbor not in self.snake:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []

    def dfs(self):
        start = self.snake[0]
        goal = self.food
        stack = [(start, [])]
        visited = set()

        while stack:
            current, path = stack.pop()
            if current == goal:
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and neighbor not in self.snake:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))

        return []

    def heuristic(self, node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def a_star(self):
        start = self.snake[0]
        goal = self.food
        open_set = []
        heapq.heappush(open_set, (0, start, []))
        g_score = {start: 0}
        visited = set()

        while open_set:
            _, current, path = heapq.heappop(open_set)
            if current == goal:
                return path

            for neighbor in self.get_neighbors(current):
                # Skip neighbors that have already been visited or are part of the snake
                if neighbor in visited or neighbor in self.snake:
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor, path + [neighbor]))

            visited.add(current)

        # No path found; return an empty list or trigger a fallback
        print("a star timeout! Resetting game...")
        return []

