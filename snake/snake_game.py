import pygame
import random
from collections import deque
import heapq

WINDOW_SIZE = 500
CELL_SIZE = 20
NUM_CELLS = WINDOW_SIZE // CELL_SIZE
REWARD_EAT = 1
REWARD_STAY_ALIVE = 0
REWARD_DIE = -1
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)

class SnakeGame:
    def __init__(self, render=False):
        self.render_game = render  # Changed to render_game to avoid conflict
        if self.render_game:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(NUM_CELLS // 2, NUM_CELLS // 2)]
        self.direction = (0, 1)
        self.food = self.place_food()
        self.game_over = False
        return self.get_state()

    def place_food(self):
        while True:
            food = (random.randint(0, NUM_CELLS - 1), random.randint(0, NUM_CELLS - 1))
            if food not in self.snake:
                return food

    def step(self, action=None):
        if action is not None:  # maintain direction if no action
            if action == 0 and self.direction != (1, 0):  # action K_UP
                self.direction = (-1, 0)
            elif action == 1 and self.direction != (-1, 0):  # action K_DOWN
                self.direction = (1, 0)
            elif action == 2 and self.direction != (0, 1):  # action K_LEFT
                self.direction = (0, -1)
            elif action == 3 and self.direction != (0, -1):  # action K_RIGHT
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
        if keys[pygame.K_UP] and self.direction != (1, 0):
            return 0
        elif keys[pygame.K_DOWN] and self.direction != (-1, 0):
            return 1
        elif keys[pygame.K_LEFT] and self.direction != (0, 1):
            return 2
        elif keys[pygame.K_RIGHT] and self.direction != (0, -1):
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
                if neighbor in visited or neighbor in self.snake:
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor, path + [neighbor]))

            visited.add(current)

        return []
