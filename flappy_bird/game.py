import pygame
import random

class FlappyBirdGame:
    def __init__(self):
        pygame.init()
        self.width, self.height = 400, 600
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird")
        
        # Game settings
        self.bird_x = 60
        self.bird_y = 300
        self.bird_width, self.bird_height = 50, 35
        self.bird_vel_y = 0
        self.gravity = 0.6
        self.jump = -10
        
        self.pipe_width = 70
        self.pipe_gap = 200
        self.pipe_vel_x = -5
        self.pipe_min_height = 100
        self.pipe_max_height = 400
        
        self.pipes = [self.generate_pipe()]
        self.score = 0
        self.game_over = False

        # Load images (customize paths accordingly)
        self.background_img = pygame.image.load("background.png").convert()
        self.bird_img = [pygame.image.load("bird1.png").convert_alpha(), pygame.image.load("bird2.png").convert_alpha()]
        self.pipe_img = pygame.image.load("pipe.png").convert_alpha()

        # Scale images
        self.background_img = pygame.transform.scale(self.background_img, (self.width, self.height))
        self.bird_img = [pygame.transform.scale(img, (self.bird_width, self.bird_height)) for img in self.bird_img]
        self.pipe_img = pygame.transform.scale(self.pipe_img, (self.pipe_width, 400))

    def generate_pipe(self):
        height = random.randint(self.pipe_min_height, self.pipe_max_height)
        return {"x": self.width, "top": height - self.pipe_gap // 2, "bottom": height + self.pipe_gap // 2}

    def reset(self):
        self.bird_y = 300
        self.bird_vel_y = 0
        self.pipes = [self.generate_pipe()]
        self.score = 0
        self.game_over = False

    def get_state(self):
        nearest_pipe = min(self.pipes, key=lambda p: p['x'] + self.pipe_width > self.bird_x)
        # Simplified state representation: [bird's y-position, vertical distance to next pipe's gap]
        state = (
            self.bird_y / self.height,  # Normalize y-position
            (nearest_pipe['top'] + nearest_pipe['bottom']) / 2 - self.bird_y  # Distance to gap center
        )
        return state

    def step(self, action):
        if action == 1:
            self.bird_vel_y = self.jump

        self.bird_vel_y += self.gravity
        self.bird_y += self.bird_vel_y

        for pipe in self.pipes:
            pipe["x"] += self.pipe_vel_x

        if self.pipes[-1]["x"] < self.width - 300:
            self.pipes.append(self.generate_pipe())
        if self.pipes[0]["x"] < -self.pipe_width:
            self.pipes.pop(0)
            self.score += 1

        # Check for collisions
        self.game_over = False
        for pipe in self.pipes:
            if (self.bird_x + self.bird_width > pipe["x"] and self.bird_x < pipe["x"] + self.pipe_width):
                if self.bird_y < pipe["top"] or self.bird_y + self.bird_height > pipe["bottom"]:
                    self.game_over = True

        if self.bird_y > self.height - self.bird_height or self.bird_y < 0:
            self.game_over = True

        reward = 1 if not self.game_over else -100
        return self.get_state(), reward

    def render(self):
        self.win.blit(self.background_img, (0, 0))
        for pipe in self.pipes:
            self.win.blit(self.pipe_img, (pipe["x"], pipe["top"] - self.pipe_img.get_height()))
            self.win.blit(self.pipe_img, (pipe["x"], pipe["bottom"]))
        self.win.blit(self.bird_img[0], (self.bird_x, self.bird_y))
        pygame.display.update()

    def is_done(self):
        return self.game_over

    def close(self):
        pygame.quit()
    