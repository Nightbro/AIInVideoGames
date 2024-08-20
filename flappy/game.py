import pygame
import random

# Initialize Pygame
pygame.init()

# Screen Dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BROWN = (150, 75, 0)


# Load Images
BIRD_IMAGE = pygame.image.load("images\\bird1.png")
BIRD_IMAGE_FLY = pygame.image.load("images\\bird2.png")
BACKGROUND_IMAGE = pygame.image.load("images\\background.png")
PIPE_IMAGE = pygame.image.load("images\\pipe.png")
#GROUND_IMAGE = pygame.image.load("images\\ground.png")


# Scale Images if necessary
BIRD_IMAGE = pygame.transform.scale(BIRD_IMAGE, (50, 35)) 
BIRD_IMAGE_FLY = pygame.transform.scale(BIRD_IMAGE_FLY, (50, 35)) 
PIPE_IMAGE = pygame.transform.scale(PIPE_IMAGE, (80, 400))
BACKGROUND_IMAGE = pygame.transform.scale(BACKGROUND_IMAGE, (400, 600)) 
#GROUND_IMAGE = pygame.transform.scale(GROUND_IMAGE, (SCREEN_WIDTH, 20))

# Classes

class Player:
    def __init__(self):
        self.x = 100
        self.y = 300
        self.sizeX= 50
        self.sizeY= 35
        self.velocity = 0
        self.gravity = 0.8
        self.lift = -20
        self.radius = 20
        self.alive = True
        self.image = BIRD_IMAGE

    def update(self):
        self.velocity += self.gravity
        self.velocity *= 0.9  # Air resistance
        self.y += self.velocity

        #if self.y >= SCREEN_HEIGHT - GROUND_IMAGE.get_height():
        #    self.y = SCREEN_HEIGHT - GROUND_IMAGE.get_height()
        if self.y >= SCREEN_HEIGHT - 20:
            self.y = SCREEN_HEIGHT - 20 
            self.velocity = 0
            self.alive = False  # Bird is dead if it hits the ground

        # Prevent the bird from falling out of the screen
        if self.y > SCREEN_HEIGHT - self.radius:
            self.y = SCREEN_HEIGHT - self.radius
            self.velocity = 0
            self.alive = False
        elif self.y < 0:
            self.y = 0
            self.velocity = 0
            self.alive = False 


    def jump(self):
        self.velocity += self.lift

    def draw(self, screen):
        #pygame.draw.circle(screen, (255, 255, 0), (self.x, int(self.y)), self.radius)
        #screen.blit(self.image, (self.x, self.y))
        
        rotated_image = pygame.transform.rotate(BIRD_IMAGE, 0)
        if self.velocity < 0:
            rotation = max(-25, self.velocity * 3)
            rotated_image = pygame.transform.rotate(BIRD_IMAGE_FLY, 0)
        else:
            rotation = min(90, self.velocity * 3)
            rotated_image = pygame.transform.rotate(BIRD_IMAGE, -rotation)
        
        screen.blit(rotated_image, (self.x, self.y))



class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 80
        self.gap = 150
        self.top = random.randint(50, 400)
        self.bottom = self.top + self.gap
        self.speed = 5

    def update(self):
        self.x -= self.speed

    def offscreen(self):
        return self.x < -self.width

    def draw(self, screen):
        # Draw top pipe (flipped image)
        top_pipe_image = pygame.transform.flip(PIPE_IMAGE, False, True)
        screen.blit(top_pipe_image, (self.x, self.top - top_pipe_image.get_height()))

        # Draw bottom pipe
        screen.blit(PIPE_IMAGE, (self.x, self.bottom))


class PipePair:
    def __init__(self):
        self.pipes = [Pipe(SCREEN_WIDTH)]

    def add_new_pipe(self):
        self.pipes.append(Pipe(SCREEN_WIDTH))

    def update(self):
        for pipe in self.pipes:
            pipe.update()
        
        if self.pipes[0].offscreen():
            self.pipes.pop(0)
        
        if self.pipes[-1].x < SCREEN_WIDTH / 2:
            self.pipes.append(Pipe(SCREEN_WIDTH))        

    def draw(self, screen):
        for pipe in self.pipes:
            pipe.draw(screen)

    def check_collision(self, player):
        for pipe in self.pipes:
            if player.x + player.sizeX > pipe.x and player.x < pipe.x + pipe.width:
                if player.y < pipe.top or player.y + player.sizeY > pipe.bottom:
                    player.alive = False  # Bird hits the pipe

        # Check ground collision
        if player.y >= SCREEN_HEIGHT - 20:
        #if player.y >= SCREEN_HEIGHT - GROUND_IMAGE.get_height():
            player.alive = False


class Ground:
    def __init__(self):
        self.x = 0
        self.y = SCREEN_HEIGHT - 20
        self.width = SCREEN_WIDTH
        self.height = 20
        self.speed = 5

    def update(self):
        self.x -= self.speed
        if self.x <= -self.width:
            self.x = 0

    def draw(self, screen):
        pygame.draw.rect(screen, BROWN, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, BROWN, (self.x + self.width, self.y, self.width, self.height))
        
    #def draw(self, screen):
        #screen.blit(GROUND_IMAGE, (self.x, self.y))
        #screen.blit(GROUND_IMAGE, (self.x + SCREEN_WIDTH, self.y))


# Main Game Loop

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    player = Player()
    pipe_pair = PipePair()
    ground = Ground()

    running = True
    while running:
        screen.fill((135, 206, 235))  # Sky blue background

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    player.jump()

        # Update game objects
        player.update()
        pipe_pair.update()
        ground.update()

        # Collision detection
        pipe_pair.check_collision(player)
        if not player.alive:
            running = False  # End the game if the player hits a pipe

        # Draw game objects
        pipe_pair.draw(screen)
        player.draw(screen)
        ground.draw(screen)

        pygame.display.update()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
