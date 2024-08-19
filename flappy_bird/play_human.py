import pygame
from game import FlappyBirdGame

def play_human():
    game = FlappyBirdGame()
    running = True
    clock = pygame.time.Clock()

    while running:
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.step(1)  # Jump when space is pressed

        game.step(0)  # Continue game logic even without jumping
        game.render()

        if game.is_done():
            game.reset()  # Reset the game automatically after the bird dies

    game.close()

if __name__ == "__main__":
    play_human()
