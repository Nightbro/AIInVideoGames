import pygame
from game import Game2048

WINDOW_SIZE = 400
CELL_SIZE = WINDOW_SIZE // 4
GRID_COLOR = (187, 173, 160)
EMPTY_CELL_COLOR = (205, 193, 180)
TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

FONT_COLOR = (119, 110, 101)

class Game2048GUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("2048")
        self.font = pygame.font.SysFont("arial", 40)
        self.game = Game2048()
        self.render()

    def render(self):
        self.screen.fill(GRID_COLOR)
        for i in range(4):
            for j in range(4):
                value = self.game.board[i, j]
                color = TILE_COLORS.get(value, EMPTY_CELL_COLOR)
                pygame.draw.rect(self.screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                if value != 0:
                    text_surface = self.font.render(str(value), True, FONT_COLOR)
                    text_rect = text_surface.get_rect(center=(j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2))
                    self.screen.blit(text_surface, text_rect)
                pygame.draw.rect(self.screen, GRID_COLOR, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        pygame.display.flip()

    def handle_key(self, key):
        if key == pygame.K_UP:
            state, score, game_over, changed = self.game.move(0)
        elif key == pygame.K_DOWN:
            state, score, game_over, changed = self.game.move(1)
        elif key == pygame.K_LEFT:
            state, score, game_over, changed = self.game.move(2)
        elif key == pygame.K_RIGHT:
            state, score, game_over, changed = self.game.move(3)
        if changed:
            self.render()
        if game_over:
            pygame.quit()

    def play(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
        pygame.quit()

if __name__ == "__main__":
    gui = Game2048GUI()
    gui.play()
