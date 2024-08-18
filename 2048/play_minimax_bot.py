import numpy as np
import random
from game import Game2048

class MinimaxBot:
    def __init__(self, depth=3):
        self.depth = depth

    def get_move(self, game):
        best_move = None
        best_score = -float('inf')
        for move in range(4):
            cloned_game = self.clone_game(game)
            _, score, _, changed = cloned_game.move(move)
            if changed:
                move_score = self.minimax(cloned_game, self.depth - 1, False)
                if move_score > best_score:
                    best_score = move_score
                    best_move = move
        return best_move

    def minimax(self, game, depth, is_maximizing):
        if depth == 0 or game.game_over:
            return game.score

        if is_maximizing:
            best_score = -float('inf')
            for move in range(4):
                cloned_game = self.clone_game(game)
                _, score, _, changed = cloned_game.move(move)
                if changed:
                    move_score = self.minimax(cloned_game, depth - 1, False)
                    best_score = max(best_score, move_score)
            return best_score
        else:
            # Simulate random tile placement
            best_score = float('inf')
            empty_cells = [(i, j) for i in range(4) for j in range(4) if game.board[i, j] == 0]
            for cell in empty_cells:
                cloned_game = self.clone_game(game)
                i, j = cell
                cloned_game.board[i, j] = 2 if random.random() < 0.9 else 4
                move_score = self.minimax(cloned_game, depth - 1, True)
                best_score = min(best_score, move_score)
            return best_score

    def clone_game(self, game):
        cloned_game = Game2048()
        cloned_game.board = game.board.copy()
        cloned_game.score = game.score
        return cloned_game

if __name__ == "__main__":
    game = Game2048()
    bot = MinimaxBot(depth=3)
    
    while not game.game_over:
        move = bot.get_move(game)
        if move is not None:
            state, score, game_over, changed = game.move(move)
            game.print_board()