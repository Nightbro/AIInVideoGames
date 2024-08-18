import numpy as np
import random

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.game_over = False
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.game_over = False
        self.add_random_tile()
        self.add_random_tile()
        return self.get_state()

    def add_random_tile(self):
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i, j] = 2 if random.random() < 0.9 else 4

    def move(self, direction):
        old_board = self.board.copy()
        if direction == 0:  # Up
            self.board = self.board.transpose()
            self._move_left()
            self.board = self.board.transpose()
        elif direction == 1:  # Down
            self.board = self.board.transpose()
            self._move_right()
            self.board = self.board.transpose()
        elif direction == 2:  # Left
            self._move_left()
        elif direction == 3:  # Right
            self._move_right()
        
        if not np.array_equal(old_board, self.board):
            self.add_random_tile()
            if not self.can_move():
                self.game_over = True
            return self.get_state(), self.score, self.game_over, True
        else:
            return self.get_state(), self.score, self.game_over, False

    def _move_left(self):
        for i in range(4):
            self.board[i] = self._merge_row(self.board[i])

    def _move_right(self):
        for i in range(4):
            self.board[i] = self._merge_row(self.board[i][::-1])[::-1]

    def _merge_row(self, row):
        new_row = [i for i in row if i != 0]
        merged_row = []
        skip = False
        for i in range(len(new_row)):
            if skip:
                skip = False
                continue
            if i + 1 < len(new_row) and new_row[i] == new_row[i + 1]:
                merged_row.append(new_row[i] * 2)
                self.score += new_row[i] * 2
                skip = True
            else:
                merged_row.append(new_row[i])
        merged_row += [0] * (4 - len(merged_row))
        return merged_row

    def can_move(self):
        if np.any(self.board == 0):
            return True
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j + 1] or self.board[j, i] == self.board[j + 1, i]:
                    return True
        return False

    def get_state(self):
        return self.board.flatten()

    def print_board(self):
        print(self.board)
        print(f"Score: {self.score}")


