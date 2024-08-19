import numpy as np
import random
import time
from game import Game2048

class MCTSNode:
    def __init__(self, game, move=None, parent=None):
        self.game = self.clone_game(game)
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def clone_game(self, game):
        cloned_game = Game2048()
        cloned_game.board = game.board.copy()
        cloned_game.score = game.score
        cloned_game.game_over = game.game_over
        return cloned_game

    def expand(self):
        if self.is_fully_expanded():
            return
        for move in range(4):
            cloned_game = self.clone_game(self.game)
            _, score, game_over, changed = cloned_game.move(move)
            if changed:
                child_node = MCTSNode(cloned_game, move, self)
                self.children.append(child_node)

    def is_fully_expanded(self):
        return len(self.children) == 4

    def best_child(self, exploration_weight=1.41):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            uct_score = child.wins / (child.visits + 1) + exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1))
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    def simulate(self):
        cloned_game = self.clone_game(self.game)
        while not cloned_game.game_over:
            valid_moves = [0, 1, 2, 3]
            move = random.choice(valid_moves)
            _, _, game_over, _ = cloned_game.move(move)
            if game_over:
                break
        return cloned_game.score

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

class MCTSBot:
    def __init__(self, simulations=100, time_limit=5):
        self.simulations = simulations
        self.time_limit = time_limit

    def get_move(self, game):
        root = MCTSNode(game)
        root.expand()

        start_time = time.time()
        simulation_count = 0

        while time.time() - start_time < self.time_limit and simulation_count < self.simulations:
            node = self.select_node(root)
            if not node.is_fully_expanded():
                node.expand()
            leaf = self.select_node(node)
            result = leaf.simulate()
            leaf.backpropagate(result)
            simulation_count += 1

        best_move = root.best_child(exploration_weight=0).move
        return best_move

    def select_node(self, node):
        while not node.game.game_over and node.is_fully_expanded():
            node = node.best_child()
        return node

if __name__ == "__main__":
    game = Game2048()
    bot = MCTSBot(simulations=100, time_limit=5)

    while not game.game_over:
        move = bot.get_move(game)
        if move is not None:
            state, score, game_over, changed = game.move(move)
            game.print_board()
