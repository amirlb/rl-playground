from math import inf
import random


class Game:
    def starting_position(self):
        raise NotImplementedError

    def legal_moves(self, board):
        """List of legal moves, and game value if terminal"""
        raise NotImplementedError

    def play_move(self, board, move):
        raise NotImplementedError


class Player:
    def choose_move(self, board):
        raise NotImplementedError


class RandomPlayer(Player):
    def __init__(self, game):
        self.game = game

    def choose_move(self, board):
        moves, _ = self.game.legal_moves(board)
        return random.choice(moves)


class ValueEstimator:
    def value(self, board):
        """Float between -1 and 1"""
        raise NotImplementedError


class ValuePlayer(Player):
    def __init__(self, game, value_estimator: ValueEstimator):
        self.game = game
        self.value_estimator = value_estimator

    def choose_move(self, board):
        moves, value = [], -inf
        legal_moves, _ = self.game.legal_moves(board)
        for move in legal_moves:
            v = -self.value_estimator.value(self.game.play_move(board, move))
            if v > value:
                moves = [move]
                value = v
            elif v == value:
                moves.append(move)
        return random.choice(moves)


class OptimalValueEstimator(ValueEstimator):
    """Only use for games with a small number of states"""

    VALUES = {}

    def __init__(self, game):
        self.game = game

    def value(self, board):
        if board not in self.VALUES:
            self.VALUES[board] = self.compute_value(board)
        return self.VALUES[board]

    def compute_value(self, board):
        legal_moves, value = self.game.legal_moves(board)
        if value is not None:
            return value
        return max(-self.value(self.game.play_move(board, move)) for move in legal_moves)


def simulate_game(game, player1, player2):
    board = game.starting_position()
    positions = []
    player_value = 1
    while True:
        positions.append(board)
        _, value = game.legal_moves(board)
        if value is not None:
            return (positions, value * player_value)
        board = game.play_move(board, player1.choose_move(board))
        player1, player2, player_value = player2, player1, -player_value


def gather_stats(game, player1, player2, n_games):
    results = [0, 0, 0]
    player_value = 1
    for _ in range(n_games):
        if random.random() < 0.5:
            player1, player2, player_value = player2, player1, -player_value
        _, result = simulate_game(game, player1, player2)
        results[1 - result * player_value] += 1
    return results
