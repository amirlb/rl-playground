from dataclasses import dataclass
from math import inf
import random
from typing import TypeVar, Protocol

Board = TypeVar("Board")
Move = TypeVar("Move")


class Game(Protocol):
    def starting_position(self) -> Board:
        raise NotImplementedError

    def legal_moves(self, board: Board) -> tuple[list[Move], float | None]:
        """List of legal moves, and game value if terminal"""
        raise NotImplementedError

    def play_move(self, board: Board, move: Move) -> Board:
        raise NotImplementedError


class Player:
    def choose_move(self, board: Board) -> Move:
        raise NotImplementedError


class RandomPlayer(Player):
    def __init__(self, game: Game):
        self.game = game

    def choose_move(self, board: Board) -> Move:
        moves, _ = self.game.legal_moves(board)
        return random.choice(moves)


class ValueEstimator:
    def value(self, board: Board) -> float:
        """Float between -1 and 1"""
        raise NotImplementedError


class ValuePlayer(Player):
    def __init__(self, game: Game, value_estimator: ValueEstimator, exploration_rate: float = 0.0):
        self.game = game
        self.value_estimator = value_estimator
        self.exploration_rate = exploration_rate

    def choose_move(self, board: Board) -> Move:
        moves, value = [], -inf
        legal_moves, _ = self.game.legal_moves(board)
        if self.exploration_rate > 0 and random.random() < self.exploration_rate:
            return random.choice(legal_moves)
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

    def __init__(self, game):
        self.game = game
        self.values = {}
        self.compute_value(game.starting_position())

    def value(self, board: Board) -> float:
        return self.values[board]

    def compute_value(self, board: Board) -> None:
        if board in self.values:
            return
        legal_moves, value = self.game.legal_moves(board)
        if value is not None:
            self.values[board] = value
            return
        for move in legal_moves:
            self.compute_value(self.game.play_move(board, move))
        self.values[board] = max(-self.values[self.game.play_move(board, move)] for move in legal_moves)


class GameTreeIndexer(dict):
    def __init__(self, game: Game):
        super().__init__({board: i for i, board in enumerate(OptimalValueEstimator(game).values.keys())})


@dataclass
class Episode:
    decisions: list[tuple[Board, Move]]
    last_position: Board
    value: float


def simulate_game(game: Game, player1: Player, player2: Player) -> Episode:
    board = game.starting_position()
    decisions = []
    player_value = 1
    while True:
        _, value = game.legal_moves(board)
        if value is not None:
            return Episode(decisions, board, value * player_value)
        move = player1.choose_move(board)
        decisions.append((board, move))
        board = game.play_move(board, move)
        player1, player2, player_value = player2, player1, -player_value


def gather_stats(game: Game, player1: Player, player2: Player, n_games: int) -> list[int]:
    results = [0, 0, 0]
    for _ in range(n_games):
        if random.random() < 0.5:
            episode = simulate_game(game, player1, player2)
            results[1 + episode.value] += 1
        else:
            episode = simulate_game(game, player2, player1)
            results[1 - episode.value] += 1
    return results
