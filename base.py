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


@dataclass
class Episode:
    decisions: list[tuple[Board, Move]]
    last_position: Board
    value: float


class ValueOptimizer:
    def train(self, episodes: list[Episode], estimator: ValueEstimator) -> None:
        raise NotImplementedError


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


def evaluate_against_random(game: Game, player: Player, name: str, n_games: int = 1000) -> float:
    loss, draw, win = gather_stats(game, player, RandomPlayer(game), n_games)
    print(f"Out of {n_games} games, {name}: wins: {win/n_games:5.1%}, draws: {draw/n_games:5.1%}, loss: {loss/n_games:5.1%}")


def train_self_play(game: Game, n_iters: int, n_games_per_iter: int, estimator: ValueEstimator, exploration_rate: float, optimizer: ValueOptimizer) -> None:
    evaluate_against_random(game, ValuePlayer(game, estimator), "Before training")
    explorer = ValuePlayer(game, estimator, exploration_rate)
    for i in range(n_iters):
        episodes = [simulate_game(game, explorer, explorer) for _ in range(n_games_per_iter)]
        optimizer.train(episodes, estimator)
        evaluate_against_random(game, ValuePlayer(game, estimator), f"After {i + 1} iterations")
