import torch
import torch.nn as nn
from base import Board, Episode, Game, ValueEstimator, ValueOptimizer


class FullGameTree(ValueEstimator):
    def __init__(self, game: Game):
        self.game = game
        self.values = {}
        self._compute_value(game.starting_position())
        self.index = {board: i for i, board in enumerate(self.values.keys())}

    def value(self, board: Board) -> float:
        return self.values[board]

    def _compute_value(self, board: Board) -> None:
        if board in self.values:
            return
        legal_moves, value = self.game.legal_moves(board)
        if value is not None:
            self.values[board] = value
            return
        for move in legal_moves:
            self._compute_value(self.game.play_move(board, move))
        self.values[board] = max(-self.values[self.game.play_move(board, move)] for move in legal_moves)


class VectorValueEstimator(ValueEstimator):
    def __init__(self, game: Game):
        self.index = FullGameTree(game).index
        self.values = [0] * len(self.index)

    def value(self, board: Board) -> float:
        return self.values[self.index[board]]


class PytorchVectorValueEstimator(ValueEstimator):
    def __init__(self, game: Game):
        self.index = FullGameTree(game).index
        self.values = nn.Parameter(torch.zeros(len(self.index), dtype=torch.float32))

    def value(self, board: Board) -> float:
        return self.values[self.index[board]]

    def parameters(self):
        return [self.values]

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        return self.values[indices]

    def embed(self, boards: list[Board]) -> torch.Tensor:
        return torch.tensor([self.index[board] for board in boards], dtype=torch.long)


class OptimizerNaiveAdd(ValueOptimizer):
    # Additive update

    def __init__(self, max_diff: float):
        self.max_diff = max_diff

    def train(self, episodes: list[Episode], estimator: VectorValueEstimator) -> None:
        for ep in episodes:
            for i, (board, _) in enumerate(ep.decisions):
                self.update_towards(estimator, board, ep.value * (-1)**i)
            self.update_towards(estimator, ep.last_position, ep.value * (-1)**len(ep.decisions))

    def update_towards(self, estimator: VectorValueEstimator, board: Board, new_value: float) -> None:
        i = estimator.index[board]
        d = new_value - estimator.values[i]
        if d > self.max_diff:
            d = self.max_diff
        elif d < -self.max_diff:
            d = -self.max_diff
        estimator.values[i] += d


class OptimizerNaiveGrad(ValueOptimizer):
    # Hand-implemented gradient descent

    def __init__(self, m: float):
        self.m = m

    def train(self, episodes: list[Episode], estimator: VectorValueEstimator) -> None:
        for ep in episodes:
            for i, (board, _) in enumerate(ep.decisions):
                self.update_towards(estimator, board, ep.value * (-1)**i)
            self.update_towards(estimator, ep.last_position, ep.value * (-1)**len(ep.decisions))

    def update_towards(self, estimator: VectorValueEstimator, board: Board, new_value: float) -> None:
        i = estimator.index[board]
        estimator.values[i] = (1 - self.m) * estimator.values[i] + self.m * new_value


class OptimizerNaiveTD(ValueOptimizer):
    # Naive TD on the value function

    def __init__(self, m: float):
        self.m = m

    def train(self, episodes: list[Episode], estimator: PytorchVectorValueEstimator) -> None:
        for ep in episodes:
            trajectory = [board for board, _ in ep.decisions] + [ep.last_position]
            self.update_towards(estimator, ep.last_position, ep.value * (-1)**len(ep.decisions))
            for i in reversed(range(1, len(trajectory))):
                self.update_towards(estimator, trajectory[i-1], -estimator.value(trajectory[i]))

    def update_towards(self, estimator: VectorValueEstimator, board: Board, new_value: float) -> None:
        i = estimator.index[board]
        estimator.values[i] = (1 - self.m) * estimator.values[i] + self.m * new_value
