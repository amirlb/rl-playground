import random
import torch
from base import Episode, Game, GameTreeIndexer, OptimalValueEstimator, Player, RandomPlayer, ValueEstimator, ValuePlayer, gather_stats, simulate_game


TTTPosition = tuple[int, int]
TTTMove = int


class TicTacToeGame(Game):
    LINES = [0b000000111, 0b000111000, 0b111000000, 0b001001001, 0b010010010, 0b100100100, 0b100010001, 0b001010100]

    @classmethod
    def starting_position(cls) -> TTTPosition:
        return (0, 0)

    @classmethod
    def fmt(cls, board: TTTPosition) -> str:
        my, other = board
        return " ".join("".join("X" if my & (1 << i) else "O" if other & (1 << i) else "." for i in range(j, j+3)) for j in range(0, 9, 3))

    @classmethod
    def legal_moves(cls, board: TTTPosition) -> tuple[list[TTTMove], float | None]:
        my, other = board
        whole = my | other
        if whole == 0b111111111:
            return [], 0
        for line in cls.LINES:
            if my & line == line:
                return [], 1
            if other & line == line:
                return [], -1
        return [i for i in range(9) if whole & (1 << i) == 0], None

    @classmethod
    def play_move(cls, board: TTTPosition, move: TTTMove) -> TTTPosition:
        my, other = board
        whole = my | other
        if whole & (1 << move):
            raise ValueError(f"Cell {move} is not empty")
        return (other, my | (1 << move))


class VectorValueEstimator(ValueEstimator):
    def __init__(self):
        self.index = GameTreeIndexer(TicTacToeGame)
        self.values = [0] * len(self.index)
        # opt = OptimalValueEstimator(TicTacToeGame)
        # for board, ind in self.index.items():
        #     self.values[ind] = opt.value(board)

    def value(self, board: TTTPosition) -> float:
        return self.values[self.index[board]]


class PytorchVectorValueEstimator(ValueEstimator):
    def __init__(self):
        self.index = GameTreeIndexer(TicTacToeGame)
        self.values = torch.zeros(len(self.index))

    def value(self, board: TTTPosition) -> float:
        return self.values[self.index[board]]

    def update(self, board: TTTPosition, value: float) -> None:
        self.values[self.index[board]] = value


class ValueOptimizer:
    def train(self, episodes: list[Episode], estimator: ValueEstimator) -> None:
        raise NotImplementedError


def train_self_play(game: Game, n_iters: int, n_games_per_iter: int, estimator: ValueEstimator, exploration_rate: float, optimizer: ValueOptimizer) -> None:
    evaluate_player(ValuePlayer(game, estimator), "Before training")
    explorer = ValuePlayer(game, estimator, exploration_rate)
    for i in range(n_iters):
        episodes = [simulate_game(game, explorer, explorer) for _ in range(n_games_per_iter)]
        optimizer.train(episodes, estimator)
        evaluate_player(ValuePlayer(game, estimator), f"After {i + 1} iterations")


class OptimizerNaive1(ValueOptimizer):
    def __init__(self, max_diff: float):
        self.max_diff = max_diff

    def train(self, episodes: list[Episode], estimator: VectorValueEstimator) -> None:
        for ep in episodes:
            for i, (board, _) in enumerate(ep.decisions):
                self.update_towards(estimator, board, ep.value * (-1)**i)
            self.update_towards(estimator, ep.last_position, ep.value * (-1)**len(ep.decisions))

    def update_towards(self, estimator: VectorValueEstimator, board: TTTPosition, new_value: float) -> None:
        i = estimator.index[board]
        d = new_value - estimator.values[i]
        if d > self.max_diff:
            d = self.max_diff
        elif d < -self.max_diff:
            d = -self.max_diff
        estimator.values[i] += d
        assert -1 <= estimator.values[i] <= 1


class OptimizerNaive2(ValueOptimizer):
    def __init__(self, m: float):
        self.m = m

    def train(self, episodes: list[Episode], estimator: VectorValueEstimator) -> None:
        for ep in episodes:
            for i, (board, _) in enumerate(ep.decisions):
                self.update_towards(estimator, board, ep.value * (-1)**i)
            self.update_towards(estimator, ep.last_position, ep.value * (-1)**len(ep.decisions))

    def update_towards(self, estimator: VectorValueEstimator, board: TTTPosition, new_value: float) -> None:
        i = estimator.index[board]
        estimator.values[i] = (1 - self.m) * estimator.values[i] + self.m * new_value
        assert -1 <= estimator.values[i] <= 1


def evaluate_player(player: Player, name: str, n_games: int = 1000) -> float:
    loss, draw, win = gather_stats(TicTacToeGame, player, RandomPlayer(TicTacToeGame), n_games)
    print(f"Out of {n_games} games, {name}: wins: {win/n_games:5.1%}, draws: {draw/n_games:5.1%}, loss: {loss/n_games:5.1%}")


def main():
    random.seed(123)
    evaluate_player(RandomPlayer(TicTacToeGame), "Random")
    evaluate_player(ValuePlayer(TicTacToeGame, OptimalValueEstimator(TicTacToeGame)), "Optimal")
    estimator = VectorValueEstimator()
    # train_self_play(TicTacToeGame, n_iters=500, n_games_per_iter=10, exploration_rate=0.2, optimizer=OptimizerNaive1(max_diff=0.1), estimator=estimator)
    # train_self_play(TicTacToeGame, n_iters=500, n_games_per_iter=10, exploration_rate=0.2, optimizer=OptimizerNaive2(m=0.1), estimator=estimator)
    train_self_play(TicTacToeGame, n_iters=500, n_games_per_iter=10, exploration_rate=0.2, optimizer=OptimizerNaive2(m=0.1), estimator=estimator)
    evaluate_player(ValuePlayer(TicTacToeGame, estimator), "After training")


if __name__ == '__main__':
    main()
