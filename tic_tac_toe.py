import random
import torch
from base import Episode, Game, GameTreeIndexer, OptimalValueEstimator, Player, RandomPlayer, ValueEstimator, ValuePlayer, gather_stats, simulate_game
import torch.nn as nn


TTTPosition = tuple[int, int]
TTTMove = int


# Select best device: MPS (Mac NPU), CUDA, or CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    default_device = torch.device('mps')
elif torch.cuda.is_available():
    default_device = torch.device('cuda')
else:
    default_device = torch.device('cpu')
print(f"Using device: {default_device}")


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
        self.values = nn.Parameter(torch.zeros(len(self.index), dtype=torch.float32))

    def value(self, board: TTTPosition) -> float:
        return self.values[self.index[board]]


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


class OptimizerNaive3(ValueOptimizer):
    def __init__(self, lr: float):
        self.lr = lr

    def train(self, episodes: list[Episode], estimator: PytorchVectorValueEstimator) -> None:
        indices, targets = [], []
        for ep in episodes:
            for i, (board, _) in enumerate(ep.decisions):
                indices.append(estimator.index[board])
                targets.append(ep.value * (-1)**i)
            indices.append(estimator.index[ep.last_position])
            targets.append(ep.value * (-1)**len(ep.decisions))
        indices = torch.tensor(indices, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.float32)

        optimizer = torch.optim.SGD([estimator.values], lr=self.lr)
        preds = estimator.values[indices]
        loss = nn.MSELoss()(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        y = self.ln1(x)
        y, _ = self.attn(y, y, y)
        x = x + y
        y = self.mlp(self.ln2(x))
        return x + y


class DLValueEstimator(nn.Module, ValueEstimator):
    """Transformer-based network for score estimation."""
    def __init__(self, dim=16, n_blocks=2, n_heads=4, mlp_dim=32):
        super().__init__()
        # input features are fixed 16-dim: one-hot pos (9), self bit, opp bit, zeros (5)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_dim) for _ in range(n_blocks)
        ])
        # output score estimate (-1 to 1)
        self.head = nn.Linear(dim * 9, 1)

    def forward(self, x):  # x: (batch, 9, dim)
        x = x.permute(1, 0, 2)            # (9, batch, dim)
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(1, 0, 2)            # (batch, 9, dim)
        batch_size, seq_len, d = x.shape
        x = x.reshape(batch_size, seq_len * d)  # (batch, 9*dim)
        score = self.head(x)              # (batch, 1)
        return torch.tanh(score.squeeze(-1))          # (batch,)

    def value(self, board: TTTPosition) -> float:
        return self.forward(self._embed([board]).to(default_device)).item()

    @staticmethod
    def _embed(boards: list[TTTPosition]) -> torch.Tensor:
        return torch.stack([DLValueEstimator._embed_board(board) for board in boards])

    @staticmethod
    def _embed_board(board: TTTPosition) -> torch.Tensor:
        my, other = board
        return torch.tensor([
            [0] * i + [1] + [0] * (8-i) + [int(my & (1 << i))] + [int(other & (1 << i))] + [0] * 5
            for i in range(9)
        ], dtype=torch.float32)


class DLOptimizer(ValueOptimizer):
    def __init__(self, lr: float, epochs: int, batch_size: int = 32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, episodes: list[Episode], estimator: DLValueEstimator) -> None:
        data = []
        for ep in episodes:
            for i, (board, _) in enumerate(ep.decisions):
                data.append((board, ep.value * (-1)**i))
            data.append((ep.last_position, ep.value * (-1)**len(ep.decisions)))
        random.shuffle(data)
        states = estimator._embed([board for board, _ in data]).to(default_device)
        targets = torch.tensor([value for _, value in data], dtype=torch.float32).to(default_device)
        optimizer = torch.optim.SGD(estimator.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                preds = estimator(states[i:i+self.batch_size])
                loss = criterion(preds, targets[i:i+self.batch_size])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def evaluate_player(player: Player, name: str, n_games: int = 1000) -> float:
    loss, draw, win = gather_stats(TicTacToeGame, player, RandomPlayer(TicTacToeGame), n_games)
    print(f"Out of {n_games} games, {name}: wins: {win/n_games:5.1%}, draws: {draw/n_games:5.1%}, loss: {loss/n_games:5.1%}")


def main():
    random.seed(123)
    evaluate_player(RandomPlayer(TicTacToeGame), "Random")
    evaluate_player(ValuePlayer(TicTacToeGame, OptimalValueEstimator(TicTacToeGame)), "Optimal")
    # estimator = VectorValueEstimator()
    # train_self_play(TicTacToeGame, n_iters=500, n_games_per_iter=10, exploration_rate=0.2, optimizer=OptimizerNaive1(max_diff=0.1), estimator=estimator)
    # train_self_play(TicTacToeGame, n_iters=50, n_games_per_iter=100, exploration_rate=0.2, optimizer=OptimizerNaive2(m=0.1), estimator=estimator)
    # estimator = PytorchVectorValueEstimator()
    # train_self_play(TicTacToeGame, n_iters=50, n_games_per_iter=100, exploration_rate=0.2, optimizer=OptimizerNaive3(lr=0.05), estimator=estimator)
    estimator = DLValueEstimator(n_blocks=1).to(default_device)
    train_self_play(TicTacToeGame, n_iters=50, n_games_per_iter=100, exploration_rate=0.2, optimizer=DLOptimizer(lr=0.1, epochs=3), estimator=estimator)


if __name__ == '__main__':
    main()
