import random
import torch
import argparse
import torch.nn as nn

from base import Episode, Game, RandomPlayer, ValueEstimator, ValueOptimizer, ValuePlayer, evaluate_against_random, train_self_play
from small_games import FullGameTree, OptimizerNaiveAdd, OptimizerNaiveGrad, OptimizerNaiveTD, PytorchVectorValueEstimator, VectorValueEstimator


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


class TicTacToe(Game):
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


class TicTacToeNet(nn.Module, ValueEstimator):
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
        return self.forward(self.embed([board]).to(default_device)).item()

    def embed(self, boards: list[TTTPosition]) -> torch.Tensor:
        return torch.stack([self._embed_board(board) for board in boards])

    def _embed_board(self, board: TTTPosition) -> torch.Tensor:
        my, other = board
        return torch.tensor([
            [0] * i + [1] + [0] * (8-i) + [int(my & (1 << i))] + [int(other & (1 << i))] + [0] * (self.dim - 11)
            for i in range(9)
        ], dtype=torch.float32)


class TorchOptimizer(ValueOptimizer):
    def __init__(self, lr: float, epochs: int, batch_size: int):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, episodes: list[Episode], estimator: TicTacToeNet) -> None:
        data = []
        for ep in episodes:
            for i, (board, _) in enumerate(ep.decisions):
                data.append((board, ep.value * (-1)**i))
            data.append((ep.last_position, ep.value * (-1)**len(ep.decisions)))
        random.shuffle(data)
        states = estimator.embed([board for board, _ in data]).to(default_device)
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


class TorchOptimizerTD(ValueOptimizer):
    # TD on the value function using SGD

    def __init__(self, lr: float, epochs: int, batch_size: int):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, episodes: list[Episode], estimator: nn.Module) -> None:
        indices, targets = [], []
        intermed1, intermed2 = [], []
        for ep in episodes:
            for i, (board, _) in enumerate(ep.decisions):
                intermed1.append(estimator.index[board])
                if i > 0:
                    intermed2.append(estimator.index[board])
            indices.append(estimator.index[ep.last_position])
            targets.append(ep.value * (-1)**len(ep.decisions))
            intermed2.append(estimator.index[ep.last_position])
        indices = torch.tensor(indices, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.float32)
        intermed1 = torch.tensor(intermed1, dtype=torch.long)
        intermed2 = torch.tensor(intermed2, dtype=torch.long)
        intermed_targets = torch.zeros(len(intermed2), dtype=torch.float32)

        optimizer = torch.optim.SGD(estimator.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        for _ in range(self.epochs):
            # TODO: batch
            preds = estimator(indices)
            optimizer.zero_grad()
            criterion(preds, targets).backward()
            preds = estimator.values[intermed1] + estimator.values[intermed2]
            criterion(preds, intermed_targets).backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description='Train TicTacToe AI using various methods')
    parser.add_argument('--variant', type=str, choices=['vec_naive_add', 'vec_naive_grad', 'vec_grad', 'vec_td', 'vec_naive_td', 'dl'], 
                      default='dl', help='Which training variant to use')
    parser.add_argument('--exploration-rate', type=float, default=0.4,
                      help='Exploration rate for training')
    parser.add_argument('--n-iters', type=int, default=50,
                      help='Number of training iterations')
    parser.add_argument('--n-games-per-iter', type=int, default=100,
                      help='Number of games to play per iteration')
    parser.add_argument('--lr', type=float, default=0.05,
                      help='Learning rate for optimizers that use it')
    parser.add_argument('--max-diff', type=float, default=0.1,
                      help='Maximum difference for OptimizerNaiveAdd')
    parser.add_argument('--m', type=float, default=0.1,
                      help='Momentum parameter for OptimizerNaiveGrad and OptimizerNaiveTD')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of epochs for TorchOptimizer')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for TorchOptimizer')
    parser.add_argument('--n-blocks', type=int, default=2,
                      help='Number of transformer blocks for TicTacToeNet')
    parser.add_argument('--n-heads', type=int, default=4,
                      help='Number of attention heads for TicTacToeNet')
    parser.add_argument('--dim', type=int, default=16,
                      help='Dimension of transformer for TicTacToeNet')
    parser.add_argument('--mlp-dim', type=int, default=32,
                      help='Dimension of MLP in transformer for TicTacToeNet')
    args = parser.parse_args()

    random.seed(123)
    evaluate_against_random(TicTacToe, RandomPlayer(TicTacToe), "Random")
    evaluate_against_random(TicTacToe, ValuePlayer(TicTacToe, FullGameTree(TicTacToe)), "Optimal")

    if args.variant == 'vec_naive_add':
        estimator = VectorValueEstimator(TicTacToe)
        train_self_play(TicTacToe, n_iters=args.n_iters, n_games_per_iter=args.n_games_per_iter,
                       exploration_rate=args.exploration_rate, 
                       optimizer=OptimizerNaiveAdd(max_diff=args.max_diff), 
                       estimator=estimator)
    elif args.variant == 'vec_naive_grad':
        estimator = VectorValueEstimator(TicTacToe)
        train_self_play(TicTacToe, n_iters=args.n_iters, n_games_per_iter=args.n_games_per_iter,
                       exploration_rate=args.exploration_rate, 
                       optimizer=OptimizerNaiveGrad(m=args.m), 
                       estimator=estimator)
    elif args.variant == 'vec_grad':
        estimator = PytorchVectorValueEstimator(TicTacToe)
        train_self_play(TicTacToe, n_iters=args.n_iters, n_games_per_iter=args.n_games_per_iter,
                       exploration_rate=args.exploration_rate, 
                       optimizer=TorchOptimizer(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size), 
                       estimator=estimator)
    elif args.variant == 'vec_td':
        estimator = PytorchVectorValueEstimator(TicTacToe)
        train_self_play(TicTacToe, n_iters=args.n_iters, n_games_per_iter=args.n_games_per_iter,
                       exploration_rate=args.exploration_rate, 
                       optimizer=TorchOptimizerTD(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size), 
                       estimator=estimator)
    elif args.variant == 'vec_naive_td':
        estimator = VectorValueEstimator(TicTacToe)
        train_self_play(TicTacToe, n_iters=args.n_iters, n_games_per_iter=args.n_games_per_iter,
                       exploration_rate=args.exploration_rate, 
                       optimizer=OptimizerNaiveTD(m=args.m), 
                       estimator=estimator)
    elif args.variant == 'dl':
        estimator = TicTacToeNet(n_blocks=args.n_blocks, n_heads=args.n_heads, 
                                   dim=args.dim, mlp_dim=args.mlp_dim).to(default_device)
        train_self_play(TicTacToe, n_iters=args.n_iters, n_games_per_iter=args.n_games_per_iter,
                       exploration_rate=args.exploration_rate, 
                       optimizer=TorchOptimizer(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size), 
                       estimator=estimator)


if __name__ == '__main__':
    main()
