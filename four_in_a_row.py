import random
import torch
import argparse
import torch.nn as nn
from base import Episode, Game, ValueEstimator, ValuePlayer, RandomPlayer, ValueOptimizer, evaluate_against_random, train_self_play

# Select best device: MPS (Mac NPU), CUDA, or CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    default_device = torch.device('mps')
elif torch.cuda.is_available():
    default_device = torch.device('cuda')
else:
    default_device = torch.device('cpu')
print(f"Using device: {default_device}")


# Board representation: (my_pieces, opponent_pieces)
# Each is a bitmask representing the cells occupied by the player.
# Cell indices (6 rows, 7 columns):
# 35 36 37 38 39 40 41  (row 5)
# 28 29 30 31 32 33 34  (row 4)
# 21 22 23 24 25 26 27  (row 3)
# 14 15 16 17 18 19 20  (row 2)
#  7  8  9 10 11 12 13  (row 1)
#  0  1  2  3  4  5  6  (row 0)
FIARPosition = tuple[int, int]
FIARMove = int  # Column index 0-6

class FourInARow(Game):
    WIDTH = 7
    HEIGHT = 6
    SIZE = WIDTH * HEIGHT
    TOP_ROW_MASK = ((1 << WIDTH) - 1) << (WIDTH * (HEIGHT - 1))

    @classmethod
    def starting_position(cls) -> FIARPosition:
        return (0, 0)

    @classmethod
    def fmt(cls, board: FIARPosition) -> str:
        my, other = board
        lines = []
        for r in reversed(range(cls.HEIGHT)):
            line = []
            for c in range(cls.WIDTH):
                idx = r * cls.WIDTH + c
                if my & (1 << idx):
                    line.append('X')
                elif other & (1 << idx):
                    line.append('O')
                else:
                    line.append('.')
            lines.append(" ".join(line))
        return "\\n".join(lines)

    @classmethod
    def _check_win(cls, player_board: int) -> bool:
        # Horizontal
        m = player_board & (player_board >> 1) & (player_board >> 2) & (player_board >> 3)
        # Need to mask out wins wrapping around columns - check every 4 positions if they cross boundaries
        for r in range(cls.HEIGHT):
            for c in range(cls.WIDTH - 3):
                 mask = (0b1111 << (r * cls.WIDTH + c))
                 if (player_board & mask) == mask: return True

        # Vertical
        m = player_board & (player_board >> cls.WIDTH) & (player_board >> (2 * cls.WIDTH)) & (player_board >> (3 * cls.WIDTH))
        if m != 0: return True

        # Diagonal Down-Right (\)
        m = player_board & (player_board >> (cls.WIDTH - 1)) & (player_board >> (2 * (cls.WIDTH - 1))) & (player_board >> (3 * (cls.WIDTH - 1)))
        # Masking needed for diagonals
        for r in range(cls.HEIGHT - 3):
            for c in range(cls.WIDTH - 3):
                 mask = (1 << (r*cls.WIDTH+c)) | (1 << ((r+1)*cls.WIDTH+c+1)) | \
                        (1 << ((r+2)*cls.WIDTH+c+2)) | (1 << ((r+3)*cls.WIDTH+c+3))
                 if (player_board & mask) == mask: return True


        # Diagonal Up-Right (/)
        m = player_board & (player_board >> (cls.WIDTH + 1)) & (player_board >> (2 * (cls.WIDTH + 1))) & (player_board >> (3 * (cls.WIDTH + 1)))
        # Masking needed for diagonals
        for r in range(3, cls.HEIGHT):
             for c in range(cls.WIDTH - 3):
                 mask = (1 << (r*cls.WIDTH+c)) | (1 << ((r-1)*cls.WIDTH+c+1)) | \
                        (1 << ((r-2)*cls.WIDTH+c+2)) | (1 << ((r-3)*cls.WIDTH+c+3))
                 if (player_board & mask) == mask: return True

        return False

    @classmethod
    def legal_moves(cls, board: FIARPosition) -> tuple[list[FIARMove], float | None]:
        my, other = board
        
        if cls._check_win(my):
             return [], 1 # Current player just made the move, previous player (other) won. -> Wait, perspective is current player. If `my` has winning lines, it means the *previous* move by `other` created this state, so `other` wins -> -1
             # Let's re-evaluate. `legal_moves` is called *before* the current player moves. 
             # If `other` made the last move and won, `other` board passed to this function would have a winning line.
             # So, check `other` for win first.
        if cls._check_win(other):
             return [], -1 # Previous player (other) won

        whole = my | other
        if whole & cls.TOP_ROW_MASK == cls.TOP_ROW_MASK:
            # Board is full
             return [], 0 # Draw

        moves = []
        for c in range(cls.WIDTH):
            # Check if the top cell of the column is empty
            if not (whole & (1 << (c + (cls.HEIGHT - 1) * cls.WIDTH))):
                moves.append(c)
        
        # If no moves possible, it's a draw (already checked above, but as safeguard)
        if not moves: 
             return [], 0

        return moves, None

    @classmethod
    def play_move(cls, board: FIARPosition, move: FIARMove) -> FIARPosition:
        my, other = board
        whole = my | other

        # Find the lowest empty row in the chosen column 'move'
        for r in range(cls.HEIGHT):
            idx = r * cls.WIDTH + move
            if not (whole & (1 << idx)):
                # Place piece by setting the bit in 'my' board
                new_my = my | (1 << idx)
                return (other, new_my) # Swap players

        # Should not happen if legal_moves is checked first
        raise ValueError(f"Column {move} is full, cannot play move.")


# --- Neural Network ---

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True) # Use batch_first=True
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        y = self.ln1(x)
        # Note: MHA expects query, key, value. For self-attention, they are the same.
        attn_output, _ = self.attn(y, y, y)
        x = x + attn_output
        y = self.mlp(self.ln2(x))
        return x + y

class ConnectFourNet(nn.Module, ValueEstimator):
    """Transformer-based network for Connect Four score estimation."""
    def __init__(self, dim=32, n_blocks=3, n_heads=4, mlp_dim=64):
        super().__init__()
        self.width = FourInARow.WIDTH
        self.height = FourInARow.HEIGHT
        self.n_cells = self.width * self.height
        self.dim = dim

        # Embedding layer for each cell state (empty, mine, opponent's) -> dim
        # We'll use 2 features per cell: my_piece, other_piece
        self.cell_embed = nn.Linear(2, dim) 
        
        # Positional encoding (simple learned embedding for each position)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_cells, dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_dim) for _ in range(n_blocks)
        ])
        # Output score estimate (-1 to 1)
        # Pool features across all cells before final linear layer
        self.head = nn.Linear(dim * self.n_cells, 1) 

    def embed(self, boards: list[FIARPosition]) -> torch.Tensor:
        batch_size = len(boards)
        # Create input tensor: (batch_size, n_cells, 2)
        input_features = torch.zeros(batch_size, self.n_cells, 2, device=default_device)
        for i, (my, other) in enumerate(boards):
            for cell_idx in range(self.n_cells):
                if my & (1 << cell_idx):
                    input_features[i, cell_idx, 0] = 1.0
                elif other & (1 << cell_idx):
                    input_features[i, cell_idx, 1] = 1.0
        
        # Embed cell features: (batch_size, n_cells, dim)
        x = self.cell_embed(input_features)
        
        # Add positional embedding: (batch_size, n_cells, dim)
        x = x + self.pos_embed 
        return x

    def forward(self, x):  # x: (batch, n_cells, dim)
        for blk in self.blocks:
            x = blk(x)
        
        # Flatten features: (batch, n_cells * dim)
        batch_size, seq_len, d = x.shape
        x = x.reshape(batch_size, seq_len * d) 
        
        # Final head for score
        score = self.head(x)              # (batch, 1)
        return torch.tanh(score.squeeze(-1)) # (batch,)

    # --- ValueEstimator interface ---
    def value(self, board: FIARPosition) -> float:
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            embedded_boards = self.embed([board]).to(default_device)
            val = self.forward(embedded_boards).item()
        return val

    def value_batch(self, boards: list[FIARPosition]) -> list[float]:
        if not boards:
            return []
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            embedded_boards = self.embed(boards).to(default_device)
            vals = self.forward(embedded_boards)
        return vals.cpu().tolist()

    def parameters(self): # Required by some optimizers potentially
        return super().parameters()


# --- Optimizer ---
# Reusing TorchOptimizer from small_games, but defining it here for clarity if needed
class TorchOptimizer(ValueOptimizer):
    def __init__(self, lr: float, epochs: int, batch_size: int):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, episodes: list[Episode], estimator: ConnectFourNet) -> None:
        estimator.train() # Set to training mode
        data = []
        # Collect all (board, final_value_perspective) pairs from episodes
        for ep in episodes:
            final_value = ep.value
            # Iterate backwards assigning credit
            current_value = final_value
            # Add last state first
            data.append((ep.last_position, current_value))
            # Step back through decisions
            for i, (board, _) in enumerate(reversed(ep.decisions)):
                 # Perspective flips at each step backwards
                 current_value *= -1 
                 data.append((board, current_value))
                 
        if not data: return # No data to train on

        random.shuffle(data)
        
        # Prepare dataset for PyTorch
        states_embedded = estimator.embed([board for board, _ in data]).to(default_device)
        targets = torch.tensor([value for _, value in data], dtype=torch.float32, device=default_device)
        
        optimizer = torch.optim.Adam(estimator.parameters(), lr=self.lr) # Use Adam
        criterion = nn.MSELoss()
        
        estimator.train() # Ensure model is in training mode
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            indices = torch.randperm(len(states_embedded)) # Shuffle indices each epoch
            for i in range(0, len(states_embedded), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_states = states_embedded[batch_indices]
                batch_targets = targets[batch_indices]
                
                preds = estimator(batch_states) # Pass embedded states
                loss = criterion(preds, batch_targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / (len(states_embedded)/self.batch_size)}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Train Four-in-a-Row AI using a Transformer network.')
    parser.add_argument('--exploration-rate', type=float, default=0.3, help='Exploration rate for training')
    parser.add_argument('--n-iters', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--n-games-per-iter', type=int, default=100, help='Number of games to play per iteration')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate') # Lower LR for Adam
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--n-blocks', type=int, default=3, help='Number of transformer blocks')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dim', type=int, default=32, help='Transformer dimension')
    parser.add_argument('--mlp-dim', type=int, default=64, help='MLP dimension in transformer')
    # Removed n-workers as parallel simulation is now in base.train_self_play if needed
    # parser.add_argument('--n-workers', type=int, default=4, help='Number of parallel workers for self-play simulation')
    args = parser.parse_args()

    random.seed(123)
    torch.manual_seed(123)
    
    game = FourInARow()
    
    evaluate_against_random(game, RandomPlayer(game), "Random Baseline")
    # Optimal player is not feasible for Connect Four

    estimator = ConnectFourNet(
        dim=args.dim, 
        n_blocks=args.n_blocks, 
        n_heads=args.n_heads, 
        mlp_dim=args.mlp_dim
    ).to(default_device)

    optimizer = TorchOptimizer(
        lr=args.lr, 
        epochs=args.epochs, 
        batch_size=args.batch_size
    )

    print("Starting training...")
    train_self_play(
        game=game, 
        n_iters=args.n_iters, 
        n_games_per_iter=args.n_games_per_iter, 
        estimator=estimator, 
        exploration_rate=args.exploration_rate, 
        optimizer=optimizer
        # n_workers=args.n_workers # Pass if train_self_play accepts it
    )

if __name__ == '__main__':
    main()
