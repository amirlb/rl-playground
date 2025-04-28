""" four_in_a_row.py

A moderately-efficient library for four-in-a-row (7x6) with perspective invariance.
Includes Board, RandomAgent, MCTSAgent, Elo evaluation and plotting.
"""

import random
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Select best device: MPS (Mac NPU), CUDA, or CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    default_device = torch.device('mps')
elif torch.cuda.is_available():
    default_device = torch.device('cuda')
else:
    default_device = torch.device('cpu')
print(f"Using device: {default_device}")

class Board:
    ROWS = 6
    COLS = 7

    def __init__(self):
        # 0 empty, 1 current player's piece, -1 opponent's piece
        self.grid = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)]

    def clone(self):
        new = Board()
        new.grid = [row.copy() for row in self.grid]
        return new

    def legal_moves(self):
        return [c for c in range(self.COLS) if self.grid[0][c] == 0]

    def play_move(self, col):
        # drop piece for current player (1)
        for r in range(self.ROWS - 1, -1, -1):
            if self.grid[r][col] == 0:
                self.grid[r][col] = 1
                win = self._check_win_cell(r, col)
                draw = (all(self.grid[0][c] != 0 for c in range(self.COLS)) and not win)
                # invert perspective: flip all non-zero cells
                for i in range(self.ROWS):
                    for j in range(self.COLS):
                        self.grid[i][j] = -self.grid[i][j]
                return win, draw
        raise ValueError(f"Column {col} is full")

    def _check_win_cell(self, r, c):
        # check 4 in a row including cell (r,c) for value 1
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            # check both directions
            for step in (1, -1):
                rr, cc = r, c
                while True:
                    rr += dr * step
                    cc += dc * step
                    if 0 <= rr < self.ROWS and 0 <= cc < self.COLS and self.grid[rr][cc] == 1:
                        count += 1
                    else:
                        break
            if count >= 4:
                return True
        return False

    def __str__(self):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        rows = []
        for row in self.grid:
            rows.append(' '.join(symbols[v] for v in row))
        return '\n'.join(rows)

class RandomAgent:
    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def select_move(self, board):
        moves = board.legal_moves()
        return self.random.choice(moves)

def simulate_play(board, agent):
    # simulate until terminal; return 1 if starting side (whose perspective is on board at start) wins, -1 if loses, 0 if draw
    sim_identity = 1  # 1 for starting current player's side
    while True:
        move = agent.select_move(board)
        win, draw = board.play_move(move)
        if win:
            return 1 if sim_identity == 1 else -1
        if draw:
            return 0
        # flip identity: perspective has been inverted
        sim_identity *= -1

class MCTSAgent:
    def __init__(self, playout_agent, n_playouts):
        self.playout_agent = playout_agent
        self.n_playouts = n_playouts

    def select_move(self, board):
        moves = board.legal_moves()
        best_move = None
        best_val = -float('inf')
        for move in moves:
            total = 0
            for _ in range(self.n_playouts):
                bcopy = board.clone()
                win, draw = bcopy.play_move(move)
                if win:
                    result = 1
                elif draw:
                    result = 0
                else:
                    result = -simulate_play(bcopy, self.playout_agent)
                total += result
            avg = total / self.n_playouts
            if avg > best_val:
                best_val = avg
                best_move = move
        return best_move

def evaluate_elo(n_playouts_list, n_games=50):
    random_agent = RandomAgent(seed=0)
    elo_list = []
    for n in n_playouts_list:
        mcts_agent = MCTSAgent(RandomAgent(seed=1), n)
        wins = draws = losses = 0
        for i in range(n_games):
            board = Board()
            # alternate who starts
            if i % 2 == 0:
                agents = [mcts_agent, random_agent]
            else:
                agents = [random_agent, mcts_agent]
            turn = 0
            while True:
                current = agents[turn % 2]
                move = current.select_move(board)
                win, draw = board.play_move(move)
                if win:
                    if current is mcts_agent:
                        wins += 1
                    else:
                        losses += 1
                    break
                if draw:
                    draws += 1
                    break
                turn += 1
        score = wins + 0.5 * draws
        avg_score = score / n_games
        if avg_score == 1:
            rating = float('inf')
        elif avg_score == 0:
            rating = -float('inf')
        else:
            rating = -400 * math.log10(1 / avg_score - 1)
        elo_list.append(rating)
    return elo_list

def plot_elo(n_playouts_list, elo_list):
    plt.figure()
    plt.plot(n_playouts_list, elo_list, marker='o')
    plt.xlabel('Number of playouts per move')
    plt.ylabel('ELO vs Random')
    plt.title('ELO vs MCTS playouts')
    plt.grid(True)
    plt.show()

# Add all-vs-all tournament functions
def all_vs_all(agents, names, n_games=50):
    """Return result matrix of win fractions for each agent pair."""
    n = len(agents)
    results = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            wins_i = draws = wins_j = 0
            for g in range(n_games):
                board = Board()
                if g % 2 == 0:
                    pair = (agents[i], agents[j])
                else:
                    pair = (agents[j], agents[i])
                turn = 0
                winner = None
                while True:
                    current = pair[turn % 2]
                    move = current.select_move(board)
                    win, draw = board.play_move(move)
                    if win:
                        winner = current
                        break
                    if draw:
                        break
                    turn += 1
                if winner is None:
                    draws += 1
                elif winner is agents[i]:
                    wins_i += 1
                else:
                    wins_j += 1
            score_i = (wins_i + 0.5*draws) / n_games
            score_j = (wins_j + 0.5*draws) / n_games
            results[i][j] = score_i
            results[j][i] = score_j
    return results

def estimate_elo_matrix(results):
    """Estimate Elo ratings via least-squares on pairwise results."""
    pairs = []
    diffs = []
    n = len(results)
    for i in range(n):
        for j in range(i+1, n):
            p = results[i][j]
            if p <= 0 or p >= 1:
                continue
            diff = -400 * math.log10(1 / p - 1)
            pairs.append((i, j))
            diffs.append(diff)
    if not pairs:
        return [0]*n
    A = np.zeros((len(pairs), n))
    b = np.array(diffs)
    for k, (i, j) in enumerate(pairs):
        A[k, i] = 1
        A[k, j] = -1
    R, *_ = np.linalg.lstsq(A, b, rcond=None)
    R = R - np.mean(R)
    return R.tolist()

def plot_elo_matrix(names, ratings):
    plt.figure()
    plt.bar(names, ratings)
    plt.xticks(rotation=45)
    plt.ylabel('Elo rating')
    plt.title('All-vs-all Elo ratings')
    plt.tight_layout()
    plt.show()

def main():
    torch.set_num_threads(11)
    torch.set_num_interop_threads(11)

    n_playouts_list = [1, 5, 10, 20, 50]
    elo_values = evaluate_elo(n_playouts_list, n_games=50)
    print("Playouts:", n_playouts_list)
    print("ELO ratings:", elo_values)
    plot_elo(n_playouts_list, elo_values)

    # all-vs-all tournament
    agents = [RandomAgent(seed=0)] + [MCTSAgent(RandomAgent(seed=1), n) for n in n_playouts_list]
    names = ['Random'] + [f'MCTS_{n}' for n in n_playouts_list]
    print("Running all-vs-all tournament...")
    results = all_vs_all(agents, names, n_games=30)
    ratings = estimate_elo_matrix(results)
    for name, r in zip(names, ratings):
        print(f"{name}: {r:.1f}")
    plot_elo_matrix(names, ratings)

    # DL self-play training
    print("Starting DL self-play training...")
    elos, versions = train_selfplay(n_iters=100, games_per_iter=100, epochs=3, batch_size=32, lr=1e-5, buffer_max=10000)
    print("DL Training Elo progression:", elos)
    plot_dl_progress(elos)

    # Deep learning agent modules

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

class FourInARowNet(nn.Module):
    """Transformer-based network for win-probability estimation."""
    def __init__(self, dim=64, n_blocks=4, n_heads=4, mlp_dim=128):
        super().__init__()
        # input features are fixed 64-dim: one-hot pos (42), self bit, opp bit, zeros (20)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_dim) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(dim * 42, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (batch, 42, dim)
        x = x.permute(1, 0, 2)            # (42, batch, dim)
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(1, 0, 2)            # (batch, 42, dim)
        batch_size, seq_len, d = x.shape
        x = x.reshape(batch_size, seq_len * d)  # (batch, 42*dim)
        prob = self.head(x)               # (batch, 1)
        return prob.squeeze(-1)           # (batch,)

class DLAgent:
    """Agent using the network to pick moves (no search)."""
    def __init__(self, net, device='cpu', with_exploration=False):
        self.net = net
        self.device = device
        self.net.eval()
        self.with_exploration = with_exploration

    def select_move(self, board):
        moves = board.legal_moves()
        net_probs = {}
        for m in moves:
            bcopy = board.clone()
            win, draw = bcopy.play_move(m)
            state = board_to_tensor(bcopy).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p_tensor = self.net(state)  # (1,) probability of win
            net_probs[m] = p_tensor[0].item()
        if self.with_exploration:
            probs = {m: 0.01 + p**0.5 for m, p in net_probs.items()}
            x = random.random() * sum(probs.values())
            for m, p in probs.items():
                x -= p
                if x < 0:
                    return m
            raise Exception("Failed to select move")
        else:
            return max(net_probs, key=net_probs.get)

# Utilities for DL self-play and training

def board_to_tensor(board):
    """Convert board to tensor of shape (42,64):
    one-hot pos(42), self-piece(1), opp-piece(1), zeros(20)."""
    # flatten grid
    flat = [cell for row in board.grid for cell in row]
    # positional one-hot (42 x 42)
    pos = torch.eye(42, dtype=torch.float32)
    # piece presence bits
    self_bits = torch.tensor([1.0 if v==1 else 0.0 for v in flat], dtype=torch.float32).unsqueeze(1)
    opp_bits = torch.tensor([1.0 if v==-1 else 0.0 for v in flat], dtype=torch.float32).unsqueeze(1)
    # padding zeros to reach 64 dims
    pad = torch.zeros(42, 20, dtype=torch.float32)
    # concatenate to (42, 64)
    x = torch.cat([pos, self_bits, opp_bits, pad], dim=1)
    return x

def print_numbered_board(moves):
    """Print a single final board with moves numbered on each cell."""
    rows, cols = Board.ROWS, Board.COLS
    n = len(moves)
    width = len(str(n))
    # track stack heights per column
    heights = [0] * cols
    # init grid with dots
    grid = [[ '.' * width for _ in range(cols)] for _ in range(rows)]
    for idx, col in enumerate(moves):
        # compute row from bottom
        row = rows - 1 - heights[col]
        heights[col] += 1
        grid[row][col] = str(idx+1).rjust(width)
    # print rows top-down
    for r in range(rows):
        print(' '.join(grid[r]))
    print()

def generate_selfplay_data(agent, n_games, show_games=False):
    """Run self-play, collect (state, label) pairs for win-loss classification."""
    data = []
    for game_idx in range(n_games):
        board = Board()
        sim_id = 1
        states = []  # (Board, sim_id)
        moves = []
        while True:
            states.append((board.clone(), sim_id))
            move = agent.select_move(board)
            moves.append(move)
            win, draw = board.play_move(move)
            if win or draw:
                # include final board position (after move) in states
                # perspective has been inverted, so flip sim_id
                states.append((board.clone(), -sim_id))
                # print every 20th game
                if show_games and game_idx % 20 == 19:
                    print(f"---- Self-play Game {game_idx+1} ----")
                    print_numbered_board(moves)
                result = 1 if win else 0
                for st, sid in states:
                    label = result if sid == 1 else (1 - result)
                    data.append((board_to_tensor(st), label))
                break
            sim_id *= -1
    return data

def evaluate_agent_vs(agent, opponent, n_games=50):
    """Return Elo rating difference of agent vs opponent."""
    wins = draws = losses = 0
    for i in range(n_games):
        board = Board()
        players = [agent, opponent] if i % 2 == 0 else [opponent, agent]
        turn = 0
        while True:
            cur = players[turn % 2]
            move = cur.select_move(board)
            win, draw = board.play_move(move)
            if win:
                if cur is agent: wins += 1
                else: losses += 1
                break
            if draw:
                draws += 1
                break
            turn += 1
    p = (wins + 0.5 * draws) / n_games
    if p <= 0: return -float('inf')
    if p >= 1: return float('inf')
    return -400 * math.log10(1 / p - 1)

def train_selfplay(n_iters=5, games_per_iter=100, epochs=3, batch_size=32, lr=1e-3, buffer_max=10000):
    """Run self-play training, return Elo progression and saved state_dicts."""
    # use default selected device (MPS/CUDA/CPU)
    device = default_device
    net = FourInARowNet(n_blocks=2).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # use binary cross-entropy for single-sigmoid output
    criterion = nn.BCELoss()
    buffer = []
    elos = []
    versions = []
    random_agent = RandomAgent(seed=0)
    # Evaluate current net vs random
    score = evaluate_agent_vs(DLAgent(net, device), random_agent, n_games=50)
    elos.append(score)
    versions.append(net.state_dict())
    print(f"Before training: Elo vs Random = {score:.1f}")
    for it in range(1, n_iters + 1):
        try:
            agent = DLAgent(net, device, with_exploration=True)
            data = generate_selfplay_data(agent, games_per_iter)
            buffer += data
            if len(buffer) > buffer_max:
                buffer = buffer[-buffer_max:]
            # Training epochs
            for epoch in range(1, epochs+1):
                net.train()
                random.shuffle(buffer)
                running_loss = 0.0
                correct = 0
                total = 0
                for i in range(0, len(buffer), batch_size):
                    batch = buffer[i:i+batch_size]
                    states = torch.stack([s for s, _ in batch]).to(device)
                    labels = torch.tensor([l for _, l in batch], dtype=torch.float32).to(device)
                    preds = net(states)
                    loss = criterion(preds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # accumulate loss and accuracy
                    running_loss += loss.item() * labels.size(0)
                    preds_label = (preds >= 0.5).float()
                    correct += (preds_label == labels).sum().item()
                    total += labels.size(0)
                epoch_loss = running_loss / total
                epoch_acc = correct / total
                print(f"Iteration {it}, Epoch {epoch}: loss={epoch_loss:.4f}, acc={epoch_acc:.3f}")
            net.eval()
            # Evaluate current net vs random
            score = evaluate_agent_vs(DLAgent(net, device), random_agent, n_games=200)
            elos.append(score)
            versions.append(net.state_dict())
            print(f"Iteration {it}: Elo vs Random = {score:.1f}")
        except KeyboardInterrupt:
            print("Training interrupted by user")
            break
    return elos, versions

def plot_dl_progress(elos):
    plt.figure()
    plt.plot(range(len(elos)), elos, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Elo vs Random')
    plt.title('DL Agent Training Progress')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
