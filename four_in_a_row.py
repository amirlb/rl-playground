""" four_in_a_row.py

A moderately-efficient library for four-in-a-row (7x6) with perspective invariance.
Includes Board, RandomAgent, MCTSAgent, Elo evaluation and plotting.
"""

import random
import math
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == '__main__':
    main()
