# Four-in-a-Row RL Playground

This repository contains a moderate-efficiency Python library and experiment suite for the classic Four-in-a-Row (Connect 4) game (7 columns × 6 rows). It supports:

- **Game representation** with perspective invariance (board always from the current player's view).
- **Agents**:
  - `RandomAgent` — uniform random moves.
  - `MCTSAgent` — vanilla Monte Carlo playouts for move evaluation.
  - `DLAgent` — transformer-based neural network with self-play training and logistic win probability head.
- **Elo evaluation** utilities:
  - `evaluate_elo` — Elo vs. random for MCTS agents.
  - `all_vs_all` + `estimate_elo_matrix` — round-robin Elo rating among multiple agents.
  - `train_selfplay` + `plot_dl_progress` — self-play training loop tracking Elo of the DL agent.

---

## File Structure

```
rl_playground/
├── four_in_a_row.py    # Main library & experiments
└── README.md           # This document
```

All code lives in `four_in_a_row.py`; there are no additional modules or packages.

---

## Quick Start

1. **Run the full pipeline**:
   ```bash
   python four_in_a_row.py
   ```
   By default, this will:
   - Evaluate various MCTS agents (1,5,10,20,50 playouts) vs. Random. Plot Elo vs. playouts.
   - Run an all-vs-all round-robin between Random and those MCTS agents. Print and (optionally) plot Elo ratings.
   - Train the DLAgent for a fixed number of self-play iterations and plot its Elo improvement against Random.

2. **Modify parameters**:
   - Edit `main()` in `four_in_a_row.py` to adjust the lists of playouts, number of games, self-play iterations, training epochs, batch size, learning rate, etc.

3. **Use APIs interactively**:
   ```python
   from four_in_a_row import Board, RandomAgent, MCTSAgent, DLAgent, train_selfplay, evaluate_elo

   board = Board()
   ra = RandomAgent(seed=42)
   move = ra.select_move(board)

   # MCTS with 20 playouts
   ma = MCTSAgent(RandomAgent(seed=0), n_playouts=20)

   # Self-play train a DLAgent
   elos, versions = train_selfplay(n_iters=10, games_per_iter=200)
   ```

---

## Core Classes & Functions

### Board
- `Board()` — 7×6 grid with entries in {1,0,−1}. 1 is current player, −1 is opponent.
- `.legal_moves()` → list of non-full columns.
- `.play_move(col)` → `(win, draw)` and flips perspective.
- `.clone()` → deep copy.

### Agents
- `RandomAgent(seed)` — random move selection.
- `MCTSAgent(playout_agent, n_playouts)` — evaluate each legal move by simulating `n_playouts` games using `playout_agent`.
- `DLAgent(net, device, with_exploration=False)` — use a PyTorch net to score moves; can optionally sample moves for exploration.

### Experiment Utilities
- `evaluate_elo(n_playouts_list, n_games)` — Elo vs random for MCTS.
- `all_vs_all(agents, names, n_games)` → matrix of win rates.
- `estimate_elo_matrix(results)` → least-squares Elo ratings.
- `train_selfplay(n_iters, games_per_iter, epochs, batch_size, lr, buffer_max)` → self-play + training pipeline for DLAgent, returns Elo progression and saved states.
- Plotting functions: `plot_elo`, `plot_elo_matrix`, `plot_dl_progress`.

---

## Continuing Development

- **Adding new agents**: subclass a new `Agent` with `.select_move(board)`.
- **Improving MCTS**: incorporate UCT/PUCT, neural network priors, value heads.
- **Extending the network**: swap in different architectures, convolutional nets, deeper transformers, multi-head value/policy.
- **Hyperparameter tuning**: expose more CLI args or use a config file.
- **Data export**: log self-play trajectories into disk files for offline analysis or replay.

Keep this README as your reference for object interfaces and experiment entry points. Good luck!
