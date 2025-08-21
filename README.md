# Monte-Carlo Chess Suite

## Overview

Monte-Carlo Chess Suite is an academic-grade research platform for benchmarking and analyzing Monte Carlo search algorithms in computer chess. The suite implements a range of state-of-the-art search strategies, including classical Alpha-Beta (AB) minimax, several Monte Carlo Tree Search (MCTS) variants, and integration with Stockfish for calibration and benchmarking. The project is designed for rigorous experimentation, reproducibility, and comparative analysis using chess.com-style accuracy metrics.

## Algorithms Implemented

- **UNIFORM**: Pure random move selection (baseline)
- **UCB**: Upper Confidence Bound for Trees
- **UCT**: Standard Monte Carlo Tree Search
- **UCT_RAVE**: UCT with Rapid Action Value Estimation (AMAF)
- **PUTC**: PUCT variant with tunable exploration and priors
- **PUTC_RAVE**: PUCT with AMAF/RAVE backup
- **PUTC_GRAVE**: PUCT with GRAVE-style global move statistics
- **AB**: Classical Alpha-Beta search (negamax, TT, LMR, null-move, killers/history)
- **STOCKFISH**: For calibration and benchmarking (via UCI)

## Features & Upgrades

- RAVE/GRAVE integration, classical evaluation, AB search with advanced pruning
- Eval-leaf for MCTS, time-per-move control, resign/adjudication, randomized openings
- Tournament logging, quality analysis tooling, and Markdown/CSV reporting
- Accuracy-first benchmarking: centipawn loss mapped to chess.com-style accuracy percentage

## Benchmarking & Metrics

Experiments are run using round-robin tournaments, parameter sweeps, and calibration against Stockfish at controlled strength (Elo, depth, movetime). Primary metrics:

- **Accuracy% (Acc%)**: 100·k/(k+loss_cp), k=200 (tunable)
- **ACPL**: Average centipawn loss
- **Best-move rate, error breakdown, blunders/mistakes per 100 moves**

See `logs/MASTER_REPORT.md` and `logs/PROGRESS.md` for detailed experiment logs, parameter grids, and results.

## Installation

### Requirements

- Python 3.7+
- python-chess v0.28.0+
- PySimpleGUI 4.4.1+
- Pyperclip

### Setup

1. Clone the repository and download all files, including `Images/`, `Icon/`, and any required engines/books.
2. (Optional) Place your preferred UCI chess engine (e.g., Stockfish) in the `Engines/` directory.
3. Install dependencies:
	 ```bash
	 pip install python-chess pysimplegui pyperclip
	 ```
4. (Optional) Build the Windows executable using PyInstaller.

## Usage

### GUI Mode

Run the chess GUI:

```bash
python python_easy_chess_gui.py
```

### Headless Experiments

- Run matches (no GUI):
	```bash
	python tournament.py --strategies PUTC_GRAVE PUTC --games 4 --playouts 40 --max-steps 300 --log-csv logs/experiments.csv --pgn-dir logs/pgn
	```
- Aggregate results:
	```bash
	python scripts/aggregate_results.py --csv logs/experiments.csv --subject PUTC_GRAVE
	```
- Quality analysis:
	```bash
	python analyze_pgn.py --acc-k 200 --out-csv logs/sweep_500ms_elo1320/analysis_acc.csv
	python scripts/rollup_quality.py --input logs/sweep_500ms_elo1320/analysis_acc.csv
	```

## End-to-end experiment pipeline

The repository includes a reproducible workflow from match generation to accuracy analysis and rollups. Commands below are PowerShell-friendly on Windows.

1) Run tournaments and capture PGNs/CSV

- Round robin across strategies with MCTS using eval priors/leaves and AB leaves at depth 2:
	```powershell
	python tournament.py --strategies UNIFORM UCB UCT UCT_RAVE PUTC PUTC_RAVE PUTC_GRAVE AB `
		--games 2 --playouts 40 --max-steps 300 --processes 2 `
		--use-eval-priors --use-eval-leaf --ab-leaf-depth 2 --rollout-steps 0 `
		--c-puct 1.0 --ucb-c 0.8 --grave-K 100 `
		--log-csv logs/experiments.csv --pgn-dir logs/pgn
	```

- Versus Stockfish (calibration). Example: our strategy as White vs SF depth=1, skill=0, movetime=3ms:
	```powershell
	python tournament.py --strategies PUTC_GRAVE STOCKFISH `
		--games 4 --playouts 40 --max-steps 250 `
		--stockfish-path "C:\\Program Files\\stockfish\\stockfish-windows-x86-64-avx2.exe" `
		--sf-depth 1 --sf-threads 1 --sf-hash-mb 64 --sf-skill 0 --sf-movetime-ms 3 `
		--log-csv logs/experiments.csv --pgn-dir logs/pgn
	```

2) Compute Accuracy% and per-move diagnostics from PGNs

- Analyze a directory of PGNs using Stockfish at a fixed analysis budget; skip opening plies to reduce book noise:
	```powershell
	python analyze_pgn.py --pgn logs/pgn `
		--stockfish-path "C:\\Program Files\\stockfish\\stockfish-windows-x86-64-avx2.exe" `
		--sf-movetime-ms 200 --sf-threads 1 --sf-hash-mb 64 `
		--multipv 1 --skip-plies 6 --inacc-thr 50 --mistake-thr 100 --blunder-thr 200 `
		--acc-k 200 --out-csv logs/quality/analysis_acc.csv `
		--out-move-csv logs/quality/moves_acc.csv
	```

3) Roll up strategy-level quality metrics

- Summarize per-game accuracy and error rates by strategy:
	```powershell
	python scripts/rollup_quality.py --input logs/quality/analysis_acc.csv
	```

4) Optional: aggregate WDL/Elo/time from tournament CSVs

- Compute win% and Elo estimates over a subject strategy:
	```powershell
	python scripts/aggregate_results.py --csv logs/experiments.csv --subject PUTC_GRAVE
	```

5) Optional: parameter sweeps (overnight)

- Weak-Stockfish sweep harness with robust venv activation is provided in `scripts/overnight_sweeps.ps1` and `scripts/run_param_sweep_overnight.ps1`. Example direct call:
	```powershell
	python scripts/param_sweep_weak_sf.py --out-dir logs/param-sweeps
	```

Artifacts will appear under `logs/` (CSV, PGN, Markdown summaries) and can be re-analyzed at any time.

## Technical details (algorithms and design)

### Alpha–Beta (AB)

- Standalone engine: negamax with transposition table (TT), killer moves and history heuristic, late-move reductions (LMR), aspiration windows, null-move pruning, and quiescence search for captures/checks.
- Time control: depth-targeted or per-move time via `--move-time-ms`.
- As MCTS leaf: when `--use-eval-leaf` with `--ab-leaf-depth k` (k≥1), AB runs at leaves to produce a stronger value; we set `--rollout-steps 0` to avoid noisy playouts and initialize child Q from the leaf eval (Q-init).

### UCT / PUCT family

- UCT: selects child maximizing `Q_i / N_i + c * sqrt(ln N_parent / (1 + N_i))` with stochastic rollouts or eval-based cutoffs.
- PUCT (PUTC): blends priors with value: `Q_i + c_puct * P_i * sqrt(N_parent) / (1 + N_i)`. Priors `P_i` come from an evaluation heuristic; progressive bias helps break ties toward promising moves.
- RAVE/AMAF (UCT_RAVE, PUTC_RAVE): backs up action values using all-moves-as-first statistics to accelerate early learning.
- GRAVE (PUTC_GRAVE): mixes node-local and global move averages to stabilize early estimates, especially in wide trees.
- Eval priors/leaves: enable with `--use-eval-priors` and `--use-eval-leaf`. For highest stability at low budgets, prefer `--ab-leaf-depth 2` and `--rollout-steps 0`.

### Evaluation and priors

- Classical eval combines material, piece-square tables (PST), mobility, and king safety; mapped to a probability via a monotone logistic, producing move priors and leaf values.
- For Accuracy% computation we rely on external Stockfish analysis (depth or movetime capped) to measure centipawn loss per move and map loss to accuracy: `accuracy = 100 * k / (k + loss_cp)` with k≈200.

### Engine integration and controls

- Stockfish: used for calibration (as an opponent) and for analysis. Tournament flags `--sf-depth/--sf-movetime-ms/--sf-skill/--sf-threads/--sf-hash-mb` configure strength. Some builds clamp very low Elo; prefer explicit depth/movetime at the low end.
- Time control: all strategies support per-move time via `--move-time-ms` or fixed playout budgets. Adjudication/resign logic reduces long dead-drawn or lost positions.
- Diversity: `--random-openings N` seeds N random plies to diversify self-play and reduce book bias.

## Reproducibility and defaults

- Seeds: tournament runner varies seed per game; pass `--seed` for deterministic Zobrist tables and opening choices.
- Default Stockfish path: `C:\\Program Files\\stockfish\\stockfish-windows-x86-64-avx2.exe`. Override with `--stockfish-path` or in analysis via `--stockfish-path`.
- Logs: PGNs in `logs/pgn/`, CSVs like `logs/experiments.csv`, and quality reports under `logs/quality/` or sweep-specific folders.

### Key Parameters

- `--c-puct`, `--ucb-c`, `--use-eval-priors`, `--use-eval-leaf`, `--ab-leaf-depth`, `--grave-K`
- `--max-steps`, `--eval-adjudicate --eval-threshold`, `--random-openings`, `--resign-threshold --resign-moves`

## Experiment Artifacts

- PGN and CSV logs: `logs/experiments.csv`, `logs/pgn/`, `logs/sweep_500ms_elo1320/`
- Markdown summaries: `logs/MASTER_REPORT.md`, `logs/PROGRESS.md`, `logs/sweep_500ms_elo1320/rollup_acc.md`

## Academic Notes

- AB is used both as a standalone engine and as a leaf evaluator in MCTS variants.
- All experiments are reproducible; parameter grids and seeds are logged for each run.
- Quality metrics are computed using Stockfish analysis at controlled settings (depth, movetime, Elo).
- See `logs/MASTER_REPORT.md` for executive summary, evolution timeline, and best parameter configurations.

## Credits

- PySimpleGUI: https://github.com/PySimpleGUI/PySimpleGUI
- python-chess: https://github.com/niklasf/python-chess
- Pyperclip: https://github.com/asweigart/pyperclip
- The Week in Chess: https://theweekinchess.com/
- PyInstaller: https://github.com/pyinstaller/pyinstaller
- pgn-extract: https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/
- Python-Easy-Chess-GUI: https://github.com/fsmosca/Python-Easy-Chess-GUI

### D. How to
#### To start the gui
* Execute python_easy_chess_gui.py<br>
Typical command line:<br>
`python python_easy_chess_gui.py`
* Execute the exe when using exe file

#### Headless runner and experiments
- Run matches (no GUI):
	- `python tournament.py --strategies PUTC_GRAVE PUTC --games 4 --playouts 40 --max-steps 300 --log-csv logs/experiments.csv --pgn-dir logs/pgn`
- Aggregate results:
	- `python scripts/aggregate_results.py --csv logs/experiments.csv --subject PUTC_GRAVE`
- Useful flags to tune strength/decisiveness:
	- `--c-puct`, `--ucb-c`, `--use-eval-priors`, `--use-eval-leaf`, `--ab-leaf-depth`, `--grave-K`
	- `--max-steps`, `--eval-adjudicate --eval-threshold`, `--random-openings`, `--resign-threshold --resign-moves`

- Quality benchmarking:
	- Primary metric is Accuracy% computed by `analyze_pgn.py` (from centipawn loss); use `scripts/rollup_quality.py` to aggregate by strategy.
	- Use UNIFORM as a lower boundary and Stockfish by Elo or explicit settings (depth/skill) as an upper boundary.


### E. Credits
* PySimpleGUI<br>
https://github.com/PySimpleGUI/PySimpleGUI
* Python-Chess<br>
https://github.com/niklasf/python-chess
* Pyperclip<br>
https://github.com/asweigart/pyperclip
* The Week in Chess<br>
https://theweekinchess.com/
* PyInstaller<br>
https://github.com/pyinstaller/pyinstaller
* pgn-extract<br>
https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/
* Python-Easy-Chess-GUI<br>
https://github.com/fsmosca/Python-Easy-Chess-GUI

This project was graded 15/20
