from __future__ import annotations
import itertools
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Ensure repo root (parent of scripts/) is on sys.path so we can import tournament.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tournament as T

# Stockfish configuration (explicit settings)
SF_THREADS = 1
SF_HASH_MB = 16
SF_SKILL = 0
SF_DEPTH = 1

# Common match knobs
MAX_STEPS = 120
EVAL_ADJUDICATE = True
EVAL_THR = 200
RESIGN_THR = 1200
RESIGN_MOVES = 4
RANDOM_OPENINGS = 1
PROCESSES = 1
SEED = 0

# Default games per side per configuration
GAMES_PER_SIDE = 2  # total games per config = 4

# Parameter grids per algorithm
GRIDS: Dict[str, Dict[str, List[Any]]] = {
    'AB': {
        'ab_depth_white': [3, 4],
        'ab_depth_black': [3, 4],
    },
    'UCB': {
        'playouts': [200, 300],
        'ucb_c': [0.8, 1.2],
    },
    'UCT': {
        'playouts': [80, 120],
    },
    'UCT_RAVE': {
        'playouts': [80, 120],
    },
    'PUTC': {
        'playouts': [90, 120],
        'c_puct': [0.6, 0.8],
        'use_eval_priors': [True],
        'use_eval_leaf': [True],
        'ab_leaf_depth': [1],
        'rollout_steps': [0],
    },
    'PUTC_RAVE': {
        'playouts': [90, 120],
        'c_puct': [0.6, 0.8],
    },
    'PUTC_GRAVE': {
        'playouts': [90, 120],
        'c_puct': [0.6, 0.8],
        'grave_K': [50, 100],
        'use_eval_priors': [True],
        'use_eval_leaf': [True],
        'ab_leaf_depth': [1],
        'rollout_steps': [0],
    },
}


def _wdl_for(res: Dict[Tuple[str, str], Dict[str, int]], strat: str) -> Tuple[int, int, int, int]:
    wdl = {'wins': 0, 'draws': 0, 'losses': 0}
    key_w = (strat, 'STOCKFISH')
    key_b = ('STOCKFISH', strat)
    if key_w in res:
        rw = res[key_w]
        wdl['wins'] += rw['1-0']
        wdl['losses'] += rw['0-1']
        wdl['draws'] += rw['1/2-1/2']
    if key_b in res:
        rb = res[key_b]
        wdl['wins'] += rb['0-1']
        wdl['losses'] += rb['1-0']
        wdl['draws'] += rb['1/2-1/2']
    total = wdl['wins'] + wdl['draws'] + wdl['losses']
    return wdl['wins'], wdl['draws'], wdl['losses'], total


def _score(w, d, l) -> float:
    # 1 for win, 0.5 for draw, 0 for loss
    return (w + 0.5 * d) / max(1, (w + d + l))


def run_grid_for_strategy(strategy: str, grid: Dict[str, List[Any]], games_per_side: int = GAMES_PER_SIDE) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    rows: List[Dict[str, Any]] = []
    for combo in itertools.product(*vals):
        cfg = dict(zip(keys, combo))
        play = int(cfg.get('playouts', 100))
        ucb_c = float(cfg.get('ucb_c', 0.8))
        c_puct = float(cfg.get('c_puct', 0.8))
        use_eval_priors = bool(cfg.get('use_eval_priors', False))
        rollout_steps = int(cfg.get('rollout_steps', 0))
        use_eval_leaf = bool(cfg.get('use_eval_leaf', True))
        ab_leaf_depth = int(cfg.get('ab_leaf_depth', 1))
        grave_K = float(cfg.get('grave_K', 100.0))
        ab_depth_white = int(cfg.get('ab_depth_white', 3))
        ab_depth_black = int(cfg.get('ab_depth_black', 3))

        res = T.run_tournament(
            strategies=[strategy, 'STOCKFISH'],
            games=games_per_side,
            playouts=play,
            seed=SEED,
            max_steps=MAX_STEPS,
            adjudicate=True,
            material_threshold=1,
            log_csv=None,
            pgn_dir=None,
            processes=PROCESSES,
            ucb_c=ucb_c,
            c_puct=c_puct,
            use_eval_priors=use_eval_priors,
            rollout_steps=rollout_steps,
            use_eval_leaf=use_eval_leaf,
            ab_leaf_depth=ab_leaf_depth,
            ab_depth_white=ab_depth_white,
            ab_depth_black=ab_depth_black,
            grave_K=grave_K,
            eval_adjudicate=EVAL_ADJUDICATE,
            eval_threshold_cp=EVAL_THR,
            random_openings=RANDOM_OPENINGS,
            resign_threshold_cp=RESIGN_THR,
            resign_moves=RESIGN_MOVES,
            stockfish_path=None,
            sf_depth=SF_DEPTH,
            sf_nodes=0,
            sf_movetime_ms=0,
            sf_threads=SF_THREADS,
            sf_hash_mb=SF_HASH_MB,
            sf_skill=SF_SKILL,
            sf_limit_strength=False,
            sf_elo=None,
        )
        w, d, l, tot = _wdl_for(res, strategy)
        rows.append({
            **cfg,
            'wins': w,
            'draws': d,
            'losses': l,
            'total': tot,
            'score': _score(w, d, l),
        })
    # sort by score desc, then wins desc
    rows.sort(key=lambda r: (r['score'], r['wins']), reverse=True)
    return rows


def write_csv(rows: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    import csv
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_md(results: Dict[str, List[Dict[str, Any]]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Stockfish Parameter Sweep (explicit settings)\n")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"\nSettings: SF depth={SF_DEPTH}, skill={SF_SKILL}, threads={SF_THREADS}, hashMB={SF_HASH_MB}; max_steps={MAX_STEPS}; games/side={GAMES_PER_SIDE}\n")
    for strat, rows in results.items():
        lines.append(f"\n## {strat}\n")
        if not rows:
            lines.append("No results\n")
            continue
        best = rows[0]
        keys = [k for k in best.keys() if k not in ('wins','draws','losses','total','score')]
        conf = ", ".join(f"{k}={best[k]}" for k in keys)
        lines.append(f"Best: score={best['score']:.3f} W-D-L {best['wins']}-{best['draws']}-{best['losses']} (N={best['total']})\n")
        lines.append(f"Params: {conf}\n")
    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sweep parameters vs Stockfish with explicit settings (depth=1, skill=0)')
    parser.add_argument('--strategies', '-s', nargs='+', default=['AB','PUTC_GRAVE','PUTC','PUTC_RAVE','UCT_RAVE','UCT','UCB'])
    parser.add_argument('--games-per-side', type=int, default=GAMES_PER_SIDE)
    parser.add_argument('--out-dir', type=str, default='logs')
    args = parser.parse_args()

    results: Dict[str, List[Dict[str, Any]]] = {}
    out_dir = Path(args.out_dir)

    for strat in args.strategies:
        grid = GRIDS.get(strat)
        if not grid:
            print(f"[warn] No grid for {strat}, skipping")
            continue
    print(f"\n=== Parameter sweep for {strat} vs Stockfish (depth={SF_DEPTH}, skill={SF_SKILL}) ===")
    rows = run_grid_for_strategy(strat, grid, games_per_side=args.games_per_side)
    results[strat] = rows
    # write per-strategy CSV
    write_csv(rows, out_dir / f"param-sweep-{strat.lower()}.csv")

    # write summary
    ts = time.strftime('%Y%m%d-%H%M%S')
    write_summary_md(results, out_dir / f"param-sweep-summary-{ts}.md")
    print(f"\nSummary saved to {out_dir / f'param-sweep-summary-{ts}.md'}")


if __name__ == '__main__':
    main()
