import argparse
import itertools
import time
from pathlib import Path

import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tournament as T

# This script runs parameter sweeps vs Stockfish configured by Elo (limit strength) at 500ms/move,
# saves PGNs and CSV logs, then computes analysis CSV and an aggregated rollup. Accuracy% is the primary metric.


def run_one(
    strategy: str,
    games_per_side: int,
    playouts: int,
    move_time_ms: int,
    max_steps: int,
    stockfish_path: str | None,
    sf_elo: int,
    out_root: Path,
    processes: int = 1,
    params: dict | None = None,
):
    params = params or {}
    tag = "_".join([f"{k}-{v}" for k, v in sorted(params.items())]) if params else "default"
    pgn_dir = out_root / f"pgn_{strategy}_{tag}"
    csv_path = out_root / f"csv_{strategy}_{tag}.csv"

    ucb_c = float(params.get('ucb_c', 0.8))
    c_puct = float(params.get('c_puct', 1.2))
    use_eval_priors = bool(params.get('use_eval_priors', False))
    rollout_steps = int(params.get('rollout_steps', 200))
    use_eval_leaf = bool(params.get('use_eval_leaf', False))
    ab_leaf_depth = int(params.get('ab_leaf_depth', 0))
    ab_depth = int(params.get('ab_depth', 3))
    grave_K = float(params.get('grave_K', 100.0))

    print(f"\n=== Running {strategy} vs SF Elo {sf_elo} @ {move_time_ms}ms | params: {params} ===")
    T.run_tournament(
        strategies=[strategy, 'STOCKFISH'],
        games=games_per_side,
        playouts=playouts,
        seed=0,
        max_steps=max_steps,
        adjudicate=True,
        material_threshold=1,
        log_csv=csv_path,
        pgn_dir=pgn_dir,
        processes=processes,
        ucb_c=ucb_c,
        c_puct=c_puct,
        use_eval_priors=use_eval_priors,
        rollout_steps=rollout_steps,
        use_eval_leaf=use_eval_leaf,
        ab_leaf_depth=ab_leaf_depth,
        ab_depth_white=ab_depth,
        ab_depth_black=ab_depth,
        grave_K=grave_K,
        eval_adjudicate=True,
        eval_threshold_cp=150,
        random_openings=2,
        resign_threshold_cp=600,
        resign_moves=3,
        stockfish_path=stockfish_path,
        sf_depth=1,
        sf_nodes=0,
        sf_movetime_ms=move_time_ms,
        sf_threads=1,
        sf_hash_mb=64,
        sf_skill=None,
        sf_limit_strength=True,
        sf_elo=sf_elo,
        move_time_ms=move_time_ms,
    )


def main():
    ap = argparse.ArgumentParser(description="Stockfish Elo sweep at fixed time per move (limit_strength)")
    ap.add_argument('--sf-elo', type=int, default=1400, help='UCI_Elo for Stockfish (with limit_strength)')
    ap.add_argument('--games-per-side', type=int, default=10, help='Games per color per parameter point')
    ap.add_argument('--move-time-ms', type=int, default=500, help='Per-move time for both engines')
    ap.add_argument('--max-steps', type=int, default=150, help='Ply cutoff')
    ap.add_argument('--playouts', type=int, default=0, help='Playouts for MCTS when not using time (0 uses time)')
    ap.add_argument('--stockfish-path', type=str, default='', help='Path to Stockfish executable')
    ap.add_argument('--out-dir', type=str, default='sweep_500ms', help='Output directory under logs/')
    args = ap.parse_args()

    out_root = Path('logs') / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # Define parameter grids per strategy
    grids: dict[str, list[dict]] = {
        'UNIFORM': [ {} ],
        'UCB': [ {'ucb_c': c} for c in [0.6, 0.8, 1.2, 1.6] ],
        'UCT_RAVE': [ {} ],  # RAVE constant is internal; time is controlled globally
        'PUTC': [
            {'c_puct': c, 'use_eval_priors': True, 'rollout_steps': rs, 'use_eval_leaf': True, 'ab_leaf_depth': d}
            for c in [1.0, 1.4]
            for rs in [0, 100]
            for d in [0, 2]
        ],
        'PUTC_GRAVE': [
            {'c_puct': c, 'use_eval_priors': True, 'rollout_steps': rs, 'use_eval_leaf': True, 'ab_leaf_depth': d, 'grave_K': K}
            for c in [1.0, 1.4]
            for rs in [0, 100]
            for d in [0, 2]
            for K in [75.0, 150.0]
        ],
        'AB': [ {'ab_depth': d} for d in [3, 4, 5] ],
    }

    # Run sweeps
    start = time.time()
    for strat, plist in grids.items():
        for params in plist:
            run_one(
                strategy=strat,
                games_per_side=args.games_per_side,
                playouts=args.playouts,
                move_time_ms=args.move_time_ms,
                max_steps=args.max_steps,
                stockfish_path=(args.stockfish_path if args.stockfish_path else None),
                sf_elo=args.sf_elo,
                out_root=out_root,
                processes=1,
                params=params,
            )
    elapsed = time.time() - start
    print(f"\nAll sweeps completed in {elapsed/60.0:.1f} minutes")

    # Aggregate analysis
    # 1) Analyze all PGNs produced
    pgn_root = out_root
    analysis_csv = out_root / 'quality_analysis.csv'
    from subprocess import run as _run
    _run([
        'python', 'analyze_pgn.py', '--pgn', str(pgn_root), '--sf-movetime-ms', '200', '--sf-threads', '1', '--sf-hash-mb', '64', '--skip-plies', '6', '--out-csv', str(analysis_csv)
    ], check=False)

    # 2) Rollup by strategy
    rollup_csv = out_root / 'rollup.csv'
    rollup_md = out_root / 'rollup.md'
    _run([
        'python', 'scripts/rollup_quality.py', '--input', str(analysis_csv), '--out-csv', str(rollup_csv), '--out-md', str(rollup_md)
    ], check=False)
    print(f"\nSummary written: {rollup_csv} and {rollup_md}")


if __name__ == '__main__':
    main()
