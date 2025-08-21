import itertools
import time
import sys
from pathlib import Path
import argparse
from subprocess import run, CalledProcessError

# Simple sweep harness around tournament.py
# Example:
#   python sweep.py --games 2 --playouts 40 --seeds 0 1 2 --ucb-c 0.5 0.8 1.2 --c-puct 0.5 0.8 1.2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=2)
    p.add_argument('--playouts', type=int, nargs='+', default=[40])
    p.add_argument('--max-steps', type=int, default=300)
    p.add_argument('--processes', type=int, default=1)
    p.add_argument('--material-threshold', type=int, default=1)
    p.add_argument('--adjudicate', action='store_true', help='Enable adjudication at cutoff')
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1])
    p.add_argument('--ucb-c', dest='ucb_c', type=float, nargs='+', default=[0.8])
    p.add_argument('--c-puct', dest='c_puct', type=float, nargs='+', default=[0.8])
    p.add_argument('--grave-K', dest='grave_K', type=float, nargs='+', default=[50.0, 100.0, 200.0])
    p.add_argument('--use-eval-priors', action='store_true')
    p.add_argument('--use-eval-leaf', action='store_true')
    p.add_argument('--rollout-steps', type=int, default=0)
    p.add_argument('--ab-leaf-depth', type=int, nargs='+', default=[0])
    p.add_argument('--eval-adjudicate', action='store_true')
    p.add_argument('--eval-threshold', type=int, nargs='+', default=[150])
    p.add_argument('--random-openings', type=int, nargs='+', default=[4])
    p.add_argument('--resign-threshold', type=int, nargs='+', default=[0])
    p.add_argument('--resign-moves', type=int, nargs='+', default=[0])
    p.add_argument('--strategies', nargs='+', default=['PUTC_GRAVE','PUTC'])
    p.add_argument('--pgn-dir', type=str, default='logs/pgn')
    p.add_argument('--csv', type=str, default='logs/experiments.csv')
    p.add_argument('--move-time-ms', type=int, default=0, help='Per-move time budget (ms) for MCTS variants; 0 uses playouts')
    args = p.parse_args()

    combos = list(itertools.product(args.seeds, args.playouts, args.ucb_c, args.c_puct, args.grave_K, args.ab_leaf_depth, args.eval_threshold, args.random_openings, args.resign_threshold, args.resign_moves))
    print(f"Total runs: {len(combos)}")

    for seed, playouts, ucb_c, c_puct, grave_K, ab_leaf_depth, eval_thr, rand_open, resign_thr, resign_moves in combos:
        cmd = [
            sys.executable, 'tournament.py',
            '--strategies', *args.strategies,
            '--games', str(args.games),
            '--playouts', str(playouts),
            '--max-steps', str(args.max_steps),
            '--processes', str(args.processes),
            '--seed', str(seed),
            '--ucb-c', str(ucb_c),
            '--c-puct', str(c_puct),
            '--grave-K', str(grave_K),
            '--rollout-steps', str(args.rollout_steps),
            '--pgn-dir', args.pgn_dir,
            '--log-csv', args.csv,
        ]
        if args.move_time_ms and args.move_time_ms > 0:
            cmd += ['--move-time-ms', str(args.move_time_ms)]
        if not args.adjudicate:
            cmd.append('--no-adjudicate')
        if args.use_eval_priors:
            cmd.append('--use-eval-priors')
        if args.use_eval_leaf:
            cmd.append('--use-eval-leaf')
        if args.eval_adjudicate:
            cmd.append('--eval-adjudicate')
        # multi-choice args
        cmd += ['--ab-leaf-depth', str(ab_leaf_depth)]
        cmd += ['--eval-threshold', str(eval_thr)]
        cmd += ['--random-openings', str(rand_open)]
        cmd += ['--resign-threshold', str(resign_thr)]
        cmd += ['--resign-moves', str(resign_moves)]
        print('Running:', ' '.join(cmd))
        start = time.time()
        try:
            run(cmd, check=True)
        except CalledProcessError as e:
            print('Run failed:', e)
        print('Elapsed:', round(time.time()-start, 2), 's')


if __name__ == '__main__':
    main()
