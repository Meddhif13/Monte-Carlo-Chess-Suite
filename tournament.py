"""Simple headless tournament runner for the MChess repo.

Usage examples (PowerShell):
python tournament.py --games 4 --playouts 50

This script runs programmatic matches (no GUI) between strategies:
- UNIFORM: random legal move
- UCB: uses UCB.py::UCB(board, n)
- UCT: uses UCT_IA.py::BestMoveUCT(board, h, piece_hash, hashTurn, n)

It initializes a zobrist-like piece_hash and hashTurn compatible with the project's code
and keeps the zobrist hash updated using the play() helper in UCT_IA.
"""

import argparse
import random
import time
import chess
import csv
import math
from pathlib import Path
from typing import Optional
import chess.pgn
import chess.engine
from ab import best_move_ab, best_move_ab_with_stats
from eval import evaluate
d_DEFAULT_STOCKFISH_PATH = r"C:\\Program Files\\stockfish\\stockfish-windows-x86-64-avx2.exe"
SF_ELO_MIN = 1320
SF_ELO_MAX = 3190


# Import AIs and helpers from the repo
from UCB import UCB
import UCT_IA
import PUTC
import UCT_RAVE
import PUTC_RAVE
import PUTC_GRAVE

# File-local mapping used by other repo modules
d = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
    "8": 0,
    "7": 1,
    "6": 2,
    "5": 3,
    "4": 4,
    "3": 5,
    "2": 6,
    "1": 7
}


def make_piece_hash(seed=None):
    rnd = random.Random(seed)
    # 12 piece types (6 white, 6 black) x 8 x 8
    return [[[rnd.getrandbits(64) for _ in range(8)] for _ in range(8)] for _ in range(12)]


def make_hash_turn(seed=None):
    rnd = random.Random(seed)
    return rnd.getrandbits(64)


def compute_zobrist(board, piece_hash):
    h = 0
    for square in chess.SQUARES:
        piece = board.piece_type_at(square)
        color = board.color_at(square)
        if piece is not None:
            indice_color = 0 if color else 1  # True == white -> indice 0
            uci = chess.square_name(square)
            x = d[uci[0]]
            y = d[uci[1]]
            h ^= piece_hash[(piece - 1) + 6 * indice_color][x][y]
    return h


def _material_eval(board: chess.Board) -> int:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    total = 0
    for _, pc in board.piece_map().items():
        v = vals.get(pc.piece_type, 0)
        total += v if pc.color else -v
    return total  # >0 favors white, <0 favors black


def _adjudicate_result(board: chess.Board, threshold: int = 1, eval_adjudicate: bool = False, eval_threshold_cp: int = 200) -> str:
    # First try material-based adjudication
    m = _material_eval(board)
    if m > threshold:
        return '1-0'
    if m < -threshold:
        return '0-1'
    # Optionally use classical eval to adjudicate if material is close
    if eval_adjudicate:
        cp = evaluate(board)  # positive favors White
        if cp > eval_threshold_cp:
            return '1-0'
        if cp < -eval_threshold_cp:
            return '0-1'
    return '1/2-1/2'


def choose_move(strategy, board, h, piece_hash, hashTurn, playouts, processes: int = 1, ucb_c: float = 0.8, c_puct: float = 0.8, use_eval_priors: bool = False, rollout_steps: int = 200, use_eval_leaf: bool = False, ab_leaf_depth: int = 0, ab_depth_white: int = 3, ab_depth_black: int = 3, grave_K: float = 100.0, stockfish_engine: Optional[chess.engine.SimpleEngine] = None, sf_depth: int = 12, sf_nodes: int = 0, sf_movetime_ms: int = 0, move_time_ms: int | None = None):
    if strategy == 'UNIFORM':
        moves = [m for m in board.legal_moves]
        return random.choice(moves)
    elif strategy == 'UCB':
        return UCB(board, playouts, c=ucb_c, time_ms=move_time_ms)
    elif strategy == 'UCT':
        return UCT_IA.BestMoveUCT(board, h, piece_hash, hashTurn, playouts, processes=processes, time_ms=move_time_ms)
    elif strategy == 'UCT_RAVE':
        return UCT_RAVE.BestMoveUCT_RAVE(board, h, piece_hash, hashTurn, playouts, processes=processes, time_ms=move_time_ms)
    elif strategy == 'PUTC':
        return PUTC.BestMovePUTC(board, h, piece_hash, hashTurn, playouts, processes=processes, c_puct=c_puct, use_eval_priors=use_eval_priors, rollout_steps=rollout_steps, use_eval_leaf=use_eval_leaf, ab_leaf_depth=ab_leaf_depth, time_ms=move_time_ms)
    elif strategy == 'PUTC_RAVE':
        return PUTC_RAVE.BestMovePUTC_RAVE(board, h, piece_hash, hashTurn, playouts, processes=processes, c_puct=c_puct, time_ms=move_time_ms)
    elif strategy == 'PUTC_GRAVE':
        return PUTC_GRAVE.BestMovePUTC_GRAVE(board, h, piece_hash, hashTurn, playouts, processes=processes, c_puct=c_puct, use_eval_priors=use_eval_priors, rollout_steps=rollout_steps, use_eval_leaf=use_eval_leaf, ab_leaf_depth=ab_leaf_depth, K=grave_K, time_ms=move_time_ms)
    elif strategy == 'AB':
        depth = ab_depth_white if board.turn else ab_depth_black
        mv, stats = best_move_ab_with_stats(board, depth=depth, time_ms=move_time_ms)
        # Stash stats on board object for logging by caller
        setattr(board, "_ab_stats", stats)
        return mv
    elif strategy in ('STOCKFISH', 'SF'):
        if stockfish_engine is None:
            raise ValueError('Stockfish engine not initialized; provide --stockfish-path when using STOCKFISH strategy')
        # Build a search limit: prefer nodes, else movetime, else depth
        if sf_nodes and sf_nodes > 0:
            limit = chess.engine.Limit(nodes=sf_nodes)
        elif sf_movetime_ms and sf_movetime_ms > 0:
            limit = chess.engine.Limit(time=sf_movetime_ms / 1000.0)
        else:
            limit = chess.engine.Limit(depth=sf_depth)
        res = stockfish_engine.play(board, limit)
        return res.move
    else:
        raise ValueError(f'Unknown strategy {strategy}')


def play_match(white_strat, black_strat, playouts, seed=None, verbose=False, max_steps: int = 200, adjudicate: bool = True, material_threshold: int = 1, save_pgn: Optional[Path] = None, processes: int = 1, ucb_c: float = 0.8, c_puct: float = 0.8, use_eval_priors: bool = False, rollout_steps: int = 200, use_eval_leaf: bool = False, ab_leaf_depth: int = 0, ab_depth_white: int = 3, ab_depth_black: int = 3, grave_K: float = 100.0, eval_adjudicate: bool = False, eval_threshold_cp: int = 200, random_openings: int = 0, resign_threshold_cp: int = 0, resign_moves: int = 0, stockfish_path: Optional[str] = None, sf_depth: int = 12, sf_nodes: int = 0, sf_movetime_ms: int = 0, sf_threads: int = 1, sf_hash_mb: int = 64, sf_skill: Optional[int] = None, sf_limit_strength: bool = False, sf_elo: Optional[int] = None, move_time_ms: int | None = None):
    board = chess.Board()
    # create zobrist table and hash turn for this match
    piece_hash = make_piece_hash(seed)
    hashTurn = make_hash_turn(seed)
    h = compute_zobrist(board, piece_hash)

    # Prepare PGN game if logging
    game = None
    node = None
    if save_pgn is not None:
        game = chess.pgn.Game()
        game.headers["Event"] = f"{white_strat} vs {black_strat}"
        game.headers["White"] = white_strat
        game.headers["Black"] = black_strat
        game.headers["Date"] = time.strftime("%Y.%m.%d")

    # Optional randomized openings to diversify positions
    if random_openings and random_openings > 0:
        for _ in range(random_openings):
            if board.is_game_over(claim_draw=False):
                break
            moves = [m for m in board.legal_moves]
            if not moves:
                break
            mv = random.choice(moves)
            h = UCT_IA.play(board, h, mv, piece_hash, hashTurn)
            if save_pgn is not None and game is not None:
                node = game.add_variation(mv) if node is None else node.add_variation(mv)

    # Optionally initialize Stockfish engine if needed
    engine: Optional[chess.engine.SimpleEngine] = None
    needs_sf = (white_strat in ('STOCKFISH', 'SF')) or (black_strat in ('STOCKFISH', 'SF'))
    if needs_sf:
        if not stockfish_path:
            stockfish_path = d_DEFAULT_STOCKFISH_PATH
        p = Path(stockfish_path)
        if not p.exists():
            raise FileNotFoundError(f'Stockfish not found at path: {stockfish_path}')
        engine = chess.engine.SimpleEngine.popen_uci(str(p))
        # Configure engine options
        try:
            opts = {
                'Threads': int(max(1, sf_threads)),
                'Hash': int(max(16, sf_hash_mb)),
            }
            if sf_skill is not None:
                opts['Skill Level'] = int(sf_skill)
            if sf_limit_strength:
                opts['UCI_LimitStrength'] = True
                if sf_elo is not None:
                    opts['UCI_Elo'] = int(sf_elo)
            engine.configure(opts)
        except Exception:
            # Non-fatal: continue with defaults
            pass

    # Play until game over or step cutoff
    steps = 0
    adv_white = 0
    adv_black = 0
    try:
        while not board.is_game_over(claim_draw=False) and steps < max_steps:
            strategy = white_strat if board.turn else black_strat
            side = 'white' if board.turn else 'black'
            move = choose_move(
                strategy, board, h, piece_hash, hashTurn, playouts,
                processes=processes, ucb_c=ucb_c, c_puct=c_puct,
                use_eval_priors=use_eval_priors, rollout_steps=rollout_steps,
                use_eval_leaf=use_eval_leaf, ab_leaf_depth=ab_leaf_depth,
                ab_depth_white=ab_depth_white, ab_depth_black=ab_depth_black,
                grave_K=grave_K,
                stockfish_engine=engine if strategy in ('STOCKFISH', 'SF') else None,
                sf_depth=sf_depth, sf_nodes=sf_nodes, sf_movetime_ms=sf_movetime_ms,
                move_time_ms=(None if (move_time_ms is None or move_time_ms <= 0) else move_time_ms),
            )
            if move is None:
                break
            # Use UCT_IA.play helper to update zobrist and push move
            h = UCT_IA.play(board, h, move, piece_hash, hashTurn)
            if game is not None:
                node = game.add_variation(move) if node is None else node.add_variation(move)
            steps += 1

            if verbose:
                print(f'{side} {strategy} -> {move}')

            # Optional resign logic based on classical eval sustained over N plies
            if resign_threshold_cp and resign_moves and resign_threshold_cp > 0 and resign_moves > 0:
                cp = evaluate(board)
                if cp > resign_threshold_cp:
                    adv_white += 1
                    adv_black = 0
                elif cp < -resign_threshold_cp:
                    adv_black += 1
                    adv_white = 0
                else:
                    adv_white = 0
                    adv_black = 0
                if adv_white >= resign_moves:
                    if game is not None and save_pgn is not None:
                        game.headers["Result"] = '1-0'
                        with open(str(save_pgn), "a", encoding="utf-8") as f:
                            print(game, file=f)
                            print(file=f)
                    return '1-0', steps
                if adv_black >= resign_moves:
                    if game is not None and save_pgn is not None:
                        game.headers["Result"] = '0-1'
                        with open(str(save_pgn), "a", encoding="utf-8") as f:
                            print(game, file=f)
                            print(file=f)
                    return '0-1', steps
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

    if board.is_game_over(claim_draw=False):
        res = board.result(claim_draw=False)
        if game is not None and save_pgn is not None:
            game.headers["Result"] = res
            with open(str(save_pgn), "a", encoding="utf-8") as f:
                print(game, file=f)
                print(file=f)
        return res, steps
    # adjudicate by material/eval if cutoff reached
    res = _adjudicate_result(board, threshold=material_threshold, eval_adjudicate=eval_adjudicate, eval_threshold_cp=eval_threshold_cp) if adjudicate else '1/2-1/2'
    if game is not None and save_pgn is not None:
        game.headers["Result"] = res
        with open(str(save_pgn), "a", encoding="utf-8") as f:
            print(game, file=f)
            print(file=f)
    return res, steps


def run_tournament(strategies, games, playouts, seed=None, max_steps: int = 200, adjudicate: bool = True, material_threshold: int = 1, log_csv: Path | None = None, pgn_dir: Path | None = None, processes: int = 1, ucb_c: float = 0.8, c_puct: float = 0.8, use_eval_priors: bool = False, rollout_steps: int = 200, use_eval_leaf: bool = False, ab_leaf_depth: int = 0, ab_depth_white: int = 3, ab_depth_black: int = 3, grave_K: float = 100.0, eval_adjudicate: bool = False, eval_threshold_cp: int = 200, random_openings: int = 0, resign_threshold_cp: int = 0, resign_moves: int = 0, stockfish_path: Optional[str] = None, sf_depth: int = 12, sf_nodes: int = 0, sf_movetime_ms: int = 0, sf_threads: int = 1, sf_hash_mb: int = 64, sf_skill: Optional[int] = None, sf_limit_strength: bool = False, sf_elo: Optional[int] = None, move_time_ms: int | None = None):
    # Round-robin: for every ordered pair (A,B) run `games` matches with A as white
    results = {(a, b): {'1-0': 0, '0-1': 0, '1/2-1/2': 0} for a in strategies for b in strategies if a != b}

    for a in strategies:
        for b in strategies:
            if a == b:
                continue
            for g in range(games):
                # vary seed per game for some diversity
                s = None if seed is None else seed + g
                save_pgn = None
                if pgn_dir is not None:
                    pgn_dir.mkdir(parents=True, exist_ok=True)
                    save_pgn = pgn_dir / f"{a}_vs_{b}.pgn"
                t0 = time.time()
                res, steps = play_match(a, b, playouts, seed=s, max_steps=max_steps, adjudicate=adjudicate, material_threshold=material_threshold, save_pgn=save_pgn, processes=processes, ucb_c=ucb_c, c_puct=c_puct, use_eval_priors=use_eval_priors, rollout_steps=rollout_steps, use_eval_leaf=use_eval_leaf, ab_leaf_depth=ab_leaf_depth, ab_depth_white=ab_depth_white, ab_depth_black=ab_depth_black, grave_K=grave_K, eval_adjudicate=eval_adjudicate, eval_threshold_cp=eval_threshold_cp, random_openings=random_openings, resign_threshold_cp=resign_threshold_cp, resign_moves=resign_moves, stockfish_path=stockfish_path, sf_depth=sf_depth, sf_nodes=sf_nodes, sf_movetime_ms=sf_movetime_ms, sf_threads=sf_threads, sf_hash_mb=sf_hash_mb, sf_skill=sf_skill, sf_limit_strength=sf_limit_strength, sf_elo=sf_elo, move_time_ms=move_time_ms)
                game_seconds = time.time() - t0
                results[(a, b)][res] += 1
                print(f'Game {a} (white) vs {b} (black): {res}')
                if log_csv is not None:
                    log_csv.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_csv, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        # If file is empty, write a header row for future aggregation
                        if f.tell() == 0:
                            w.writerow([
                                "timestamp", "white", "black", "playouts", "seed", "max_steps", "adjudicate",
                                "material_threshold", "processes", "ucb_c", "c_puct", "use_eval_priors",
                                "rollout_steps", "use_eval_leaf", "ab_leaf_depth", "ab_depth_white", "ab_depth_black",
                                "grave_K", "eval_adjudicate", "eval_threshold_cp", "random_openings",
                                "resign_threshold_cp", "resign_moves", "stockfish_path", "sf_depth", "sf_nodes", "sf_movetime_ms", "sf_threads", "sf_hash_mb", "sf_skill", "sf_limit_strength", "sf_elo",
                                "moves", "game_seconds", "total_playouts", "nps_playouts",
                                "nodes", "depth", "elapsed", "result"
                            ])
                        # Optional AB stats placeholder (none by default)
                        nodes = ''
                        depth_reached = ''
                        elapsed_ms = ''
                        # Derived metrics
                        moves = steps
                        total_playouts = (playouts or 0) * (moves or 0)
                        nps_playouts = (total_playouts / game_seconds) if game_seconds and game_seconds > 0 else ''
                        w.writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"), a, b, playouts, s, max_steps, adjudicate,
                            material_threshold, processes, ucb_c, c_puct, use_eval_priors, rollout_steps,
                            use_eval_leaf, ab_leaf_depth, ab_depth_white, ab_depth_black, grave_K,
                            eval_adjudicate, eval_threshold_cp, random_openings,
                            resign_threshold_cp, resign_moves, (stockfish_path or ''), sf_depth, sf_nodes, sf_movetime_ms, sf_threads, sf_hash_mb, (sf_skill if sf_skill is not None else ''), int(sf_limit_strength), (sf_elo if sf_elo is not None else ''),
                            moves, f"{game_seconds:.3f}", total_playouts, f"{nps_playouts:.1f}" if nps_playouts != '' else '',
                            nodes, depth_reached, elapsed_ms, res
                        ])
    return results


def _elo_from_score(score: float) -> float:
    # score in [0,1]; clamp to avoid inf
    s = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


def estimate_elo_vs_stockfish(results: dict) -> None:
    # Sum UCB vs STOCKFISH over both colors
    pairs = [
        ('UCB', 'STOCKFISH'),
        ('STOCKFISH', 'UCB'),
    ]
    total_games = 0
    ucb_points = 0.0
    for a, b in pairs:
        key = (a, b)
        if key not in results:
            continue
        r = results[key]
        g = r['1-0'] + r['0-1'] + r['1/2-1/2']
        if g == 0:
            continue
        total_games += g
        if a == 'UCB':
            ucb_points += r['1-0'] * 1.0 + r['1/2-1/2'] * 0.5
        else:
            # STOCKFISH as white, so UCB points are reversed
            ucb_points += r['0-1'] * 1.0 + r['1/2-1/2'] * 0.5
    if total_games == 0:
        print('No UCB vs STOCKFISH games found for Elo estimation.')
        return
    score = ucb_points / total_games
    elo = _elo_from_score(score)
    # approximate 95% CI via normal approximation
    se = math.sqrt(max(score * (1 - score) / total_games, 1e-12))
    lo_s = min(max(score - 1.96 * se, 1e-6), 1 - 1e-6)
    hi_s = min(max(score + 1.96 * se, 1e-6), 1 - 1e-6)
    elo_lo = _elo_from_score(lo_s)
    elo_hi = _elo_from_score(hi_s)
    print(f"\nEstimated Elo of UCB relative to STOCKFISH: {elo:.1f} (95% CI: {elo_lo:.1f} .. {elo_hi:.1f}) over {total_games} games")


def estimate_elo_for_strategy_vs_sf(strategy: str, results: dict) -> Optional[tuple[float, float, float, int]]:
    pairs = [
        (strategy, 'STOCKFISH'),
        ('STOCKFISH', strategy),
    ]
    total_games = 0
    points = 0.0
    for a, b in pairs:
        key = (a, b)
        if key not in results:
            continue
        r = results[key]
        g = r['1-0'] + r['0-1'] + r['1/2-1/2']
        if g == 0:
            continue
        total_games += g
        if a == strategy:
            points += r['1-0'] * 1.0 + r['1/2-1/2'] * 0.5
        else:
            # STOCKFISH as white, so strategy points are reversed
            points += r['0-1'] * 1.0 + r['1/2-1/2'] * 0.5
    if total_games == 0:
        return None
    score = points / total_games
    elo = _elo_from_score(score)
    se = math.sqrt(max(score * (1 - score) / total_games, 1e-12))
    lo_s = min(max(score - 1.96 * se, 1e-6), 1 - 1e-6)
    hi_s = min(max(score + 1.96 * se, 1e-6), 1 - 1e-6)
    elo_lo = _elo_from_score(lo_s)
    elo_hi = _elo_from_score(hi_s)
    return elo, elo_lo, elo_hi, total_games


def _default_md_path(prefix: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return Path('logs') / f"{prefix}-{ts}.md"


def write_markdown_tournament_summary(results: dict, args, md_path: Optional[str] = None) -> Path:
    # Prepare path
    out = Path(md_path) if md_path else _default_md_path('tournament')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Build content
    lines: list[str] = []
    lines.append(f"# Tournament Summary\n")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"\n## Settings\n")
    lines.append(f"- Strategies: {', '.join(args.strategies)}")
    lines.append(f"- Games per ordered pair: {args.games}")
    lines.append(f"- Playouts: {args.playouts}; Move time (ms): {args.move_time_ms}")
    lines.append(f"- Max steps: {args.max_steps}; Adjudicate: {not args.no_adjudicate}; Material threshold: {args.material_threshold}")
    lines.append(f"- Processes: {args.processes}; UCB c: {args.ucb_c}; c_puct: {args.c_puct}")
    lines.append(f"- Eval priors: {args.use_eval_priors}; Rollout steps: {args.rollout_steps}; Eval leaf: {args.use_eval_leaf}; AB leaf depth: {args.ab_leaf_depth}")
    lines.append(f"- AB depths (W/B): {args.ab_depth}/{args.ab_depth if args.ab_depth_black is None else args.ab_depth_black}; GRAVE K: {args.grave_K}")
    lines.append(f"- Eval adjudicate: {args.eval_adjudicate}; Eval threshold cp: {args.eval_threshold}")
    lines.append(f"- Random openings: {args.random_openings}; Resign: thr {args.resign_threshold} cp for {args.resign_moves} plies")
    lines.append(f"- Stockfish: path={(args.stockfish_path or d_DEFAULT_STOCKFISH_PATH)}; depth={args.sf_depth}; nodes={args.sf_nodes}; movetime(ms)={args.sf_movetime_ms}; threads={args.sf_threads}; hashMB={args.sf_hash_mb}; skill={args.sf_skill}; limit_strength={args.sf_limit_strength}; elo={args.sf_elo}\n")

    # Pairings table
    lines.append("## Pairings\n")
    lines.append("| White | Black | W | B | D | Total |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for (a, b), r in results.items():
        total = r['1-0'] + r['0-1'] + r['1/2-1/2']
        lines.append(f"| {a} | {b} | {r['1-0']} | {r['0-1']} | {r['1/2-1/2']} | {total} |")

    # Elo estimates vs Stockfish for any strategy present
    if any(s in ('STOCKFISH', 'SF') for s in args.strategies):
        lines.append("\n## Approx Elo vs Stockfish\n")
        for s in args.strategies:
            if s in ('STOCKFISH', 'SF'):
                continue
            est = estimate_elo_for_strategy_vs_sf(s, results)
            if est:
                elo, elo_lo, elo_hi, n = est
                lines.append(f"- {s}: {elo:.1f} (95% CI {elo_lo:.1f}..{elo_hi:.1f}) over {n} games")
            else:
                lines.append(f"- {s}: no games vs Stockfish")

    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nMarkdown summary saved to: {out}")
    return out


def write_markdown_sweep_summary(sweep: dict[str, list[dict]], args, md_path: Optional[str] = None) -> Path:
    out = Path(md_path) if md_path else _default_md_path('sweep')
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Stockfish Sweep Summary\n")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n## Settings\n")
    lines.append(f"- Strategies: {', '.join([s for s in args.strategies if s not in ('STOCKFISH','SF')])}")
    # Print either depth or Elo range depending on which sweep is active
    if getattr(args, 'ssf_elo_sweep', False):
        lines.append(f"- Elo range: {args.ssf_elo_start}..{args.ssf_elo_end} step {args.ssf_elo_step}; Games per side per step: {args.ssf_games_per_side}")
    else:
        lines.append(f"- Depth range: {args.ssf_depth_start}..{args.ssf_depth_end}; Games per side per depth: {args.ssf_games_per_side}")
    lines.append(f"- Playouts: {args.playouts}; Max steps: {args.max_steps}; Adjudicate: {not args.no_adjudicate}; Eval adjudicate: {args.eval_adjudicate}; Eval threshold: {args.eval_threshold}")
    lines.append(f"- Stockfish: path={(args.stockfish_path or d_DEFAULT_STOCKFISH_PATH)}; threads={args.sf_threads}; hashMB={args.sf_hash_mb}; skill={args.sf_skill}; limit_strength={args.sf_limit_strength}; elo={args.sf_elo}\n")

    for strat, rows in sweep.items():
        lines.append(f"\n### {strat}\n")
        # Detect mode by keys in the row
        mode = 'elo' if (rows and 'elo' in rows[0] and 'depth' not in rows[0]) else 'depth'
        if mode == 'depth':
            lines.append("| Depth | Wins | Draws | Losses | Total | RelElo~ |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for r in rows:
                rel_disp = '' if r.get('elo') is None else f"{r['elo']:.0f}"
                lines.append(f"| {r['depth']} | {r['wins']} | {r['draws']} | {r['losses']} | {r['total']} | {rel_disp} |")
        else:
            lines.append("| SF Elo | Wins | Draws | Losses | Total | RelElo~ |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for r in rows:
                rel_disp = '' if r.get('relElo') is None else f"{r['relElo']:.0f}"
                lines.append(f"| {r['elo']} | {r['wins']} | {r['draws']} | {r['losses']} | {r['total']} | {rel_disp} |")

    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nMarkdown sweep summary saved to: {out}")
    return out

def run_stockfish_depth_sweep(
    strategies: list[str],
    games_per_side: int,
    depth_start: int,
    depth_end: int,
    playouts: int,
    seed: Optional[int],
    max_steps: int,
    adjudicate: bool,
    material_threshold: int,
    log_csv: Path | None,
    pgn_dir: Path | None,
    processes: int,
    ucb_c: float,
    c_puct: float,
    use_eval_priors: bool,
    rollout_steps: int,
    use_eval_leaf: bool,
    ab_leaf_depth: int,
    ab_depth_white: int,
    ab_depth_black: int,
    grave_K: float,
    eval_adjudicate: bool,
    eval_threshold_cp: int,
    random_openings: int,
    resign_threshold_cp: int,
    resign_moves: int,
    stockfish_path: Optional[str],
    sf_threads: int,
    sf_hash_mb: int,
    sf_skill: Optional[int],
    sf_limit_strength: bool,
    sf_elo: Optional[int],
    move_time_ms: int | None = None,
) -> dict:
    summary: dict[str, list[dict]] = {}
    for strat in strategies:
        if strat in ('STOCKFISH', 'SF'):
            continue
        print(f"\n=== Sweep for {strat} vs Stockfish ===")
        summary[strat] = []
        proceed = True
        for depth in range(depth_start, depth_end + 1):
            if not proceed:
                break
            # Run games_per_side for each color
            run_strats = [strat, 'STOCKFISH']
            res = run_tournament(
                run_strats,
                games_per_side,
                playouts,
                seed=seed,
                max_steps=max_steps,
                adjudicate=adjudicate,
                material_threshold=material_threshold,
                log_csv=log_csv,
                pgn_dir=pgn_dir,
                processes=processes,
                ucb_c=ucb_c,
                c_puct=c_puct,
                use_eval_priors=use_eval_priors,
                rollout_steps=rollout_steps,
                use_eval_leaf=use_eval_leaf,
                ab_leaf_depth=ab_leaf_depth,
                ab_depth_white=ab_depth_white,
                ab_depth_black=ab_depth_black,
                grave_K=grave_K,
                eval_adjudicate=eval_adjudicate,
                eval_threshold_cp=eval_threshold_cp,
                random_openings=random_openings,
                resign_threshold_cp=resign_threshold_cp,
                resign_moves=resign_moves,
                stockfish_path=stockfish_path or d_DEFAULT_STOCKFISH_PATH,
                sf_depth=depth,
                sf_nodes=0,
                sf_movetime_ms=0,
                sf_threads=sf_threads,
                sf_hash_mb=sf_hash_mb,
                sf_skill=sf_skill,
                sf_limit_strength=sf_limit_strength,
                sf_elo=sf_elo,
                move_time_ms=move_time_ms,
            )
            # Collate results for this depth
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
                # when Stockfish is white, strategy wins correspond to 0-1
                wdl['wins'] += rb['0-1']
                wdl['losses'] += rb['1-0']
                wdl['draws'] += rb['1/2-1/2']
            total = wdl['wins'] + wdl['draws'] + wdl['losses']
            elo_tuple = estimate_elo_for_strategy_vs_sf(strat, res)
            print(f"Depth {depth}: {strat} vs SF -> W-D-L {wdl['wins']}-{wdl['draws']}-{wdl['losses']} (N={total})")
            if elo_tuple:
                elo, elo_lo, elo_hi, n = elo_tuple
                print(f"  Elo({strat}-SF) ~ {elo:.1f} 95% CI [{elo_lo:.1f},{elo_hi:.1f}] over {n} games")
            summary[strat].append({'depth': depth, **wdl, 'total': total, 'elo': (elo_tuple[0] if elo_tuple else None)})
            # Stopping condition: if losses > total/2, stop progressing
            if wdl['losses'] > total / 2:
                print(f"Stopping {strat} at depth {depth} (lost majority)")
                proceed = False
                break
    return summary


def run_stockfish_elo_sweep(
    strategies: list[str],
    games_per_side: int,
    elo_start: int,
    elo_end: int,
    elo_step: int,
    playouts: int,
    seed: Optional[int],
    max_steps: int,
    adjudicate: bool,
    material_threshold: int,
    log_csv: Path | None,
    pgn_dir: Path | None,
    processes: int,
    ucb_c: float,
    c_puct: float,
    use_eval_priors: bool,
    rollout_steps: int,
    use_eval_leaf: bool,
    ab_leaf_depth: int,
    ab_depth_white: int,
    ab_depth_black: int,
    grave_K: float,
    eval_adjudicate: bool,
    eval_threshold_cp: int,
    random_openings: int,
    resign_threshold_cp: int,
    resign_moves: int,
    stockfish_path: Optional[str],
    sf_threads: int,
    sf_hash_mb: int,
    sf_skill: Optional[int],
    move_time_ms: int | None = None,
) -> dict:
    summary: dict[str, list[dict]] = {}
    # Clamp requested Elo range to typical Stockfish bounds to avoid surprising results
    if elo_start < SF_ELO_MIN or elo_end < SF_ELO_MIN:
        print(f"[warn] Requested Elo start/end ({elo_start}-{elo_end}) is below engine min; clamping to {SF_ELO_MIN}.")
    if elo_end > SF_ELO_MAX:
        print(f"[warn] Requested Elo end ({elo_end}) exceeds engine max; clamping to {SF_ELO_MAX}.")
    elo_start = max(elo_start, SF_ELO_MIN)
    elo_end = min(elo_end, SF_ELO_MAX)
    for strat in strategies:
        if strat in ('STOCKFISH', 'SF'):
            continue
        print(f"\n=== Elo sweep for {strat} vs Stockfish ===")
        summary[strat] = []
        proceed = True
        for elo in range(elo_start, elo_end + 1, max(1, elo_step)):
            if not proceed:
                break
            run_strats = [strat, 'STOCKFISH']
            res = run_tournament(
                run_strats,
                games_per_side,
                playouts,
                seed=seed,
                max_steps=max_steps,
                adjudicate=adjudicate,
                material_threshold=material_threshold,
                log_csv=log_csv,
                pgn_dir=pgn_dir,
                processes=processes,
                ucb_c=ucb_c,
                c_puct=c_puct,
                use_eval_priors=use_eval_priors,
                rollout_steps=rollout_steps,
                use_eval_leaf=use_eval_leaf,
                ab_leaf_depth=ab_leaf_depth,
                ab_depth_white=ab_depth_white,
                ab_depth_black=ab_depth_black,
                grave_K=grave_K,
                eval_adjudicate=eval_adjudicate,
                eval_threshold_cp=eval_threshold_cp,
                random_openings=random_openings,
                resign_threshold_cp=resign_threshold_cp,
                resign_moves=resign_moves,
                stockfish_path=stockfish_path or d_DEFAULT_STOCKFISH_PATH,
                sf_depth=1,
                sf_nodes=0,
                sf_movetime_ms=0,
                sf_threads=sf_threads,
                sf_hash_mb=sf_hash_mb,
                sf_skill=sf_skill,
                sf_limit_strength=True,
                sf_elo=elo,
                move_time_ms=move_time_ms,
            )
            # Collate results for this Elo
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
            elo_tuple = estimate_elo_for_strategy_vs_sf(strat, res)
            print(f"Elo {elo}: {strat} vs SF -> W-D-L {wdl['wins']}-{wdl['draws']}-{wdl['losses']} (N={total})")
            if elo_tuple:
                rel, rel_lo, rel_hi, n = elo_tuple
                print(f"  Relative Elo({strat}-SF@{elo}) ~ {rel:.1f} 95% CI [{rel_lo:.1f},{rel_hi:.1f}] over {n} games")
            summary[strat].append({'elo': elo, **wdl, 'total': total, 'relElo': (elo_tuple[0] if elo_tuple else None)})
            # Stop if majority losses
            if wdl['losses'] > total / 2:
                print(f"Stopping {strat} at Elo {elo} (lost majority)")
                proceed = False
                break
    return summary


def print_summary(results):
    print('\nTournament summary:')
    for k, v in results.items():
        a, b = k
        print(f'{a} (white) vs {b} (black): {v["1-0"]} - {v["0-1"]} - {v["1/2-1/2"]}  (w-b-d)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run headless tournament between strategies')
    parser.add_argument('--strategies', '-s', nargs='+', default=['UNIFORM', 'UCB', 'UCT'],
                        help='List of strategies to include (UNIFORM, UCB, UCT, UCT_RAVE, PUTC, PUTC_RAVE, PUTC_GRAVE, AB, STOCKFISH)')
    parser.add_argument('--games', '-g', type=int, default=4, help='Number of games per ordered pair')
    parser.add_argument('--playouts', '-p', type=int, default=50, help='Playouts passed to UCB/UCT/PUTC')
    parser.add_argument('--processes', type=int, default=1, help='Processes for root-parallel MCTS (UCT/PUTC)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (optional)')
    parser.add_argument('--ucb-c', type=float, default=0.8, help='Exploration constant for UCB')
    parser.add_argument('--move-time-ms', type=int, default=0, help='Per-move time budget (ms) for UCB/UCT/PUTC variants; 0 uses playouts')
    parser.add_argument('--max-steps', type=int, default=200, help='Ply cutoff before adjudication/draw')
    parser.add_argument('--no-adjudicate', action='store_true', help='Treat cutoff as draw instead of adjudicating')
    parser.add_argument('--material-threshold', type=int, default=1, help='Material threshold used in adjudication')
    parser.add_argument('--log-csv', type=str, default='', help='Path to append per-game CSV logs')
    parser.add_argument('--pgn-dir', type=str, default='', help='Directory to save PGN files per pairing')
    parser.add_argument('--c-puct', type=float, default=0.8, help='PUCT exploration constant for PUTC/UCT variants')
    parser.add_argument('--use-eval-priors', action='store_true', help='Use heuristic eval to form priors for PUCT/PUTC_RAVE')
    parser.add_argument('--rollout-steps', type=int, default=200, help='Rollout steps before leaf eval; 0 to disable rollouts (direct eval)')
    parser.add_argument('--use-eval-leaf', action='store_true', help='Use classical eval at MCTS leaves instead of heuristic cutoff')
    parser.add_argument('--ab-leaf-depth', type=int, default=0, help='If >0, use AB(Minimax) at MCTS leaves (depth) to evaluate states')
    parser.add_argument('--ab-depth', type=int, default=3, help='AB search depth when choosing AB strategy as a player (White unless overridden)')
    parser.add_argument('--ab-depth-black', type=int, default=None, help='Optional AB depth for Black (defaults to ab-depth)')
    parser.add_argument('--grave-K', type=float, default=100.0, help='GRAVE blending constant K (higher -> more global move average influence)')
    parser.add_argument('--eval-adjudicate', action='store_true', help='Use classical eval to adjudicate at cutoff if material is near equal')
    parser.add_argument('--eval-threshold', type=int, default=200, help='Centipawn threshold for eval-based adjudication at cutoff')
    parser.add_argument('--random-openings', type=int, default=0, help='Number of random half-moves to play before engines move')
    parser.add_argument('--resign-threshold', type=int, default=0, help='Centipawn threshold to trigger resign if held for N plies (0 disables)')
    parser.add_argument('--resign-moves', type=int, default=0, help='Number of consecutive plies beyond threshold to resign (0 disables)')
    parser.add_argument('--log-md', type=str, default='', help='Optional path to write a Markdown summary of results')
    # Stockfish sweep mode
    parser.add_argument('--ssf-sweep', action='store_true', help='Run a sweep vs Stockfish across increasing depths with stop-on-majority-loss per strategy')
    parser.add_argument('--ssf-elo-sweep', action='store_true', help='Run a sweep vs Stockfish across increasing UCI Elo (limit strength) with stop-on-majority-loss per strategy')
    parser.add_argument('--ssf-depth-start', type=int, default=1, help='Start depth for Stockfish sweep')
    parser.add_argument('--ssf-depth-end', type=int, default=4, help='End depth for Stockfish sweep')
    parser.add_argument('--ssf-games-per-side', type=int, default=4, help='Games per color per depth in sweep (total per depth = 2x)')
    parser.add_argument('--ssf-elo-start', type=int, default=800, help='Start UCI Elo for Stockfish Elo sweep')
    parser.add_argument('--ssf-elo-end', type=int, default=2000, help='End UCI Elo for Stockfish Elo sweep')
    parser.add_argument('--ssf-elo-step', type=int, default=200, help='UCI Elo step for Stockfish Elo sweep')
    parser.add_argument('--stockfish-path', type=str, default='', help='Path to Stockfish executable for STOCKFISH strategy')
    parser.add_argument('--sf-depth', type=int, default=12, help='Stockfish search depth (used if nodes/movetime not set)')
    parser.add_argument('--sf-nodes', type=int, default=0, help='Stockfish node limit (overrides depth if >0)')
    parser.add_argument('--sf-movetime-ms', type=int, default=0, help='Stockfish move time in milliseconds (overrides depth if >0 and nodes==0)')
    parser.add_argument('--sf-threads', type=int, default=1, help='Stockfish Threads option')
    parser.add_argument('--sf-hash-mb', type=int, default=64, help='Stockfish Hash size in MB')
    parser.add_argument('--sf-skill', type=int, default=None, help='Stockfish Skill Level (0-20), optional')
    parser.add_argument('--sf-limit-strength', action='store_true', help='Enable Stockfish UCI_LimitStrength')
    parser.add_argument('--sf-elo', type=int, default=None, help='Stockfish UCI_Elo when limit strength is enabled')

    args = parser.parse_args()

    random.seed(args.seed)

    start = time.time()
    if args.ssf_elo_sweep:
        sweep = run_stockfish_elo_sweep(
            args.strategies,
            games_per_side=args.ssf_games_per_side,
            elo_start=args.ssf_elo_start,
            elo_end=args.ssf_elo_end,
            elo_step=args.ssf_elo_step,
            playouts=args.playouts,
            seed=args.seed,
            max_steps=args.max_steps,
            adjudicate=(not args.no_adjudicate),
            material_threshold=args.material_threshold,
            log_csv=(Path(args.log_csv) if args.log_csv else None),
            pgn_dir=(Path(args.pgn_dir) if args.pgn_dir else None),
            processes=args.processes,
            ucb_c=args.ucb_c,
            c_puct=args.c_puct,
            use_eval_priors=args.use_eval_priors,
            rollout_steps=args.rollout_steps,
            use_eval_leaf=args.use_eval_leaf,
            ab_leaf_depth=args.ab_leaf_depth,
            ab_depth_white=args.ab_depth,
            ab_depth_black=(args.ab_depth if args.ab_depth_black is None else args.ab_depth_black),
            grave_K=args.grave_K,
            eval_adjudicate=args.eval_adjudicate,
            eval_threshold_cp=args.eval_threshold,
            random_openings=args.random_openings,
            resign_threshold_cp=args.resign_threshold,
            resign_moves=args.resign_moves,
            stockfish_path=(args.stockfish_path if args.stockfish_path else None),
            sf_threads=args.sf_threads,
            sf_hash_mb=args.sf_hash_mb,
            sf_skill=args.sf_skill,
            move_time_ms=(None if args.move_time_ms <= 0 else args.move_time_ms),
        )
        print("\nElo sweep completed.")
        write_markdown_sweep_summary(sweep, args, md_path=(args.log_md if args.log_md else None))
        print('\nElapsed (s):', time.time() - start)
        raise SystemExit(0)

    if args.ssf_sweep:
        # Use provided strategies (excluding STOCKFISH automatically)
        sweep = run_stockfish_depth_sweep(
            args.strategies,
            games_per_side=args.ssf_games_per_side,
            depth_start=args.ssf_depth_start,
            depth_end=args.ssf_depth_end,
            playouts=args.playouts,
            seed=args.seed,
            max_steps=args.max_steps,
            adjudicate=(not args.no_adjudicate),
            material_threshold=args.material_threshold,
            log_csv=(Path(args.log_csv) if args.log_csv else None),
            pgn_dir=(Path(args.pgn_dir) if args.pgn_dir else None),
            processes=args.processes,
            ucb_c=args.ucb_c,
            c_puct=args.c_puct,
            use_eval_priors=args.use_eval_priors,
            rollout_steps=args.rollout_steps,
            use_eval_leaf=args.use_eval_leaf,
            ab_leaf_depth=args.ab_leaf_depth,
            ab_depth_white=args.ab_depth,
            ab_depth_black=(args.ab_depth if args.ab_depth_black is None else args.ab_depth_black),
            grave_K=args.grave_K,
            eval_adjudicate=args.eval_adjudicate,
            eval_threshold_cp=args.eval_threshold,
            random_openings=args.random_openings,
            resign_threshold_cp=args.resign_threshold,
            resign_moves=args.resign_moves,
            stockfish_path=(args.stockfish_path if args.stockfish_path else None),
            sf_threads=args.sf_threads,
            sf_hash_mb=args.sf_hash_mb,
            sf_skill=args.sf_skill,
            sf_limit_strength=args.sf_limit_strength,
            sf_elo=args.sf_elo,
            move_time_ms=(None if args.move_time_ms <= 0 else args.move_time_ms),
        )
        print("\nSweep completed.")
        # Write Markdown summary
        write_markdown_sweep_summary(sweep, args, md_path=(args.log_md if args.log_md else None))
        print('\nElapsed (s):', time.time() - start)
        raise SystemExit(0)

    res = run_tournament(
        args.strategies,
        args.games,
        args.playouts,
        seed=args.seed,
        max_steps=args.max_steps,
        adjudicate=(not args.no_adjudicate),
        material_threshold=args.material_threshold,
        log_csv=(Path(args.log_csv) if args.log_csv else None),
        pgn_dir=(Path(args.pgn_dir) if args.pgn_dir else None),
        processes=args.processes,
        ucb_c=args.ucb_c,
        c_puct=args.c_puct,
        use_eval_priors=args.use_eval_priors,
        rollout_steps=args.rollout_steps,
        use_eval_leaf=args.use_eval_leaf,
        ab_leaf_depth=args.ab_leaf_depth,
        ab_depth_white=args.ab_depth,
    ab_depth_black=(args.ab_depth if args.ab_depth_black is None else args.ab_depth_black),
    grave_K=args.grave_K,
    eval_adjudicate=args.eval_adjudicate,
    eval_threshold_cp=args.eval_threshold,
    random_openings=args.random_openings,
    resign_threshold_cp=args.resign_threshold,
    resign_moves=args.resign_moves,
    stockfish_path=(args.stockfish_path if args.stockfish_path else None),
    sf_depth=args.sf_depth,
    sf_nodes=args.sf_nodes,
    sf_movetime_ms=args.sf_movetime_ms,
    sf_threads=args.sf_threads,
    sf_hash_mb=args.sf_hash_mb,
    sf_skill=args.sf_skill,
    sf_limit_strength=args.sf_limit_strength,
    sf_elo=args.sf_elo,
    move_time_ms=(None if args.move_time_ms <= 0 else args.move_time_ms),
    )
    print_summary(res)
    # Write Markdown summary
    write_markdown_tournament_summary(res, args, md_path=(args.log_md if args.log_md else None))
    # Auto-estimate UCB Elo if pairing with STOCKFISH exists
    if 'UCB' in args.strategies and any(s in ('STOCKFISH', 'SF') for s in args.strategies):
        estimate_elo_vs_stockfish(res)
    print('\nElapsed (s):', time.time() - start)
