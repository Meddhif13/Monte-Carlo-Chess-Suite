import argparse
import time
from pathlib import Path
from typing import Optional, List, Tuple, Any

import chess
import chess.pgn
import chess.engine

DEFAULT_STOCKFISH_PATH = r"C:\\Program Files\\stockfish\\stockfish-windows-x86-64-avx2.exe"


def move_accuracy_from_loss(loss_cp: float, k: float = 200.0) -> float:
    """
    Map centipawn loss for a move to an accuracy percentage [0, 100].
    Heuristic, chess.com-like monotone mapping: 0 cp => 100, 50 cp => ~80 (for k=200),
    100 cp => ~66.7, 200 cp => 50, 400 cp => ~33.3.
    """
    loss = max(0.0, float(loss_cp))
    return 100.0 * (k / (k + loss))


def cp_from_score_for_color(score: Any, color: bool) -> Optional[int]:
    """
    Convert python-chess engine Score to centipawns from the perspective of `color`.
    Works across versions by duck-typing.
    """
    try:
        # Try modern POV helpers first
        if hasattr(score, "pov"):
            sc = score.pov(chess.WHITE if color else chess.BLACK)
        elif hasattr(score, "white") and hasattr(score, "black"):
            sc = score.white() if color else score.black()
        else:
            sc = score

        # Centipawn value is preferred
        if hasattr(sc, "cp") and sc.cp is not None:
            return int(sc.cp)
        # Mate score fallback
        if hasattr(sc, "mate") and sc.mate() is not None:
            return 100000 if sc.mate() > 0 else -100000
    except Exception:
        return None
    return None


def analyze_game(
    game: chess.pgn.Game,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    multipv: int = 1,
    skip_plies: int = 0,
    inacc_thr: int = 50,
    mistake_thr: int = 100,
    blunder_thr: int = 200,
    acc_k: float = 200.0,
    per_move_writer: Optional[Tuple] = None,
) -> dict:
    board = game.board()
    node = game

    white_losses: List[int] = []
    black_losses: List[int] = []
    white_accs: List[float] = []
    black_accs: List[float] = []
    white_best = 0
    black_best = 0
    white_moves = 0
    black_moves = 0
    white_inacc = white_mistakes = white_blunders = 0
    black_inacc = black_mistakes = black_blunders = 0

    ply = 0
    while node.variations:
        move = node.variations[0].move
        color = board.turn  # side to move before playing 'move'
        # Optional: skip early plies to avoid opening book noise
        if ply >= skip_plies:
            # Analyze current position to get engine best and eval
            info = engine.analyse(board, limit, multipv=multipv)
            if isinstance(info, list):
                top = info[0]
            else:
                top = info
            best_move = None
            if isinstance(top, dict) and "pv" in top and top["pv"]:
                best_move = top["pv"][0]
            best_score = None
            if isinstance(top, dict) and "score" in top:
                best_score = cp_from_score_for_color(top["score"], color)

            # Analyze the position after the played move
            board.push(move)
            try:
                info_after = engine.analyse(board, limit, multipv=1)
                if isinstance(info_after, list):
                    info_after = info_after[0]
                played_score = None
                if isinstance(info_after, dict) and "score" in info_after:
                    # Score for position after the move; evaluate from the original player's POV
                    played_score = cp_from_score_for_color(info_after["score"], color)
            finally:
                board.pop()

            # Compute centipawn loss (positive means worse than best)
            loss = 0
            if best_score is not None and played_score is not None:
                loss = max(0, best_score - played_score)

            if color:
                white_moves += 1
                white_losses.append(loss)
                white_accs.append(move_accuracy_from_loss(loss, k=acc_k))
                if best_move is not None and move == best_move:
                    white_best += 1
                if loss >= blunder_thr:
                    white_blunders += 1
                elif loss >= mistake_thr:
                    white_mistakes += 1
                elif loss >= inacc_thr:
                    white_inacc += 1
            else:
                black_moves += 1
                black_losses.append(loss)
                black_accs.append(move_accuracy_from_loss(loss, k=acc_k))
                if best_move is not None and move == best_move:
                    black_best += 1
                if loss >= blunder_thr:
                    black_blunders += 1
                elif loss >= mistake_thr:
                    black_mistakes += 1
                elif loss >= inacc_thr:
                    black_inacc += 1

            # Optional: write per-move detail row
            if per_move_writer is not None:
                try:
                    # per_move_writer is (csv_writer, pgn_name, game_id)
                    w, pgn_name, gid = per_move_writer
                    san = board.san(move)
                    best_san = board.san(best_move) if best_move is not None else ""
                    w.writerow([
                        pgn_name, gid, ply, "W" if color else "B", san, best_san, loss,
                        "blunder" if loss >= blunder_thr else ("mistake" if loss >= mistake_thr else ("inaccuracy" if loss >= inacc_thr else "ok")),
                        f"{move_accuracy_from_loss(loss, k=acc_k):.1f}",
                    ])
                except Exception:
                    pass
        
        # Play the move and continue
        board.push(move)
        node = node.variations[0]
        ply += 1

    def avg(xs: List[float] | List[int]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    result = game.headers.get("Result", "*")
    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")

    return {
        "white": white,
        "black": black,
        "result": result,
        "moves": ply,
        "white_acpl": avg(white_losses),
        "black_acpl": avg(black_losses),
    "white_accuracy": avg(white_accs),
    "black_accuracy": avg(black_accs),
        "white_best_pct": (100.0 * white_best / white_moves) if white_moves else 0.0,
        "black_best_pct": (100.0 * black_best / black_moves) if black_moves else 0.0,
        "white_inacc": white_inacc, "white_mistakes": white_mistakes, "white_blunders": white_blunders,
        "black_inacc": black_inacc, "black_mistakes": black_mistakes, "black_blunders": black_blunders,
    }


def iter_pgn_games(path: Path):
    if path.is_dir():
        # recurse into subdirectories to collect all PGNs
        for pgn in sorted(path.rglob("*.pgn")):
            with open(pgn, encoding="utf-8") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    yield pgn.name, game
    else:
        with open(path, encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield path.name, game


def main():
    ap = argparse.ArgumentParser(description="Analyze PGN games with Stockfish for ACPL and best-move stats")
    ap.add_argument("--pgn", required=True, help="PGN file or directory containing PGNs")
    ap.add_argument("--stockfish-path", default=DEFAULT_STOCKFISH_PATH, help="Path to Stockfish executable")
    ap.add_argument("--sf-depth", type=int, default=12, help="Stockfish depth (used if movetime==0)")
    ap.add_argument("--sf-movetime-ms", type=int, default=0, help="Stockfish movetime in ms (overrides depth if >0)")
    ap.add_argument("--sf-threads", type=int, default=1, help="Stockfish Threads option")
    ap.add_argument("--sf-hash-mb", type=int, default=64, help="Stockfish Hash size in MB")
    ap.add_argument("--multipv", type=int, default=1, help="MultiPV setting (top lines to compute)")
    ap.add_argument("--skip-plies", type=int, default=0, help="Skip the first N plies for accuracy stats (opening)")
    ap.add_argument("--out-csv", default="", help="Output CSV path for per-game summary (default logs/analysis-<ts>.csv)")
    ap.add_argument("--inacc-thr", type=int, default=50, help="Centipawn threshold for inaccuracy classification")
    ap.add_argument("--mistake-thr", type=int, default=100, help="Centipawn threshold for mistake classification")
    ap.add_argument("--blunder-thr", type=int, default=200, help="Centipawn threshold for blunder classification")
    ap.add_argument("--acc-k", type=float, default=200.0, help="Accuracy mapping scale (higher => more forgiving)")
    ap.add_argument("--out-move-csv", default="", help="Optional per-move CSV output (one row per move with loss and classification)")
    args = ap.parse_args()

    pgn_path = Path(args.pgn)
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN path not found: {pgn_path}")

    sf_path = Path(args.stockfish_path)
    if not sf_path.exists():
        raise FileNotFoundError(f"Stockfish not found at: {sf_path}")

    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    try:
        engine.configure({
            "Threads": int(max(1, args.sf_threads)),
            "Hash": int(max(16, args.sf_hash_mb)),
        })
        limit = chess.engine.Limit(depth=args.sf_depth) if args.sf_movetime_ms <= 0 else chess.engine.Limit(time=args.sf_movetime_ms/1000.0)

        import csv
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_csv = Path(args.out_csv) if args.out_csv else Path("logs") / f"analysis-{ts}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Optional per-move CSV
        move_writer = None
        if args.out_move_csv:
            out_moves = Path(args.out_move_csv)
            out_moves.parent.mkdir(parents=True, exist_ok=True)
            mf = open(out_moves, "w", newline="", encoding="utf-8")
            mw = csv.writer(mf)
            mw.writerow(["pgn", "game_id", "ply", "side", "san", "best_san", "loss_cp", "class", "accuracy"])
            move_writer = (mw, None, None, mf)  # include file handle for cleanup

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pgn", "white", "black", "result", "moves", "white_acpl", "black_acpl", "white_accuracy", "black_accuracy", "white_best_pct", "black_best_pct", "white_inacc", "white_mistakes", "white_blunders", "black_inacc", "black_mistakes", "black_blunders"])    
            gid = 0
            for pgn_name, game in iter_pgn_games(pgn_path):
                try:
                    per_mv = None
                    if move_writer is not None:
                        # patch writer with current pgn name and game id
                        mw, _, _, _ = move_writer
                        per_mv = (mw, pgn_name, gid)
                    stats = analyze_game(
                        game, engine, limit,
                        multipv=args.multipv,
                        skip_plies=args.skip_plies,
                        inacc_thr=args.inacc_thr,
                        mistake_thr=args.mistake_thr,
                        blunder_thr=args.blunder_thr,
                        acc_k=args.acc_k,
                        per_move_writer=per_mv,
                    )
                    w.writerow([
                        pgn_name,
                        stats["white"], stats["black"], stats["result"], stats["moves"],
                        f"{stats['white_acpl']:.1f}", f"{stats['black_acpl']:.1f}",
                        f"{stats['white_accuracy']:.1f}", f"{stats['black_accuracy']:.1f}",
                        f"{stats['white_best_pct']:.1f}", f"{stats['black_best_pct']:.1f}",
                        stats["white_inacc"], stats["white_mistakes"], stats["white_blunders"],
                        stats["black_inacc"], stats["black_mistakes"], stats["black_blunders"],
                    ])
                    gid += 1
                except Exception as e:
                    # Continue on errors for individual games
                    print(f"[warn] Failed to analyze game in {pgn_name}: {e}")
                    continue
        print(f"Analysis written to: {out_csv}")
        if move_writer is not None:
            try:
                _, _, _, mf = move_writer
                mf.close()
                print(f"Per-move details written to: {args.out_move_csv}")
            except Exception:
                pass
    finally:
        try:
            engine.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
