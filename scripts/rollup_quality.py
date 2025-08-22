import argparse
import csv
from pathlib import Path
from collections import defaultdict

# Input format expected: analyze_pgn.py CSV with headers:
# pgn, white, black, result, moves, white_acpl, black_acpl, white_accuracy, black_accuracy,
# white_best_pct, black_best_pct, white_inacc, white_mistakes, white_blunders, black_inacc, black_mistakes, black_blunders


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-game quality metrics by strategy and emit CSV/Markdown")
    ap.add_argument("--input", required=True, help="Input CSV (from analyze_pgn.py) or directory containing CSVs")
    ap.add_argument("--out-csv", default="", help="Output aggregated CSV path (default logs/rollup.csv)")
    ap.add_argument("--out-md", default="", help="Output Markdown summary path (default logs/rollup.md)")
    args = ap.parse_args()

    src = Path(args.input)
    files = []
    if src.is_dir():
        files = sorted(src.glob("*.csv"))
    else:
        files = [src]

    # Accumulators per strategy
    agg = defaultdict(lambda: {
        "games": 0,
        "moves": 0,              # total plies (half-moves) across all games the strategy appeared in (both colors counted separately)
        "wins": 0,
        "draws": 0,
        "losses": 0,
        # Game-averaged accumulators (legacy)
        "acpl_game_sum": 0.0,
        "acc_game_sum": 0.0,
        "best_pct_game_sum": 0.0,
        # Move-weighted accumulators (new, estimated from per-game move counts)
        "acpl_loss_total": 0.0,   # sum of centipawn losses over that side's moves (estimated)
        "acc_weighted_total": 0.0,# sum of per-move accuracies
        "side_moves": 0,          # estimated total number of moves (by that side) seen
        "best_moves_est": 0.0,    # estimated count of best moves from per-game percentage * side_moves_game
        # Error counts
        "inacc": 0,
        "mistakes": 0,
        "blunders": 0,
    })

    def update_for_side(strategy: str, is_white: bool, row: dict):
        # Result handling from perspective of the given side
        res = row.get("result", "*")
        if is_white:
            if res == "1-0":
                agg[strategy]["wins"] += 1
            elif res == "0-1":
                agg[strategy]["losses"] += 1
            elif res == "1/2-1/2":
                agg[strategy]["draws"] += 1
        else:
            if res == "0-1":
                agg[strategy]["wins"] += 1
            elif res == "1-0":
                agg[strategy]["losses"] += 1
            elif res == "1/2-1/2":
                agg[strategy]["draws"] += 1

        total_plies = int(row.get("moves", 0))  # total half-moves in game
        agg[strategy]["games"] += 1
        agg[strategy]["moves"] += total_plies

        # Estimate number of moves played by this side in the game
        side_moves_game = (total_plies + 1) // 2 if is_white else total_plies // 2
        if side_moves_game < 0:
            side_moves_game = 0

        if is_white:
            acpl = safe_float(row.get("white_acpl", 0.0))
            acc = safe_float(row.get("white_accuracy", 0.0))
            best_pct = safe_float(row.get("white_best_pct", 0.0))
            inacc = int(row.get("white_inacc", 0))
            mistakes = int(row.get("white_mistakes", 0))
            blunders = int(row.get("white_blunders", 0))
        else:
            acpl = safe_float(row.get("black_acpl", 0.0))
            acc = safe_float(row.get("black_accuracy", 0.0))
            best_pct = safe_float(row.get("black_best_pct", 0.0))
            inacc = int(row.get("black_inacc", 0))
            mistakes = int(row.get("black_mistakes", 0))
            blunders = int(row.get("black_blunders", 0))

        # Game-level (legacy style averaging)
        agg[strategy]["acpl_game_sum"] += acpl
        agg[strategy]["acc_game_sum"] += acc
        agg[strategy]["best_pct_game_sum"] += best_pct

        # Move-weighted (preferred) accumulators
        agg[strategy]["acpl_loss_total"] += acpl * side_moves_game
        agg[strategy]["acc_weighted_total"] += acc * side_moves_game
        agg[strategy]["side_moves"] += side_moves_game
        agg[strategy]["best_moves_est"] += (best_pct / 100.0) * side_moves_game

        # Error tallies
        agg[strategy]["inacc"] += inacc
        agg[strategy]["mistakes"] += mistakes
        agg[strategy]["blunders"] += blunders

    # Ingest rows
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                w = row.get("white", "?")
                b = row.get("black", "?")
                update_for_side(w, True, row)
                update_for_side(b, False, row)

    # Write aggregated CSV
    out_csv = Path(args.out_csv) if args.out_csv else Path("logs") / "rollup.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "strategy", "games", "moves", "wins", "draws", "losses",
            # Game-averaged metrics (legacy)
            "acpl_game_mean", "accuracy_game_mean", "best_move_pct_game_mean",
            # Move-weighted metrics (preferred)
            "acpl_move_mean", "accuracy_move_mean", "best_move_pct_move_mean",
            "inacc_per100", "mistakes_per100", "blunders_per100"
        ])
        for strat, s in sorted(agg.items()):
            games = s["games"] or 1
            total_plies = s["moves"] or 1
            side_moves = s["side_moves"] or 1
            acpl_game_mean = s["acpl_game_sum"] / games
            acc_game_mean = s["acc_game_sum"] / games
            best_pct_game_mean = s["best_pct_game_sum"] / games
            # Move-weighted means
            acpl_move_mean = s["acpl_loss_total"] / side_moves if side_moves else 0.0
            accuracy_move_mean = s["acc_weighted_total"] / side_moves if side_moves else 0.0
            best_move_pct_move_mean = (s["best_moves_est"] / side_moves * 100.0) if side_moves else 0.0
            per100 = lambda x: 100.0 * x / side_moves
            w.writerow([
                strat, s["games"], s["moves"], s["wins"], s["draws"], s["losses"],
                f"{acpl_game_mean:.1f}", f"{acc_game_mean:.1f}", f"{best_pct_game_mean:.1f}",
                f"{acpl_move_mean:.1f}", f"{accuracy_move_mean:.1f}", f"{best_move_pct_move_mean:.1f}",
                f"{per100(s['inacc']):.2f}", f"{per100(s['mistakes']):.2f}", f"{per100(s['blunders']):.2f}",
            ])

    # Write Markdown summary
    out_md = Path(args.out_md) if args.out_md else Path("logs") / "rollup.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Quality rollup by strategy\n\n")
        f.write(f"Source: {args.input}\n\n")
        f.write("Game-averaged metrics (legacy): ACPL/Acc% averaged per game (unweighted). Move-weighted metrics prefer per-move fidelity.\n\n")
        f.write("| Strategy | Games | W | D | L | ACPL(g) | Acc%(g) | Best%(g) | ACPL(m) | Acc%(m) | Best%(m) | Inacc/100 | Mistakes/100 | Blunders/100 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for strat, s in sorted(agg.items()):
            games = s["games"] or 1
            side_moves = s["side_moves"] or 1
            acpl_game_mean = s["acpl_game_sum"] / games
            acc_game_mean = s["acc_game_sum"] / games
            best_pct_game_mean = s["best_pct_game_sum"] / games
            acpl_move_mean = s["acpl_loss_total"] / side_moves if side_moves else 0.0
            accuracy_move_mean = s["acc_weighted_total"] / side_moves if side_moves else 0.0
            best_move_pct_move_mean = (s["best_moves_est"] / side_moves * 100.0) if side_moves else 0.0
            per100 = lambda x: 100.0 * x / side_moves
            f.write(
                f"| {strat} | {s['games']} | {s['wins']} | {s['draws']} | {s['losses']} | "
                f"{acpl_game_mean:.1f} | {acc_game_mean:.1f} | {best_pct_game_mean:.1f} | "
                f"{acpl_move_mean:.1f} | {accuracy_move_mean:.1f} | {best_move_pct_move_mean:.1f} | "
                f"{per100(s['inacc']):.2f} | {per100(s['mistakes']):.2f} | {per100(s['blunders']):.2f} |\n"
            )
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
