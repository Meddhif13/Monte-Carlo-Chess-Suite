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
        "moves": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "acpl_sum": 0.0,
    "acc_sum": 0.0,
        "best_sum": 0.0,
        "best_moves": 0,  # count moves considered in best% denominator
        "inacc": 0,
        "mistakes": 0,
        "blunders": 0,
    })

    def update_for_side(strategy, is_white, row):
        # result handling
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

        moves = int(row.get("moves", 0))
        agg[strategy]["games"] += 1
        agg[strategy]["moves"] += moves

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

        agg[strategy]["acpl_sum"] += acpl
        agg[strategy]["acc_sum"] += acc
        agg[strategy]["best_sum"] += best_pct
        agg[strategy]["best_moves"] += moves  # approximate denominator alignment
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
            "acpl_mean", "accuracy_mean", "best_move_pct_mean",
            "inacc_per100", "mistakes_per100", "blunders_per100"
        ])
        for strat, s in sorted(agg.items()):
            games = s["games"] or 1
            moves = s["moves"] or 1
            acpl_mean = s["acpl_sum"] / games
            acc_mean = s["acc_sum"] / games
            best_pct_mean = s["best_sum"] / games
            per100 = lambda x: 100.0 * x / moves
            w.writerow([
                strat, s["games"], s["moves"], s["wins"], s["draws"], s["losses"],
                f"{acpl_mean:.1f}", f"{acc_mean:.1f}", f"{best_pct_mean:.1f}",
                f"{per100(s['inacc']):.2f}", f"{per100(s['mistakes']):.2f}", f"{per100(s['blunders']):.2f}",
            ])

    # Write Markdown summary
    out_md = Path(args.out_md) if args.out_md else Path("logs") / "rollup.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Quality rollup by strategy\n\n")
        f.write(f"Source: {args.input}\n\n")
        f.write("| Strategy | Games | W | D | L | ACPL | Acc% | Best% | Inacc/100 | Mistakes/100 | Blunders/100 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for strat, s in sorted(agg.items()):
            games = s["games"] or 1
            moves = s["moves"] or 1
            acpl_mean = s["acpl_sum"] / games
            acc_mean = s["acc_sum"] / games
            best_pct_mean = s["best_sum"] / games
            per100 = lambda x: 100.0 * x / moves
            f.write(
                f"| {strat} | {s['games']} | {s['wins']} | {s['draws']} | {s['losses']} | "
                f"{acpl_mean:.1f} | {acc_mean:.1f} | {best_pct_mean:.1f} | {per100(s['inacc']):.2f} | {per100(s['mistakes']):.2f} | {per100(s['blunders']):.2f} |\n"
            )
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
