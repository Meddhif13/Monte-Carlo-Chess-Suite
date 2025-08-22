"""Plotting utilities for Monte-Carlo Chess Suite experiments.

Generates publication-quality figures from tournament and quality analysis CSVs.

Usage examples (PowerShell):

# 1. Accuracy & error profile (rollup_acc.csv)
python scripts/plot_results.py --rollup-acc logs/sweep_500ms_elo1320/rollup_acc.csv --out-dir figs

# 2. Parameter sweep (example: UCB ucb_c values) from individual csv_* files
python scripts/plot_results.py --sweep-dir logs/sweep_500ms_elo1320 --pattern "csv_UCB_ucb_c-*.csv" --out-dir figs

# 3. Compare AB depths (rollup subset + per-depth CSV)
python scripts/plot_results.py --sweep-dir logs/sweep_500ms_elo1320 --pattern "csv_AB_ab_depth-*.csv" --out-dir figs

# 4. All standard figures in one go
python scripts/plot_results.py --rollup-acc logs/sweep_500ms_elo1320/rollup_acc.csv --sweep-dir logs/sweep_500ms_elo1320 --all --out-dir figs
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)


def _export(fig, out_base: Path):
    """Save figure to PNG & SVG; also legend-free duplicate if legend present.

    out_base: path without extension (e.g., figs/accuracy_bar)
    Creates:
      out_base.png
      out_base.svg
      out_base_clean.png/.svg (legend removed if existed)
    """
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix('.png'), dpi=200)
    fig.savefig(out_base.with_suffix('.svg'))
    leg = fig.legends
    # Some legends attached via axes
    if not leg:
        for ax in fig.axes:
            if ax.get_legend():
                leg.append(ax.get_legend())
    if leg:
        # Remove and save clean copy
        for L in leg:
            L.remove()
        fig.tight_layout()
        fig.savefig(out_base.with_name(out_base.name + '_clean').with_suffix('.png'), dpi=200)
        fig.savefig(out_base.with_name(out_base.name + '_clean').with_suffix('.svg'))


def _safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Failed to read {path}: {e}")
        return None


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns present: {candidates}")


def plot_accuracy_bar(df: pd.DataFrame, out: Path):
    # Prefer move-weighted means; fall back to legacy column names.
    acc_col = _pick_col(df, ["accuracy_move_mean", "accuracy_mean", "accuracy_game_mean"])
    order = df.sort_values(acc_col, ascending=False)["strategy"].tolist()
    fig, ax = plt.subplots(figsize=(8,4.2))
    sns.barplot(data=df, x="strategy", y=acc_col, order=order, ax=ax, hue=None)
    label_src = "move-weighted" if acc_col.startswith("accuracy_move") else ("legacy" if acc_col=="accuracy_mean" else "game-avg")
    ax.set_ylabel(f"Accuracy% ({label_src})")
    ax.set_xlabel("Strategy")
    ax.set_title("Strategy Accuracy (500ms / Stockfish Elo≈1320)")
    for p in ax.patches:
        if hasattr(p, 'get_height') and hasattr(p, 'get_x') and hasattr(p, 'get_width'):
            h = p.get_height()  # type: ignore[attr-defined]
            ax.annotate(f"{h:.1f}",(p.get_x() + p.get_width() / 2, h),ha='center', va='bottom', fontsize=9)  # type: ignore[attr-defined]
    fig.tight_layout(); _export(fig, out / "accuracy_bar"); plt.close(fig)


def plot_error_stack(df: pd.DataFrame, out: Path):
    acc_col = _pick_col(df, ["accuracy_move_mean", "accuracy_mean", "accuracy_game_mean"])
    melt_df = df.melt(id_vars=["strategy"], value_vars=["inacc_per100","mistakes_per100","blunders_per100"], var_name="error_type", value_name="rate")
    mapping = {"inacc_per100":"Inaccuracies","mistakes_per100":"Mistakes","blunders_per100":"Blunders"}
    melt_df["error_type"] = melt_df["error_type"].map(mapping)
    order = df.sort_values(acc_col, ascending=False)["strategy"].tolist()
    fig, ax = plt.subplots(figsize=(8,4.8))
    for et in ["Inaccuracies","Mistakes","Blunders"]:
        sub = melt_df[melt_df.error_type==et]
        sns.barplot(data=sub, x="strategy", y="rate", order=order, ax=ax, label=et)
    ax.set_ylabel("Errors per 100 moves")
    ax.set_xlabel("Strategy")
    ax.set_title("Error Profile by Strategy")
    ax.legend(frameon=False)
    fig.tight_layout(); _export(fig, out / "error_profile"); plt.close(fig)


def plot_accuracy_vs_acpl(df: pd.DataFrame, out: Path):
    acc_col = _pick_col(df, ["accuracy_move_mean", "accuracy_mean", "accuracy_game_mean"])
    acpl_col = _pick_col(df, ["acpl_move_mean", "acpl_mean", "acpl_game_mean"])
    fig, ax = plt.subplots(figsize=(6,5))
    sns.scatterplot(data=df, x=acpl_col, y=acc_col, hue="strategy", s=120, ax=ax)
    for _, r in df.iterrows():
        ax.text(r[acpl_col]+10, r[acc_col], r.strategy, fontsize=9)
    ax.set_xlabel(f"ACPL ({'move-weighted' if 'move' in acpl_col else 'game-avg'})")
    ax.set_ylabel(f"Accuracy% ({'move-weighted' if 'move' in acc_col else 'game-avg'})")
    ax.set_title("Accuracy vs ACPL")
    fig.tight_layout(); _export(fig, out / "accuracy_vs_acpl"); plt.close(fig)


def _extract_param_from_filename(name: str, key: str):
    """Extract numeric parameter value from filename.

    Handles patterns like:
      csv_UCB_ucb_c-0.8.csv
      csv_PUTC_GRAVE_ab_leaf_depth-2_c_puct-1.4_grave_K-150.0_....csv

    Avoids capturing trailing '.' before extension (previous regex greedily captured '0.6.').
    """
    m = re.search(fr"{re.escape(key)}-([0-9]+(?:\.[0-9]+)?)", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        # Fallback: strip trailing non-digits/dots then retry
        cleaned = re.sub(r"[^0-9.].*$", "", m.group(1))
        try:
            return float(cleaned)
        except Exception:
            return None


def plot_param_sweep(files: list[Path], key: str, out: Path, metric: str = "result"):
    rows = []
    for f in files:
        df = _safe_read_csv(f)
        if df is None or df.empty: continue
        val = _extract_param_from_filename(f.name, key)
        # result column may appear as last; assume WDL encoded in 'result'
        w = sum(df.result=='1-0')
        d = sum(df.result=='1/2-1/2')
        l = sum(df.result=='0-1')
        games = w + d + l
        score = (w + 0.5*d)/games if games else 0
        rows.append({key:val, 'wins':w, 'draws':d, 'losses':l, 'games':games, 'score':score})
    if not rows:
        print(f"[warn] No rows for param sweep {key}")
        return
    sdf = pd.DataFrame(rows).sort_values(key)
    fig, ax1 = plt.subplots(figsize=(6,4))
    sns.lineplot(data=sdf, x=key, y='score', marker='o', ax=ax1, color='tab:blue')
    ax1.set_ylabel('Score (W + 0.5D)/Games', color='tab:blue')
    ax1.set_xlabel(key)
    ax2 = ax1.twinx()
    sns.barplot(data=sdf, x=key, y='games', alpha=0.2, ax=ax2, color='gray')
    ax2.set_ylabel('Games')
    ax1.set_title(f"Parameter sweep: {key}")
    fig.tight_layout(); _export(fig, out / f"sweep_{key}"); plt.close(fig)


def plot_ab_depth(files: list[Path], out: Path):
    rows = []
    for f in files:
        df = _safe_read_csv(f)
        if df is None or df.empty: continue
        depth = _extract_param_from_filename(f.name, 'ab_depth')
        w = sum(df.result=='1-0'); d = sum(df.result=='1/2-1/2'); l = sum(df.result=='0-1')
        games = w+d+l
        score = (w + 0.5*d)/games if games else 0
        rows.append({'ab_depth':depth, 'wins':w,'draws':d,'losses':l,'games':games,'score':score})
    if not rows: return
    df = pd.DataFrame(rows).sort_values('ab_depth')
    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(data=df, x='ab_depth', y='score', marker='o', ax=ax)
    ax.set_ylabel('Score'); ax.set_xlabel('AB Search Depth'); ax.set_title('Alpha-Beta Depth vs Score (calibration)')
    for _, r in df.iterrows():
        ax.text(r.ab_depth, r.score+0.01, f"{r.score:.2f}", ha='center', fontsize=9)
    fig.tight_layout(); _export(fig, out / 'ab_depth_score'); plt.close(fig)


def plot_move_vs_game_comparison(df: pd.DataFrame, out: Path):
    required = {"strategy","accuracy_game_mean","accuracy_move_mean","acpl_game_mean","acpl_move_mean"}
    if not required.issubset(df.columns):
        return  # silently skip if refined columns absent
    # Accuracy comparison
    acc_long = df.melt(id_vars=['strategy'], value_vars=['accuracy_game_mean','accuracy_move_mean'],
                       var_name='metric', value_name='accuracy')
    acc_long['metric'] = acc_long['metric'].map({'accuracy_game_mean':'Game-weighted','accuracy_move_mean':'Move-weighted'})
    acpl_long = df.melt(id_vars=['strategy'], value_vars=['acpl_game_mean','acpl_move_mean'],
                        var_name='metric', value_name='acpl')
    acpl_long['metric'] = acpl_long['metric'].map({'acpl_game_mean':'Game-weighted','acpl_move_mean':'Move-weighted'})
    order = df.sort_values('accuracy_move_mean', ascending=False)['strategy']
    fig, axes = plt.subplots(1,2, figsize=(12,4.2), sharey=False)
    sns.barplot(data=acc_long, x='strategy', y='accuracy', hue='metric', order=order, ax=axes[0])
    axes[0].set_ylabel('Accuracy%'); axes[0].set_xlabel('Strategy'); axes[0].set_title('Accuracy: Move vs Game weighting')
    axes[0].tick_params(axis='x', rotation=45)
    sns.barplot(data=acpl_long, x='strategy', y='acpl', hue='metric', order=order, ax=axes[1])
    axes[1].set_ylabel('ACPL'); axes[1].set_xlabel('Strategy'); axes[1].set_title('ACPL: Move vs Game weighting')
    axes[1].tick_params(axis='x', rotation=45)
    # Merge legends into a single one if duplicates
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[1].get_legend():
        axes[1].get_legend().remove()
    axes[0].legend(handles, labels, frameon=False)
    fig.tight_layout(); _export(fig, out / 'move_vs_game_comparison'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollup-acc', help='Path to rollup_acc.csv for accuracy & error plots')
    ap.add_argument('--sweep-dir', help='Directory containing csv_* parameter sweep files')
    ap.add_argument('--pattern', default='csv_UCB_ucb_c-*.csv', help='Glob-like substring to filter sweep files')
    ap.add_argument('--out-dir', required=True, help='Output directory for figures')
    ap.add_argument('--all', action='store_true', help='Generate all standard plots')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Rollup-based plots
    if args.rollup_acc:
        roll = _safe_read_csv(Path(args.rollup_acc))
        if roll is not None and not roll.empty:
            plot_accuracy_bar(roll, out)
            plot_error_stack(roll, out)
            plot_accuracy_vs_acpl(roll, out)
            plot_move_vs_game_comparison(roll, out)
        else:
            print('[warn] Rollup accuracy CSV empty or unreadable.')

    # Parameter sweeps
    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
        files = [p for p in sweep_dir.glob('*.csv') if args.pattern in p.name or re.search(args.pattern.replace('*','.*'), p.name)]
        # Heuristics: detect key in pattern
        if 'ucb_c' in args.pattern:
            plot_param_sweep(files, 'ucb_c', out)
        if 'ab_depth' in args.pattern:
            plot_ab_depth(files, out)
        if 'c_puct' in args.pattern:
            plot_param_sweep([f for f in files if 'c_puct' in f.name], 'c_puct', out)
        if 'grave_K' in args.pattern:
            plot_param_sweep([f for f in files if 'grave_K' in f.name], 'grave_K', out)

    # Convenience: generate a standard batch
    if args.all and args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
        # AB depth
        plot_ab_depth(list(sweep_dir.glob('csv_AB_ab_depth-*.csv')), out)
        # UCB c sweep
        plot_param_sweep(list(sweep_dir.glob('csv_UCB_ucb_c-*.csv')), 'ucb_c', out)
        # PUTC c_puct sweeps
        plot_param_sweep([p for p in sweep_dir.glob('csv_PUTC_*c_puct-*') if 'ab_leaf_depth-2' in p.name], 'c_puct', out)
        # GRAVE K sweeps
        plot_param_sweep([p for p in sweep_dir.glob('csv_PUTC_GRAVE_*grave_K-*') if 'ab_leaf_depth-2' in p.name], 'grave_K', out)

    print(f"Figures written to: {out}")


if __name__ == '__main__':
    main()
