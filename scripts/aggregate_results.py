import csv
import math
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Simple CSV aggregator for tournament logs produced by tournament.py
# Expected CSV headers include: timestamp, white, black, result, playouts, seed, c_puct, ucb_c, ab_leaf_depth, use_eval_leaf, use_eval_priors, rollout_steps, grave_K, eval_adjudicate, eval_threshold_cp, random_openings, moves, game_seconds, total_playouts, nps_playouts, nodes, depth, elapsed

RESULT_MAP = {
    '1-0': 1.0,
    '0-1': 0.0,
    '1/2-1/2': 0.5,
    '0.5-0.5': 0.5,
    'D': 0.5,
}


def read_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, newline='', encoding='utf-8') as f:
        r0 = csv.reader(f)
        first = next(r0, None)
        f.seek(0)
        fieldnames = None
        has_header = False
        if first:
            lower_tokens = [t.strip().lower() for t in first]
            header_keys = {"white", "black", "result", "playouts"}
            if any(tok in header_keys for tok in lower_tokens):
                has_header = True
        if not has_header:
            # Determine schema by column count in first row
            ncols = len(first) if first else 0
            if ncols == 9:
                # Very old schema: timestamp, white, black, playouts, seed, max_steps, adjudicate, processes, ucb_c, result
                fieldnames = [
                    "timestamp", "white", "black", "playouts", "seed", "max_steps", "adjudicate",
                    "processes", "ucb_c", "result"
                ]
            elif ncols == 10:
                fieldnames = [
                    "timestamp", "white", "black", "playouts", "seed", "max_steps", "adjudicate",
                    "material_threshold", "processes", "result"
                ]
            else:
                # Default to current extended schema (latest)
                fieldnames = [
                    "timestamp", "white", "black", "playouts", "seed", "max_steps", "adjudicate",
                    "material_threshold", "processes", "ucb_c", "c_puct", "use_eval_priors", "rollout_steps",
                    "use_eval_leaf", "ab_leaf_depth", "ab_depth_white", "ab_depth_black", "grave_K",
                    "eval_adjudicate", "eval_threshold_cp", "random_openings",
                    "moves", "game_seconds", "total_playouts", "nps_playouts",
                    "nodes", "depth", "elapsed", "result"
                ]
        reader = csv.DictReader(f, fieldnames=(None if has_header else fieldnames))
        # If header exists, skip it automatically; DictReader handles that.
        for r in reader:
            # If we injected fieldnames, the first line is data; if header existed, DictReader used it.
            rows.append(r)
    # Drop rows that look malformed (e.g., result missing)
    rows = [r for r in rows if r.get('result')]
    return rows


def key_for_group(row: Dict[str, str]) -> Tuple:
    # Group by matchup + key params
    return (
        row.get('white','').strip(),
        row.get('black','').strip(),
        row.get('playouts','').strip(),
        row.get('seed','').strip(),
        row.get('c_puct','').strip(),
        row.get('ucb_c','').strip(),
        row.get('ab_leaf_depth','').strip(),
        row.get('use_eval_leaf','').strip(),
        row.get('use_eval_priors','').strip(),
        row.get('rollout_steps','').strip(),
        row.get('grave_K','').strip(),
        row.get('eval_adjudicate','').strip(),
        row.get('eval_threshold_cp','').strip(),
        row.get('random_openings','').strip(),
    )


def result_to_score(result: str, white_is_subject: bool = True) -> float:
    r = RESULT_MAP.get(result, None)
    if r is None:
        # Try to parse e.g. '1-0', '0-1', '1/2-1/2'
        if result in ('1-0', '0-1'):
            r = 1.0 if result == '1-0' else 0.0
        elif result in ('1/2-1/2', '0.5-0.5'):
            r = 0.5
        else:
            return float('nan')
    return r if white_is_subject else (1.0 - r)


def elo_from_score(p: float) -> float:
    # Convert expected score to Elo difference (subject - opponent)
    if p <= 0.0:
        return -float('inf')
    if p >= 1.0:
        return float('inf')
    return -400.0 * math.log10(1.0 / p - 1.0)


def agg(csv_path: str, subject: str) -> None:
    rows = read_rows(csv_path)
    groups: Dict[Tuple, List[Dict[str,str]]] = defaultdict(list)
    for r in rows:
        groups[key_for_group(r)].append(r)

    print(f"Loaded {len(rows)} rows; {len(groups)} groups")

    for key, items in groups.items():
        white, black, playouts, seed, c_puct, ucb_c, ab_leaf_depth, use_eval_leaf, use_eval_priors, rollout_steps, grave_K, eval_adj, eval_thr, rand_open = key
        # Subject perspective scores
        scores: List[float] = []
        times: List[float] = []
        nodes_list: List[float] = []
        nps_playouts_list: List[float] = []
        who = None
        if subject == white:
            who = 'white'
        elif subject == black:
            who = 'black'
        else:
            continue
        for it in items:
            s = result_to_score(it.get('result',''), white_is_subject=(who=='white'))
            if not math.isnan(s):
                scores.append(s)
            # Prefer game_seconds if available, else fallback to elapsed
            try:
                t = it.get('game_seconds', '')
                if t == '' or t is None:
                    t = it.get('elapsed','')
                times.append(float(t or 0.0))
            except Exception:
                pass
            try:
                nodes_list.append(float(it.get('nodes','') or 0.0))
            except Exception:
                pass
            try:
                nps_playouts_list.append(float(it.get('nps_playouts','') or 0.0))
            except Exception:
                pass
        if not scores:
            continue
        n = len(scores)
        mean = sum(scores)/n
        # 95% CI for Bernoulli mean via normal approx
        se = math.sqrt(max(mean*(1-mean), 1e-9)/max(n,1))
        ci_low = max(0.0, mean - 1.96*se)
        ci_high = min(1.0, mean + 1.96*se)
        elo = elo_from_score(mean)
        t_avg = sum(times)/len(times) if times else float('nan')
        nodes_avg = sum(nodes_list)/len(nodes_list) if nodes_list else float('nan')
        nps = (sum(nodes_list)/sum(times)) if times and nodes_list and sum(times)>0 else float('nan')
        nps_playouts_avg = sum(nps_playouts_list)/len(nps_playouts_list) if nps_playouts_list else float('nan')
        print("\nGroup:")
        print(f"  matchup: {white} vs {black}")
        print(f"  params: playouts={playouts}, c_puct={c_puct}, ucb_c={ucb_c}, ab_leaf_depth={ab_leaf_depth}, eval_leaf={use_eval_leaf}, eval_priors={use_eval_priors}, rollout_steps={rollout_steps}, grave_K={grave_K}, eval_adj={eval_adj}, eval_thr={eval_thr}, rand_open={rand_open}, seed={seed}")
        print(f"  n={n}, win%={mean*100:.1f} [{ci_low*100:.1f}, {ci_high*100:.1f}] as {who}, elo≈{elo:.1f}")
        if not math.isnan(t_avg):
            extra = f", nps≈{nps:.0f}" if not math.isnan(nps) else ""
            extra2 = f", nps_playouts≈{nps_playouts_avg:.0f}" if not math.isnan(nps_playouts_avg) else ""
            print(f"  time avg={t_avg:.2f}s, nodes avg={nodes_avg:.0f}{extra}{extra2}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='logs/experiments.csv')
    p.add_argument('--subject', default='PUTC_GRAVE', help='Name of the strategy to view from its perspective')
    args = p.parse_args()
    agg(args.csv, args.subject)
