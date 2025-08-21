import copy
import math
import multiprocessing as mp
import chess

from PUTC import play, _rollout_move, _black_prob_eval
from eval import evaluate, cp_to_black_prob
from ab import negamax as _ab_negamax

BLACK = False
WHITE = True


def score(board: chess.Board) -> float:
    res = board.result(claim_draw=True)
    if res == "0-1":
        return 1.0
    if res == "1-0":
        return 0.0
    return 0.5


class Node:
    __slots__ = ("N", "nplayouts", "nwins", "priors", "q_init", "moves_uci")
    def __init__(self, moves_len: int):
        self.N = 1.0
        self.nplayouts = [0.0 for _ in range(moves_len)]
        self.nwins = [0.0 for _ in range(moves_len)]
        self.priors = [1.0/moves_len for _ in range(moves_len)] if moves_len > 0 else []
        self.q_init = [0.5 for _ in range(moves_len)]
        self.moves_uci = []


def look(h, Table):
    try:
        return Table[h]
    except Exception:
        return None


def add(board: chess.Board, h, Table, priors, q_init, moves):
    t = Node(len(priors))
    t.priors = priors
    t.q_init = q_init
    t.moves_uci = [mv.uci() for mv in moves]
    Table[h] = t


def _compute_priors(board: chess.Board, moves, use_eval_priors: bool = False):
    if use_eval_priors:
        # Policy prior should reflect side-to-move preferences at the parent node
        vals = []
        side_to_move = board.turn  # True=White, False=Black at parent
        for mv in moves:
            board.push(mv)
            try:
                cp = evaluate(board)  # positive favors White
                p_black = cp_to_black_prob(cp)
                # Convert to "good for side to move": White prefers low p_black, Black prefers high p_black
                prior = (1.0 - p_black) if side_to_move == WHITE else p_black
                vals.append(prior)
            finally:
                board.pop()
        s = sum(vals)
        if s <= 1e-9:
            n = len(moves)
            return [1.0/n for _ in range(n)] if n else []
        return [v/s for v in vals]
    # Simple structural priors: promotions>captures>checks>center
    scores = []
    for mv in moves:
        s = 0.0
        if mv.promotion:
            s += 5.0
        if board.is_capture(mv):
            s += 3.0
        if board.gives_check(mv):
            s += 1.5
        to = chess.square_name(mv.to_square)
        if to[0] in {"d","e"}:
            s += 0.5
        if to[1] in {"4","5"}:
            s += 0.5
        scores.append(s)
    total = sum(scores)
    if total <= 1e-9:
        n = len(moves)
        return [1.0/n for _ in range(n)] if n else []
    return [x/total for x in scores]


def _leaf_value(board: chess.Board, use_eval_leaf: bool, ab_leaf_depth: int) -> float:
    if ab_leaf_depth and ab_leaf_depth > 0:
        color = 1 if board.turn == chess.WHITE else -1
        cp_signed = _ab_negamax(board, ab_leaf_depth, -10**9, 10**9, color)
        cp_white = cp_signed * color
        return cp_to_black_prob(int(cp_white))
    if use_eval_leaf:
        cp = evaluate(board)
        return cp_to_black_prob(cp)
    return _black_prob_eval(board)


def playout(b: chess.Board, h, piece_hash, hashTurn, max_steps: int, use_eval_leaf: bool, ab_leaf_depth: int):
    steps = 0
    trail = []
    while True:
        if b.is_game_over():
            return score(b), h, trail
        mv = _rollout_move(b)
        trail.append(mv.uci())
        h = play(b, h, mv, piece_hash, hashTurn)
        steps += 1
        if steps >= max_steps:
            return _leaf_value(b, use_eval_leaf, ab_leaf_depth), h, trail


def PUCT_GRAVE(board: chess.Board, h, piece_hash, hashTurn, Table, move_stats: dict, c_puct=0.8, use_eval_priors: bool = False, rollout_steps: int = 200, use_eval_leaf: bool = False, ab_leaf_depth: int = 0, K: float = 100.0):
    if board.is_game_over():
        return score(board), h, []

    t = look(h, Table)
    moves = [m for m in board.legal_moves]
    if t is not None:
        # Resync: if move set changed, rebuild arrays but preserve stats for matching moves by UCI
        if len(moves) != len(t.nplayouts) or len(t.nwins) != len(moves) or len(t.priors) != len(moves) or len(getattr(t, 'q_init', [])) != len(moves):
            old_map = {uci: idx for idx, uci in enumerate(getattr(t, 'moves_uci', []) or [])}
            new_len = len(moves)
            new_nplayouts = [0.0 for _ in range(new_len)]
            new_nwins = [0.0 for _ in range(new_len)]
            new_priors = _compute_priors(board, moves, use_eval_priors)
            # Recompute q_init from leaf eval for each child (fresh, parent perspective)
            side_to_move = board.turn
            new_q_init = []
            for i, mv in enumerate(moves):
                # carry over stats if same move existed before
                u = mv.uci()
                if u in old_map:
                    j = old_map[u]
                    if j < len(t.nplayouts):
                        new_nplayouts[i] = t.nplayouts[j]
                    if j < len(t.nwins):
                        new_nwins[i] = t.nwins[j]
                board.push(mv)
                try:
                    if ab_leaf_depth and ab_leaf_depth > 0:
                        color = 1 if board.turn == chess.WHITE else -1
                        cp_signed = _ab_negamax(board, ab_leaf_depth, -10**9, 10**9, color)
                        cp_white = cp_signed * color
                        pb = cp_to_black_prob(int(cp_white))
                    else:
                        cp = evaluate(board)
                        pb = cp_to_black_prob(cp)
                    q_child = (1.0 - pb) if side_to_move == WHITE else pb
                    new_q_init.append(q_child)
                finally:
                    board.pop()
            t.nplayouts = new_nplayouts
            t.nwins = new_nwins
            t.priors = new_priors
            t.q_init = new_q_init
            t.moves_uci = [mv.uci() for mv in moves]
        bestValue = -1e9
        best = 0
        N_parent = t.N
        for i, mv in enumerate(moves):
            n_i = t.nplayouts[i]
            q_i = (t.nwins[i]/n_i) if n_i > 0 else t.q_init[i]
            if board.turn == WHITE:
                q_i = 1 - q_i
            # GRAVE value for move
            plays, wins = move_stats.get(mv.uci(), (0.0, 0.0))
            q_g = (wins/plays) if plays > 0 else 0.5
            if board.turn == WHITE:
                q_g = 1 - q_g
            beta = K / (K + n_i)
            q_mix = (1 - beta) * q_i + beta * q_g
            P = t.priors[i]
            # Progressive bias: small extra term from priors to break ties quicker
            U = c_puct * P * math.sqrt(max(1.0, N_parent)) / (1 + n_i) + 0.05 * P
            val = q_mix + U
            if val > bestValue:
                bestValue = val
                best = i
        chosen = moves[best]
        h = play(board, h, chosen, piece_hash, hashTurn)
        res, h, suffix = PUCT_GRAVE(board, h, piece_hash, hashTurn, Table, move_stats, c_puct, use_eval_priors, rollout_steps, use_eval_leaf, ab_leaf_depth, K)
        # backprop local
        t.N += 1
        t.nplayouts[best] += 1
        t.nwins[best] += res
        # backprop global move stats (GRAVE/AMAF-style)
        for uci in suffix:
            p, w = move_stats.get(uci, (0.0, 0.0))
            move_stats[uci] = (p + 1.0, w + res)
        return res, h, [chosen.uci()] + suffix
    else:
        priors = _compute_priors(board, moves, use_eval_priors)
        # Initialize Q with leaf eval for each child (parent perspective)
        side_to_move = board.turn
        q_init = []
        for mv in moves:
            board.push(mv)
            try:
                if ab_leaf_depth and ab_leaf_depth > 0:
                    color = 1 if board.turn == chess.WHITE else -1
                    cp_signed = _ab_negamax(board, ab_leaf_depth, -10**9, 10**9, color)
                    cp_white = cp_signed * color
                    pb = cp_to_black_prob(int(cp_white))
                else:
                    cp = evaluate(board)
                    pb = cp_to_black_prob(cp)
                # Convert to current player's success prob
                q_child = (1.0 - pb) if side_to_move == WHITE else pb
                q_init.append(q_child)
            finally:
                board.pop()
        add(board, h, Table, priors, q_init, moves)
        if rollout_steps is not None and rollout_steps <= 0:
            return _leaf_value(board, use_eval_leaf, ab_leaf_depth), h, []
        res, h, tail = playout(board, h, piece_hash, hashTurn, rollout_steps if rollout_steps is not None else 200, use_eval_leaf, ab_leaf_depth)
        # also update global move stats on playout tail
        for uci in tail:
            p, w = move_stats.get(uci, (0.0, 0.0))
            move_stats[uci] = (p + 1.0, w + res)
        return res, h, tail


def _grave_worker(args):
    fen, h, piece_hash, hashTurn, nb_playout, c_puct, use_eval_priors, rollout_steps, use_eval_leaf, ab_leaf_depth, K = args
    board = chess.Board(fen)
    Table = {}
    move_stats = {}
    for _ in range(nb_playout):
        b1 = copy.deepcopy(board)
        h1 = h
        PUCT_GRAVE(b1, h1, piece_hash, hashTurn, Table, move_stats, c_puct, use_eval_priors, rollout_steps, use_eval_leaf, ab_leaf_depth, K)
    t = look(h, Table)
    moves = [i for i in board.legal_moves]
    if t is None:
        return [0.0] * len(moves)
    return list(t.nplayouts)


def BestMovePUTC_GRAVE(board, h, piece_hash, hashTurn, nb_playout, processes: int = 1, c_puct: float = 0.8, use_eval_priors: bool = False, rollout_steps: int = 200, use_eval_leaf: bool = False, ab_leaf_depth: int = 0, K: float = 100.0, time_ms: int | None = None):
    moves = [i for i in board.legal_moves]
    if not moves:
        return None
    if processes is None or processes <= 1 or (time_ms is not None and time_ms > 0):
        Table = {}
        move_stats = {}
        if time_ms is not None and time_ms > 0:
            import time as _t
            start = _t.time()
            while (_t.time() - start) * 1000.0 < time_ms:
                b1 = copy.deepcopy(board)
                h1 = h
                PUCT_GRAVE(b1, h1, piece_hash, hashTurn, Table, move_stats, c_puct, use_eval_priors, rollout_steps, use_eval_leaf, ab_leaf_depth, K)
        else:
            for _ in range(nb_playout):
                b1 = copy.deepcopy(board)
                h1 = h
                PUCT_GRAVE(b1, h1, piece_hash, hashTurn, Table, move_stats, c_puct, use_eval_priors, rollout_steps, use_eval_leaf, ab_leaf_depth, K)
        t = look(h, Table)
        counts = t.nplayouts if t is not None else [0.0] * len(moves)
    else:
        share = [nb_playout // processes] * processes
        for i in range(nb_playout % processes):
            share[i] += 1
        args = [(board.fen(), h, piece_hash, hashTurn, s, c_puct, use_eval_priors, rollout_steps, use_eval_leaf, ab_leaf_depth, K) for s in share]
        with mp.Pool(processes=processes) as pool:
            parts = pool.map(_grave_worker, args)
        counts = [0.0] * len(moves)
        for arr in parts:
            for i, v in enumerate(arr):
                counts[i] += v
    best_idx = max(range(len(moves)), key=lambda i: counts[i])
    return moves[best_idx]
