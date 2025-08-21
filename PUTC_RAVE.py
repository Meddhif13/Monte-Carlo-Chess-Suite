import copy
import random
import math
import multiprocessing as mp
import chess

from PUTC import play, _rollout_move, _black_prob_eval

BLACK = False
WHITE = True


def score(board):
    res = board.result(claim_draw=True)
    if res == "0-1":
        return 1
    if res == "1-0":
        return 0
    return 0.5


def add(board, h, Table):
    moves = [i for i in board.legal_moves]
    nplayouts = [0.0 for _ in moves]
    nwins = [0.0 for _ in moves]
    priors = [1.0 / len(moves) for _ in moves] if moves else []
    amaf = {}  # move_uci -> (plays, wins)
    Table[h] = [1, nplayouts, nwins, priors, amaf]


def look(h, Table):
    try:
        return Table[h]
    except Exception:
        return None


def playout(b, h, piece_hash, hashTurn, max_steps: int = 200):
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
            return _black_prob_eval(b), h, trail


def PUCT_RAVE(board, h, piece_hash, hashTurn, Table, c_puct=0.8, beta_const: float = 300.0):
    if board.is_game_over():
        return score(board), h, []

    t = look(h, Table)
    if t is not None:
        moves = [i for i in board.legal_moves]
        if len(moves) != len(t[1]) or len(t[2]) != len(moves) or len(t[3]) != len(moves):
            t[1] = [0.0 for _ in moves]
            t[2] = [0.0 for _ in moves]
            t[3] = [1.0 / len(moves) for _ in moves] if moves else []
        bestValue = -1e9
        best = 0
        N_parent = t[0]
        amaf = t[4]
        for i, mv in enumerate(moves):
            n_i = t[1][i]
            q = (t[2][i] / n_i) if n_i > 0 else 0.5
            if board.turn == WHITE:
                q = 1 - q
            # AMAF term
            plays, wins = amaf.get(mv.uci(), (0.0, 0.0))
            q_amaf = (wins / plays) if plays > 0 else 0.5
            if board.turn == WHITE:
                q_amaf = 1 - q_amaf
            beta = beta_const / (beta_const + n_i)
            q_mix = (1 - beta) * q + beta * q_amaf
            # PUCT with priors
            P = t[3][i]
            U = c_puct * P * math.sqrt(N_parent) / (1 + n_i)
            val = q_mix + U
            if val > bestValue:
                bestValue = val
                best = i

        chosen = moves[best]
        h = play(board, h, chosen, piece_hash, hashTurn)
        res, h, suffix = PUCT_RAVE(board, h, piece_hash, hashTurn, Table, c_puct=c_puct, beta_const=beta_const)
        # backprop
        t[0] += 1
        t[1][best] += 1
        t[2][best] += res
        for uci in suffix:
            p, w = amaf.get(uci, (0.0, 0.0))
            amaf[uci] = (p + 1.0, w + res)
        return res, h, [chosen.uci()] + suffix
    else:
        add(board, h, Table)
        res, h, tail = playout(board, h, piece_hash, hashTurn)
        return res, h, tail


def _puct_rave_worker(args):
    fen, h, piece_hash, hashTurn, nb_playout, c_puct, beta_const = args
    board = chess.Board(fen)
    Table = {}
    for _ in range(nb_playout):
        b1 = copy.deepcopy(board)
        h1 = h
        PUCT_RAVE(b1, h1, piece_hash, hashTurn, Table, c_puct=c_puct, beta_const=beta_const)
    t = look(h, Table)
    moves = [i for i in board.legal_moves]
    if t is None:
        return [0.0] * len(moves)
    return list(t[1])


def BestMovePUTC_RAVE(board, h, piece_hash, hashTurn, nb_playout, processes: int = 1, c_puct: float = 0.8, beta_const: float = 300.0, time_ms: int | None = None):
    moves = [i for i in board.legal_moves]
    if not moves:
        return None
    if processes is None or processes <= 1 or (time_ms is not None and time_ms > 0):
        Table = {}
        if time_ms is not None and time_ms > 0:
            import time as _t
            start = _t.time()
            while (_t.time() - start) * 1000.0 < time_ms:
                b1 = copy.deepcopy(board)
                h1 = h
                PUCT_RAVE(b1, h1, piece_hash, hashTurn, Table, c_puct=c_puct, beta_const=beta_const)
        else:
            for _ in range(nb_playout):
                b1 = copy.deepcopy(board)
                h1 = h
                PUCT_RAVE(b1, h1, piece_hash, hashTurn, Table, c_puct=c_puct, beta_const=beta_const)
        t = look(h, Table)
        counts = t[1] if t is not None else [0.0] * len(moves)
    else:
        share = [nb_playout // processes] * processes
        for i in range(nb_playout % processes):
            share[i] += 1
        args = [(board.fen(), h, piece_hash, hashTurn, s, c_puct, beta_const) for s in share]
        with mp.Pool(processes=processes) as pool:
            parts = pool.map(_puct_rave_worker, args)
        counts = [0.0] * len(moves)
        for arr in parts:
            for i, v in enumerate(arr):
                counts[i] += v
    best_idx = 0
    best_val = counts[0]
    for i in range(1, len(moves)):
        if counts[i] > best_val:
            best_val = counts[i]
            best_idx = i
    return moves[best_idx]
