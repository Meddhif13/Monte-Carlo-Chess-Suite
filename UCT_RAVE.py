import copy
import random
from math import sqrt, log
import multiprocessing as mp
import chess

BLACK = False
WHITE = True

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


def score(board: chess.Board) -> float:
    res = board.result(claim_draw=True)
    if res == "0-1":
        return 1.0
    if res == "1-0":
        return 0.0
    return 0.5


def get_color_code(col):
    if col is None:
        return 0
    return 1 if col else 2


def update_hashcode_zobriest(piece, board, h, hashTurn, piece_hash, move):
    to_col = board.color_at(move.to_square)
    to_col = get_color_code(to_col)
    to_piece = board.piece_type_at(move.to_square)

    from_uci = chess.square_name(move.from_square)
    x1 = d[from_uci[0]]
    y1 = d[from_uci[1]]
    to_uci = chess.square_name(move.to_square)
    x2 = d[to_uci[0]]
    y2 = d[to_uci[1]]

    indice_color = 0 if board.turn else 1

    h = h ^ piece_hash[(piece - 1) + 6 * indice_color][x1][y1]
    h = h ^ piece_hash[(piece - 1) + 6 * indice_color][x2][y2]
    h = h ^ hashTurn
    if to_col == 1:
        h = h ^ piece_hash[(to_piece - 1)][x2][y2]
    elif to_col == 2:
        h = h ^ piece_hash[(to_piece - 1) + 6][x2][y2]
    return h


def play(board, h, best_move, piece_hash, hashTurn):
    piece = board.piece_type_at(best_move.from_square)
    h = update_hashcode_zobriest(piece, board, h, hashTurn, piece_hash, best_move)
    board.push(best_move)
    return h


def _rollout_move(b: chess.Board) -> chess.Move:
    moves = [m for m in b.legal_moves]
    if not moves:
        raise ValueError("No legal moves in rollout")
    last = b.move_stack[-1] if len(b.move_stack) > 0 else None
    last_to = last.to_square if last is not None else None

    promotions, recaptures, captures, checks, quiets = [], [], [], [], []
    for mv in moves:
        if mv.promotion:
            promotions.append(mv)
            continue
        if b.is_capture(mv):
            if last_to is not None and mv.to_square == last_to:
                recaptures.append(mv)
            else:
                captures.append(mv)
            continue
        if b.gives_check(mv):
            checks.append(mv)
            continue
        quiets.append(mv)

    if promotions:
        return random.choice(promotions)
    if recaptures:
        return random.choice(recaptures)
    if captures:
        return random.choice(captures)
    if checks:
        return random.choice(checks)

    def quiet_score(mv: chess.Move) -> float:
        to = chess.square_name(mv.to_square)
        file, rank = to[0], to[1]
        s = 1.0
        if to in {"d4", "e4", "d5", "e5"}:
            s += 2.0
        if file in {"c", "d", "e", "f"} and rank in {"3", "4", "5", "6"}:
            s += 1.0
        piece = b.piece_type_at(mv.from_square)
        from_rank = chess.square_name(mv.from_square)[1]
        if piece in (chess.KNIGHT, chess.BISHOP) and from_rank in ("1", "8"):
            s += 0.5
        return s

    weights = [quiet_score(mv) for mv in quiets] if quiets else [1.0]
    r = random.random() * sum(weights)
    acc = 0.0
    for mv, w in zip(quiets, weights):
        acc += w
        if r <= acc:
            return mv
    return random.choice(moves)


def playout(b: chess.Board, h, piece_hash, hashTurn, max_steps: int = 200):
    steps = 0
    played_moves = []  # list of UCI strings in order
    while True:
        if b.is_game_over():
            return score(b), h, played_moves
        mv = _rollout_move(b)
        played_moves.append(mv.uci())
        h = play(b, h, mv, piece_hash, hashTurn)
        steps += 1
        if steps >= max_steps:
            return 0.5, h, played_moves


def add(board: chess.Board, h, Table):
    moves = [m for m in board.legal_moves]
    nplayouts = [0.0 for _ in moves]
    nwins = [0.0 for _ in moves]
    amaf = {}  # move_uci -> [plays, wins]
    Table[h] = [1, nplayouts, nwins, amaf]


def look(h, Table):
    try:
        return Table[h]
    except Exception:
        return None


def UCT_RAVE(board: chess.Board, h, piece_hash, hashTurn, Table, beta_const: float = 300.0):
    if board.is_game_over():
        return score(board), h, []

    t = look(h, Table)
    if t is not None:
        moves = [m for m in board.legal_moves]
        # ensure arrays length
        if len(moves) != len(t[1]) or len(t[2]) != len(moves):
            t[1] = [0.0 for _ in moves]
            t[2] = [0.0 for _ in moves]
        bestValue = -1e9
        best = 0
        N_parent = t[0]
        amaf = t[3]

        for i, mv in enumerate(moves):
            n_i = t[1][i]
            q = (t[2][i] / n_i) if n_i > 0 else 0.5
            if board.turn == WHITE:
                q = 1 - q
            # AMAF stat
            plays, wins = amaf.get(mv.uci(), (0.0, 0.0))
            q_amaf = (wins / plays) if plays > 0 else 0.5
            if board.turn == WHITE:
                q_amaf = 1 - q_amaf
            beta = beta_const / (beta_const + n_i)
            ucb = 0.4 * sqrt(log(max(1.0, N_parent)) / (1.0 + n_i))
            val = (1 - beta) * q + beta * q_amaf + ucb
            if val > bestValue:
                bestValue = val
                best = i

        chosen = moves[best]
        h = play(board, h, chosen, piece_hash, hashTurn)
        res, h, suffix_moves = UCT_RAVE(board, h, piece_hash, hashTurn, Table, beta_const)
        # backprop
        t[0] += 1
        t[1][best] += 1
        t[2][best] += res
        # update AMAF for moves seen later in the simulation
        for uci in suffix_moves:
            p, w = t[3].get(uci, (0.0, 0.0))
            t[3][uci] = (p + 1.0, w + res)
        return res, h, [chosen.uci()] + suffix_moves
    else:
        add(board, h, Table)
        res, h, tail = playout(board, h, piece_hash, hashTurn)
        return res, h, tail


def _uct_rave_worker(args):
    fen, h, piece_hash, hashTurn, nb_playout, beta_const = args
    board = chess.Board(fen)
    Table = {}
    for _ in range(nb_playout):
        b1 = copy.deepcopy(board)
        h1 = h
        UCT_RAVE(b1, h1, piece_hash, hashTurn, Table, beta_const=beta_const)
    t = look(h, Table)
    moves = [i for i in board.legal_moves]
    if t is None:
        return [0.0] * len(moves)
    return list(t[1])


def BestMoveUCT_RAVE(board, h, piece_hash, hashTurn, nb_playout, processes: int = 1, beta_const: float = 300.0, time_ms: int | None = None):
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
                UCT_RAVE(b1, h1, piece_hash, hashTurn, Table, beta_const=beta_const)
        else:
            for _ in range(nb_playout):
                b1 = copy.deepcopy(board)
                h1 = h
                UCT_RAVE(b1, h1, piece_hash, hashTurn, Table, beta_const=beta_const)
        t = look(h, Table)
        counts = t[1] if t is not None else [0.0] * len(moves)
    else:
        share = [nb_playout // processes] * processes
        for i in range(nb_playout % processes):
            share[i] += 1
        args = [(board.fen(), h, piece_hash, hashTurn, s, beta_const) for s in share]
        with mp.Pool(processes=processes) as pool:
            parts = pool.map(_uct_rave_worker, args)
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
