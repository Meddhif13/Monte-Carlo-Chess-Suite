import copy
import random
import math
import multiprocessing as mp
import chess
from eval import evaluate, cp_to_black_prob
from ab import negamax as _ab_negamax

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


def get_color_code(col):
    if (col == None):
        code = 0
    elif (col):
        code = 1
    else:
        code = 2
    return code


def score(board):
    # return 1 if black wins, 0 if white wins, 0.5 for draw (consistent with UCT_IA)
    res = board.result(claim_draw=True)
    if res == "0-1":
        return 1
    if res == "1-0":
        return 0
    return 0.5


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


def playout(b, h, piece_hash, hashTurn, max_steps: int = 200, use_eval_leaf: bool = False, ab_leaf_depth: int = 0):
    steps = 0
    while True:
        moves = [i for i in b.legal_moves]
        if b.is_game_over():
            return score(b), h
        mv = _rollout_move(b)
        # update h and push move using the same hashing helper as repository
        h = play(b, h, mv, piece_hash, hashTurn)
        steps += 1
        if steps >= max_steps:
            # Prefer AB leaf if requested
            if ab_leaf_depth and ab_leaf_depth > 0:
                color = 1 if b.turn == chess.WHITE else -1
                cp_signed = _ab_negamax(b, ab_leaf_depth, -10**9, 10**9, color)
                # Convert to white-positive cp
                cp_white = cp_signed * color
                return cp_to_black_prob(int(cp_white)), h
            if use_eval_leaf:
                cp = evaluate(b)
                return cp_to_black_prob(cp), h
            # Use a stronger heuristic eval at cutoff
            return _black_prob_eval(b), h


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

    h = h ^ piece_hash[(piece - 1) + 6*indice_color][x1][y1]
    h = h ^ piece_hash[(piece - 1) + 6*indice_color][x2][y2]
    h = h ^ hashTurn
    if(to_col == 1):
        h = h ^ piece_hash[(to_piece - 1)][x2][y2]
    elif(to_col == 2):
        h = h ^ piece_hash[(to_piece - 1) + 6][x2][y2]

    return h


def play(board, h, best_move, piece_hash, hashTurn):
    piece = board.piece_type_at(best_move.from_square)
    h = update_hashcode_zobriest(piece, board, h, hashTurn, piece_hash, best_move)
    board.push(best_move)
    return h


def add(board, h, Table, use_eval_priors: bool = False, init_q_from_ab: bool = False, ab_leaf_depth: int = 0):
    moves = [i for i in board.legal_moves]
    nplayouts = [0.0 for _ in range(len(moves))]
    nwins = [0.0 for _ in range(len(moves))]
    priors = _compute_priors(board, moves, use_eval_priors=use_eval_priors)
    # Optionally initialize Q with AB leaf evaluation to accelerate convergence
    if init_q_from_ab and ab_leaf_depth and ab_leaf_depth > 0:
        for i, mv in enumerate(moves):
            board.push(mv)
            try:
                color = 1 if board.turn == chess.WHITE else -1
                cp_signed = _ab_negamax(board, ab_leaf_depth, -10**9, 10**9, color)
                cp_white = cp_signed * color
                q_black = cp_to_black_prob(int(cp_white))
                # seed as 1 virtual visit
                nplayouts[i] = 1.0
                nwins[i] = q_black
            finally:
                board.pop()
    Table[h] = [1, nplayouts, nwins, priors]


def look(h, Table):
    try:
        return Table[h]
    except Exception:
        return None


def PUCT(board, h, piece_hash, hashTurn, Table, c_puct=0.8, use_eval_priors: bool = False, rollout_steps: int = 200, use_eval_leaf: bool = False, ab_leaf_depth: int = 0):
    if board.is_game_over():
        return score(board), h

    t = look(h, Table)
    if t is not None:
        moves = [i for i in board.legal_moves]
        # Resync arrays if move count differs to avoid IndexError on transpositions
        if len(moves) != len(t[1]) or len(t[2]) != len(moves) or len(t[3]) != len(moves):
            t[1] = [0.0 for _ in range(len(moves))]
            t[2] = [0.0 for _ in range(len(moves))]
            t[3] = _compute_priors(board, moves, use_eval_priors=use_eval_priors)
        bestValue = -1e9
        best = 0
        N_parent = t[0]
        for i in range(len(moves)):
            if t[1][i] > 0:
                Q = t[2][i] / t[1][i]
                if board.turn == WHITE:
                    Q = 1 - Q
            else:
                Q = 0.5  # unknown

            P = t[3][i]
            U = c_puct * P * math.sqrt(N_parent) / (1 + t[1][i])
            val = Q + U
            if val > bestValue:
                bestValue = val
                best = i

        res = 0.0
        if len(moves) > 0:
            h = play(board, h, moves[best], piece_hash, hashTurn)
            res, h = PUCT(board, h, piece_hash, hashTurn, Table, c_puct, use_eval_priors=use_eval_priors, rollout_steps=rollout_steps, use_eval_leaf=use_eval_leaf, ab_leaf_depth=ab_leaf_depth)
            t[0] += 1
            t[1][best] += 1
            t[2][best] += res
        return res, h
    else:
        add(board, h, Table, use_eval_priors=use_eval_priors, init_q_from_ab=True, ab_leaf_depth=ab_leaf_depth)
        # Direct leaf eval if rollout_steps <= 0
        if rollout_steps is not None and rollout_steps <= 0:
            if ab_leaf_depth and ab_leaf_depth > 0:
                color = 1 if board.turn == chess.WHITE else -1
                cp_signed = _ab_negamax(board, ab_leaf_depth, -10**9, 10**9, color)
                cp_white = cp_signed * color
                return cp_to_black_prob(int(cp_white)), h
            if use_eval_leaf:
                cp = evaluate(board)
                return cp_to_black_prob(cp), h
            return _black_prob_eval(board), h
        score_playout, h = playout(board, h, piece_hash, hashTurn, max_steps=rollout_steps if rollout_steps is not None else 200, use_eval_leaf=use_eval_leaf, ab_leaf_depth=ab_leaf_depth)
        return score_playout, h

def _puct_worker(args):
    fen, h, piece_hash, hashTurn, nb_playout, c_puct, use_eval_priors, rollout_steps, use_eval_leaf, ab_leaf_depth = args
    board = chess.Board(fen)
    Table = {}
    for _ in range(nb_playout):
        b1 = copy.deepcopy(board)
        h1 = h
        PUCT(b1, h1, piece_hash, hashTurn, Table, c_puct=c_puct, use_eval_priors=use_eval_priors, rollout_steps=rollout_steps, use_eval_leaf=use_eval_leaf, ab_leaf_depth=ab_leaf_depth)
    t = look(h, Table)
    moves = [i for i in board.legal_moves]
    if t is None:
        return [0.0] * len(moves)
    return list(t[1])


def BestMovePUTC(board, h, piece_hash, hashTurn, nb_playout, processes: int = 1, c_puct: float = 0.8, use_eval_priors: bool = False, rollout_steps: int = 200, use_eval_leaf: bool = False, ab_leaf_depth: int = 0, time_ms: int | None = None):
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
                PUCT(b1, h1, piece_hash, hashTurn, Table, c_puct=c_puct, use_eval_priors=use_eval_priors, rollout_steps=rollout_steps, use_eval_leaf=use_eval_leaf, ab_leaf_depth=ab_leaf_depth)
        else:
            for _ in range(nb_playout):
                b1 = copy.deepcopy(board)
                h1 = h
                PUCT(b1, h1, piece_hash, hashTurn, Table, c_puct=c_puct, use_eval_priors=use_eval_priors, rollout_steps=rollout_steps, use_eval_leaf=use_eval_leaf, ab_leaf_depth=ab_leaf_depth)
        t = look(h, Table)
        counts = t[1] if t is not None else [0.0] * len(moves)
    else:
        share = [nb_playout // processes] * processes
        for i in range(nb_playout % processes):
            share[i] += 1
        args = [(board.fen(), h, piece_hash, hashTurn, s, c_puct, use_eval_priors, rollout_steps, use_eval_leaf, ab_leaf_depth) for s in share]
        with mp.Pool(processes=processes) as pool:
            parts = pool.map(_puct_worker, args)
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


def _compute_priors(board, moves, use_eval_priors: bool = False):
    if use_eval_priors:
        # Evaluate child positions with heuristic and map to [0,1] from side-to-move perspective
        vals = []
        for mv in moves:
            board.push(mv)
            try:
                bp = _black_prob_eval(board)
                if board.turn == WHITE:  # after push, turn flipped; this bp is for current side (to move) which is opponent of parent
                    # Now it's opponent's turn; prior should reflect parent-to-move goodness
                    # Parent was opposite color, so invert
                    prior = 1.0 - bp
                else:
                    prior = bp
                vals.append(prior)
            finally:
                board.pop()
        s = sum(vals)
        if s <= 1e-9:
            n = len(moves)
            return [1.0 / n for _ in range(n)] if n else []
        return [v / s for v in vals]
    # Promotion > capture > check > center move > others
    center_files = {'d', 'e'}
    center_ranks = {'4', '5'}
    scores = []
    for mv in moves:
        s = 0.0
        if mv.promotion:
            s += 5.0
        if board.is_capture(mv):
            s += 3.0
        if board.gives_check(mv):
            s += 1.5
        to_sq = chess.square_name(mv.to_square)
        if to_sq[0] in center_files:
            s += 0.5
        if to_sq[1] in center_ranks:
            s += 0.5
        scores.append(s)
    total = sum(scores)
    if total <= 1e-9:
        n = len(moves)
        return [1.0 / n for _ in range(n)] if n else []
    return [x / total for x in scores]


# --- Heuristic evaluation used at rollout cutoff (mirrors UCT_IA style) ---
def _material_eval(board: chess.Board) -> float:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    total = 0.0
    for _, pc in board.piece_map().items():
        v = vals.get(pc.piece_type, 0)
        total += v if pc.color else -v
    return total


def _pst_eval(board: chess.Board) -> float:
    P_PST = [
        0, 0, 0, 0, 0, 0, 0, 0,
        0.1, 0.1, 0, -0.1, -0.1, 0, 0.1, 0.1,
        0.05, 0, 0.1, 0.15, 0.15, 0.1, 0, 0.05,
        0.05, 0.05, 0.1, 0.2, 0.2, 0.1, 0.05, 0.05,
        0.05, 0.05, 0.1, 0.2, 0.2, 0.1, 0.05, 0.05,
        0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05,
        0.2, 0.2, 0.2, 0.25, 0.25, 0.2, 0.2, 0.2,
        0, 0, 0, 0, 0, 0, 0, 0,
    ]
    N_PST = [
        -0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5,
        -0.4, -0.2, 0, 0, 0, 0, -0.2, -0.4,
        -0.3, 0, 0.1, 0.15, 0.15, 0.1, 0, -0.3,
        -0.3, 0.05, 0.15, 0.2, 0.2, 0.15, 0.05, -0.3,
        -0.3, 0, 0.15, 0.2, 0.2, 0.15, 0, -0.3,
        -0.3, 0.05, 0.1, 0.15, 0.15, 0.1, 0.05, -0.3,
        -0.4, -0.2, 0, 0.05, 0.05, 0, -0.2, -0.4,
        -0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5,
    ]
    B_PST = [
        -0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2,
        -0.1, 0, 0, 0, 0, 0, 0, -0.1,
        -0.1, 0, 0.05, 0.1, 0.1, 0.05, 0, -0.1,
        -0.1, 0.05, 0.1, 0.15, 0.15, 0.1, 0.05, -0.1,
        -0.1, 0, 0.1, 0.15, 0.15, 0.1, 0, -0.1,
        -0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, -0.1,
        -0.1, 0.05, 0, 0, 0, 0, 0.05, -0.1,
        -0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2,
    ]
    R_PST = [0]*64
    Q_PST = [0]*64
    K_PST = [
        0.2, 0.3, 0.1, 0, 0, 0.1, 0.3, 0.2,
        0.2, 0.2, 0, 0, 0, 0, 0.2, 0.2,
        0.1, 0, -0.1, -0.2, -0.2, -0.1, 0, 0.1,
        0, -0.1, -0.2, -0.3, -0.3, -0.2, -0.1, 0,
        0, -0.1, -0.2, -0.3, -0.3, -0.2, -0.1, 0,
        0.1, 0, -0.1, -0.2, -0.2, -0.1, 0, 0.1,
        0.2, 0.2, 0, 0, 0, 0, 0.2, 0.2,
        0.2, 0.3, 0.1, 0, 0, 0.1, 0.3, 0.2,
    ]
    def pst_val(piece_type: int, square: int, color: bool) -> float:
        idx = square if color else chess.square_mirror(square)
        if piece_type == chess.PAWN:
            return P_PST[idx]
        if piece_type == chess.KNIGHT:
            return N_PST[idx]
        if piece_type == chess.BISHOP:
            return B_PST[idx]
        if piece_type == chess.ROOK:
            return R_PST[idx]
        if piece_type == chess.QUEEN:
            return Q_PST[idx]
        if piece_type == chess.KING:
            return K_PST[idx]
        return 0.0
    s = 0.0
    for sq, pc in board.piece_map().items():
        s += pst_val(pc.piece_type, sq, pc.color) if pc.color else -pst_val(pc.piece_type, sq, pc.color)
    return s


def _mobility_eval(board: chess.Board) -> float:
    w = 0
    b = 0
    turn = board.turn
    try:
        board.turn = True
        w = board.legal_moves.count()
        board.turn = False
        b = board.legal_moves.count()
    finally:
        board.turn = turn
    return (w - b) / 30.0


def _king_safety_eval(board: chess.Board) -> float:
    bonus = 0.0
    for color in (True, False):
        kp = board.king(color)
        if kp is None:
            continue
        file = chess.square_file(kp)
        rank = chess.square_rank(kp)
        dirr = 1 if color else -1
        shield = 0
        for df in (-1, 0, 1):
            f = file + df
            r = rank + dirr
            if 0 <= f < 8 and 0 <= r < 8:
                sq = chess.square(f, r)
                pc = board.piece_at(sq)
                if pc and pc.piece_type == chess.PAWN and pc.color == color:
                    shield += 1
        bonus += (0.1 * shield) if color else -(0.1 * shield)
    return bonus


def _black_prob_eval(board: chess.Board) -> float:
    m = _material_eval(board)
    pst = _pst_eval(board)
    mob = _mobility_eval(board)
    ks = _king_safety_eval(board)
    white_score = m + pst + 0.5 * mob + 0.3 * ks
    try:
        from math import exp
        return 1.0 / (1.0 + exp(0.8 * white_score))
    except Exception:
        return 0.5
