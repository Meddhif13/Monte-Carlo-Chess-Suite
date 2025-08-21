import copy
import random
import math
import time
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


def add(board, h, Table):
    """
    Ajoute un board et son hash dans la table de transposition.

    :param board:
    :param h:
    :param Table:
    :return:
    """
    nplayouts = [0.0 for x in range(len([i for i in board.legal_moves]))]  # propre au board
    nwins = [0.0 for x in range(len([i for i in board.legal_moves]))]

    Table[h] = [1, nplayouts, nwins]


def look(h, Table):
    """
    Cherche un board dans la table de transposition.
    :param h:
    :param Table:
    :return:
    """
    try:
        t = Table[h]
    except:
        t = None
    return t


def score(board):
    """
    Favoriser le noir

    :param board:
    :return:
    """
    res = board.result(claim_draw=True)
    if res == "0-1":
        return 1
    if res == "1-0":
        return 0
    return 0.5


def _material_eval(board):
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    total = 0
    for sq, pc in board.piece_map().items():
        v = vals.get(pc.piece_type, 0)
        total += v if pc.color else -v
    return total  # >0 favors white, <0 favors black


def _pst_eval(board) -> float:
    # Piece-Square Tables (simple, midgame-ish). Values in pawns.
    # Indexed by 0..63 with a-file at left and rank 1 at bottom; we'll mirror for black.
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
        v = pst_val(pc.piece_type, sq, pc.color)
        s += v if pc.color else -v
    return s


def _mobility_eval(board) -> float:
    # simple mobility: legal moves difference scaled
    w = 0
    b = 0
    turn = board.turn
    try:
        board.turn = chess.WHITE
        w = board.legal_moves.count()
        board.turn = chess.BLACK
        b = board.legal_moves.count()
    finally:
        board.turn = turn
    # normalize roughly to pawns
    return (w - b) / 30.0


def _king_safety_eval(board) -> float:
    # crude: bonus if king has at least one friendly pawn in front on adjacent files
    bonus = 0.0
    for color in (chess.WHITE, chess.BLACK):
        kp = board.king(color)
        if kp is None:
            continue
        file = chess.square_file(kp)
        rank = chess.square_rank(kp)
        dir = 1 if color == chess.WHITE else -1
        shield = 0
        for df in (-1, 0, 1):
            f = file + df
            r = rank + dir
            if 0 <= f < 8 and 0 <= r < 8:
                sq = chess.square(f, r)
                pc = board.piece_at(sq)
                if pc and pc.piece_type == chess.PAWN and pc.color == color:
                    shield += 1
        if color:
            bonus += 0.1 * shield
        else:
            bonus -= 0.1 * shield
    return bonus


def _black_prob_eval(board) -> float:
    # combine material + PST + mobility + king safety (white-positive scale)
    m = _material_eval(board)
    pst = _pst_eval(board)
    mob = _mobility_eval(board)
    ks = _king_safety_eval(board)
    white_score = m + pst + 0.5 * mob + 0.3 * ks
    # map to probability with logistic
    try:
        from math import exp
        return 1.0 / (1.0 + exp(0.8 * white_score))
    except Exception:
        return 0.5


def _rollout_move(b: chess.Board) -> chess.Move:
    """Heuristic rollout move selection.
    Priority: promotions > recaptures > captures > checks > quiets with center bias.
    """
    moves = [m for m in b.legal_moves]
    if not moves:
        raise ValueError("No legal moves in rollout")

    last = b.move_stack[-1] if len(b.move_stack) > 0 else None
    last_to = last.to_square if last is not None else None

    promotions = []
    recaptures = []
    captures = []
    checks = []
    quiets = []
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

    # Quiet move heuristic: soft preference for central control and development
    def quiet_score(mv: chess.Move) -> float:
        to = chess.square_name(mv.to_square)
        file = to[0]
        rank = to[1]
        score_q = 1.0
        if to in {"d4", "e4", "d5", "e5"}:
            score_q += 2.0
        if file in {"c", "d", "e", "f"} and rank in {"3", "4", "5", "6"}:
            score_q += 1.0
        # Knights/Bishops developing off back rank
        piece = b.piece_type_at(mv.from_square)
        from_rank = chess.square_name(mv.from_square)[1]
        if piece in (chess.KNIGHT, chess.BISHOP) and from_rank in ("1", "8"):
            score_q += 0.5
        return score_q

    weights = [quiet_score(mv) for mv in quiets] if quiets else [1.0]
    r = random.random() * sum(weights)
    acc = 0.0
    for mv, w in zip(quiets, weights):
        acc += w
        if r <= acc:
            return mv
    # Fallback
    return random.choice(moves)


def playout(b, h, piece_hash, hashTurn, max_steps: int = 200):
    """
    Joue une partie aléatoire (avec biais captures/échecs) et limite de profondeur.
    Retourne un score en [0,1] où 1 favorise noir (cohérent avec score()) si cutoff atteint.
    """
    steps = 0
    while True:
        if b.is_game_over():
            return score(b), h
        # Heuristic rollout move choice (promotions/recaptures/captures/checks/quiet heuristic)
        chosen = _rollout_move(b)
        h = play(b, h, chosen, piece_hash, hashTurn)
        steps += 1
        if steps >= max_steps:
            return _black_prob_eval(b), h


def get_color_code(col):
    """

    Détermine l'indice de la couleur (pour la table de hashage)
    :param col:
    :return:
    """
    if (col == None):
        code = 0
    elif (col):
        code = 1
    else:
        code = 2
    return code


def update_hashcode(piece, board, h, hashTable, hashTurn, move):
    """
    Update hashcode of a board.

    Need to call this function before using board.push(move).

    :param board:
    :param h:
    :param move:
    :return:
    """
    col = board.color_at(move.to_square)
    col = get_color_code(col)

    from_uci = chess.square_name(move.from_square)
    x1 = d[from_uci[0]]
    y1 = d[from_uci[1]]
    to_uci = chess.square_name(move.to_square)
    x2 = d[to_uci[0]]
    y2 = d[to_uci[1]]

    move_color = get_color_code(board.turn)

    if col != None:
        h = h ^ hashTable[col][x2][y2][piece-1]

    h = h ^ hashTable[move_color][x2][y2][piece-1]
    h = h ^ hashTable[move_color][x1][y1][piece-1]
    h = h ^ hashTurn

    return h

def update_hashcode_zobriest(piece, board, h, hashTurn, piece_hash, move):
    """
        Update hashcode of a board with Zobriest Hashing

        Need to call this function before using board.push(move).

        :param board:
        :param h:
        :param move:
        :return:
        """

    to_col = board.color_at(move.to_square)
    to_col = get_color_code(to_col)
    to_piece = board.piece_type_at(move.to_square)

    from_uci = chess.square_name(move.from_square)
    x1 = d[from_uci[0]]
    y1 = d[from_uci[1]]
    to_uci = chess.square_name(move.to_square)
    x2 = d[to_uci[0]]
    y2 = d[to_uci[1]]


    indice_color = 0 if board.turn else 1 #True = White

    h = h ^ piece_hash[(piece - 1) + 6*indice_color][x1][y1]
    h = h ^ piece_hash[(piece - 1) + 6*indice_color][x2][y2]
    h = h ^ hashTurn
    if(to_col == 1):
        h = h ^ piece_hash[(to_piece - 1)][x2][y2]
    elif(to_col == 2):
        h = h ^ piece_hash[(to_piece - 1) + 6][x2][y2]

    return h

def play(board, h, best_move, piece_hash, hashTurn):
    """

    Joue un move et update le hashcode du board.
    :param board:
    :param h:
    :param best_move:
    :param piece_hash:
    :return:
    """
    piece = board.piece_type_at(best_move.from_square)
    h = update_hashcode_zobriest(piece, board, h, hashTurn, piece_hash, best_move)
    board.push(best_move)
    return h


def UCT(board, h, piece_hash, hashTurn, Table):
    """
    IA de l'UCT

    :param board:
    :param h:
    :param piece_hash:
    :param Table:
    :return:
    """
    if board.is_game_over():
        return score(board), h

    t = look(h, Table)
    if t != None:  # Selection and expansion step
        bestValue = -1000000.0
        best = 0

        moves = [i for i in board.legal_moves]
        # Resync arrays if move count differs
        if len(moves) != len(t[1]) or len(t[2]) != len(moves):
            t[1] = [0.0 for _ in range(len(moves))]
            t[2] = [0.0 for _ in range(len(moves))]
        for i in range(0, len(moves)):

            val = 1000000.0
            if t[1][i] > 0:
                Q = t[2][i] / t[1][i]
                if board.turn == WHITE:
                    Q = 1 - Q
                val = Q + 0.4 * math.sqrt(math.log(t[0]) / t[1][i])
            if val > bestValue:
                bestValue = val
                best = i

        res = 0.0
        if len(moves) > 0:
            h = play(board, h, moves[best], piece_hash, hashTurn)
            res, h = UCT(board, h, piece_hash, hashTurn, Table)
            t[0] += 1
            t[1][best] += 1
            t[2][best] += res
        return res, h
    else:  # Sampling step
        add(board, h, Table)
        score_playout, h = playout(board, h, piece_hash, hashTurn)
        return score_playout, h


def _uct_worker(args):
    fen, h, piece_hash, hashTurn, nb_playout = args
    board = chess.Board(fen)
    Table = {}
    for _ in range(nb_playout):
        b1 = copy.deepcopy(board)
        h1 = h
        UCT(b1, h1, piece_hash, hashTurn, Table)
    t = look(h, Table)
    moves = [i for i in board.legal_moves]
    if t is None:
        return [0.0] * len(moves)
    return list(t[1])


def BestMoveUCT(board, h, piece_hash, hashTurn, nb_playout, processes: int = 1, time_ms: int | None = None):
    """
    Détermine le best move selon UCT.
    :param board:
    :param h:
    :param piece_hash:
    :param nb_playout:
    :return:
    """
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
                UCT(b1, h1, piece_hash, hashTurn, Table)
        else:
            for _ in range(nb_playout):  # update the TT with per-move stats for this board
                b1 = copy.deepcopy(board)
                h1 = h
                UCT(b1, h1, piece_hash, hashTurn, Table)
        t = look(h, Table)
        if t is None:
            return moves[0]
        counts = t[1]
    else:
        # Root-parallelize by summing visits from independent workers
        share = [nb_playout // processes] * processes
        for i in range(nb_playout % processes):
            share[i] += 1
        args = [(board.fen(), h, piece_hash, hashTurn, s) for s in share]
        with mp.Pool(processes=processes) as pool:
            parts = pool.map(_uct_worker, args)
        # sum elementwise
        counts = [0.0] * len(moves)
        for arr in parts:
            for i, v in enumerate(arr):
                counts[i] += v

    # Choose move with maximum visit count
    best_idx = 0
    best_val = counts[0]
    for i in range(1, len(moves)):
        if counts[i] > best_val:
            best_val = counts[i]
            best_idx = i
    return moves[best_idx]
