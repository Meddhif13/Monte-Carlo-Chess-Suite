import copy
import random
import math
import time
import chess

def UCB(board, n, c: float = 0.8, time_ms: int | None = None):
    """
    Algorithme d'UCB pour trouver le best move.

    :param board: de MChess
    :param n: nombre de playouts
    :return: best move à partir du board
    """
    moves = [i for i in board.legal_moves]

    sumScores = [0.0 for x in range(len(moves))]
    nbVisits = [0 for x in range(len(moves))]
    root_is_white = board.turn
    start = time.time()
    deadline = (start + time_ms/1000.0) if (time_ms is not None and time_ms > 0) else None
    i = 0
    # Loop by playouts or time, whichever specified; always run at least one sim per iteration
    while True:
        bestScore = 0
        bestMove = moves[0]
        place = 0
        for m in range(len(moves)):  # on calcule le score de chaque coût
            if nbVisits[m] > 0:
                score = sumScores[m] / nbVisits[m] + c * math.sqrt(math.log(max(1, i)) / nbVisits[m])
            else:
                score = 10000000  # on explore tout !! du dernier au premier
            if score > bestScore:
                bestScore = score
                bestMove = moves[m]
                place = m
        b = copy.deepcopy(board)
        b.push(bestMove)  # on joue le meilleur score
        r = playout(b)
        r_adj = r if root_is_white else (1 - r)

        sumScores[place] += r_adj  # on met à jour les poids
        nbVisits[place] += 1
        i += 1
        if deadline is not None:
            if time.time() >= deadline:
                break
        else:
            if i >= n:
                break
    bestScore = 0
    bestMove = moves[0]
    for m in range(1, len(moves)):  # on renvoie le meilleur move
        score = (sumScores[m] / nbVisits[m]) if nbVisits[m] > 0 else -1
        if score > bestScore:
            bestScore = score
            bestMove = moves[m]
    return bestMove

def score(board):
    """Return 1 for white win, 0 for black win, 0.5 draw."""
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return 0.0
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
            promotions.append(mv); continue
        if b.is_capture(mv):
            if last_to is not None and mv.to_square == last_to:
                recaptures.append(mv)
            else:
                captures.append(mv)
            continue
        if b.gives_check(mv):
            checks.append(mv); continue
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


def playout(b):
    """Heuristic rollout with tactical bias; returns 1 for white win, 0 for black, 0.5 draw."""
    while True:
        moves = [i for i in b.legal_moves]
        if b.is_game_over():
            return score(b)
        move = _rollout_move(b)
        b.push(move)
