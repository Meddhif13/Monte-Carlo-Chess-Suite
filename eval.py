from __future__ import annotations
import chess

# Simplified classical evaluation in centipawns (positive is good for White)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
NEAR_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4, chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}


def _material(board: chess.Board) -> int:
    score = 0
    for pt, val in PIECE_VALUES.items():
        if pt == chess.KING:
            continue
        score += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return score


def _phase(board: chess.Board) -> float:
    # 0 opening .. 1 endgame based on non-pawn material
    total = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        total += PIECE_VALUES[pt] * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
    # Full non-pawn material ~ 2*(2*320 + 2*330 + 2*500 + 900) = 2*(320*2 + 330*2 + 500*2 + 900)
    full = 2 * (2*PIECE_VALUES[chess.KNIGHT] + 2*PIECE_VALUES[chess.BISHOP] + 2*PIECE_VALUES[chess.ROOK] + PIECE_VALUES[chess.QUEEN])
    if full <= 0:
        return 1.0
    x = max(0.0, min(1.0, total / full))
    return 1.0 - x


def _center_bias(board: chess.Board) -> int:
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        sgn = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.KNIGHT, color):
            score += sgn * (15 if sq in CENTER_SQUARES else (8 if sq in NEAR_CENTER else 0))
        for sq in board.pieces(chess.BISHOP, color):
            score += sgn * (10 if sq in CENTER_SQUARES else (6 if sq in NEAR_CENTER else 0))
        for sq in board.pieces(chess.QUEEN, color):
            score += sgn * (6 if sq in CENTER_SQUARES else (3 if sq in NEAR_CENTER else 0))
        ksq = board.king(color)
        if ksq is not None:
            rank = chess.square_rank(ksq)
            file = chess.square_file(ksq)
            dist_center = max(abs(file - 3.5), abs(rank - 3.5))
            opening_weight = max(0.0, 1.0 - _phase(board))
            score -= sgn * int(12 * opening_weight * (2.5 - dist_center))
    return score


def _mobility(board: chess.Board) -> int:
    def moves_count(b: chess.Board, col: bool) -> int:
        if b.turn != col:
            b.push(chess.Move.null())
            n = b.legal_moves.count()
            b.pop()
            return n
        return b.legal_moves.count()
    w = moves_count(board, chess.WHITE)
    b = moves_count(board, chess.BLACK)
    return 2 * (w - b)


def _pawn_structure(board: chess.Board) -> int:
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        sgn = 1 if color == chess.WHITE else -1
        pawns = board.pieces(chess.PAWN, color)
        files = [0]*8
        for sq in pawns:
            files[chess.square_file(sq)] += 1
        # doubled
        for f in range(8):
            if files[f] > 1:
                score -= sgn * 10 * (files[f] - 1)
        # isolated
        for f in range(8):
            if files[f] > 0:
                left = files[f-1] if f-1 >= 0 else 0
                right = files[f+1] if f+1 < 8 else 0
                if left == 0 and right == 0:
                    score -= sgn * 12
        # passed
        for sq in pawns:
            if _is_passed_pawn(board, sq, color):
                rank = chess.square_rank(sq)
                advance = rank if color == chess.WHITE else (7 - rank)
                score += sgn * (20 + 3 * advance)
    return score


def _is_passed_pawn(board: chess.Board, sq: chess.Square, color: bool) -> bool:
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    opp = not color
    for df in (-1, 0, 1):
        f = file + df
        if f < 0 or f > 7:
            continue
        r_iter = range(rank+1, 8) if color == chess.WHITE else range(rank-1, -1, -1)
        for r in r_iter:
            if chess.square(f, r) in board.pieces(chess.PAWN, opp):
                return False
    return True


def _king_pawn_shield(board: chess.Board) -> int:
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        sgn = 1 if color == chess.WHITE else -1
        ksq = board.king(color)
        if ksq is None:
            continue
        rank = chess.square_rank(ksq)
        file = chess.square_file(ksq)
        missing = 0
        if color == chess.WHITE:
            rows = [min(7, rank + 1)]
        else:
            rows = [max(0, rank - 1)]
        for df in (-1, 0, 1):
            f = min(7, max(0, file + df))
            for r in rows:
                if chess.square(f, r) not in board.pieces(chess.PAWN, color):
                    missing += 1
        score -= sgn * (6 * missing)
    return score


def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        # side to move has no legal move and is in check -> mated
        return -99999 if board.turn == chess.WHITE else 99999
    # Cheap draw checks only; avoid expensive can_claim_threefold_repetition in hot paths
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3):
        return 0
    score = 0
    score += _material(board)
    score += _center_bias(board)
    score += _mobility(board)
    score += _pawn_structure(board)
    score += _king_pawn_shield(board)
    # Bishop pair bonus
    w_b = len(board.pieces(chess.BISHOP, chess.WHITE))
    b_b = len(board.pieces(chess.BISHOP, chess.BLACK))
    if w_b >= 2:
        score += 25
    if b_b >= 2:
        score -= 25
    # Rook activity on open/semi-open files
    for color in (chess.WHITE, chess.BLACK):
        sgn = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.ROOK, color):
            file = chess.square_file(sq)
            has_own_pawn = any(chess.square(file, r) in board.pieces(chess.PAWN, color) for r in range(8))
            has_enemy_pawn = any(chess.square(file, r) in board.pieces(chess.PAWN, not color) for r in range(8))
            if not has_own_pawn and not has_enemy_pawn:
                score += sgn * 15  # open file
            elif not has_own_pawn and has_enemy_pawn:
                score += sgn * 8   # semi-open
    return score


def cp_to_black_prob(cp: int, scale: float = 0.004) -> float:
    # cp > 0 favors white => lower black win probability.
    # logistic: P(black) = 1 / (1 + exp(scale * cp))
    from math import exp
    # Clamp cp to avoid overflow
    cp = max(-2000, min(2000, cp))
    return 1.0 / (1.0 + exp(scale * cp))
