from __future__ import annotations
import time
import chess
import chess.polyglot as poly
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
from eval import evaluate

MATE_SCORE = 100000
INF = 10**9


@dataclass
class TTEntry:
    depth: int
    flag: int  # 0=EXACT, 1=LOWER, 2=UPPER
    score: int
    best_move: Optional[chess.Move]


@dataclass
class SearchCtx:
    deadline: Optional[float] = None
    max_nodes: Optional[int] = None
    nodes: int = 0
    tt: Dict[int, TTEntry] = field(default_factory=dict)
    killers: Dict[int, List[chess.Move]] = field(default_factory=dict)  # ply -> [k1, k2]
    history: Dict[Tuple[int, int], int] = field(default_factory=dict)  # (from,to) -> score
    last_score: int = 0


def _time_up(ctx: SearchCtx) -> bool:
    if ctx.max_nodes is not None and ctx.nodes >= ctx.max_nodes:
        return True
    if ctx.deadline is not None and time.time() > ctx.deadline:
        return True
    return False


def _move_key(m: chess.Move) -> Tuple[int, int]:
    return (m.from_square, m.to_square)


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square)
        attacker = board.piece_type_at(move.from_square)
        if victim is None or attacker is None:
            return 0
        return (victim * 10) - attacker
    if move.promotion:
        return 90 + move.promotion
    return 0


def order_moves(board: chess.Board, moves, ctx: SearchCtx, tt_move: Optional[chess.Move], ply: int):
    klist = ctx.killers.get(ply, [])
    def score(m: chess.Move) -> Tuple[int, int, int, int]:
        is_tt = 1 if (tt_move is not None and m == tt_move) else 0
        is_killer = 1 if m in klist else 0
        hist = ctx.history.get(_move_key(m), 0)
        return (
            is_tt,
            mvv_lva_score(board, m),
            is_killer,
            hist,
        )
    return sorted(moves, key=score, reverse=True)


def quiescence(board: chess.Board, alpha: int, beta: int, color: int, qdepth: int = 4, ctx: Optional[SearchCtx] = None) -> int:
    if ctx:
        ctx.nodes += 1
        if _time_up(ctx):
            return 0
    stand_pat = color * evaluate(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    if qdepth <= 0:
        return stand_pat

    # Only consider captures and promotions; prefer higher MVV-LVA first
    caps = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    for move in order_moves(board, caps, ctx or SearchCtx(), None, 0):
        board.push(move)
        if board.is_repetition(3):
            score = 0
        else:
            score = -quiescence(board, -beta, -alpha, -color, qdepth-1, ctx)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


EXACT, LOWER, UPPER = 0, 1, 2


def negamax(board: chess.Board, depth: int, alpha: int, beta: int, color: int, ctx: Optional[SearchCtx] = None, ply: int = 0) -> int:
    if ctx:
        ctx.nodes += 1
        if _time_up(ctx):
            return 0

    # Mate/stalemate checks
    if depth <= 0:
        return quiescence(board, alpha, beta, color, qdepth=8, ctx=ctx)

    # Null-move pruning (simple): skip if in check or low depth
    # Helps at low depths to quickly refute positions where a pass still keeps score high
    if depth >= 3 and not board.is_check():
        # Heuristic: try a null move and see if beta cutoff occurs
        R = 2
        board.push(chess.Move.null())
        try:
            # After null, side-to-move flips; color flips via the negamax call
            null_score = -negamax(board, depth - 1 - R, -beta, -beta + 1, -color, ctx, ply + 1)
        finally:
            board.pop()
        if null_score >= beta:
            return beta

    key = poly.zobrist_hash(board)
    if ctx and key in ctx.tt:
        e = ctx.tt[key]
        if e.depth >= depth:
            val = e.score
            if e.flag == EXACT:
                return val
            elif e.flag == LOWER and val > alpha:
                alpha = val
            elif e.flag == UPPER and val < beta:
                beta = val
            if alpha >= beta:
                return val

    best = -INF
    best_move = None
    tt_move = (ctx.tt[key].best_move if (ctx and key in ctx.tt and ctx.tt[key].best_move is not None) else None)
    legal = list(board.legal_moves)
    moves = order_moves(board, legal, ctx or SearchCtx(), tt_move, ply)

    for idx, move in enumerate(moves):
        is_check = board.gives_check(move)
        board.push(move)
        if board.is_repetition(3):
            score = 0
        else:
            ext = 1 if is_check else 0
            new_depth = depth - 1 + ext
            # Late move reductions: reduce depth for non-check, quiet, late moves
            if not is_check and not board.is_capture(move) and new_depth >= 2 and idx >= 3:
                new_depth -= 1
            score = -negamax(board, new_depth, -beta, -alpha, -color, ctx, ply+1)
        board.pop()

        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            # update killers/history on fail-high
            if ctx is not None:
                kl = ctx.killers.get(ply, [])
                if move not in kl:
                    ctx.killers[ply] = ([move] + kl)[:2]
                if not board.is_capture(move):
                    mk = _move_key(move)
                    ctx.history[mk] = ctx.history.get(mk, 0) + depth*depth
            break

    # Store TT entry
    if ctx is not None:
        flag = EXACT
        if best <= alpha:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        ctx.tt[key] = TTEntry(depth=depth, flag=flag, score=best, best_move=best_move)
    return best


def best_move_ab_with_stats(board: chess.Board, depth: int = 3, time_ms: Optional[int] = None, max_nodes: Optional[int] = None) -> Tuple[Optional[chess.Move], dict]:
    color = 1 if board.turn == chess.WHITE else -1
    ctx = SearchCtx(
        deadline=(time.time() + time_ms/1000.0) if time_ms else None,
        max_nodes=max_nodes,
    )

    best_score = -INF
    best_move = None

    # Aspiration windows around previous score
    window = 50  # centipawns
    prev = 0
    start_ts = time.time()
    depth_reached = 0
    for d in range(1, depth + 1):
        alpha, beta = prev - window, prev + window
        while True:
            if _time_up(ctx):
                break
            cur_best = best_move
            cur_score = -INF

            # Probe TT move if available at root
            key = poly.zobrist_hash(board)
            tt_move = ctx.tt[key].best_move if key in ctx.tt and ctx.tt[key].best_move else None
            legal = list(board.legal_moves)
            moves = order_moves(board, legal, ctx, tt_move, 0)

            for move in moves:
                if _time_up(ctx):
                    break
                board.push(move)
                if board.is_repetition(3):
                    score = 0
                else:
                    score = -negamax(board, d - 1, -beta, -alpha, -color, ctx, 1)
                board.pop()
                if score > cur_score or cur_best is None:
                    cur_score = score
                    cur_best = move
                if cur_score > alpha:
                    alpha = cur_score
            # Check aspiration window results
            if cur_score <= alpha:
                # fail low -> widen below
                alpha -= window
                window *= 2
                continue
            if cur_score >= beta:
                # fail high -> widen above
                beta += window
                window *= 2
                continue
            # success
            best_move, best_score = cur_best, cur_score
            prev = cur_score
            ctx.last_score = cur_score
            depth_reached = d
            break
        if _time_up(ctx):
            break
    elapsed_ms = int((time.time() - start_ts) * 1000)
    stats = {"nodes": ctx.nodes, "elapsed_ms": elapsed_ms, "depth": depth_reached}
    return best_move, stats


def best_move_ab(board: chess.Board, depth: int = 3, time_ms: Optional[int] = None, max_nodes: Optional[int] = None) -> Optional[chess.Move]:
    mv, _ = best_move_ab_with_stats(board, depth=depth, time_ms=time_ms, max_nodes=max_nodes)
    return mv
