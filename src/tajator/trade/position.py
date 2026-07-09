"""Deterministic trade management: the plan frozen at entry is enforced here.

Rules, checked in this order every tick (stop always first):
1. Mental stop on the EQUITY price -> flatten immediately. Never LLM-negotiable.
2. Runner phase (last piece): exit at break-even, or on adverse VWAP cross
   after price had traded beyond VWAP in our favor.
3. Scaling phase: when the trade has moved at least half its planned equity
   risk in our favor and the current piece's target reference is touched, emit
   a scale candidate — the LLM may say "hold one more bar", but the
   deterministic fallback (and any LLM error) scales the piece. EMA9 is chart
   context only; it is too shallow for an automatic exit.

One-contract positions follow the notes' one-contract process: full exit
at the first target, no runner phase.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ..models import Direction, OpenPosition, PositionPlan, Snapshot

SCALE_REFS = ["ema50_vwap", "hod_lod"]


class ManageAction(BaseModel):
    kind: Literal["hold", "stop_exit", "runner_exit", "scale_candidate", "broker_stop_filled"]
    target_ref: str | None = None
    reason: str = ""


def split_pieces(qty: int) -> list[int]:
    """Split contracts into scale-out pieces, extras loaded up front."""
    n = min(qty, len(SCALE_REFS) + 1)
    base, rem = divmod(qty, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def build_plan(
    direction: Direction,
    level_price: float,
    stop_price: float,
    entry_equity_price: float,
    entry_premium: float,
    qty: int,
) -> PositionPlan:
    pieces = split_pieces(qty)
    refs = SCALE_REFS[: len(pieces) - 1] + ["runner"] if len(pieces) > 1 else [SCALE_REFS[0]]
    return PositionPlan(
        direction=direction,
        level_price=level_price,
        stop_price=stop_price,
        entry_equity_price=entry_equity_price,
        entry_premium=entry_premium,
        total_qty=qty,
        pieces=pieces,
        target_refs=refs,
    )


def update_extreme(position: OpenPosition, price: float) -> None:
    best = position.favorable_extreme
    if position.plan.direction == "call":
        position.favorable_extreme = max(best, price) if best is not None else price
    else:
        position.favorable_extreme = min(best, price) if best is not None else price


def current_piece_qty(position: OpenPosition) -> int:
    return position.plan.pieces[position.pieces_sold]


def active_stop_price(position: OpenPosition) -> float:
    """The enforced stop. After any profit-taking, protect the remainder at break-even."""
    if position.profit_taken or position.pieces_sold > 0:
        return position.plan.entry_equity_price
    return position.plan.stop_price


def evaluate(position: OpenPosition, snapshot: Snapshot) -> ManageAction:
    plan = position.plan
    price = snapshot.price
    long = plan.direction == "call"

    # 1. Mental stop — first, always.
    stop_price = active_stop_price(position)
    stop_hit = price <= stop_price if long else price >= stop_price
    if stop_hit:
        stop_label = "break-even stop" if stop_price == plan.entry_equity_price else "mental stop"
        return ManageAction(
            kind="stop_exit",
            reason=f"equity {price} through {stop_label} {stop_price} — exit everything",
        )

    in_runner_phase = len(plan.pieces) > 1 and position.pieces_sold == len(plan.pieces) - 1

    if in_runner_phase:
        be_hit = price <= plan.entry_equity_price if long else price >= plan.entry_equity_price
        if be_hit:
            return ManageAction(
                kind="runner_exit",
                reason=f"runner back to break-even {plan.entry_equity_price} — protect profits",
            )
        if snapshot.vwap is not None and position.favorable_extreme is not None:
            was_beyond = (
                position.favorable_extreme > snapshot.vwap
                if long
                else position.favorable_extreme < snapshot.vwap
            )
            crossed_back = price < snapshot.vwap if long else price > snapshot.vwap
            if was_beyond and crossed_back:
                side = "lost" if long else "reclaimed"
                return ManageAction(
                    kind="runner_exit", reason=f"price {side} VWAP {snapshot.vwap:.2f} against runner"
                )
        return ManageAction(kind="hold")

    # 3. Scaling phase.
    ref = plan.target_refs[position.pieces_sold]
    target = _target_value(ref, snapshot, plan.direction)
    if target is None:
        return ManageAction(kind="hold")
    if _favorable_move(price, plan) < _min_scale_move(plan):
        return ManageAction(kind="hold")
    touched = price >= target if long else price <= target
    if touched:
        return ManageAction(
            kind="scale_candidate",
            target_ref=ref,
            reason=f"price {price} touched {ref} target {target:.2f}",
        )
    return ManageAction(kind="hold")


def _favorable_move(price: float, plan: PositionPlan) -> float:
    if plan.direction == "call":
        return price - plan.entry_equity_price
    return plan.entry_equity_price - price


def _min_scale_move(plan: PositionPlan) -> float:
    return abs(plan.entry_equity_price - plan.stop_price) / 2


def _target_value(ref: str, snapshot: Snapshot, direction: Direction) -> float | None:
    if ref == "ema50_vwap":
        vals = [v for v in (snapshot.ema50, snapshot.vwap) if v is not None]
        if not vals:
            return None
        # The nearer reference in the profit direction triggers first.
        return min(vals) if direction == "call" else max(vals)
    if ref == "hod_lod":
        return snapshot.hod if direction == "call" else snapshot.lod
    return None
