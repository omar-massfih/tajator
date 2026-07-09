"""Order execution against the Broker protocol. Pure mechanics, no judgment."""

from __future__ import annotations

import logging
import math
from typing import Literal

from ..broker.base import Broker
from ..config import Settings
from ..models import Decision, Direction, ExecutedAction, OpenPosition, Snapshot
from .contracts import select_contract
from .position import active_stop_price, build_plan, current_piece_qty, update_extreme

log = logging.getLogger(__name__)


def size_entry(premium: float, settings: Settings) -> int:
    if premium <= 0:
        return 0
    affordable = math.floor(settings.max_premium_usd / (premium * 100))
    return min(settings.max_contracts, affordable)


def stop_order_ref(settings: Settings, symbol: str) -> str:
    """orderRef stamped on protective stops — how startup recognizes our own."""
    return f"{settings.order_ref_prefix}-stop:{symbol}"


def execute_entry(
    broker: Broker,
    settings: Settings,
    decision: Decision,
    direction: Direction,
    snapshot: Snapshot,
) -> tuple[OpenPosition | None, ExecutedAction | None, str | None]:
    """Returns (position, action, skip_reason) — exactly one of position/skip_reason set."""
    chain = broker.get_option_chain(snapshot.symbol)
    contract = select_contract(chain, snapshot.symbol, snapshot.price, direction, broker.now())
    if contract is None:
        return None, None, "no usable contract in option chain"

    premium = broker.get_option_premium(contract)
    if premium is None or premium <= 0:
        return None, None, f"no premium quote for {contract.local_name}"
    qty = size_entry(premium, settings)
    if qty == 0:
        return None, None, (
            f"one {contract.local_name} costs ${premium * 100:.0f} — over MAX_PREMIUM_USD budget"
        )

    fill = broker.buy_option(contract, qty)
    # A reconciled partial fill returns fewer contracts than requested — the
    # plan must cover what the account actually holds, not what was asked for.
    plan = build_plan(
        direction=direction,
        level_price=decision.level_price,
        stop_price=decision.stop_price,
        entry_equity_price=snapshot.price,
        entry_premium=fill.premium,
        qty=fill.qty,
    )
    position = OpenPosition(
        contract=contract, plan=plan, qty_remaining=fill.qty, opened_at=fill.ts
    )
    update_extreme(position, snapshot.price)
    restore_protective_stop(broker, settings, position, snapshot.symbol)
    action = ExecutedAction(
        kind="entry", qty=fill.qty, premium=fill.premium, equity_price=snapshot.price,
        ts=fill.ts, reason=decision.reasoning,
    )
    return position, action, None


def restore_protective_stop(
    broker: Broker, settings: Settings, position: OpenPosition, symbol: str
) -> None:
    """Place the broker-side stop when it should exist but doesn't. A failure
    must never fail the trade — the mental stop still enforces the plan and
    this is retried every tick."""
    if not settings.protective_stop_enabled:
        return
    if position.qty_remaining <= 0 or position.protective_stop is not None:
        return
    try:
        position.protective_stop = broker.place_protective_stop(
            position.contract,
            position.qty_remaining,
            active_stop_price(position),
            position.plan.direction,
            stop_order_ref(settings, symbol),
        )
    except Exception:  # noqa: BLE001 — backstop only; the mental stop remains
        log.exception(
            "could not place protective stop for %dx %s",
            position.qty_remaining, position.contract.local_name,
        )


def retire_protective_stop(
    broker: Broker, position: OpenPosition, snapshot: Snapshot
) -> ExecutedAction | None:
    """Cancel-and-confirm the resting stop before ANY agent-driven sell — the
    double-sell guard. If the cancel raced the stop's fill, the fill is adopted
    here (qty_remaining shrinks) and returned for journaling. Raises when the
    broker cannot confirm a terminal state — the caller must not sell then."""
    stop = position.protective_stop
    if stop is None:
        return None
    result = broker.cancel_protective_stop(
        position.contract, stop, expected_held=position.qty_remaining
    )
    position.protective_stop = None
    if not result.filled_qty:
        return None
    position.qty_remaining -= result.filled_qty
    return ExecutedAction(
        kind="stop_exit",
        qty=result.filled_qty,
        premium=result.avg_price,
        equity_price=snapshot.price,
        ts=broker.now(),
        reason="protective stop filled while being cancelled",
    )


def execute_scale_out(
    broker: Broker,
    settings: Settings,
    position: OpenPosition,
    snapshot: Snapshot,
    reason: str,
) -> list[ExecutedAction]:
    actions: list[ExecutedAction] = []
    pre = retire_protective_stop(broker, position, snapshot)
    if pre is not None:
        actions.append(pre)
    qty = min(current_piece_qty(position), position.qty_remaining)
    if qty > 0:
        fill = broker.sell_option(position.contract, qty)
        if fill.qty == qty:
            position.pieces_sold += 1  # a partially sold piece is retried on the next signal
        if fill.qty > 0:
            if not position.profit_taken:
                position.profit_lock_price = snapshot.price
            position.profit_taken = True
        position.qty_remaining -= fill.qty
        actions.append(
            ExecutedAction(
                kind="scale_out", qty=fill.qty, premium=fill.premium,
                equity_price=snapshot.price, ts=fill.ts, reason=reason,
            )
        )
    restore_protective_stop(broker, settings, position, snapshot.symbol)
    return actions


def execute_exit(
    broker: Broker,
    settings: Settings,
    position: OpenPosition,
    snapshot: Snapshot,
    kind: Literal["stop_exit", "runner_exit", "manual_exit"],
    reason: str,
) -> list[ExecutedAction]:
    actions: list[ExecutedAction] = []
    pre = retire_protective_stop(broker, position, snapshot)
    if pre is not None:
        actions.append(pre)
    qty = position.qty_remaining
    if qty > 0:
        fill = broker.sell_option(position.contract, qty)
        position.qty_remaining = qty - fill.qty  # nonzero after a partial exit
        actions.append(
            ExecutedAction(
                kind=kind, qty=fill.qty, premium=fill.premium, equity_price=snapshot.price,
                ts=fill.ts, reason=reason,
            )
        )
    # a partial exit leaves contracts that must stay stop-protected
    restore_protective_stop(broker, settings, position, snapshot.symbol)
    return actions
