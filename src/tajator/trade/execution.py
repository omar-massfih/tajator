"""Order execution against the Broker protocol. Pure mechanics, no judgment."""

from __future__ import annotations

import math
from typing import Literal

from ..broker.base import Broker
from ..config import Settings
from ..models import Decision, Direction, ExecutedAction, OpenPosition, Snapshot
from .contracts import select_contract
from .position import build_plan, current_piece_qty, update_extreme


def size_entry(premium: float, settings: Settings) -> int:
    if premium <= 0:
        return 0
    affordable = math.floor(settings.max_premium_usd / (premium * 100))
    return min(settings.max_contracts, affordable)


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
    action = ExecutedAction(
        kind="entry", qty=fill.qty, premium=fill.premium, equity_price=snapshot.price,
        ts=fill.ts, reason=decision.reasoning,
    )
    return position, action, None


def execute_scale_out(
    broker: Broker, position: OpenPosition, snapshot: Snapshot, reason: str
) -> ExecutedAction:
    qty = min(current_piece_qty(position), position.qty_remaining)
    fill = broker.sell_option(position.contract, qty)
    if fill.qty == qty:
        position.pieces_sold += 1  # a partially sold piece is retried on the next signal
    position.qty_remaining -= fill.qty
    return ExecutedAction(
        kind="scale_out", qty=fill.qty, premium=fill.premium, equity_price=snapshot.price,
        ts=fill.ts, reason=reason,
    )


def execute_exit(
    broker: Broker,
    position: OpenPosition,
    snapshot: Snapshot,
    kind: Literal["stop_exit", "runner_exit", "manual_exit"],
    reason: str,
) -> ExecutedAction:
    qty = position.qty_remaining
    fill = broker.sell_option(position.contract, qty)
    position.qty_remaining = qty - fill.qty  # nonzero after a partial exit
    return ExecutedAction(
        kind=kind, qty=fill.qty, premium=fill.premium, equity_price=snapshot.price,
        ts=fill.ts, reason=reason,
    )
