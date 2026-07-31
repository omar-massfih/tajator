"""Order execution against the Broker protocol. Pure mechanics, no judgment."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Literal

from ..broker.base import Broker
from ..config import Settings
from ..models import (
    Decision,
    Direction,
    ExecutedAction,
    OpenPosition,
    OptionQuote,
    SetupCandidate,
    Snapshot,
)
from .contracts import select_contract
from .position import active_stop_price, build_plan, current_piece_qty, update_extreme

log = logging.getLogger(__name__)


def size_entry(premium: float, settings: Settings, *, reserve_pct: float = 0.0) -> int:
    if premium <= 0:
        return 0
    reserved = premium * (1 + reserve_pct)
    affordable = math.floor(settings.max_premium_usd / (reserved * 100))
    return min(settings.max_contracts, affordable)


def validate_entry_preflight(
    quote: OptionQuote,
    underlying: float | None,
    snapshot: Snapshot,
    candidate: SetupCandidate,
    stop_price: float,
    settings: Settings,
) -> str | None:
    """Return a skip reason when liquidity or the original signal degraded."""
    quote_reason = validate_option_liquidity(quote, settings)
    if quote_reason:
        return quote_reason
    if underlying is None or underlying <= 0:
        return "no fresh underlying price for entry preflight"

    level = candidate.level.price
    approach = settings.approach_band_pct * level
    overshoot = settings.overshoot_band_pct * level
    drift_limit = max(
        settings.max_entry_drift_min_cents / 100,
        (snapshot.atr or 0.0) * settings.max_entry_drift_atr,
    )
    if candidate.direction == "call":
        if underlying <= stop_price:
            return f"underlying {underlying:.2f} already crossed call stop {stop_price:.2f}"
        if underlying - snapshot.price > drift_limit:
            return f"call signal moved away by {underlying - snapshot.price:.2f}"
        if not level - overshoot <= underlying <= level + approach:
            return f"underlying {underlying:.2f} left support approach zone"
    else:
        if underlying >= stop_price:
            return f"underlying {underlying:.2f} already crossed put stop {stop_price:.2f}"
        if snapshot.price - underlying > drift_limit:
            return f"put signal moved away by {snapshot.price - underlying:.2f}"
        if not level - approach <= underlying <= level + overshoot:
            return f"underlying {underlying:.2f} left resistance approach zone"
    return None


def validate_option_liquidity(
    quote: OptionQuote, settings: Settings, now: datetime | None = None
) -> str | None:
    """Validate the quote facts shared by production entries and test-order."""
    if not quote.valid:
        return "entry quote has no valid positive bid/ask"
    if quote.delayed:
        return "entry option quote is delayed"
    now = now or datetime.now(quote.ts.tzinfo)
    age = max(0.0, (now - quote.ts).total_seconds())
    if age > settings.max_option_quote_age_s:
        return f"entry quote is stale ({age:.1f}s old)"
    midpoint, spread = quote.midpoint, quote.spread
    allowed_spread = min(
        settings.max_option_spread_cents / 100,
        settings.max_option_spread_pct * midpoint,
    )
    if spread > allowed_spread:
        return f"option spread {spread:.2f} exceeds {allowed_spread:.2f} maximum"
    return None


def stop_order_ref(settings: Settings, symbol: str) -> str:
    """orderRef stamped on protective stops — how startup recognizes our own."""
    return f"{settings.order_ref_prefix}-stop:{symbol}"


def execute_entry(
    broker: Broker,
    settings: Settings,
    decision: Decision,
    direction: Direction,
    snapshot: Snapshot,
    candidate: SetupCandidate | None = None,
) -> tuple[OpenPosition | None, ExecutedAction | None, str | None]:
    """Returns (position, action, skip_reason) — exactly one of position/skip_reason set."""
    chain = broker.get_option_chain(snapshot.symbol)
    contract = select_contract(chain, snapshot.symbol, snapshot.price, direction, broker.now())
    if contract is None:
        return None, None, "no usable contract in option chain"

    preflight_quote = None
    preflight_underlying = None
    if broker.uses_live_execution_guards:
        try:
            preflight_quote, preflight_underlying = broker.get_entry_market_snapshot(contract)
        except Exception as exc:  # fail closed: an entry is optional
            reason = f"entry market snapshot request failed: {exc}"
            broker.record_execution_preflight(
                ts=broker.now(), symbol=snapshot.symbol, contract=contract,
                signal_ts=snapshot.ts, signal_equity_price=snapshot.price,
                underlying_price=None, option_quote=None, accepted=False, reason=reason,
            )
            return None, None, reason
        reason = validate_entry_preflight(
            preflight_quote, preflight_underlying, snapshot, candidate,
            decision.stop_price, settings,
        ) if candidate is not None and decision.stop_price is not None else "entry candidate/stop missing"
        broker.record_execution_preflight(
            ts=preflight_quote.ts,
            symbol=snapshot.symbol,
            contract=contract,
            signal_ts=snapshot.ts,
            signal_equity_price=snapshot.price,
            underlying_price=preflight_underlying,
            option_quote=preflight_quote,
            accepted=reason is None,
            reason=reason or "accepted",
        )
        if reason:
            return None, None, reason
        premium = preflight_quote.ask
        reserve_pct = settings.entry_budget_reserve_pct
    else:
        premium = broker.get_option_premium(contract)
        reserve_pct = 0.0
    if premium is None or premium <= 0:
        return None, None, f"no premium quote for {contract.local_name}"
    qty = size_entry(premium, settings, reserve_pct=reserve_pct)
    if qty == 0:
        reserved_cost = premium * (1 + reserve_pct) * 100
        return None, None, (
            f"one {contract.local_name} reserves ${reserved_cost:.0f} — over MAX_PREMIUM_USD budget"
        )

    fill = (
        broker.buy_option_from_snapshot(
            contract, qty, preflight_quote, preflight_underlying,
        )
        if preflight_quote is not None
        else broker.buy_option(contract, qty)
    )
    quality = fill.execution_quality
    if quality is not None:
        quality.signal_ts = snapshot.ts
        quality.preflight_ts = preflight_quote.ts if preflight_quote is not None else None
        quality.underlying_signal = snapshot.price
    entry_equity_price = (
        quality.underlying_fill if quality is not None and quality.underlying_fill is not None
        else preflight_underlying if preflight_underlying is not None else snapshot.price
    )
    fill.equity_price = entry_equity_price
    fill.stop_price = decision.stop_price
    fill.regime = snapshot.regime
    fill.level_quality_score = candidate.quality_score if candidate is not None else 0.0
    # A reconciled partial fill returns fewer contracts than requested — the
    # plan must cover what the account actually holds, not what was asked for.
    plan = build_plan(
        direction=direction,
        level_price=decision.level_price,
        stop_price=decision.stop_price,
        entry_equity_price=entry_equity_price,
        entry_premium=fill.premium,
        qty=fill.qty,
        hod_at_entry=snapshot.hod,
        lod_at_entry=snapshot.lod,
    )
    position = OpenPosition(
        contract=contract, plan=plan, qty_remaining=fill.qty, opened_at=fill.ts
    )
    update_extreme(position, entry_equity_price)
    restore_protective_stop(broker, settings, position, snapshot.symbol)
    action = ExecutedAction(
        kind="entry", qty=fill.qty, premium=fill.premium, equity_price=entry_equity_price,
        ts=fill.ts, reason=decision.reasoning, execution_quality=quality,
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
        quality = fill.execution_quality
        equity_price = (
            quality.underlying_fill if quality is not None and quality.underlying_fill is not None
            else snapshot.price
        )
        fill.equity_price = equity_price
        fill.exit_reason = reason
        if fill.qty == qty:
            position.pieces_sold += 1  # a partially sold piece is retried on the next signal
        if fill.qty > 0:
            # RUNNER_STOP=first_target locks the scale-out price; the default
            # leaves profit_lock_price unset so active_stop falls back to
            # break-even and the runner keeps room toward hod/lod.
            if not position.profit_taken and settings.runner_stop == "first_target":
                position.profit_lock_price = snapshot.price
            position.profit_taken = True
        position.qty_remaining -= fill.qty
        actions.append(
            ExecutedAction(
                kind="scale_out", qty=fill.qty, premium=fill.premium,
                equity_price=equity_price, ts=fill.ts, reason=reason,
                execution_quality=quality,
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
        quality = fill.execution_quality
        equity_price = (
            quality.underlying_fill if quality is not None and quality.underlying_fill is not None
            else snapshot.price
        )
        fill.equity_price = equity_price
        fill.exit_reason = reason
        position.qty_remaining = qty - fill.qty  # nonzero after a partial exit
        actions.append(
            ExecutedAction(
                kind=kind, qty=fill.qty, premium=fill.premium, equity_price=equity_price,
                ts=fill.ts, reason=reason, execution_quality=quality,
            )
        )
    # a partial exit leaves contracts that must stay stop-protected
    restore_protective_stop(broker, settings, position, snapshot.symbol)
    return actions
