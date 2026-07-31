"""Graph nodes. Each is a closure over the RuntimeContext (broker, journal, settings).

Every node is deterministic. The flat branch detects an Opening-Range Breakout
and enters it; the position branch manages the trade with the fixed
stop/scale/runner rules. (The node keys ``llm_decide``/``llm_manage`` are kept
for graph-wiring stability; they no longer call any model.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..broker.base import Broker, OrderFailed
from ..config import Settings
from ..journal import Journal
from ..market.indicators import build_snapshot
from ..market.orb import detect_orb
from ..models import (
    Decision,
    ExecutedAction,
)
from ..notify import Notifier, NullNotifier
from ..risk import guardrails
from ..trade import position as pos
from ..trade.execution import (
    execute_entry,
    execute_exit,
    execute_scale_out,
    restore_protective_stop,
)
from .state import AgentState


@dataclass
class RuntimeContext:
    settings: Settings
    broker: Broker
    journal: Journal
    symbol: str
    notifier: Notifier = field(default_factory=NullNotifier)
    metrics: dict[str, int] = field(default_factory=dict)


def make_nodes(ctx: RuntimeContext) -> dict[str, Any]:
    settings = ctx.settings.for_symbol(ctx.symbol)

    def fetch_data(state: AgentState) -> dict:
        bars = ctx.broker.get_bars(ctx.symbol)
        prev_high, prev_low = ctx.broker.get_prev_day_range(ctx.symbol)
        return {
            "bars": bars,
            "prev_day_high": prev_high,
            "prev_day_low": prev_low,
        }

    def compute_context(state: AgentState) -> dict:
        bars = state["bars"]
        if not bars:
            raise RuntimeError("broker returned no bars (data farm down or market closed)")
        snapshot = build_snapshot(ctx.symbol, bars, atr_window=settings.atr_window_bars)
        # ORB anchors on the opening range, not prev-day/premarket levels.
        return {"snapshot": snapshot, "levels": []}

    # ----- position-open branch ---------------------------------------------

    def sync_protective_stop(position, snapshot) -> dict | None:
        """Reconcile the resting broker-side stop before managing: it may have
        fired (or been touched externally) since the last tick. Returns a
        state update ending the tick when the stop closed the position."""
        stop = position.protective_stop
        sold, avg = 0, None
        if stop is not None:
            status = ctx.broker.poll_protective_stop(position.contract, stop)
            if status.state == "filled":
                position.protective_stop = None
                sold, avg = min(status.filled_qty, position.qty_remaining), status.avg_price
            elif status.state == "partial":
                # cancel-and-confirm is the authoritative fill count, and it
                # guarantees no second stop order is left working
                result = ctx.broker.cancel_protective_stop(
                    position.contract, stop, expected_held=position.qty_remaining
                )
                position.protective_stop = None
                sold, avg = min(result.filled_qty, position.qty_remaining), result.avg_price
            elif status.state == "gone":
                ctx.journal.write(
                    "warning", ts=snapshot.ts, symbol=ctx.symbol,
                    warning=f"protective stop order {stop.order_id} disappeared at the broker "
                    "with no fills (cancelled externally?) — re-placing",
                )
                position.protective_stop = None
        if sold:
            position.qty_remaining -= sold
            action = ExecutedAction(
                kind="stop_exit", qty=sold,
                premium=avg if avg is not None else 0.0,
                equity_price=snapshot.price, ts=ctx.broker.now(),
                reason="broker protective stop fired",
            )
            ctx.journal.write("fill", ts=snapshot.ts, symbol=ctx.symbol, action=action, position=position)
            ctx.notifier.notify_fill(ctx.symbol, action, position)
            if position.qty_remaining == 0:
                return {
                    "manage_action": pos.ManageAction(
                        kind="broker_stop_filled", reason="broker protective stop fired"
                    ),
                    "actions": [action],
                    "position": None,
                }
        # covers entry-time placement failure and the cases above
        restore_protective_stop(ctx.broker, settings, position, ctx.symbol)
        return None

    def manage_position(state: AgentState) -> dict:
        position, snapshot = state["position"], state["snapshot"]
        done = sync_protective_stop(position, snapshot)
        if done is not None:
            return done
        pos.update_extreme(position, snapshot.price)
        action = pos.evaluate(position, snapshot)
        if action.kind != "hold":
            ctx.journal.write("manage_signal", ts=snapshot.ts, symbol=ctx.symbol, action=action)
        return {"manage_action": action}

    def llm_manage(state: AgentState) -> dict:
        # Deterministic: a scale_candidate from manage_position always scales the
        # current piece. (Node key kept for graph-wiring stability.)
        snapshot, action = state["snapshot"], state["manage_action"]
        decision = Decision(action="scale_out", reasoning=f"rule-follower: {action.reason}")
        ctx.journal.write("scale_decision", ts=snapshot.ts, symbol=ctx.symbol, mode="manage", decision=decision)
        return {"decision": decision}

    def do_scale_out(state: AgentState) -> dict:
        position, snapshot = state["position"], state["snapshot"]
        reason = state["manage_action"].reason
        actions = execute_scale_out(ctx.broker, settings, position, snapshot, reason)
        for action in actions:
            ctx.journal.write("fill", ts=snapshot.ts, symbol=ctx.symbol, action=action, position=position)
            ctx.notifier.notify_fill(ctx.symbol, action, position)
        closed = position.qty_remaining == 0
        return {"actions": actions, "position": None if closed else position}

    def do_exit(state: AgentState) -> dict:
        position, snapshot = state["position"], state["snapshot"]
        manage = state.get("manage_action")
        if manage is not None and manage.kind in ("stop_exit", "runner_exit"):
            kind, reason = manage.kind, manage.reason
        else:
            kind, reason = "manual_exit", state["decision"].reasoning
        actions = execute_exit(ctx.broker, settings, position, snapshot, kind, reason)
        for action in actions:
            ctx.journal.write("fill", ts=snapshot.ts, symbol=ctx.symbol, action=action, position=position)
            ctx.notifier.notify_fill(ctx.symbol, action, position)
        # a partial exit leaves contracts to retry on the next tick
        return {"actions": actions, "position": position if position.qty_remaining else None}

    # ----- flat branch --------------------------------------------------------

    def detect_setups(state: AgentState) -> dict:
        candidates = detect_orb(
            state["bars"], state["snapshot"],
            window_minutes=settings.orb_window_minutes,
            breakout_buffer_pct=settings.orb_breakout_buffer_pct,
            min_relative_volume=settings.orb_min_relative_volume,
            min_breakout_range_atr=settings.orb_min_breakout_range_atr,
        )
        if not candidates:
            return {"candidates": candidates}
        ctx.journal.write(
            "candidates", ts=state["snapshot"].ts, symbol=ctx.symbol,
            candidates=candidates, snapshot=state["snapshot"],
        )
        # Cheap deterministic vetoes (kill switch, time window, trade count).
        blockers = guardrails.entry_blockers(
            now=ctx.broker.now(),
            position=state.get("position"),
            trades_today=state.get("trades_today", 0),
            settings=settings,
            delayed_data=ctx.broker.is_delayed_data,
            internal_halt_reason=ctx.broker.entry_halt_reason,
        )
        if blockers:
            ctx.metrics["entry_blocker"] = ctx.metrics.get("entry_blocker", 0) + 1
            ctx.journal.write(
                "entry_pre_veto", ts=state["snapshot"].ts, symbol=ctx.symbol,
                candidates=candidates, violations=blockers,
            )
        return {"candidates": candidates, "entry_blockers": blockers}

    def llm_decide(state: AgentState) -> dict:
        # Deterministic entry: take the breakout candidate and use its stop
        # (the opposite side of the opening range). Node key kept for wiring.
        snapshot, candidates = state["snapshot"], state["candidates"]
        c = candidates[0]
        stop = c.stop_price
        if stop is None:  # defensive: fall back to a fixed buffer beyond the level
            buffer = settings.stop_buffer_cents / 100
            stop = c.level.price - buffer if c.direction == "call" else c.level.price + buffer
        decision = Decision(
            action=f"enter_{c.direction}", level_price=c.level.price, stop_price=round(stop, 2),
            confidence="medium", reasoning=f"rule-follower: {c.note}",
        )
        ctx.journal.write("entry_decision", ts=snapshot.ts, symbol=ctx.symbol, mode="entry", decision=decision)
        return {"decision": decision}

    def risk_gate(state: AgentState) -> dict:
        verdict = guardrails.check(
            state["decision"],
            now=ctx.broker.now(),
            position=state.get("position"),
            trades_today=state.get("trades_today", 0),
            candidates=state["candidates"],
            settings=settings,
            internal_halt_reason=ctx.broker.entry_halt_reason,
            delayed_data=ctx.broker.is_delayed_data,
            snapshot_price=state["snapshot"].price,
        )
        if not verdict.approved:
            ctx.metrics["risk_veto"] = ctx.metrics.get("risk_veto", 0) + 1
            if any("actual entry-to-stop risk" in v for v in verdict.violations):
                ctx.metrics["actual_risk"] = ctx.metrics.get("actual_risk", 0) + 1
            ctx.journal.write(
                "risk_veto", ts=state["snapshot"].ts, symbol=ctx.symbol,
                decision=state["decision"], violations=verdict.violations,
            )
        return {"risk": verdict}

    def do_entry(state: AgentState) -> dict:
        decision, snapshot = state["decision"], state["snapshot"]
        direction = "call" if decision.action == "enter_call" else "put"
        candidate = next(
            (
                c for c in state["candidates"]
                if c.direction == direction and decision.level_price is not None
                and abs(c.level.price - decision.level_price) <= guardrails.LEVEL_MATCH_TOL * c.level.price
            ),
            None,
        )
        try:
            position, action, skip = execute_entry(
                ctx.broker, settings, decision, direction, snapshot, candidate=candidate
            )
        except OrderFailed as exc:
            # An order reached IB, so the attempt consumes a trade — otherwise a
            # setup that keeps firing would loop a new order every minute (this
            # must happen inside the node: state changes are lost if the
            # exception escapes to the runner). Only OrderFailed is caught;
            # data/qualify/premium errors placed no order and keep retrying.
            ctx.journal.write(
                "entry_order_failed", ts=snapshot.ts, symbol=ctx.symbol,
                error=str(exc), decision=decision,
            )
            ctx.notifier.notify_status(f"{ctx.symbol} entry order FAILED: {exc}")
            print(f"[{ctx.symbol}] entry order failed ({exc})")
            return {"skip_reason": str(exc), "trades_today": state.get("trades_today", 0) + 1}
        if skip is not None:
            ctx.journal.write("entry_skipped", ts=snapshot.ts, symbol=ctx.symbol, reason=skip, decision=decision)
            return {"skip_reason": skip}
        ctx.journal.write("fill", ts=snapshot.ts, symbol=ctx.symbol, action=action, position=position)
        ctx.notifier.notify_fill(ctx.symbol, action, position)
        return {
            "actions": [action],
            "position": position,
            "trades_today": state.get("trades_today", 0) + 1,
        }

    return {
        "fetch_data": fetch_data,
        "compute_context": compute_context,
        "manage_position": manage_position,
        "llm_manage": llm_manage,
        "do_scale_out": do_scale_out,
        "do_exit": do_exit,
        "detect_setups": detect_setups,
        "llm_decide": llm_decide,
        "risk_gate": risk_gate,
        "do_entry": do_entry,
    }
