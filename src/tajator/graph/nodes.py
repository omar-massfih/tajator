"""Graph nodes. Each is a closure over the RuntimeContext (broker, LLM, journal, settings).

The LLM appears in exactly two nodes (llm_decide, llm_manage); everything
else is deterministic. With use_llm=False a rule-follower stands in for the
LLM — used by `replay --no-llm` and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..broker.base import Broker, OrderFailed
from ..config import Settings
from ..journal import Journal
from ..llm.decide import build_llm, decide_entry, decide_scale, format_snapshot
from ..market.indicators import build_snapshot
from ..market.levels import detect_levels
from ..market.setups import detect_candidates
from ..models import Decision, MorningBriefing
from ..notify import Notifier, NullNotifier
from ..risk import guardrails
from ..trade import position as pos
from ..trade.execution import execute_entry, execute_exit, execute_scale_out
from .state import AgentState


@dataclass
class RuntimeContext:
    settings: Settings
    broker: Broker
    journal: Journal
    symbol: str
    use_llm: bool = True
    notifier: Notifier = field(default_factory=NullNotifier)
    _llm: Any = field(default=None, repr=False)
    _prep_llm: Any = field(default=None, repr=False)

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = build_llm(self.settings.llm_model)
        return self._llm

    @property
    def prep_llm(self) -> Any:
        if self._prep_llm is None:
            self._prep_llm = build_llm(self.settings.llm_model, output_model=MorningBriefing)
        return self._prep_llm


def make_nodes(ctx: RuntimeContext) -> dict[str, Any]:
    settings = ctx.settings

    def fetch_data(state: AgentState) -> dict:
        bars = ctx.broker.get_bars(ctx.symbol)
        prev_high, prev_low = ctx.broker.get_prev_day_range(ctx.symbol)
        return {"bars": bars, "prev_day_high": prev_high, "prev_day_low": prev_low}

    def compute_context(state: AgentState) -> dict:
        bars = state["bars"]
        if not bars:
            raise RuntimeError("broker returned no bars (data farm down or market closed)")
        snapshot = build_snapshot(ctx.symbol, bars)
        levels = detect_levels(
            bars, state.get("prev_day_high"), state.get("prev_day_low"),
            min_touch_separation=settings.double_min_touch_separation_bars,
            min_pullback_pct=settings.double_min_pullback_pct,
        )
        return {"snapshot": snapshot, "levels": levels}

    # ----- position-open branch ---------------------------------------------

    def manage_position(state: AgentState) -> dict:
        position, snapshot = state["position"], state["snapshot"]
        pos.update_extreme(position, snapshot.price)
        action = pos.evaluate(position, snapshot)
        if action.kind != "hold":
            ctx.journal.write("manage_signal", ts=snapshot.ts, symbol=ctx.symbol, action=action)
        return {"manage_action": action}

    def llm_manage(state: AgentState) -> dict:
        snapshot, action = state["snapshot"], state["manage_action"]
        if not ctx.use_llm:
            decision = Decision(action="scale_out", reasoning=f"rule-follower: {action.reason}")
        else:
            text = format_snapshot(
                state["bars"], snapshot, state["levels"], [],
                state.get("trades_today", 0), position=state["position"],
                manage_note=f"{action.reason}. Scale out this piece now, or hold one more bar?",
            )
            try:
                decision = decide_scale(ctx.llm, text)
            except Exception as exc:  # noqa: BLE001 — e.g. missing API key at LLM construction
                decision = Decision(action="scale_out", reasoning=f"LLM unavailable ({exc}); scaling by default")
        ctx.journal.write("llm_decision", ts=snapshot.ts, symbol=ctx.symbol, mode="manage", decision=decision)
        return {"decision": decision}

    def do_scale_out(state: AgentState) -> dict:
        position, snapshot = state["position"], state["snapshot"]
        reason = state["manage_action"].reason
        action = execute_scale_out(ctx.broker, position, snapshot, reason)
        ctx.journal.write("fill", ts=snapshot.ts, symbol=ctx.symbol, action=action, position=position)
        ctx.notifier.notify_fill(ctx.symbol, action, position)
        closed = position.qty_remaining == 0
        return {"actions": [action], "position": None if closed else position}

    def do_exit(state: AgentState) -> dict:
        position, snapshot = state["position"], state["snapshot"]
        manage = state.get("manage_action")
        if manage is not None and manage.kind in ("stop_exit", "runner_exit"):
            kind, reason = manage.kind, manage.reason
        else:
            kind, reason = "manual_exit", state["decision"].reasoning
        action = execute_exit(ctx.broker, position, snapshot, kind, reason)
        ctx.journal.write("fill", ts=snapshot.ts, symbol=ctx.symbol, action=action, position=position)
        ctx.notifier.notify_fill(ctx.symbol, action, position)
        # a partial exit leaves contracts to retry on the next tick
        return {"actions": [action], "position": position if position.qty_remaining else None}

    # ----- flat branch --------------------------------------------------------

    def detect_setups(state: AgentState) -> dict:
        candidates = detect_candidates(
            state["bars"], state["levels"], state["snapshot"],
            min_dist_from_open_pct=settings.min_level_dist_from_open_pct,
        )
        if not candidates:
            return {"candidates": candidates}
        ctx.journal.write(
            "candidates", ts=state["snapshot"].ts, symbol=ctx.symbol,
            candidates=candidates, snapshot=state["snapshot"],
        )
        # Cheap deterministic vetoes (kill switch, time window, trade count)
        # before paying for an LLM call that risk_gate would reject anyway.
        blockers = guardrails.entry_blockers(
            now=ctx.broker.now(),
            position=state.get("position"),
            trades_today=state.get("trades_today", 0),
            settings=settings,
        )
        if blockers:
            ctx.journal.write(
                "entry_pre_veto", ts=state["snapshot"].ts, symbol=ctx.symbol,
                candidates=candidates, violations=blockers,
            )
        return {"candidates": candidates, "entry_blockers": blockers}

    def llm_decide(state: AgentState) -> dict:
        snapshot, candidates = state["snapshot"], state["candidates"]
        if not ctx.use_llm:
            c = candidates[0]
            buffer = settings.stop_buffer_cents / 100
            stop = c.level.price - buffer if c.direction == "call" else c.level.price + buffer
            decision = Decision(
                action=f"enter_{c.direction}", level_price=c.level.price, stop_price=round(stop, 2),
                confidence="medium", reasoning=f"rule-follower: {c.note}",
            )
        else:
            text = format_snapshot(
                state["bars"], snapshot, state["levels"], candidates,
                state.get("trades_today", 0),
            )
            try:
                decision = decide_entry(ctx.llm, text)
            except Exception as exc:  # noqa: BLE001 — e.g. missing API key at LLM construction
                decision = Decision(action="wait", reasoning=f"LLM unavailable ({exc}); waiting")
        ctx.journal.write("llm_decision", ts=snapshot.ts, symbol=ctx.symbol, mode="entry", decision=decision)
        return {"decision": decision}

    def risk_gate(state: AgentState) -> dict:
        verdict = guardrails.check(
            state["decision"],
            now=ctx.broker.now(),
            position=state.get("position"),
            trades_today=state.get("trades_today", 0),
            candidates=state["candidates"],
            settings=settings,
        )
        if not verdict.approved:
            ctx.journal.write(
                "risk_veto", ts=state["snapshot"].ts, symbol=ctx.symbol,
                decision=state["decision"], violations=verdict.violations,
            )
        return {"risk": verdict}

    def do_entry(state: AgentState) -> dict:
        decision, snapshot = state["decision"], state["snapshot"]
        direction = "call" if decision.action == "enter_call" else "put"
        try:
            position, action, skip = execute_entry(ctx.broker, settings, decision, direction, snapshot)
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
