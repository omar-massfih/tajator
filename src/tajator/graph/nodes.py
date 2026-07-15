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
from ..llm.decide import build_llm, build_pattern_llm, decide_entry, decide_scale, format_snapshot
from ..llm.pattern_data import build_pattern_data, decide_pattern, validate_pattern_entry
from ..market.indicators import build_snapshot
from ..market.levels import detect_levels
from ..market.setups import detect_candidates
from ..market.timeframes import build_daily_context, build_five_minute_context, rank_candidates
from ..models import (
    Decision,
    ExecutedAction,
    MorningBriefing,
    MultiTimeframeContext,
    PatternAnalysis,
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
    use_llm: bool = True
    pattern_data: bool = False
    notifier: Notifier = field(default_factory=NullNotifier)
    _llm: Any = field(default=None, repr=False)
    _prep_llm: Any = field(default=None, repr=False)
    _pattern_llm: Any = field(default=None, repr=False)
    metrics: dict[str, int] = field(default_factory=dict)
    _daily_context_cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _last_pattern_bar_ts: Any = field(default=None, repr=False)

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

    @property
    def pattern_llm(self) -> Any:
        if self._pattern_llm is None:
            self._pattern_llm = build_pattern_llm(self.settings.llm_model)
        return self._pattern_llm


def make_nodes(ctx: RuntimeContext) -> dict[str, Any]:
    settings = ctx.settings.for_symbol(ctx.symbol)

    def fetch_data(state: AgentState) -> dict:
        bars = ctx.broker.get_bars(ctx.symbol)
        prev_high, prev_low = ctx.broker.get_prev_day_range(ctx.symbol)
        daily_bars = ctx.broker.get_daily_bars(ctx.symbol) if settings.multi_timeframe_context else []
        return {
            "bars": bars,
            "daily_bars": daily_bars,
            "prev_day_high": prev_high,
            "prev_day_low": prev_low,
        }

    def compute_context(state: AgentState) -> dict:
        bars = state["bars"]
        if not bars:
            raise RuntimeError("broker returned no bars (data farm down or market closed)")
        snapshot = build_snapshot(ctx.symbol, bars, atr_window=settings.atr_window_bars)
        if settings.multi_timeframe_context:
            cache_key = snapshot.ts.astimezone(guardrails.ET).date().isoformat()
            daily = ctx._daily_context_cache.get(cache_key)
            if daily is None:
                daily = build_daily_context(
                    state.get("daily_bars") or [], snapshot.ts, snapshot.price,
                )
                ctx._daily_context_cache = {cache_key: daily}
            multi = MultiTimeframeContext(
                enabled=True,
                daily=daily,
                five_minute=build_five_minute_context(bars),
            )
            snapshot = snapshot.model_copy(update={"multi_timeframe": multi})
        levels = detect_levels(
            bars, state.get("prev_day_high"), state.get("prev_day_low"),
            min_touch_separation=settings.double_min_touch_separation_bars,
            min_pullback_pct=settings.double_min_pullback_pct,
            swing_window=settings.swing_window_bars,
            cluster_tol=settings.level_cluster_tol_pct,
        )
        return {"snapshot": snapshot, "levels": levels}

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
        if ctx.pattern_data:
            bars = state["bars"]
            bar_ts = bars[-1].ts
            now_et = bar_ts.astimezone(guardrails.ET)
            session_minute = now_et.hour * 60 + now_et.minute - (9 * 60 + 30)
            due = (
                len(bars) >= settings.pattern_data_min_bars
                and session_minute >= 0
                and session_minute % settings.pattern_data_scan_interval_bars == 0
                and bar_ts != ctx._last_pattern_bar_ts
            )
            if not due:
                return {"candidates": [], "pattern_scan_due": False}
            ctx._last_pattern_bar_ts = bar_ts
            blockers = guardrails.entry_blockers(
                now=ctx.broker.now(),
                position=state.get("position"),
                trades_today=state.get("trades_today", 0),
                settings=settings,
                delayed_data=ctx.broker.is_delayed_data,
            )
            if blockers:
                ctx.metrics["entry_blocker"] = ctx.metrics.get("entry_blocker", 0) + 1
                ctx.journal.write(
                    "entry_pre_veto", ts=state["snapshot"].ts, symbol=ctx.symbol,
                    candidates=[], violations=blockers, source="pattern_data",
                )
            return {
                "candidates": [],
                "pattern_scan_due": True,
                "entry_blockers": blockers,
            }

        now_et = state["snapshot"].ts.astimezone(guardrails.ET).time()
        confirmation = settings.entry_confirmation
        if settings.opening_confirmation_until and now_et < settings.opening_confirmation_until:
            confirmation = "touch_rejection"
        candidates = detect_candidates(
            state["bars"], state["levels"], state["snapshot"],
            min_dist_from_open_pct=settings.min_level_dist_from_open_pct,
            approach_band=settings.approach_band_pct,
            overshoot_band=settings.overshoot_band_pct,
            speed_window=settings.speed_window_bars,
            min_speed_pct=settings.min_speed_pct,
            fast_approach_mult=settings.fast_approach_speed_mult,
            rejection_wick_frac=settings.rejection_wick_min_frac,
            reaction_lookback=settings.reaction_lookback_bars,
            long_wick_min_frac=settings.long_wick_min_frac,
            trade_flipped_levels=settings.trade_flipped_levels,
            entry_confirmation=confirmation,
        )
        candidates = rank_candidates(candidates, state["snapshot"].multi_timeframe)
        if candidates:
            ctx.journal.write(
                "candidate_features", ts=state["snapshot"].ts, symbol=ctx.symbol,
                candidates=candidates, regime=state["snapshot"].regime,
                atr=state["snapshot"].atr,
                multi_timeframe=state["snapshot"].multi_timeframe,
            )
        filtered: list[tuple[Any, str]] = []
        if settings.allowed_regimes:
            kept = []
            for c in candidates:
                if c.regime in settings.allowed_regimes:
                    kept.append(c)
                else:
                    filtered.append((c, f"regime {c.regime} not allowed"))
            candidates = kept
        if settings.blocked_direction_regimes:
            blocked = set(settings.blocked_direction_regimes)
            kept = []
            for c in candidates:
                key = f"{c.direction}:{c.regime}"
                if key in blocked:
                    filtered.append((c, f"direction/regime {key} blocked"))
                else:
                    kept.append(c)
            candidates = kept
        if settings.min_level_quality_score is not None:
            kept = []
            for c in candidates:
                if c.quality_score >= settings.min_level_quality_score:
                    kept.append(c)
                else:
                    filtered.append((c, f"quality {c.quality_score} below minimum"))
            candidates = kept
        if filtered:
            for _, reason in filtered:
                if reason.startswith("direction/regime"):
                    key = "direction_regime"
                else:
                    key = "regime" if reason.startswith("regime") else "quality"
                ctx.metrics[key] = ctx.metrics.get(key, 0) + 1
            ctx.journal.write(
                "strategy_filter_veto", ts=state["snapshot"].ts, symbol=ctx.symbol,
                dropped=[{"candidate": c, "reason": reason} for c, reason in filtered],
            )
        # Levels under a stop-out cooldown are dropped before the LLM ever
        # sees them — and since risk_gate only admits detected candidates,
        # the LLM cannot re-enter them either.
        candidates, cooled = guardrails.cooldown_filter(
            candidates, state.get("cooldown_levels") or []
        )
        if cooled:
            ctx.journal.write(
                "cooldown_veto", ts=state["snapshot"].ts, symbol=ctx.symbol,
                dropped=cooled, cooldown_levels=state.get("cooldown_levels"),
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
            delayed_data=ctx.broker.is_delayed_data,
        )
        if blockers:
            ctx.metrics["entry_blocker"] = ctx.metrics.get("entry_blocker", 0) + 1
            ctx.journal.write(
                "entry_pre_veto", ts=state["snapshot"].ts, symbol=ctx.symbol,
                candidates=candidates, violations=blockers,
            )
        return {"candidates": candidates, "entry_blockers": blockers}

    def llm_decide(state: AgentState) -> dict:
        snapshot, candidates = state["snapshot"], state["candidates"]
        update: dict[str, Any] = {}
        if ctx.pattern_data:
            pattern_data = build_pattern_data(
                ctx.symbol,
                state["bars"],
                limit=settings.pattern_data_lookback_bars,
            )
            context = (
                f"{ctx.symbol} at {snapshot.ts:%Y-%m-%d %H:%M} ET; "
                f"latest completed close {snapshot.price:.2f}; ATR "
                f"{snapshot.atr if snapshot.atr is not None else 'n/a'}. "
                "Classify only the completed numerical bars and pivots supplied below."
            )
            try:
                analysis = decide_pattern(ctx.pattern_llm, context, pattern_data)
            except Exception as exc:  # noqa: BLE001 - model construction must fail closed
                analysis = PatternAnalysis(
                    action="wait",
                    reasoning=f"pattern-data LLM unavailable, defaulting to wait: {exc}",
                )
            decision, candidate, violations = validate_pattern_entry(
                analysis, state["bars"], snapshot, settings,
            )
            pattern_candidates = [candidate] if candidate is not None else []
            if candidate is not None:
                pattern_candidates, cooled = guardrails.cooldown_filter(
                    pattern_candidates, state.get("cooldown_levels") or []
                )
                if cooled:
                    decision = Decision(
                        action="wait",
                        reasoning="pattern-data signal rejected: breakout level is under stop cooldown",
                    )
                    violations.append("breakout level is under stop cooldown")
            ctx.journal.write(
                "pattern_data_analysis",
                ts=snapshot.ts,
                symbol=ctx.symbol,
                analysis=analysis,
                validation_violations=violations,
                pattern_data={
                    "sha256": pattern_data.sha256,
                    "bar_count": pattern_data.bar_count,
                    "pivot_count": pattern_data.pivot_count,
                },
            )
            update["candidates"] = pattern_candidates
        elif not ctx.use_llm:
            c = candidates[0]
            buffer = settings.stop_buffer_cents / 100
            if settings.stop_atr_multiplier is not None and snapshot.atr is not None:
                buffer = min(
                    settings.stop_max_cents / 100,
                    max(settings.stop_min_cents / 100, snapshot.atr * settings.stop_atr_multiplier),
                )
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
        if (
            decision.action in ("enter_call", "enter_put")
            and decision.level_price is not None
            and settings.stop_atr_multiplier is not None
            and snapshot.atr is not None
        ):
            buffer = min(
                settings.stop_max_cents / 100,
                max(settings.stop_min_cents / 100, snapshot.atr * settings.stop_atr_multiplier),
            )
            stop = (
                decision.level_price - buffer
                if decision.action == "enter_call" else decision.level_price + buffer
            )
            decision = decision.model_copy(update={"stop_price": round(stop, 2)})
        ctx.journal.write("llm_decision", ts=snapshot.ts, symbol=ctx.symbol, mode="entry", decision=decision)
        return {"decision": decision, **update}

    def risk_gate(state: AgentState) -> dict:
        verdict = guardrails.check(
            state["decision"],
            now=ctx.broker.now(),
            position=state.get("position"),
            trades_today=state.get("trades_today", 0),
            candidates=state["candidates"],
            settings=settings,
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
