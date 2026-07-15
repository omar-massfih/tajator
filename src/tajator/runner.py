"""TradingSession: owns durable state and drives the graph once per minute."""

from __future__ import annotations

import logging
import time as time_mod
from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from .broker.stub import StubBroker
from .graph.build import build_graph
from .graph.nodes import RuntimeContext
from .graph.state import AgentState
from .llm.decide import decide_prep, format_prep_snapshot, no_llm_briefing
from .market.indicators import build_snapshot
from .market.levels import detect_levels
from .market.timeframes import build_multi_timeframe_context
from .models import OpenPosition
from .state_store import PersistedSession, StateStore
from .trade.execution import execute_exit

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
FLATTEN_AT = time(15, 55)  # stop tick-managing and force-flatten before the illiquid closing minutes
PREP_TIME = time(9, 0)  # 30 min before RTH_OPEN

log = logging.getLogger(__name__)


def _sleep_to_next_minute() -> None:
    now = time_mod.time()
    time_mod.sleep(60 - (now % 60) + 2)  # +2s so the just-closed bar is available


def _sleep_until(target: datetime) -> None:
    while True:
        remaining = (target - datetime.now(ET)).total_seconds()
        if remaining <= 0:
            return
        time_mod.sleep(min(remaining, 60))


def _todays_prep_and_open(now: datetime) -> tuple[datetime, datetime]:
    day = now.astimezone(ET).date()
    prep_at = datetime.combine(day, PREP_TIME, tzinfo=ET)
    open_at = datetime.combine(day, RTH_OPEN, tzinfo=ET)
    return prep_at, open_at


def _next_session_prep(now: datetime) -> datetime:
    """Prep time of the next weekday session strictly after `now`."""
    day = now.astimezone(ET).date()
    prep_at = datetime.combine(day, PREP_TIME, tzinfo=ET)
    while prep_at <= now or prep_at.weekday() >= 5:
        day += timedelta(days=1)
        prep_at = datetime.combine(day, PREP_TIME, tzinfo=ET)
    return prep_at


class TradingSession:
    def __init__(
        self,
        ctx: RuntimeContext,
        store: StateStore | None = None,
        restored: PersistedSession | None = None,
        day: date | None = None,
    ):
        self.ctx = ctx
        self.graph = build_graph(ctx)
        self.store = store
        self.position: OpenPosition | None = restored.position if restored else None
        self.trades_today: int = restored.trades_today if restored else 0
        # restored state is valid for `day` — start_new_day must not wipe it
        self._day: date | None = day if restored else None
        # (level_price, stopped_at) of recent stop-outs; those levels are
        # untradable for STOP_COOLDOWN_MINUTES. Not persisted: a mid-day
        # restart forgets cooldowns, which only risks one extra entry.
        self._stop_cooldowns: list[tuple[float, datetime]] = []

    def start_new_day(self, day: date | None = None) -> None:
        """Reset per-day state once per calendar day; an overnight position
        stays and keeps being managed. Idempotent within a day so a same-day
        restart keeps its restored trade count."""
        day = day or self.ctx.broker.now().date()
        if self._day == day:
            return
        self._day = day
        self.trades_today = 0
        self._stop_cooldowns = []
        self._persist()

    def _active_cooldown_levels(self) -> list[float]:
        minutes = self.ctx.settings.stop_cooldown_minutes
        if minutes <= 0 or not self._stop_cooldowns:
            return []
        now = self.ctx.broker.now()
        self._stop_cooldowns = [
            (price, at) for price, at in self._stop_cooldowns
            if now < at + timedelta(minutes=minutes)
        ]
        return [price for price, _ in self._stop_cooldowns]

    def _persist(self) -> None:
        if self.store is None:  # replay/backtest/prep never persist
            return
        day = self._day or self.ctx.broker.now().date()
        self.store.update(self.ctx.symbol, self.position, self.trades_today, day)

    def tick(self) -> AgentState:
        state: AgentState = {
            "position": self.position,
            "trades_today": self.trades_today,
            "cooldown_levels": self._active_cooldown_levels(),
        }
        stopped_from = self.position  # the plan whose stop could fire this tick
        out = self.graph.invoke(state)
        self.position = out.get("position")
        self.trades_today = out.get("trades_today", self.trades_today)
        if stopped_from is not None and any(
            a.kind == "stop_exit" for a in out.get("actions", [])
        ):
            self._stop_cooldowns.append(
                (stopped_from.plan.level_price, self.ctx.broker.now())
            )
        self._persist()
        return out

    # -- live ------------------------------------------------------------------

    def _tick_once(self) -> None:
        try:
            if self.ctx.broker.ensure_connected():
                self.ctx.journal.write("broker_reconnected", symbol=self.ctx.symbol)
                self.ctx.notifier.notify_status(f"{self.ctx.symbol} — IB connection was lost, reconnected")
                print(f"[{self.ctx.symbol}] IB connection was lost — reconnected")
            out = self.tick()
        except Exception as exc:  # noqa: BLE001 — a bad tick must not kill the session
            log.exception("tick failed for %s", self.ctx.symbol)
            self.ctx.journal.write("error", symbol=self.ctx.symbol, error=str(exc))
            print(f"[{self.ctx.symbol}] tick failed ({exc}) — retrying next minute")
            return
        self._print_tick(out)

    def _print_tick(self, out: AgentState) -> None:
        snap = out.get("snapshot")
        if snap is None:
            return
        pieces = []
        if out.get("candidates"):
            pieces.append(f"{len(out['candidates'])} setup candidate(s)")
        if out.get("decision") is not None:
            pieces.append(f"LLM: {out['decision'].action}")
        for a in out.get("actions", []):
            pieces.append(f"FILL {a.kind} {a.qty}x @ {a.premium:.2f}")
        if self.position is not None:
            p = self.position
            pieces.append(f"position {p.qty_remaining}x {p.contract.local_name}")
        status = " | ".join(pieces) if pieces else "flat, nothing setting up"
        print(f"[{snap.ts:%H:%M}] {snap.symbol} {snap.price:.2f}  {status}")

    def prep(self) -> None:
        """One-shot pre-market prep: compute levels and (if enabled) an LLM briefing."""
        ctx = self.ctx
        bars = ctx.broker.get_bars(ctx.symbol)
        if not bars:
            log.warning("prep: no bars yet for %s — skipping", ctx.symbol)
            return
        prev_high, prev_low = ctx.broker.get_prev_day_range(ctx.symbol)
        levels = detect_levels(
            bars, prev_high, prev_low,
            min_touch_separation=ctx.settings.double_min_touch_separation_bars,
            min_pullback_pct=ctx.settings.double_min_pullback_pct,
            swing_window=ctx.settings.swing_window_bars,
            cluster_tol=ctx.settings.level_cluster_tol_pct,
        )
        snapshot = build_snapshot(ctx.symbol, bars)
        settings = ctx.settings.for_symbol(ctx.symbol)
        if settings.multi_timeframe_context:
            multi = build_multi_timeframe_context(
                bars, ctx.broker.get_daily_bars(ctx.symbol), snapshot.ts, snapshot.price,
            )
            snapshot = snapshot.model_copy(update={"multi_timeframe": multi})
        if ctx.use_llm:
            text = format_prep_snapshot(ctx.symbol, snapshot, levels)
            briefing = decide_prep(ctx.prep_llm, ctx.symbol, levels, text)
        else:
            briefing = no_llm_briefing(ctx.symbol, levels, "prep run with --no-llm")
        ctx.journal.write(
            "pre_market_prep", ts=snapshot.ts, symbol=ctx.symbol, levels=levels, briefing=briefing
        )
        self._print_prep(snapshot, briefing)

    def _print_prep(self, snapshot, briefing) -> None:
        print(f"\n=== {snapshot.symbol} pre-market prep @ {snapshot.ts:%H:%M} ET — price {snapshot.price:.2f} ===")
        for w in briefing.watch_levels:
            tag = "TRADABLE" if w.tradable else "reference"
            direction = f" {w.direction}" if w.direction else ""
            print(f"  {w.level.price:.2f}  {w.level.kind:<10} ({w.level.label})  [{tag}{direction}] {w.note}")
        print(f"  bias: {briefing.bias}  |  {briefing.summary}")

    def _flatten_position(self, kind: Literal["manual_exit"], reason: str) -> bool:
        """Force-close self.position via execute_exit; journals, notifies, persists.

        Returns True once the position is fully closed. On a failed cancel/sell
        (real IB errors only — StubBroker never raises), the position is left
        untouched so it stays managed on the next tick / next day."""
        p = self.position
        if p is None:
            return True
        try:
            bars = self.ctx.broker.get_bars(self.ctx.symbol, lookback_minutes=5)
            snap = build_snapshot(self.ctx.symbol, bars)
            actions = execute_exit(self.ctx.broker, self.ctx.settings, p, snap, kind, reason)
        except Exception as exc:  # noqa: BLE001 — a failed flatten must still be reported
            self.ctx.journal.write(
                "error", symbol=self.ctx.symbol, error=f"flatten failed: {exc}", position=p
            )
            self.ctx.notifier.notify_status(
                f"{self.ctx.symbol}: flatten FAILED ({exc}) — "
                f"{p.qty_remaining}x still open, close manually via IBKR"
            )
            print(f"[{self.ctx.symbol}] flatten failed ({exc}) — position left open, close it manually via IBKR.")
            return False
        for action in actions:
            self.ctx.journal.write("fill", ts=snap.ts, symbol=self.ctx.symbol, action=action, position=p)
            self.ctx.notifier.notify_fill(self.ctx.symbol, action, p)
        sold = sum(a.qty for a in actions)
        avg = sum(a.qty * a.premium for a in actions) / sold if sold else 0.0
        if p.qty_remaining:
            self.ctx.journal.write("interrupt_open_position", symbol=self.ctx.symbol, position=p)
            print(
                f"[{self.ctx.symbol}] flattened only {sold}x @ {avg:.2f} — "
                f"{p.qty_remaining}x still open, close it manually via IBKR."
            )
        else:
            self.position = None
            print(f"[{self.ctx.symbol}] flattened {sold}x @ {avg:.2f}")
        self._persist()
        return p.qty_remaining == 0

    def _on_interrupt(self) -> None:
        if self.position is None:
            print("\nstopped — flat.")
            return
        p = self.position
        try:
            answer = input(
                f"\nopen position: {p.qty_remaining}x {p.contract.local_name} — flatten now? [y/N] "
            )
        except (KeyboardInterrupt, EOFError):  # second Ctrl-C, or stdin not a TTY
            answer = ""
        if answer.strip().lower() == "y":
            self._flatten_position("manual_exit", "operator interrupt")
        else:
            self.ctx.journal.write("interrupt_open_position", symbol=self.ctx.symbol, position=p)
            if p.protective_stop is not None:
                print(
                    f"position left open — protective stop {p.protective_stop.order_id} stays "
                    f"working GTC at {p.protective_stop.stop_price} (or close manually via IBKR)."
                )
            else:
                print("position left open — close it manually via IBKR.")

    # -- replay ------------------------------------------------------------------

    def run_replay(self, broker: StubBroker, warmup_minutes: int = 10, verbose: bool = True) -> None:
        """Step the same graph through a recorded day, one bar at a time."""
        day = broker.bars[0].ts.astimezone(ET).date()
        start = datetime.combine(day, RTH_OPEN, tzinfo=ET)
        broker.seek(start)
        for _ in range(warmup_minutes):
            broker.advance()
        while True:
            now_et = broker.now().astimezone(ET)
            if now_et.time() >= RTH_CLOSE:
                break
            out = self.tick()
            if verbose:
                self._print_tick(out)
            if not broker.advance():
                break
        self._flatten_end_of_replay(broker, verbose)
        if verbose:
            self._replay_summary(broker)

    def _flatten_end_of_replay(self, broker: StubBroker, verbose: bool) -> None:
        """Force-close a position left at the end of a replayed day.

        Without this, backtest/replay PnL would silently exclude the trade's
        entire entry cost (the ledger only counts closed round-trips)."""
        if self.position is None:
            return
        bars = broker.get_bars(self.ctx.symbol, lookback_minutes=5)
        snap = build_snapshot(self.ctx.symbol, bars)
        actions = execute_exit(
            self.ctx.broker, self.ctx.settings, self.position, snap,
            "manual_exit", "end of replay day — forced flat",
        )
        for action in actions:
            self.ctx.journal.write(
                "fill", ts=snap.ts, symbol=self.ctx.symbol, action=action, position=self.position
            )
            self.ctx.notifier.notify_fill(self.ctx.symbol, action, self.position)
        self.position = None
        self._persist()
        if verbose and actions:
            print(f"end of day — flattened {actions[-1].qty}x @ {actions[-1].premium:.2f}")

    def _replay_summary(self, broker: StubBroker) -> None:
        print("\n--- replay summary ---")
        if not broker.fills:
            print("no trades taken.")
            return
        pnl = 0.0
        for side, contract, fill in broker.fills:
            sign = -1 if side == "BUY" else 1
            pnl += sign * fill.premium * fill.qty * 100
            print(f"{fill.ts:%H:%M}  {side:<4} {fill.qty}x {contract.local_name} @ {fill.premium:.2f}")
        open_qty = self.position.qty_remaining if self.position else 0
        note = f" ({open_qty} contracts still open, excluded)" if open_qty else ""
        print(f"synthetic realized PnL: ${pnl:,.0f}{note}")
        print("note: option fills are synthetic — this validates plumbing, not profitability.")


class LiveRunner:
    """Drives one TradingSession per symbol through the same once-a-minute cadence."""

    def __init__(self, sessions: list[TradingSession]):
        self.sessions = sessions

    def run(self) -> None:
        mode = self.sessions[0].ctx.settings.trading_mode.upper()
        decisions = (
            "PATTERN DATA" if self.sessions[0].ctx.pattern_data else
            "LLM" if self.sessions[0].ctx.use_llm else "DETERMINISTIC"
        )
        symbols = ", ".join(sess.ctx.symbol for sess in self.sessions)
        banner = f"=== tajator | {symbols} | {mode} | {decisions} ==="
        if mode == "LIVE":
            banner = f"\n{'!' * 60}\n!!! LIVE TRADING — REAL MONEY !!!\n{'!' * 60}\n" + banner
        print(banner)
        notifier = self.sessions[0].ctx.notifier
        notifier.notify_status(f"tajator started | {symbols} | {mode}")
        try:
            while True:
                self._run_one_day()
        except KeyboardInterrupt:
            for sess in self.sessions:
                sess._on_interrupt()
            notifier.notify_status(f"tajator stopped | {symbols} | {mode}")

    def _run_one_day(self) -> None:
        """Wait for the next session if closed; otherwise prep, tick until the close."""
        now = datetime.now(ET)
        prep_at, open_at = _todays_prep_and_open(now)
        close_at = datetime.combine(now.date(), RTH_CLOSE, tzinfo=ET)
        flatten_at = datetime.combine(now.date(), FLATTEN_AT, tzinfo=ET)
        if now.weekday() >= 5 or now >= close_at:
            next_prep = _next_session_prep(now)
            print(f"market closed — sleeping until prep {next_prep:%a %Y-%m-%d %H:%M} ET")
            _sleep_until(next_prep)
            return  # re-derive the new day's times on the next pass
        for sess in self.sessions:
            sess.start_new_day(now.date())
        if now < open_at:
            if now < prep_at:
                print(f"waiting for pre-market prep at {prep_at:%H:%M} ET ...")
                _sleep_until(prep_at)
            print("=== pre-market prep ===")
            for sess in self.sessions:
                try:  # prep is advisory — a failure must not keep the session from trading
                    sess.prep()
                except Exception as exc:  # noqa: BLE001
                    log.exception("prep failed for %s", sess.ctx.symbol)
                    sess.ctx.journal.write("error", symbol=sess.ctx.symbol, error=f"prep failed: {exc}")
                    print(f"[{sess.ctx.symbol}] prep failed ({exc}) — continuing without briefing")
        while datetime.now(ET) < flatten_at:
            _sleep_to_next_minute()
            for sess in self.sessions:
                sess._tick_once()
        self._on_session_close()

    def _on_session_close(self) -> None:
        """Force-flatten any position still open at the FLATTEN_AT cutoff — this
        strategy has no multi-day thesis, so nothing should carry overnight."""
        for sess in self.sessions:
            if sess.position is None:
                continue
            p = sess.position
            flattened = sess._flatten_position(
                "manual_exit", f"end of day ({FLATTEN_AT:%H:%M} ET) — forced flat"
            )
            if flattened:
                continue
            stop_note = (
                f" (protective stop {p.protective_stop.order_id} stays working GTC "
                f"at {p.protective_stop.stop_price})"
                if p.protective_stop is not None
                else ""
            )
            log.warning(
                "%s: EOD flatten failed, session closed with open position %dx %s%s — it will be "
                "managed again tomorrow; close it manually via IBKR if that is not intended",
                sess.ctx.symbol, p.qty_remaining, p.contract.local_name, stop_note,
            )
            print(
                f"!!! {sess.ctx.symbol}: EOD flatten failed, market closed with open position "
                f"{p.qty_remaining}x {p.contract.local_name}{stop_note} — close manually "
                "via IBKR or leave it to be managed tomorrow"
            )
            sess.ctx.journal.write("eod_open_position", symbol=sess.ctx.symbol, position=p)
