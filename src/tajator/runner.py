"""TradingSession: owns durable state and drives the graph once per minute."""

from __future__ import annotations

import logging
import time as time_mod
from datetime import datetime, time
from zoneinfo import ZoneInfo

from .broker.stub import StubBroker
from .graph.build import build_graph
from .graph.nodes import RuntimeContext
from .graph.state import AgentState
from .llm.decide import decide_prep, format_prep_snapshot, no_llm_briefing
from .market.indicators import build_snapshot
from .market.levels import detect_levels
from .models import OpenPosition
from .trade.execution import execute_exit

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
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


class TradingSession:
    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx
        self.graph = build_graph(ctx)
        self.position: OpenPosition | None = None
        self.trades_today: int = 0

    def tick(self) -> AgentState:
        state: AgentState = {"position": self.position, "trades_today": self.trades_today}
        out = self.graph.invoke(state)
        self.position = out.get("position")
        self.trades_today = out.get("trades_today", self.trades_today)
        return out

    # -- live ------------------------------------------------------------------

    def _tick_once(self) -> None:
        try:
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
        levels = detect_levels(bars, prev_high, prev_low)
        snapshot = build_snapshot(ctx.symbol, bars)
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

    def _on_interrupt(self) -> None:
        if self.position is None:
            print("\nstopped — flat.")
            return
        p = self.position
        answer = input(
            f"\nopen position: {p.qty_remaining}x {p.contract.local_name} — flatten now? [y/N] "
        )
        if answer.strip().lower() == "y":
            bars = self.ctx.broker.get_bars(self.ctx.symbol, lookback_minutes=5)
            from .market.indicators import build_snapshot

            snap = build_snapshot(self.ctx.symbol, bars)
            action = execute_exit(self.ctx.broker, p, snap, "manual_exit", "operator interrupt")
            self.ctx.journal.write("fill", ts=snap.ts, symbol=self.ctx.symbol, action=action, position=p)
            self.position = None
            print(f"flattened {action.qty}x @ {action.premium:.2f}")
        else:
            print("position left open — close it manually via IBKR.")

    # -- replay ------------------------------------------------------------------

    def run_replay(self, broker: StubBroker, warmup_minutes: int = 10) -> None:
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
            self._print_tick(out)
            if not broker.advance():
                break
        self._replay_summary(broker)

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
        symbols = ", ".join(sess.ctx.symbol for sess in self.sessions)
        banner = f"=== tajator | {symbols} | {mode} ==="
        if mode == "LIVE":
            banner = f"\n{'!' * 60}\n!!! LIVE TRADING — REAL MONEY !!!\n{'!' * 60}\n" + banner
        print(banner)
        now = datetime.now(ET)
        prep_at, open_at = _todays_prep_and_open(now)
        if now < open_at:
            if now < prep_at:
                print(f"waiting for pre-market prep at {prep_at:%H:%M} ET ...")
                _sleep_until(prep_at)
            print("=== pre-market prep ===")
            for sess in self.sessions:
                sess.prep()
        try:
            while True:
                _sleep_to_next_minute()
                for sess in self.sessions:
                    sess._tick_once()
        except KeyboardInterrupt:
            for sess in self.sessions:
                sess._on_interrupt()
