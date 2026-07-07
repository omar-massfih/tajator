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
from .models import OpenPosition
from .trade.execution import execute_exit

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)

log = logging.getLogger(__name__)


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

    def run_live(self) -> None:
        mode = self.ctx.settings.trading_mode.upper()
        banner = f"=== tajator | {self.ctx.settings.symbol} | {mode} ==="
        if mode == "LIVE":
            banner = f"\n{'!' * 60}\n!!! LIVE TRADING — REAL MONEY !!!\n{'!' * 60}\n" + banner
        print(banner)
        try:
            while True:
                self._sleep_to_next_minute()
                out = self.tick()
                self._print_tick(out)
        except KeyboardInterrupt:
            self._on_interrupt()

    def _sleep_to_next_minute(self) -> None:
        now = time_mod.time()
        time_mod.sleep(60 - (now % 60) + 2)  # +2s so the just-closed bar is available

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

    def _on_interrupt(self) -> None:
        if self.position is None:
            print("\nstopped — flat.")
            return
        p = self.position
        answer = input(
            f"\nopen position: {p.qty_remaining}x {p.contract.local_name} — flatten now? [y/N] "
        )
        if answer.strip().lower() == "y":
            bars = self.ctx.broker.get_bars(self.ctx.settings.symbol, lookback_minutes=5)
            from .market.indicators import build_snapshot

            snap = build_snapshot(self.ctx.settings.symbol, bars)
            action = execute_exit(self.ctx.broker, p, snap, "manual_exit", "operator interrupt")
            self.ctx.journal.write("fill", ts=snap.ts, action=action, position=p)
            self.position = None
            print(f"flattened {action.qty}x @ {action.premium:.2f}")
        else:
            print("position left open — manage it in TWS.")

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
