"""No-order broker using executable TWS quotes for deterministic shadow trading."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from ..config import Settings
from ..journal import Journal
from ..models import (
    Direction,
    ExecutionQuality,
    OptionQuote,
    ProtectiveStop,
    SelectedContract,
)
from .base import Broker, ChainParams, Fill, StopCancelResult, StopStatus


class ShadowBroker(Broker):
    """Delegates market data to IBBroker but never calls an order API."""

    def __init__(self, market_data_broker, settings: Settings, journal: Journal):
        self.market = market_data_broker
        self.settings = settings
        self.journal = journal
        self.fills: list[tuple[str, SelectedContract, Fill]] = []
        self._stops: dict[int, ProtectiveStop] = {}
        self._next_stop_id = 1
        self._covered_sessions: set[tuple[str, object]] = set()

    @property
    def uses_live_execution_guards(self) -> bool:
        return True

    def ensure_connected(self) -> bool:
        return self.market.ensure_connected()

    def now(self) -> datetime:
        return self.market.now()

    def get_bars(self, symbol: str, lookback_minutes: int = 390):
        bars = self.market.get_bars(symbol, lookback_minutes)
        # A started process is not evidence of a completed observation day.
        # Mark coverage only after the graph has actually seen nearly the full
        # regular session; shadow-report uses this as its coverage denominator.
        et = ZoneInfo("America/New_York")
        rth = [
            bar for bar in bars
            if time(9, 30) <= bar.ts.astimezone(et).time() <= time(16, 0)
        ]
        if rth:
            day = rth[-1].ts.astimezone(et).date()
            key = (symbol.upper(), day)
            if (
                key not in self._covered_sessions
                and rth[0].ts.astimezone(et).time() <= time(9, 31)
                and rth[-1].ts.astimezone(et).time() >= time(15, 54)
                and len(rth) >= 380
            ):
                self._covered_sessions.add(key)
                self.journal.write(
                    "shadow_session_covered", ts=rth[-1].ts, symbol=symbol.upper(),
                    regular_bars=len(rth), no_order_placed=True,
                )
        return bars

    def get_prev_day_range(self, symbol: str):
        return self.market.get_prev_day_range(symbol)

    def get_daily_bars(self, symbol: str, lookback_days: int = 90):
        return self.market.get_daily_bars(symbol, lookback_days)

    def get_option_chain(self, symbol: str) -> ChainParams:
        return self.market.get_option_chain(symbol)

    def get_option_premium(self, contract: SelectedContract) -> float | None:
        return self.market.get_option_premium(contract)

    def get_option_quote(self, contract: SelectedContract) -> OptionQuote:
        return self.market.get_option_quote(contract)

    def get_underlying_price(self, symbol: str) -> float | None:
        return self.market.get_underlying_price(symbol)

    def record_execution_preflight(self, **payload: object) -> None:
        self.journal.write("shadow_execution_preflight", **payload)

    @property
    def is_delayed_data(self) -> bool:
        return self.market.is_delayed_data

    def _fill(self, contract: SelectedContract, qty: int, side: str) -> Fill:
        submitted = self.now()
        quote = self.get_option_quote(contract)
        if not quote.valid or quote.delayed:
            raise RuntimeError(f"shadow {side} has no executable bid/ask for {contract.local_name}")
        premium = quote.ask if side == "BUY" else quote.bid
        underlying = self.get_underlying_price(contract.symbol)
        filled_at = self.now()
        fee = max(
            self.settings.backtest_min_commission_per_order,
            qty * self.settings.backtest_commission_per_contract,
        )
        quality = ExecutionQuality(
            side=side,
            order_type="SHADOW_BID_ASK",
            status="Simulated",
            requested_qty=qty,
            filled_qty=qty,
            submitted_at=submitted,
            filled_at=filled_at,
            quote=quote,
            underlying_submit=underlying,
            underlying_fill=underlying,
            fill_price=premium,
            latency_s=max(0.0, (filled_at - submitted).total_seconds()),
            reference_price=premium,
            slippage=0.0,
            slippage_pct=0.0,
        )
        fill = Fill(
            premium=round(premium, 2), qty=qty, ts=filled_at,
            fee=round(fee, 2), equity_price=underlying,
            execution_quality=quality,
        )
        self.fills.append((side, contract, fill))
        self.journal.write(
            "shadow_execution",
            ts=filled_at,
            side=side,
            contract=contract,
            fill=fill,
            quote=quote,
            no_order_placed=True,
        )
        return fill

    def buy_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._fill(contract, qty, "BUY")

    def sell_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._fill(contract, qty, "SELL")

    def place_protective_stop(
        self,
        contract: SelectedContract,
        qty: int,
        stop_price: float,
        direction: Direction,
        order_ref: str,
    ) -> ProtectiveStop:
        stop = ProtectiveStop(
            order_id=-self._next_stop_id,
            order_ref=f"shadow:{order_ref}",
            qty=qty,
            stop_price=stop_price,
        )
        self._next_stop_id += 1
        self._stops[stop.order_id] = stop
        return stop

    def cancel_protective_stop(
        self,
        contract: SelectedContract,
        stop: ProtectiveStop,
        expected_held: int | None = None,
    ) -> StopCancelResult:
        self._stops.pop(stop.order_id, None)
        return StopCancelResult(cancelled=True)

    def poll_protective_stop(
        self, contract: SelectedContract, stop: ProtectiveStop
    ) -> StopStatus:
        return (
            StopStatus(state="working", working_qty=stop.qty)
            if stop.order_id in self._stops else StopStatus(state="gone")
        )
