"""Backtest broker: real historical option fills only — no synthetic pricing.

Subclasses StubBroker (same cursor/bar-serving/chain logic) and overrides only
option pricing: real historical 1-min option bars, fetched (and cached) via IB.
If IB has no historical data for a specific contract/day (illiquid strike, or
an expired contract that fails to qualify), pricing fails loudly instead of
silently substituting a synthetic estimate — mixing real and guessed prices
into the same backtest would make its PnL numbers meaningless.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from ..backtest.data import ET, ensure_option_bars
from ..models import Bar, SelectedContract
from .base import ChainParams, Fill
from .stub import StubBroker

# A fill priced from an option bar further than this from the decision time is
# treated as no data: better to abort than to fill off a stale/illiquid print.
MAX_FILL_STALENESS = timedelta(minutes=5)


class BacktestBroker(StubBroker):
    def __init__(
        self,
        bars: list[Bar],
        prev_day_high: float | None = None,
        prev_day_low: float | None = None,
        *,
        ib=None,
        cache_dir: Path | None = None,
        half_spread_pct: float = 0.0,
        slippage_cents: float = 0.0,
        commission_per_contract: float = 0.0,
        min_commission_per_order: float = 0.0,
    ):
        super().__init__(bars, prev_day_high, prev_day_low)
        self._ib = ib
        self._cache_dir = cache_dir
        self._option_series: dict[tuple, list[Bar] | None] = {}
        self.half_spread_pct = half_spread_pct
        self.slippage_cents = slippage_cents
        self.commission_per_contract = commission_per_contract
        self.min_commission_per_order = min_commission_per_order

    def get_option_chain(self, symbol: str) -> ChainParams:
        """Real listed strikes from IB, but expirations generated mechanically
        for the *backtest* day.

        StubBroker's default chain uses a half-dollar strike grid; for a symbol
        like SPY (whole-dollar strikes) that makes the selector pick contracts
        that were never listed, so every fill fails to price and the backtest
        aborts. The real strike set comes from IB's chain (cached by day inside
        IBBroker). Expirations still have to be synthesized — today's chain no
        longer lists the historical expirations a past session traded — so we
        keep StubBroker's near-Friday logic anchored on the backtest day."""
        if self._ib is None:
            return super().get_option_chain(symbol)
        strikes = self._ib.get_option_chain(symbol).strikes or self._chain.strikes
        day = self.now().astimezone(ET).date()
        friday = day.toordinal() + (4 - day.weekday()) % 7
        this_friday = datetime.fromordinal(friday).strftime("%Y%m%d")
        next_friday = datetime.fromordinal(friday + 7).strftime("%Y%m%d")
        return ChainParams(expirations=[this_friday, next_friday], strikes=sorted(strikes))

    def _series_for(self, contract: SelectedContract) -> list[Bar] | None:
        day = self.now().astimezone(ET).date()
        key = (contract.symbol, contract.expiry, contract.strike, contract.right, day)
        if key not in self._option_series:
            self._option_series[key] = ensure_option_bars(self._ib, contract, day, self._cache_dir)
        return self._option_series[key]

    @staticmethod
    def _fill_price(series: list[Bar], ts: datetime) -> float | None:
        """Price a fill decided at `ts` without look-ahead.

        The decision was made on the close of the bar ending at `ts`, so the
        order can only fill afterwards: use the NEXT option bar's open. At the
        end of the series (e.g. the day's forced flatten) fall back to the
        decision bar's close. Anything further than MAX_FILL_STALENESS from
        `ts` counts as no data."""
        prev = None
        for bar in series:
            if bar.ts > ts:
                if bar.ts - ts <= MAX_FILL_STALENESS:
                    return bar.open
                break
            prev = bar
        if prev is not None and ts - prev.ts <= MAX_FILL_STALENESS:
            return prev.close
        return None

    def get_option_premium(self, contract: SelectedContract) -> float | None:
        series = self._series_for(contract)
        price = self._fill_price(series, self.now()) if series else None
        if price is None:
            day = self.now().astimezone(ET).date()
            raise RuntimeError(
                f"no historical option data for {contract.local_name} on {day} — "
                "cannot price this fill from real data, aborting backtest"
            )
        return round(price, 2)

    def _execution_fill(self, contract: SelectedContract, qty: int, side: str) -> Fill:
        reference = self.get_option_premium(contract)
        adverse = reference * self.half_spread_pct + self.slippage_cents / 100
        premium = reference + adverse if side == "BUY" else max(0.01, reference - adverse)
        fee = max(self.min_commission_per_order, qty * self.commission_per_contract)
        fill = Fill(premium=round(premium, 2), qty=qty, ts=self.now(), fee=round(fee, 2))
        self.fills.append((side, contract, fill))
        return fill

    def buy_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._execution_fill(contract, qty, "BUY")

    def sell_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._execution_fill(contract, qty, "SELL")
