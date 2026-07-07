"""Backtest broker: real historical option fills only — no synthetic pricing.

Subclasses StubBroker (same cursor/bar-serving/chain logic) and overrides only
option pricing: real historical 1-min option bars, fetched (and cached) via IB.
If IB has no historical data for a specific contract/day (illiquid strike, or
an expired contract that fails to qualify), pricing fails loudly instead of
silently substituting a synthetic estimate — mixing real and guessed prices
into the same backtest would make its PnL numbers meaningless.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..backtest.data import ET, ensure_option_bars
from ..models import Bar, SelectedContract
from .stub import StubBroker


class BacktestBroker(StubBroker):
    def __init__(
        self,
        bars: list[Bar],
        prev_day_high: float | None = None,
        prev_day_low: float | None = None,
        *,
        ib=None,
        cache_dir: Path | None = None,
    ):
        super().__init__(bars, prev_day_high, prev_day_low)
        self._ib = ib
        self._cache_dir = cache_dir
        self._option_series: dict[tuple, list[Bar] | None] = {}

    def _series_for(self, contract: SelectedContract) -> list[Bar] | None:
        day = self.now().astimezone(ET).date()
        key = (contract.symbol, contract.expiry, contract.strike, contract.right, day)
        if key not in self._option_series:
            self._option_series[key] = ensure_option_bars(self._ib, contract, day, self._cache_dir)
        return self._option_series[key]

    @staticmethod
    def _bar_at_or_before(series: list[Bar], ts: datetime) -> Bar | None:
        candidate = None
        for bar in series:
            if bar.ts > ts:
                break
            candidate = bar
        return candidate

    def get_option_premium(self, contract: SelectedContract) -> float | None:
        series = self._series_for(contract)
        bar = self._bar_at_or_before(series, self.now()) if series else None
        if bar is None:
            day = self.now().astimezone(ET).date()
            raise RuntimeError(
                f"no historical option data for {contract.local_name} on {day} — "
                "cannot price this fill from real data, aborting backtest"
            )
        return round((bar.high + bar.low) / 2, 2)
