"""In-memory broker for replay and tests.

Serves bars from a preloaded day with a movable "now" cursor and fills
option orders instantly at a synthetic premium (intrinsic value plus a
flat extrinsic). All trade decisions are made on the equity chart, so a
crude option model is fine for plumbing tests — it is NOT a backtester.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ..models import Bar, SelectedContract
from .base import Broker, ChainParams, Fill

ET = ZoneInfo("America/New_York")
BASE_EXTRINSIC = 1.50  # flat synthetic time value per contract


class StubBroker(Broker):
    def __init__(
        self,
        bars: list[Bar],
        prev_day_high: float | None = None,
        prev_day_low: float | None = None,
        chain: ChainParams | None = None,
    ):
        self.bars = sorted(bars, key=lambda b: b.ts)
        self.prev_day_high = prev_day_high
        self.prev_day_low = prev_day_low
        self._chain = chain or self._default_chain()
        self.cursor = 0  # index of the latest visible bar
        self.fills: list[tuple[str, SelectedContract, Fill]] = []

    @classmethod
    def from_csv(
        cls, path: Path, prev_day_high: float | None = None, prev_day_low: float | None = None
    ) -> "StubBroker":
        bars = []
        with open(path) as f:
            for row in csv.DictReader(f):
                bars.append(
                    Bar(
                        ts=datetime.fromisoformat(row["ts"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
        return cls(bars, prev_day_high, prev_day_low)

    def _default_chain(self) -> ChainParams:
        if not self.bars:
            return ChainParams(expirations=[], strikes=[])
        spot = self.bars[0].close
        strikes = [round(spot) + d * 0.5 for d in range(-20, 21)]
        day = self.bars[0].ts.astimezone(ET).date()
        friday = day.toordinal() + (4 - day.weekday()) % 7
        this_friday = datetime.fromordinal(friday).strftime("%Y%m%d")
        next_friday = datetime.fromordinal(friday + 7).strftime("%Y%m%d")
        return ChainParams(expirations=[this_friday, next_friday], strikes=strikes)

    # -- cursor control (replay driver) --------------------------------------
    def advance(self) -> bool:
        """Move to the next bar; False when the day is exhausted."""
        if self.cursor + 1 >= len(self.bars):
            return False
        self.cursor += 1
        return True

    def seek(self, ts: datetime) -> None:
        for i, b in enumerate(self.bars):
            if b.ts >= ts:
                self.cursor = i
                return
        self.cursor = len(self.bars) - 1

    # -- Broker interface -----------------------------------------------------
    def now(self) -> datetime:
        return self.bars[self.cursor].ts

    def spot(self) -> float:
        return self.bars[self.cursor].close

    def get_bars(self, symbol: str, lookback_minutes: int = 390) -> list[Bar]:
        start = max(0, self.cursor + 1 - lookback_minutes)
        return self.bars[start : self.cursor + 1]

    def get_prev_day_range(self, symbol: str) -> tuple[float | None, float | None]:
        return self.prev_day_high, self.prev_day_low

    def get_option_chain(self, symbol: str) -> ChainParams:
        return self._chain

    def get_option_premium(self, contract: SelectedContract) -> float | None:
        spot = self.spot()
        intrinsic = spot - contract.strike if contract.right == "C" else contract.strike - spot
        return round(max(intrinsic, 0.0) + BASE_EXTRINSIC, 2)

    def buy_option(self, contract: SelectedContract, qty: int) -> Fill:
        fill = Fill(premium=self.get_option_premium(contract), qty=qty, ts=self.now())
        self.fills.append(("BUY", contract, fill))
        return fill

    def sell_option(self, contract: SelectedContract, qty: int) -> Fill:
        fill = Fill(premium=self.get_option_premium(contract), qty=qty, ts=self.now())
        self.fills.append(("SELL", contract, fill))
        return fill
