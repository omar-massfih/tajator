"""Causal underlying-only broker for long-window strategy research.

The option premium remains synthetic and must not be interpreted as PnL. Unlike
the plumbing-only StubBroker, fills occur at the next equity bar's open so the
underlying trade ledger does not receive same-bar execution.
"""

from __future__ import annotations

from datetime import time

from ..backtest.data import ET
from ..models import SelectedContract
from .base import Fill
from .stub import BASE_EXTRINSIC, StubBroker


class UnderlyingBacktestBroker(StubBroker):
    def _execution_reference(self) -> tuple[float, object]:
        current = self.bars[self.cursor]
        if self.cursor + 1 < len(self.bars):
            following = self.bars[self.cursor + 1]
            local = following.ts.astimezone(ET)
            if local.date() == current.ts.astimezone(ET).date() and local.time() <= time(16, 0):
                return following.open, following.ts
        return current.close, current.ts

    def _fill(self, contract: SelectedContract, qty: int, side: str) -> Fill:
        equity_price, ts = self._execution_reference()
        intrinsic = (
            equity_price - contract.strike
            if contract.right == "C" else contract.strike - equity_price
        )
        fill = Fill(
            premium=round(max(intrinsic, 0.0) + BASE_EXTRINSIC, 2),
            qty=qty,
            ts=ts,
            equity_price=equity_price,
        )
        self.fills.append((side, contract, fill))
        return fill

    def buy_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._fill(contract, qty, "BUY")

    def sell_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._fill(contract, qty, "SELL")
