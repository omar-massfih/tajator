"""Option contract selection: nearest strike to spot, near (not same-day) expiry."""

from __future__ import annotations

from datetime import date, datetime

from ..broker.base import ChainParams
from ..models import Direction, SelectedContract


def select_contract(
    chain: ChainParams, symbol: str, spot: float, direction: Direction, now: datetime
) -> SelectedContract | None:
    if not chain.strikes or not chain.expirations:
        return None
    strike = min(chain.strikes, key=lambda s: abs(s - spot))
    expiry = _select_expiry(chain.expirations, now.date())
    if expiry is None:
        return None
    return SelectedContract(
        symbol=symbol, expiry=expiry, strike=strike, right="C" if direction == "call" else "P"
    )


def _select_expiry(expirations: list[str], today: date) -> str | None:
    """Earliest expiry with at least one full day left — never 0DTE in v1."""
    for exp in sorted(expirations):
        if datetime.strptime(exp, "%Y%m%d").date() > today:
            return exp
    return None
