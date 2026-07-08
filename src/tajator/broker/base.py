"""Broker protocol — the rest of the system never imports a broker SDK directly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel

from ..models import Bar, SelectedContract


class ChainParams(BaseModel):
    expirations: list[str]  # YYYYMMDD, sorted ascending
    strikes: list[float]  # sorted ascending


class Fill(BaseModel):
    premium: float  # per-contract option price
    qty: int  # contracts actually filled — may be less than requested
    ts: datetime


class BrokerOptionPosition(BaseModel):
    """An option position as reported by the broker account."""

    symbol: str
    expiry: str  # YYYYMMDD
    strike: float
    right: str  # "C" | "P"
    con_id: int
    local_symbol: str
    qty: int
    avg_cost: float


class OrderFailed(RuntimeError):
    """A placed order ended in a terminal non-Filled state.

    `filled` is the reconciled contract count (0 when nothing executed);
    `suspect` is True when that count could not be confirmed against both
    execution reports and the account's position."""

    def __init__(
        self, message: str, *, side: str, requested: int, filled: int, suspect: bool
    ):
        super().__init__(message)
        self.side = side
        self.requested = requested
        self.filled = filled
        self.suspect = suspect


class Broker(ABC):
    def ensure_connected(self) -> bool:
        """Reconnect if the underlying session dropped. Returns True if a
        reconnect happened; no-op (False) for in-memory brokers."""
        return False

    @abstractmethod
    def now(self) -> datetime: ...

    @abstractmethod
    def get_bars(self, symbol: str, lookback_minutes: int = 390) -> list[Bar]:
        """Recent 1-min bars including premarket (useRTH=False)."""

    @abstractmethod
    def get_prev_day_range(self, symbol: str) -> tuple[float | None, float | None]:
        """(high, low) of the previous regular session."""

    @abstractmethod
    def get_option_chain(self, symbol: str) -> ChainParams: ...

    @abstractmethod
    def get_option_premium(self, contract: SelectedContract) -> float | None: ...

    @abstractmethod
    def buy_option(self, contract: SelectedContract, qty: int) -> Fill: ...

    @abstractmethod
    def sell_option(self, contract: SelectedContract, qty: int) -> Fill: ...
