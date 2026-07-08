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
    qty: int
    ts: datetime


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
