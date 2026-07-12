"""Broker protocol — the rest of the system never imports a broker SDK directly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from ..models import Bar, Direction, ProtectiveStop, SelectedContract


class ChainParams(BaseModel):
    expirations: list[str]  # YYYYMMDD, sorted ascending
    strikes: list[float]  # sorted ascending


class Fill(BaseModel):
    premium: float  # per-contract option price
    qty: int  # contracts actually filled — may be less than requested
    ts: datetime
    fee: float = 0.0  # total commissions/fees for this fill, in dollars
    equity_price: float | None = None
    stop_price: float | None = None
    exit_reason: str = ""
    regime: str = "unknown"
    level_quality_score: float = 0.0


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


class BrokerOpenOrder(BaseModel):
    """A resting order at the broker, with enough structure to recognize our own."""

    order_id: int
    perm_id: int | None = None
    order_ref: str = ""
    action: str
    qty: int
    order_type: str
    status: str
    con_id: int
    local_symbol: str
    symbol: str
    expiry: str = ""  # YYYYMMDD
    strike: float = 0.0
    right: str = ""  # "C" | "P"


class StopCancelResult(BaseModel):
    """Outcome of cancelling a protective stop, reconciled against executions.
    `filled_qty` counts contracts the stop sold before/despite the cancel —
    the caller must shrink its own sell by that amount (double-sell guard)."""

    cancelled: bool
    filled_qty: int = 0
    avg_price: float | None = None


class StopStatus(BaseModel):
    """Point-in-time state of a resting protective stop."""

    state: Literal["working", "filled", "partial", "gone"]
    filled_qty: int = 0
    avg_price: float | None = None
    working_qty: int = 0


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

    @property
    def is_delayed_data(self) -> bool:
        """True when quotes fell back to delayed data — entries should refuse."""
        return False

    # -- broker-side protective stop ------------------------------------------

    @abstractmethod
    def place_protective_stop(
        self,
        contract: SelectedContract,
        qty: int,
        stop_price: float,
        direction: Direction,
        order_ref: str,
    ) -> ProtectiveStop:
        """Rest a GTC market sell of `qty` contracts, triggered by the
        underlying crossing `stop_price`. Raises if the broker rejects it."""

    @abstractmethod
    def cancel_protective_stop(
        self,
        contract: SelectedContract,
        stop: ProtectiveStop,
        expected_held: int | None = None,
    ) -> StopCancelResult:
        """Cancel a resting stop and confirm the terminal state. MUST either
        return a reconciled result or raise — never 'maybe still working'.
        A raise means the caller must NOT sell (the stop may still execute).
        `expected_held` is the position the caller believes the account holds
        (usually qty_remaining); when given, the result is cross-checked
        against the account and an unexplainable mismatch raises."""

    @abstractmethod
    def poll_protective_stop(
        self, contract: SelectedContract, stop: ProtectiveStop
    ) -> StopStatus:
        """Non-mutating check of a resting stop's state."""
