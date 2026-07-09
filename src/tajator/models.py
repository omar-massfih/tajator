"""Core data models shared across the market, LLM, risk, and trade layers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["call", "put"]


class Bar(BaseModel):
    """One 1-minute equity bar, timestamped in US/Eastern."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class Level(BaseModel):
    price: float
    kind: Literal["support", "resistance"]
    label: str  # e.g. "prev_day_low", "premarket_high", "double_top", "swing_low"


class LevelWatch(BaseModel):
    """One level as judged during pre-market prep."""

    level: Level
    tradable: bool
    direction: Direction | None = None
    note: str = ""


class MorningBriefing(BaseModel):
    """Structured output of the pre-market prep LLM call. Planning only — no trade results from it."""

    symbol: str
    bias: Literal["bullish", "bearish", "neutral"]
    watch_levels: list[LevelWatch]
    cleanest_level: float | None = None
    summary: str = Field(description="1-3 sentence overall read for the morning, journaled verbatim")


class SetupCandidate(BaseModel):
    """Mechanically detected 'price approaching a level with speed' candidate."""

    direction: Direction  # call = long at support, put = short at resistance
    level: Level
    distance: float  # dollars between current price and the level
    speed: float  # net move over the approach window, in dollars
    note: str = ""


class Snapshot(BaseModel):
    """Compact indicator/market summary for one tick."""

    symbol: str
    ts: datetime
    price: float
    ema9: float | None = None
    ema50: float | None = None
    vwap: float | None = None
    hod: float | None = None
    lod: float | None = None


class Decision(BaseModel):
    """Structured output of the LLM decision node."""

    action: Literal["wait", "enter_call", "enter_put", "scale_out", "exit"]
    level_price: float | None = Field(
        default=None, description="Support/resistance level the trade is anchored to"
    )
    stop_price: float | None = Field(
        default=None,
        description="Mental stop on the EQUITY price, ~40 cents beyond the level",
    )
    confidence: Literal["low", "medium", "high"] = "low"
    reasoning: str = Field(description="Short chart-based justification, journaled verbatim")


class RiskVerdict(BaseModel):
    approved: bool
    violations: list[str] = Field(default_factory=list)


class SelectedContract(BaseModel):
    symbol: str
    expiry: str  # YYYYMMDD
    strike: float
    right: Literal["C", "P"]
    con_id: int | None = None

    @property
    def local_name(self) -> str:
        return f"{self.symbol} {self.expiry} {self.strike:g}{self.right}"


class PositionPlan(BaseModel):
    """Frozen at entry fill; the trade's contract with itself."""

    direction: Direction
    level_price: float
    stop_price: float
    entry_equity_price: float
    entry_premium: float
    total_qty: int
    pieces: list[int]  # contract counts per scale-out piece, runner last
    target_refs: list[str]  # e.g. ["ema50_vwap", "hod_lod", "runner"]


class ProtectiveStop(BaseModel):
    """A resting broker-side stop: GTC market sell of the option, triggered by
    the underlying crossing the plan's stop price. Backstop for the in-loop
    mental stop — it protects the position while tajator is down or confused."""

    order_id: int
    perm_id: int | None = None
    order_ref: str
    qty: int
    stop_price: float


class OpenPosition(BaseModel):
    contract: SelectedContract
    plan: PositionPlan
    qty_remaining: int
    pieces_sold: int = 0
    profit_taken: bool = False
    opened_at: datetime
    favorable_extreme: float | None = None  # best equity price seen, for VWAP runner rule
    protective_stop: ProtectiveStop | None = None  # resting broker-side stop, if placed


class ExecutedAction(BaseModel):
    kind: Literal["entry", "scale_out", "stop_exit", "runner_exit", "manual_exit"]
    qty: int
    premium: float
    equity_price: float
    ts: datetime
    reason: str = ""
