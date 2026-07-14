"""Core data models shared across the market, LLM, risk, and trade layers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["call", "put"]

VisionPatternName = Literal[
    "none",
    "double_top",
    "double_bottom",
    "head_and_shoulders",
    "inverse_head_and_shoulders",
    "triangle_breakout_up",
    "triangle_breakout_down",
]


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


class PriceActionFeatures(BaseModel):
    """Transparent candle and level-reaction facts attached to a setup.

    Defaults are deliberately neutral so journals/backtests written before this
    model existed can still be loaded as ``SetupCandidate`` objects.
    """

    body_fraction: float = 0.0
    upper_wick_fraction: float = 0.0
    lower_wick_fraction: float = 0.0
    close_location: float = 0.5
    close_rejection_fraction: float = 0.0
    range_atr: float | None = None
    relative_volume: float | None = None
    touched: bool = False
    reclaimed: bool = False
    break_and_reclaim: bool = False
    penetration: float = 0.0
    rejection_count: int = 0
    clean_slice: bool = False
    reaction_labels: list[str] = Field(default_factory=list)


class DailyContext(BaseModel):
    """Causal context built only from completed regular-session daily candles."""

    bias: Literal["bullish", "bearish", "neutral", "unknown"] = "unknown"
    ema20: float | None = None
    ema50: float | None = None
    ema20_slope_5: float | None = None
    atr14: float | None = None
    reference_levels: list[Level] = Field(default_factory=list)
    recent_bars: list[Bar] = Field(default_factory=list)


class FiveMinuteContext(BaseModel):
    """RTH five-minute structure; ``forming_bar`` is explicitly incomplete."""

    trend: Literal["bullish", "bearish", "neutral", "unknown"] = "unknown"
    ema9: float | None = None
    ema20: float | None = None
    atr14: float | None = None
    completed_bars: list[Bar] = Field(default_factory=list)
    forming_bar: Bar | None = None


class MultiTimeframeContext(BaseModel):
    enabled: bool = False
    daily: DailyContext = Field(default_factory=DailyContext)
    five_minute: FiveMinuteContext = Field(default_factory=FiveMinuteContext)


class HigherTimeframeScore(BaseModel):
    daily_bias: float = 0.0
    daily_confluence: float = 0.0
    five_minute_trend: float = 0.0
    five_minute_reaction: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.daily_bias + self.daily_confluence
            + self.five_minute_trend + self.five_minute_reaction,
            2,
        )


class OptionQuote(BaseModel):
    """One broker quote captured immediately before an option order."""

    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    ts: datetime
    delayed: bool = False
    source: str = "broker"

    @property
    def valid(self) -> bool:
        return bool(
            self.bid is not None and self.ask is not None
            and self.bid > 0 and self.ask > 0 and self.bid <= self.ask
        )

    @property
    def midpoint(self) -> float | None:
        return (self.bid + self.ask) / 2 if self.valid else None

    @property
    def spread(self) -> float | None:
        return self.ask - self.bid if self.valid else None


class ExecutionQuality(BaseModel):
    """Arrival quote, timing, and realized quality for one broker fill."""

    side: Literal["BUY", "SELL"]
    order_type: str = "MKT"
    order_id: int | None = None
    perm_id: int | None = None
    status: str = ""
    requested_qty: int
    filled_qty: int
    signal_ts: datetime | None = None
    preflight_ts: datetime | None = None
    submitted_at: datetime
    filled_at: datetime
    quote: OptionQuote | None = None
    underlying_signal: float | None = None
    underlying_submit: float | None = None
    underlying_fill: float | None = None
    fill_price: float
    latency_s: float
    reference_price: float | None = None
    slippage: float | None = None
    slippage_pct: float | None = None
    breaches: list[str] = Field(default_factory=list)


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
    regime: str = "unknown"
    quality_score: float = 0.0
    higher_timeframe_score: HigherTimeframeScore = Field(default_factory=HigherTimeframeScore)
    ranking_score: float = 0.0
    price_action: PriceActionFeatures = Field(default_factory=PriceActionFeatures)


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
    atr: float | None = None
    vwap_slope_5: float | None = None
    vwap_slope_15: float | None = None
    closes_above_vwap_frac: float | None = None
    regime: str = "unknown"
    multi_timeframe: MultiTimeframeContext | None = None


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


class VisionPatternAnalysis(BaseModel):
    """Structured chart read produced by the experimental vision mode.

    This is deliberately not an executable ``Decision``. The graph must first
    validate direction, confirmation, and prices, then construct a normal
    candidate that still passes the existing risk gate.
    """

    action: Literal["wait", "enter_call", "enter_put"] = "wait"
    pattern: VisionPatternName = "none"
    status: Literal["none", "forming", "confirmed"] = "none"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    breakout_price: float | None = Field(
        default=None,
        description="Underlying price level whose completed-bar break confirms the pattern",
    )
    invalidation_price: float | None = Field(
        default=None,
        description="Underlying price that invalidates the pattern thesis",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Two or fewer observable chart facts; no hidden indicators or news",
    )
    reasoning: str = Field(description="Concise visual assessment, journaled verbatim")


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
    # HOD/LOD as they stood at entry — the fixed hod_lod scale target. None on
    # plans persisted before these fields existed (falls back to the live value).
    hod_at_entry: float | None = None
    lod_at_entry: float | None = None


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
    profit_lock_price: float | None = None  # equity price protected after first scale-out
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
    execution_quality: ExecutionQuality | None = None
