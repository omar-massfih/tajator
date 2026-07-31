"""Causal daily and five-minute context derived from completed/visible bars."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from ..models import (
    Bar,
    DailyContext,
    FiveMinuteContext,
    HigherTimeframeScore,
    Level,
    MultiTimeframeContext,
    SetupCandidate,
)
from .indicators import ET, RTH_OPEN


def _ema_series(bars: list[Bar], span: int) -> pd.Series:
    closes = pd.Series([bar.close for bar in bars], dtype=float)
    return closes.ewm(span=span, adjust=False).mean()


def _atr_bars(bars: list[Bar], window: int = 14) -> float | None:
    if len(bars) < window + 1:
        return None
    frame = pd.DataFrame([bar.model_dump() for bar in bars])
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.tail(window).mean())


def _dedupe_reference_levels(levels: list[Level], tolerance: float = 0.0015) -> list[Level]:
    kept: list[Level] = []
    for level in levels:
        if all(abs(level.price - other.price) > tolerance * other.price for other in kept):
            kept.append(level)
    return kept


def _daily_reference_levels(bars: list[Bar], price: float) -> list[Level]:
    window = bars[-60:]
    pivots: list[Level] = []
    for index in range(2, len(window) - 2):
        bar = window[index]
        neighbors = window[index - 2:index] + window[index + 1:index + 3]
        if all(bar.high > other.high for other in neighbors):
            pivots.append(Level(price=bar.high, kind="resistance", label="daily_pivot_high"))
        if all(bar.low < other.low for other in neighbors):
            pivots.append(Level(price=bar.low, kind="support", label="daily_pivot_low"))

    recent20 = bars[-20:]
    if recent20:
        pivots.extend(
            [
                Level(
                    price=max(bar.high for bar in recent20),
                    kind="resistance",
                    label="daily_20d_high",
                ),
                Level(
                    price=min(bar.low for bar in recent20),
                    kind="support",
                    label="daily_20d_low",
                ),
            ]
        )
    pivots = _dedupe_reference_levels(pivots)
    below = sorted((level for level in pivots if level.price <= price), key=lambda level: -level.price)[:2]
    above = sorted((level for level in pivots if level.price > price), key=lambda level: level.price)[:2]
    return sorted(below + above, key=lambda level: level.price)


def build_daily_context(daily_bars: list[Bar], now: datetime, price: float) -> DailyContext:
    """Build context without admitting the active session or any future candle."""
    today = now.astimezone(ET).date()
    completed = sorted(
        (bar for bar in daily_bars if bar.ts.date() < today),
        key=lambda bar: bar.ts,
    )[-90:]
    if not completed:
        return DailyContext()

    ema20_values = _ema_series(completed, 20) if len(completed) >= 20 else None
    ema50_values = _ema_series(completed, 50) if len(completed) >= 50 else None
    ema20 = float(ema20_values.iloc[-1]) if ema20_values is not None else None
    ema50 = float(ema50_values.iloc[-1]) if ema50_values is not None else None
    slope = (
        float(ema20_values.iloc[-1] - ema20_values.iloc[-6])
        if ema20_values is not None and len(ema20_values) >= 6 else None
    )
    bias = "unknown"
    if ema20 is not None and ema50 is not None and slope is not None:
        close = completed[-1].close
        if close > ema20 > ema50 and slope > 0:
            bias = "bullish"
        elif close < ema20 < ema50 and slope < 0:
            bias = "bearish"
        else:
            bias = "neutral"
    return DailyContext(
        bias=bias,
        ema20=ema20,
        ema50=ema50,
        ema20_slope_5=slope,
        atr14=_atr_bars(completed),
        reference_levels=_daily_reference_levels(completed, price),
        recent_bars=completed[-6:],
    )


def _aggregate_bucket(bucket: list[Bar], start: datetime) -> Bar:
    return Bar(
        ts=start,
        open=bucket[0].open,
        high=max(bar.high for bar in bucket),
        low=min(bar.low for bar in bucket),
        close=bucket[-1].close,
        volume=sum(bar.volume for bar in bucket),
    )


def aggregate_five_minute_bars(bars: list[Bar]) -> tuple[list[Bar], Bar | None]:
    """Aggregate only today's visible RTH minutes on 09:30-aligned boundaries."""
    if not bars:
        return [], None
    latest = max(bars, key=lambda bar: bar.ts).ts.astimezone(ET)
    day = latest.date()
    visible = sorted(
        (
            bar for bar in bars
            if bar.ts.astimezone(ET).date() == day
            and bar.ts.astimezone(ET).time() >= RTH_OPEN
        ),
        key=lambda bar: bar.ts,
    )
    buckets: dict[datetime, list[Bar]] = {}
    open_at = datetime.combine(day, RTH_OPEN, tzinfo=ET)
    for bar in visible:
        minute = int((bar.ts.astimezone(ET) - open_at).total_seconds() // 60)
        if minute < 0:
            continue
        start = open_at + timedelta(minutes=(minute // 5) * 5)
        buckets.setdefault(start, []).append(bar)

    completed: list[Bar] = []
    forming: Bar | None = None
    visible_through = latest + timedelta(minutes=1)
    for start, bucket in sorted(buckets.items()):
        aggregated = _aggregate_bucket(bucket, start)
        if start + timedelta(minutes=5) <= visible_through:
            completed.append(aggregated)
        else:
            forming = aggregated
    return completed, forming


def build_five_minute_context(bars: list[Bar]) -> FiveMinuteContext:
    completed, forming = aggregate_five_minute_bars(bars)
    ema9 = ema20 = None
    trend = "unknown"
    if len(completed) >= 9:
        ema9 = float(_ema_series(completed, 9).iloc[-1])
    if len(completed) >= 20:
        ema20 = float(_ema_series(completed, 20).iloc[-1])
    if ema9 is not None and ema20 is not None:
        trend = "bullish" if ema9 > ema20 else "bearish" if ema9 < ema20 else "neutral"
    return FiveMinuteContext(
        trend=trend,
        ema9=ema9,
        ema20=ema20,
        atr14=_atr_bars(completed),
        completed_bars=completed[-6:],
        forming_bar=forming,
    )


def build_multi_timeframe_context(
    bars: list[Bar], daily_bars: list[Bar], now: datetime, price: float,
) -> MultiTimeframeContext:
    return MultiTimeframeContext(
        enabled=True,
        daily=build_daily_context(daily_bars, now, price),
        five_minute=build_five_minute_context(bars),
    )


def _five_minute_reaction(candidate: SetupCandidate, context: FiveMinuteContext) -> float:
    bar = context.forming_bar or (context.completed_bars[-1] if context.completed_bars else None)
    if bar is None or bar.high <= bar.low:
        return 0.0
    level = candidate.level.price
    close_location = (bar.close - bar.low) / (bar.high - bar.low)
    if candidate.direction == "call":
        if bar.low <= level and bar.close > level:
            return 0.5
        if bar.close < level and close_location <= 0.25:
            return -0.5
    else:
        if bar.high >= level and bar.close < level:
            return 0.5
        if bar.close > level and close_location >= 0.75:
            return -0.5
    return 0.0


def score_candidate(candidate: SetupCandidate, context: MultiTimeframeContext) -> SetupCandidate:
    daily = context.daily
    five = context.five_minute
    aligned_daily = "bullish" if candidate.direction == "call" else "bearish"
    opposed_daily = "bearish" if candidate.direction == "call" else "bullish"
    daily_bias = 0.5 if daily.bias == aligned_daily else -0.5 if daily.bias == opposed_daily else 0.0
    daily_confluence = 0.0
    if daily.atr14 and any(
        abs(reference.price - candidate.level.price) <= 0.15 * daily.atr14
        for reference in daily.reference_levels
    ):
        daily_confluence = 0.5
    aligned_five = "bullish" if candidate.direction == "call" else "bearish"
    opposed_five = "bearish" if candidate.direction == "call" else "bullish"
    five_trend = 0.5 if five.trend == aligned_five else -0.5 if five.trend == opposed_five else 0.0
    breakdown = HigherTimeframeScore(
        daily_bias=daily_bias,
        daily_confluence=daily_confluence,
        five_minute_trend=five_trend,
        five_minute_reaction=_five_minute_reaction(candidate, five),
    )
    return candidate.model_copy(
        update={
            "higher_timeframe_score": breakdown,
            "ranking_score": round(candidate.quality_score + breakdown.total, 2),
        }
    )


def rank_candidates(
    candidates: list[SetupCandidate], context: MultiTimeframeContext | None,
) -> list[SetupCandidate]:
    if context is None or not context.enabled:
        return [
            candidate.model_copy(update={"ranking_score": candidate.quality_score})
            for candidate in candidates
        ]
    scored = [score_candidate(candidate, context) for candidate in candidates]
    return sorted(scored, key=lambda candidate: (-candidate.ranking_score, abs(candidate.distance)))
