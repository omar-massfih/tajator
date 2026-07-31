from __future__ import annotations

from datetime import datetime, timedelta

from conftest import ET, make_bar, ts, walk
from tajator.market.timeframes import (
    aggregate_five_minute_bars,
    build_daily_context,
    build_five_minute_context,
    rank_candidates,
    score_candidate,
)
from tajator.models import (
    Bar,
    DailyContext,
    FiveMinuteContext,
    Level,
    MultiTimeframeContext,
    SetupCandidate,
)


def daily_bars(count: int, *, rising: bool = True) -> list[Bar]:
    start = datetime(2026, 3, 1, tzinfo=ET)
    bars = []
    for i in range(count):
        close = 100 + (i * 0.5 if rising else -i * 0.5)
        bars.append(Bar(
            ts=start + timedelta(days=i), open=close - 0.2,
            high=close + 1, low=close - 1, close=close, volume=1000,
        ))
    return bars


def test_five_minute_aggregation_excludes_premarket_and_labels_forming():
    bars = [make_bar(ts(9, 29), 99.0)] + walk(ts(9, 30), [100, 101, 102, 103, 104, 105, 106])
    completed, forming = aggregate_five_minute_bars(bars)
    assert len(completed) == 1
    assert completed[0].ts.hour == 9 and completed[0].ts.minute == 30
    assert completed[0].open == 100
    assert completed[0].close == 104
    assert forming is not None and forming.ts.minute == 35 and forming.close == 106


def test_missing_minutes_do_not_make_completed_bucket_look_forming():
    bars = walk(ts(9, 30), [100, 101, 102])
    bars.extend(walk(ts(9, 35), [103, 104]))
    completed, forming = aggregate_five_minute_bars(bars)
    assert len(completed) == 1
    assert completed[0].close == 102
    assert forming is not None and forming.ts.minute == 35


def test_daily_context_excludes_active_and_future_candles():
    bars = daily_bars(61)
    now = bars[-2].ts.replace(hour=12)
    context = build_daily_context(bars, now, bars[-2].close)
    assert context.recent_bars[-1].ts.date() < now.date()
    assert all(bar.ts.date() < now.date() for bar in context.recent_bars)


def test_daily_bias_uses_ema_stack_and_slope():
    bullish = daily_bars(70, rising=True)
    bearish = daily_bars(70, rising=False)
    after = datetime(2026, 7, 1, tzinfo=ET)
    assert build_daily_context(bullish, after, bullish[-1].close).bias == "bullish"
    assert build_daily_context(bearish, after, bearish[-1].close).bias == "bearish"


def test_five_minute_context_is_unknown_until_twenty_completed_bars():
    context = build_five_minute_context(walk(ts(9, 30), [100.0] * 60))
    assert context.trend == "unknown"
    context = build_five_minute_context(walk(ts(9, 30), [100 + i * 0.1 for i in range(105)]))
    assert context.trend == "bullish"


def candidate(direction="call", quality=3.0, distance=0.2, price=100.0):
    kind = "support" if direction == "call" else "resistance"
    return SetupCandidate(
        direction=direction,
        level=Level(price=price, kind=kind, label="prev_day_low" if direction == "call" else "prev_day_high"),
        distance=distance,
        speed=-0.5 if direction == "call" else 0.5,
        quality_score=quality,
    )


def test_score_components_and_quality_score_remains_unchanged():
    context = MultiTimeframeContext(
        enabled=True,
        daily=DailyContext(
            bias="bullish", atr14=2.0,
            reference_levels=[Level(price=100.1, kind="support", label="daily_pivot_low")],
        ),
        five_minute=FiveMinuteContext(
            trend="bullish",
            forming_bar=Bar(ts=ts(10, 0), open=101, high=101, low=99.8, close=100.2),
        ),
    )
    scored = score_candidate(candidate(), context)
    assert scored.quality_score == 3.0
    assert scored.higher_timeframe_score.daily_bias == 0.5
    assert scored.higher_timeframe_score.daily_confluence == 0.5
    assert scored.higher_timeframe_score.five_minute_trend == 0.5
    assert scored.higher_timeframe_score.five_minute_reaction == 0.5
    assert scored.ranking_score == 5.0


def test_rank_uses_total_then_proximity_and_disabled_preserves_order():
    near = candidate(quality=3.0, distance=0.1)
    far = candidate(quality=4.0, distance=0.5)
    unknown = MultiTimeframeContext(enabled=True)
    assert [item.distance for item in rank_candidates([near, far], unknown)] == [0.5, 0.1]
    disabled = rank_candidates([near, far], None)
    assert [item.distance for item in disabled] == [0.1, 0.5]


def test_clean_slice_penalizes_reaction():
    context = MultiTimeframeContext(
        enabled=True,
        five_minute=FiveMinuteContext(
            trend="unknown",
            forming_bar=Bar(ts=ts(10, 0), open=101, high=101, low=98, close=98.2),
        ),
    )
    scored = score_candidate(candidate(), context)
    assert scored.higher_timeframe_score.five_minute_reaction == -0.5
