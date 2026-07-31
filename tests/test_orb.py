"""Unit tests for the Opening-Range Breakout detector."""

from tajator.market.indicators import build_snapshot
from tajator.market.orb import detect_orb, opening_range

from conftest import DAY, walk


def _day(closes):
    return walk(DAY.replace(hour=9, minute=30), closes)


def test_opening_range_is_none_while_the_window_forms():
    # Only 10 minutes of a 15-minute window have printed.
    bars = _day([500.0, 501.0] * 5)
    assert opening_range(bars, window_minutes=15) is None


def test_opening_range_locks_after_the_window():
    # 15 range bars (500-501) then a 16th bar → window is complete.
    bars = _day([500.0, 501.0] * 7 + [500.0, 502.0])
    rng = opening_range(bars, window_minutes=15)
    assert rng == (501.0, 500.0)  # (or_high, or_low)


def test_detect_orb_emits_a_call_on_an_upside_break():
    bars = _day([500.0, 501.0] * 7 + [500.0, 502.0])  # closes 502 > 501 range high
    snapshot = build_snapshot("SPY", bars)
    candidates = detect_orb(bars, snapshot, window_minutes=15, breakout_buffer_pct=0.0005)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.direction == "call"
    assert c.level.label == "orb_high"
    assert c.level.price == 501.0
    assert c.stop_price == 500.0  # opposite side of the range


def test_detect_orb_emits_a_put_on_a_downside_break():
    bars = _day([500.0, 501.0] * 7 + [500.0, 499.0])  # closes 499 < 500 range low
    snapshot = build_snapshot("SPY", bars)
    candidates = detect_orb(bars, snapshot, window_minutes=15, breakout_buffer_pct=0.0005)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.direction == "put"
    assert c.level.label == "orb_low"
    assert c.level.price == 500.0
    assert c.stop_price == 501.0  # opposite side of the range


def test_detect_orb_holds_inside_the_range():
    bars = _day([500.0, 501.0] * 7 + [500.0, 500.5])  # 500.5 stays inside [500, 501]
    snapshot = build_snapshot("SPY", bars)
    assert detect_orb(bars, snapshot, window_minutes=15, breakout_buffer_pct=0.0005) == []


def test_detect_orb_respects_the_breakout_buffer():
    # 501.2 clears the 501 high but not the 0.5%% buffer (~2.5) — no breakout yet.
    bars = _day([500.0, 501.0] * 7 + [500.0, 501.2])
    snapshot = build_snapshot("SPY", bars)
    assert detect_orb(bars, snapshot, window_minutes=15, breakout_buffer_pct=0.005) == []


def _volume_day(breakout_vol):
    # 15 low-volume range bars, then a breakout bar whose volume we control.
    bars = walk(DAY.replace(hour=9, minute=30), [500.0, 501.0] * 7 + [500.0], vol=1000.0)
    bars += walk(bars[-1].ts.replace(minute=45), [502.0], vol=breakout_vol)
    return bars


def test_relative_volume_filter_blocks_a_quiet_breakout():
    bars = _volume_day(breakout_vol=1000.0)  # same as the average — not a surge
    snapshot = build_snapshot("SPY", bars)
    assert detect_orb(
        bars, snapshot, window_minutes=15, breakout_buffer_pct=0.0005,
        min_relative_volume=1.5,
    ) == []


def test_relative_volume_filter_admits_a_high_volume_breakout():
    bars = _volume_day(breakout_vol=5000.0)  # 5x the ~1000 average
    snapshot = build_snapshot("SPY", bars)
    out = detect_orb(
        bars, snapshot, window_minutes=15, breakout_buffer_pct=0.0005,
        min_relative_volume=1.5,
    )
    assert len(out) == 1 and out[0].direction == "call"


def test_range_atr_filter_blocks_a_narrow_breakout():
    # Long flat warmup builds a tiny ATR; a wide breakout bar should still be
    # required to exceed it. Here the breakout bar's range is small.
    bars = walk(DAY.replace(hour=9, minute=30), [500.0, 501.0] * 7 + [500.0] + [502.0] * 20)
    snapshot = build_snapshot("SPY", bars)
    # A 3x-ATR floor is far above the 1-point breakout bar's range.
    assert detect_orb(
        bars, snapshot, window_minutes=15, breakout_buffer_pct=0.0005,
        min_breakout_range_atr=3.0,
    ) == []
