from tajator.market.indicators import build_snapshot
from tajator.market.setups import detect_candidates
from tajator.models import Level

from conftest import make_bar, ts, walk

SUPPORT = Level(price=499.0, kind="support", label="prev_day_low")
RESISTANCE = Level(price=502.0, kind="resistance", label="double_top")


def candidates_for(path: list[float]):
    bars = walk(ts(9, 30), path)
    snap = build_snapshot("SPY", bars)
    return detect_candidates(bars, [SUPPORT, RESISTANCE], snap)


def test_falling_fast_into_support_fires_call():
    # 501 -> 499.4: fast drop landing just above support; the entry bar wicks
    # off the level, so even a very fast approach is admitted.
    bars = walk(ts(9, 30), [501.5, 501.2, 500.9, 500.2, 499.7])
    bars.append(make_bar(ts(9, 35), 499.4, o=499.7, h=499.7, lo=499.1))
    snap = build_snapshot("SPY", bars)
    cands = detect_candidates(bars, [SUPPORT, RESISTANCE], snap)
    assert any(c.direction == "call" and c.level.label == "prev_day_low" for c in cands)


def test_bounced_away_from_support_is_a_chase_and_blocked():
    # Dropped to support then bounced hard: price now rising away — no call.
    cands = candidates_for([500.5, 499.8, 499.1, 499.5, 500.0, 500.4])
    assert not any(c.direction == "call" for c in cands)


def test_slow_drift_into_level_lacks_speed():
    cands = candidates_for([499.45, 499.42, 499.40, 499.38, 499.36, 499.35])
    assert cands == []


def test_rising_fast_into_resistance_fires_put():
    # Fast rise into the double top; the entry bar closes off its high.
    bars = walk(ts(9, 30), [500.0, 500.3, 500.5, 501.0, 501.5])
    bars.append(make_bar(ts(9, 35), 501.8, o=501.5, h=502.0, lo=501.5))
    snap = build_snapshot("SPY", bars)
    cands = detect_candidates(bars, [SUPPORT, RESISTANCE], snap)
    assert any(c.direction == "put" and c.level.label == "double_top" for c in cands)


def test_fast_flush_without_rejection_wick_is_blocked():
    # Very fast drop whose entry bar closes on its low: no proof the level
    # held — exactly the entries the 07-06..10 backtests lost on.
    bars = walk(ts(9, 30), [501.5, 501.2, 500.9, 500.2, 499.7, 499.4])
    snap = build_snapshot("SPY", bars)
    assert detect_candidates(bars, [SUPPORT], snap) == []
    # ...REJECTION_WICK_MIN_FRAC=0 disables the gate and admits the same approach
    fired = detect_candidates(bars, [SUPPORT], snap, rejection_wick_frac=0.0)
    assert any(c.direction == "call" for c in fired)


def test_fast_squeeze_without_rejection_wick_is_blocked():
    bars = walk(ts(9, 30), [500.0, 500.3, 500.5, 501.0, 501.5, 501.8])
    snap = build_snapshot("SPY", bars)
    assert detect_candidates(bars, [RESISTANCE], snap) == []
    fired = detect_candidates(bars, [RESISTANCE], snap, rejection_wick_frac=0.0)
    assert any(c.direction == "put" for c in fired)


def test_far_from_any_level_no_candidates():
    cands = candidates_for([500.4, 500.5, 500.6, 500.5, 500.5])
    assert cands == []


def test_swing_levels_never_become_candidates():
    # Bare swing points are chart context, not a sanctioned setup — even a
    # perfect fast approach must not produce a candidate.
    swing_low = Level(price=499.0, kind="support", label="swing_low")
    swing_high = Level(price=502.0, kind="resistance", label="swing_high")

    falling = walk(ts(9, 30), [501.0, 500.9, 500.5, 500.1, 499.7, 499.3])
    snap = build_snapshot("SPY", falling)
    assert detect_candidates(falling, [swing_low], snap) == []

    rising = walk(ts(9, 30), [500.2, 500.6, 501.0, 501.4, 501.9])
    snap = build_snapshot("SPY", rising)
    assert detect_candidates(rising, [swing_high], snap) == []


def test_level_a_few_cents_from_the_open_is_vetoed():
    # Resistance 40 cents (0.08%) above the 500.2 open: no room for a move.
    near_open = Level(price=500.6, kind="resistance", label="prev_day_high")
    bars = walk(ts(9, 30), [500.2, 499.5, 499.7, 500.0, 500.3, 500.55])
    snap = build_snapshot("SPY", bars)

    assert detect_candidates(bars, [near_open], snap) == []
    # ...and disabling the filter proves the veto was the distance, not the approach
    fired = detect_candidates(bars, [near_open], snap, min_dist_from_open_pct=0.0)
    assert any(c.direction == "put" for c in fired)


def test_approach_band_override_admits_wider_approach():
    # Price 2.0 above support: outside the default 0.3% band, inside 0.5%.
    bars = walk(ts(9, 30), [502.4, 502.0, 501.8, 501.4, 501.0])
    snap = build_snapshot("SPY", bars)
    assert detect_candidates(bars, [SUPPORT], snap) == []
    fired = detect_candidates(bars, [SUPPORT], snap, approach_band=0.005)
    assert any(c.direction == "call" for c in fired)


def test_min_speed_pct_override_admits_slow_drift():
    bars = walk(ts(9, 30), [501.5, 500.5, 499.45, 499.42, 499.40, 499.38, 499.36, 499.35])
    snap = build_snapshot("SPY", bars)
    assert detect_candidates(bars, [SUPPORT], snap) == []
    fired = detect_candidates(bars, [SUPPORT], snap, min_speed_pct=0.0)
    assert any(c.direction == "call" for c in fired)


def test_speed_window_override_measures_longer_move():
    # The drop happened 4-5 bars ago; the default 3-bar window misses it.
    bars = walk(ts(9, 30), [502.0, 500.4, 500.2, 499.6, 499.5, 499.4, 499.3])
    snap = build_snapshot("SPY", bars)
    assert detect_candidates(bars, [SUPPORT], snap) == []
    fired = detect_candidates(bars, [SUPPORT], snap, speed_window=5)
    assert any(c.direction == "call" for c in fired)


def test_overshoot_band_override_admits_deeper_poke():
    # Price 0.7 below support: past the default 0.1% overshoot, inside 0.2%.
    bars = walk(ts(9, 30), [501.5, 500.6, 499.8, 499.4, 499.0, 498.6, 498.3])
    snap = build_snapshot("SPY", bars)
    assert detect_candidates(bars, [SUPPORT], snap) == []
    fired = detect_candidates(bars, [SUPPORT], snap, overshoot_band=0.002)
    assert any(c.direction == "call" for c in fired)


def test_premarket_only_bars_skip_the_open_filter():
    # No session bars yet: the day's open is unknown, so the distance filter
    # must be skipped rather than crash or block everything.
    bars = walk(ts(8, 0), [501.0, 500.9, 500.5, 500.0, 499.7, 499.4])
    snap = build_snapshot("SPY", bars)
    cands = detect_candidates(bars, [SUPPORT], snap)
    assert any(c.direction == "call" for c in cands)
