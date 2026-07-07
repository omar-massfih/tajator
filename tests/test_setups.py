from tajator.market.indicators import build_snapshot
from tajator.market.setups import detect_candidates
from tajator.models import Level

from conftest import ts, walk

SUPPORT = Level(price=499.0, kind="support", label="prev_day_low")
RESISTANCE = Level(price=502.0, kind="resistance", label="double_top")


def candidates_for(path: list[float]):
    bars = walk(ts(9, 30), path)
    snap = build_snapshot("SPY", bars)
    return detect_candidates(bars, [SUPPORT, RESISTANCE], snap)


def test_falling_fast_into_support_fires_call():
    # 501 -> 499.3: fast drop landing just above support.
    cands = candidates_for([501.0, 500.9, 500.5, 500.1, 499.7, 499.3])
    assert any(c.direction == "call" and c.level.label == "prev_day_low" for c in cands)


def test_bounced_away_from_support_is_a_chase_and_blocked():
    # Dropped to support then bounced hard: price now rising away — no call.
    cands = candidates_for([500.5, 499.8, 499.1, 499.5, 500.0, 500.4])
    assert not any(c.direction == "call" for c in cands)


def test_slow_drift_into_level_lacks_speed():
    cands = candidates_for([499.45, 499.42, 499.40, 499.38, 499.36, 499.35])
    assert cands == []


def test_rising_fast_into_resistance_fires_put():
    cands = candidates_for([500.2, 500.6, 501.0, 501.4, 501.9])
    assert any(c.direction == "put" and c.level.label == "double_top" for c in cands)


def test_far_from_any_level_no_candidates():
    cands = candidates_for([500.4, 500.5, 500.6, 500.5, 500.5])
    assert cands == []
