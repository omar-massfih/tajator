import pytest

from tajator.market.price_action import (
    analyze_level_reaction,
    candle_geometry,
    close_off_extreme_fraction,
)
from tajator.models import Bar

from conftest import make_bar, ts


@pytest.mark.parametrize(
    ("open_price", "close", "expected_close_location"),
    [(10.0, 12.0, 0.75), (12.0, 10.0, 0.25)],
)
def test_true_wick_geometry_excludes_bullish_and_bearish_body(
    open_price, close, expected_close_location
):
    bar = make_bar(ts(9, 30), close, o=open_price, h=13.0, lo=9.0)
    shape = candle_geometry(bar)
    assert shape["body_fraction"] == 0.5
    assert shape["upper_wick_fraction"] == 0.25
    assert shape["lower_wick_fraction"] == 0.25
    assert shape["close_location"] == expected_close_location


def test_doji_and_zero_range_geometry():
    doji = candle_geometry(make_bar(ts(9, 30), 11.0, o=11.0, h=13.0, lo=9.0))
    assert doji["body_fraction"] == 0.0
    assert doji["upper_wick_fraction"] == 0.5
    assert doji["lower_wick_fraction"] == 0.5
    assert doji["close_location"] == 0.5

    flat = candle_geometry(make_bar(ts(9, 31), 11.0))
    assert flat["body_fraction"] == 0.0
    assert flat["upper_wick_fraction"] == 0.0
    assert flat["lower_wick_fraction"] == 0.0
    assert flat["close_location"] == 0.5


def test_close_off_extreme_is_separate_from_true_wick():
    bar = make_bar(ts(9, 30), 12.0, o=10.0, h=12.0, lo=9.0)
    shape = candle_geometry(bar)
    assert shape["lower_wick_fraction"] == pytest.approx(1 / 3, abs=0.0001)
    assert close_off_extreme_fraction(bar, "call") == 1.0


def test_support_reaction_detects_wick_reclaim_repetition_and_higher_low():
    bars = [
        make_bar(ts(9, 30), 100.1, o=100.4, h=100.5, lo=99.95, vol=100),
        make_bar(ts(9, 31), 100.4, o=100.1, h=100.5, lo=100.05, vol=100),
        make_bar(ts(9, 32), 100.2, o=100.3, h=100.4, lo=100.0, vol=200),
    ]
    features = analyze_level_reaction(
        bars, 100.0, "call", current_atr=0.5, long_wick_min_frac=0.25
    )
    assert features.touched
    assert features.reclaimed
    assert not features.break_and_reclaim
    assert features.rejection_count == 2
    assert features.relative_volume == 2.0
    assert features.range_atr == 0.8
    assert "long_lower_wick" in features.reaction_labels
    assert "repeated_support_rejection" in features.reaction_labels
    assert "higher_low" in features.reaction_labels


def test_resistance_reaction_mirrors_support_patterns():
    bars = [
        make_bar(ts(9, 30), 99.9, o=99.6, h=100.05, lo=99.5),
        make_bar(ts(9, 31), 99.6, o=99.9, h=99.95, lo=99.5),
        make_bar(ts(9, 32), 99.8, o=99.7, h=100.0, lo=99.6),
    ]
    features = analyze_level_reaction(bars, 100.0, "put", long_wick_min_frac=0.25)
    assert features.touched
    assert features.reclaimed
    assert features.rejection_count == 2
    assert "long_upper_wick" in features.reaction_labels
    assert "repeated_resistance_rejection" in features.reaction_labels
    assert "lower_high" in features.reaction_labels


@pytest.mark.parametrize(
    ("direction", "bar", "label"),
    [
        ("call", make_bar(ts(9, 30), 100.2, o=100.1, h=100.3, lo=99.8),
         "support_break_and_reclaim"),
        ("put", make_bar(ts(9, 30), 99.8, o=99.9, h=100.2, lo=99.7),
         "resistance_break_and_reclaim"),
    ],
)
def test_break_and_reclaim(direction: str, bar: Bar, label: str):
    features = analyze_level_reaction([bar], 100.0, direction)
    assert features.break_and_reclaim
    assert label in features.reaction_labels


@pytest.mark.parametrize(
    ("direction", "bar", "label"),
    [
        ("call", make_bar(ts(9, 30), 99.8, o=100.2, h=100.3, lo=99.7),
         "clean_support_slice"),
        ("put", make_bar(ts(9, 30), 100.2, o=99.8, h=100.3, lo=99.7),
         "clean_resistance_slice"),
    ],
)
def test_clean_slice_is_negative_evidence(direction: str, bar: Bar, label: str):
    features = analyze_level_reaction([bar], 100.0, direction)
    assert features.clean_slice
    assert label in features.reaction_labels
