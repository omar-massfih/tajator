"""Deterministic candle geometry and level-reaction features.

These features describe price action around an already sanctioned level. They
never create a setup by themselves; setup detection remains in ``setups.py``.
"""

from __future__ import annotations

from statistics import mean

from ..models import Bar, PriceActionFeatures

REACTION_LOOKBACK_BARS = 5
LONG_WICK_MIN_FRAC = 0.25


def candle_geometry(
    bar: Bar,
    *,
    current_atr: float | None = None,
    average_volume: float | None = None,
) -> dict[str, float | None]:
    """Return true OHLC geometry, normalized to the candle's full range."""
    candle_range = bar.high - bar.low
    if candle_range <= 0:
        body = upper = lower = 0.0
        close_location = 0.5
    else:
        body = abs(bar.close - bar.open) / candle_range
        upper = (bar.high - max(bar.open, bar.close)) / candle_range
        lower = (min(bar.open, bar.close) - bar.low) / candle_range
        close_location = (bar.close - bar.low) / candle_range
    return {
        "body_fraction": round(max(0.0, body), 4),
        "upper_wick_fraction": round(max(0.0, upper), 4),
        "lower_wick_fraction": round(max(0.0, lower), 4),
        "close_location": round(min(1.0, max(0.0, close_location)), 4),
        "range_atr": (
            round(candle_range / current_atr, 4)
            if current_atr is not None and current_atr > 0 else None
        ),
        "relative_volume": (
            round(bar.volume / average_volume, 4)
            if average_volume is not None and average_volume > 0 else None
        ),
    }


def close_off_extreme_fraction(bar: Bar, direction: str) -> float:
    """How far the close moved from the adverse extreme, as range fraction.

    This intentionally preserves Tajator's existing fast-approach gate. It is
    not a true wick: a favorable candle body also contributes to this value.
    """
    candle_range = bar.high - bar.low
    if candle_range <= 0:
        return 0.0
    distance = bar.close - bar.low if direction == "call" else bar.high - bar.close
    return min(1.0, max(0.0, distance / candle_range))


def touch_rejected(bar: Bar, level: float, direction: str, min_frac: float) -> bool:
    """The bar traded the level, reclaimed it, and closed off its extreme."""
    touched = bar.low <= level if direction == "call" else bar.high >= level
    reclaimed = bar.close > level if direction == "call" else bar.close < level
    return touched and reclaimed and close_off_extreme_fraction(bar, direction) >= min_frac


def _prior_average_volume(bars: list[Bar], lookback: int) -> float | None:
    prior = [bar.volume for bar in bars[-lookback - 1 : -1] if bar.volume > 0]
    return mean(prior) if prior else None


def analyze_level_reaction(
    bars: list[Bar],
    level: float,
    direction: str,
    *,
    current_atr: float | None = None,
    lookback: int = REACTION_LOOKBACK_BARS,
    long_wick_min_frac: float = LONG_WICK_MIN_FRAC,
    approach_band: float = 0.003,
    overshoot_band: float = 0.001,
) -> PriceActionFeatures:
    """Describe recent price action around ``level`` for a call/put candidate."""
    if not bars:
        return PriceActionFeatures()
    window = bars[-lookback:]
    latest = window[-1]
    band = approach_band * level
    overshoot = overshoot_band * level
    geometry = candle_geometry(
        latest,
        current_atr=current_atr,
        average_volume=_prior_average_volume(bars, lookback),
    )
    close_rejection = close_off_extreme_fraction(latest, direction)

    if direction == "call":
        touched = latest.low <= level
        reclaimed = touched and latest.close > level
        broke_and_reclaimed = latest.low < level and latest.close > level
        penetration = max(0.0, level - latest.low)
        favorable_wick = float(geometry["lower_wick_fraction"] or 0.0)
        qualified = [
            bar for bar in window
            if level - overshoot <= bar.low <= level and bar.close > level
        ]
        prior_touches = [bar for bar in window[:-1] if level - overshoot <= bar.low <= level]
        structured = bool(
            prior_touches
            and latest.low > prior_touches[-1].low
            and latest.low <= level + band
            and latest.close >= level
        )
        clean_slice = touched and not reclaimed and close_rejection < long_wick_min_frac
        labels = ["support_touch"] if touched else []
        if reclaimed:
            labels.append("support_reclaim")
        if broke_and_reclaimed:
            labels.append("support_break_and_reclaim")
        if touched and favorable_wick >= long_wick_min_frac:
            labels.append("long_lower_wick")
        if len(qualified) >= 2:
            labels.append("repeated_support_rejection")
        if structured:
            labels.append("higher_low")
        if clean_slice:
            labels.append("clean_support_slice")
    else:
        touched = latest.high >= level
        reclaimed = touched and latest.close < level
        broke_and_reclaimed = latest.high > level and latest.close < level
        penetration = max(0.0, latest.high - level)
        favorable_wick = float(geometry["upper_wick_fraction"] or 0.0)
        qualified = [
            bar for bar in window
            if level <= bar.high <= level + overshoot and bar.close < level
        ]
        prior_touches = [bar for bar in window[:-1] if level <= bar.high <= level + overshoot]
        structured = bool(
            prior_touches
            and latest.high < prior_touches[-1].high
            and latest.high >= level - band
            and latest.close <= level
        )
        clean_slice = touched and not reclaimed and close_rejection < long_wick_min_frac
        labels = ["resistance_touch"] if touched else []
        if reclaimed:
            labels.append("resistance_reclaim")
        if broke_and_reclaimed:
            labels.append("resistance_break_and_reclaim")
        if touched and favorable_wick >= long_wick_min_frac:
            labels.append("long_upper_wick")
        if len(qualified) >= 2:
            labels.append("repeated_resistance_rejection")
        if structured:
            labels.append("lower_high")
        if clean_slice:
            labels.append("clean_resistance_slice")

    return PriceActionFeatures(
        **geometry,
        close_rejection_fraction=round(close_rejection, 4),
        touched=touched,
        reclaimed=reclaimed,
        break_and_reclaim=broke_and_reclaimed,
        penetration=round(penetration, 4),
        rejection_count=len(qualified),
        clean_slice=clean_slice,
        reaction_labels=labels,
    )
