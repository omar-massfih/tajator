"""Support/resistance level detection.

Levels come from (in priority order when deduping):
prev-day high/low, premarket high/low, clustered intraday swing extrema
(two+ swings at the same area = double top / double bottom), single swings.

A level's kind is relative to current price: below price = support,
above price = resistance (a broken resistance becomes support and vice versa).
"""

from __future__ import annotations

import pandas as pd

from ..models import Bar, Level
from .indicators import bars_to_df, premarket_df, session_df

SWING_WINDOW = 3  # bars on each side that must be lower/higher for a swing extreme
CLUSTER_TOL = 0.0015  # swings within 0.15% of each other form one level
DEDUPE_TOL = 0.0015

_PRIORITY = {
    "prev_day_high": 0,
    "prev_day_low": 0,
    "premarket_high": 1,
    "premarket_low": 1,
    "double_top": 2,
    "double_bottom": 2,
    "swing_high": 3,
    "swing_low": 3,
}


def _kind(price: float, current: float) -> str:
    return "support" if price <= current else "resistance"


def _swing_extrema(session: pd.DataFrame, k: int = SWING_WINDOW) -> tuple[list[float], list[float]]:
    highs, lows = [], []
    h, lo = session["high"].to_numpy(), session["low"].to_numpy()
    for i in range(k, len(session) - k):
        if h[i] == max(h[i - k : i + k + 1]):
            highs.append(float(h[i]))
        if lo[i] == min(lo[i - k : i + k + 1]):
            lows.append(float(lo[i]))
    return highs, lows


def _cluster(values: list[float], tol: float = CLUSTER_TOL) -> list[tuple[float, int]]:
    """Group nearby values; return (mean, count) per cluster."""
    clusters: list[list[float]] = []
    for v in sorted(values):
        if clusters and abs(v - clusters[-1][-1]) <= tol * v:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [(sum(c) / len(c), len(c)) for c in clusters]


def swing_levels(session: pd.DataFrame, current_price: float) -> list[Level]:
    if len(session) < 2 * SWING_WINDOW + 1:
        return []
    highs, lows = _swing_extrema(session)
    levels: list[Level] = []
    for price, count in _cluster(highs):
        label = "double_top" if count >= 2 else "swing_high"
        levels.append(Level(price=round(price, 2), kind=_kind(price, current_price), label=label))
    for price, count in _cluster(lows):
        label = "double_bottom" if count >= 2 else "swing_low"
        levels.append(Level(price=round(price, 2), kind=_kind(price, current_price), label=label))
    return levels


def detect_levels(
    bars: list[Bar],
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
) -> list[Level]:
    df = bars_to_df(bars)
    if df.empty:
        return []
    ts = bars[-1].ts
    current = bars[-1].close

    levels: list[Level] = []
    if prev_day_high:
        levels.append(Level(price=prev_day_high, kind=_kind(prev_day_high, current), label="prev_day_high"))
    if prev_day_low:
        levels.append(Level(price=prev_day_low, kind=_kind(prev_day_low, current), label="prev_day_low"))

    pre = premarket_df(df, ts)
    if not pre.empty:
        pm_high, pm_low = float(pre["high"].max()), float(pre["low"].min())
        levels.append(Level(price=pm_high, kind=_kind(pm_high, current), label="premarket_high"))
        levels.append(Level(price=pm_low, kind=_kind(pm_low, current), label="premarket_low"))

    session = session_df(df, ts)
    # Exclude the most recent bars so the move currently in progress
    # doesn't create the level it is about to be traded against.
    levels.extend(swing_levels(session.iloc[:-SWING_WINDOW] if len(session) > SWING_WINDOW else session, current))

    return _dedupe(levels)


def _dedupe(levels: list[Level]) -> list[Level]:
    kept: list[Level] = []
    for lvl in sorted(levels, key=lambda l: _PRIORITY.get(l.label, 9)):
        if all(abs(lvl.price - k.price) > DEDUPE_TOL * k.price for k in kept):
            kept.append(lvl)
    return sorted(kept, key=lambda l: l.price)
