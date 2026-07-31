"""Support/resistance level detection.

Levels come from (in priority order when deduping):
prev-day high/low, premarket high/low, intraday double tops / double bottoms,
single swings.

A double top/bottom is NOT any two swings at the same price: the touches must
be far enough apart in time with a real retrace between them — "a clear
earlier top and then returns to it" (strategy notes 07). Clusters that fail
those gates stay labeled swing_high/swing_low: chart context the LLM can see,
never trade levels.

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
DOUBLE_MIN_TOUCH_SEPARATION_BARS = 10  # minutes between the touches of a double
DOUBLE_MIN_PULLBACK_PCT = 0.002  # retrace between touches; must exceed CLUSTER_TOL

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


def _swing_extrema(
    session: pd.DataFrame, k: int = SWING_WINDOW
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """(bar_index, price) swing points — the index is needed to judge how far
    apart in time a double's touches are."""
    highs, lows = [], []
    h, lo = session["high"].to_numpy(), session["low"].to_numpy()
    for i in range(k, len(session) - k):
        if h[i] == max(h[i - k : i + k + 1]):
            highs.append((i, float(h[i])))
        if lo[i] == min(lo[i - k : i + k + 1]):
            lows.append((i, float(lo[i])))
    return highs, lows


def _cluster(
    points: list[tuple[int, float]], tol: float = CLUSTER_TOL
) -> list[list[tuple[int, float]]]:
    """Group points with nearby prices; each cluster keeps its (index, price) members."""
    clusters: list[list[tuple[int, float]]] = []
    for idx, price in sorted(points, key=lambda p: p[1]):
        if clusters and abs(price - clusters[-1][-1][1]) <= tol * price:
            clusters[-1].append((idx, price))
        else:
            clusters.append([(idx, price)])
    return clusters


def _is_qualified_double(
    cluster: list[tuple[int, float]],
    session: pd.DataFrame,
    *,
    is_top: bool,
    min_separation: int,
    min_pullback_pct: float,
) -> bool:
    """True when some pair of touches is separated by enough bars AND price
    retraced away from the level between them. Only interior bars count for
    the retrace — a touch candle's own wick cannot fabricate its pullback."""
    if len(cluster) < 2:
        return False
    touches = sorted(cluster)
    interior = session["low" if is_top else "high"].to_numpy()
    for a in range(len(touches)):
        for b in range(a + 1, len(touches)):
            (i, pi), (j, pj) = touches[a], touches[b]
            if j - i < min_separation:
                continue
            level = (pi + pj) / 2
            between = interior[i + 1 : j]
            if between.size == 0:
                continue
            retrace = level - between.min() if is_top else between.max() - level
            if retrace >= min_pullback_pct * level:
                return True
    return False


def swing_levels(
    session: pd.DataFrame,
    current_price: float,
    *,
    min_touch_separation: int = DOUBLE_MIN_TOUCH_SEPARATION_BARS,
    min_pullback_pct: float = DOUBLE_MIN_PULLBACK_PCT,
    swing_window: int = SWING_WINDOW,
    cluster_tol: float = CLUSTER_TOL,
) -> list[Level]:
    if len(session) < 2 * swing_window + 1:
        return []
    highs, lows = _swing_extrema(session, k=swing_window)
    levels: list[Level] = []
    for cluster, is_top, double, single in (
        *((c, True, "double_top", "swing_high") for c in _cluster(highs, tol=cluster_tol)),
        *((c, False, "double_bottom", "swing_low") for c in _cluster(lows, tol=cluster_tol)),
    ):
        qualified = _is_qualified_double(
            cluster, session, is_top=is_top,
            min_separation=min_touch_separation, min_pullback_pct=min_pullback_pct,
        )
        price = sum(p for _, p in cluster) / len(cluster)
        levels.append(
            Level(
                price=round(price, 2),
                kind=_kind(price, current_price),
                label=double if qualified else single,
            )
        )
    return levels


def detect_levels(
    bars: list[Bar],
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    *,
    min_touch_separation: int = DOUBLE_MIN_TOUCH_SEPARATION_BARS,
    min_pullback_pct: float = DOUBLE_MIN_PULLBACK_PCT,
    swing_window: int = SWING_WINDOW,
    cluster_tol: float = CLUSTER_TOL,
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
    levels.extend(
        swing_levels(
            session.iloc[:-swing_window] if len(session) > swing_window else session,
            current,
            min_touch_separation=min_touch_separation,
            min_pullback_pct=min_pullback_pct,
            swing_window=swing_window,
            cluster_tol=cluster_tol,
        )
    )

    return _dedupe(levels)


def _dedupe(levels: list[Level]) -> list[Level]:
    kept: list[Level] = []
    for lvl in sorted(levels, key=lambda l: _PRIORITY.get(l.label, 9)):
        if all(abs(lvl.price - k.price) > DEDUPE_TOL * k.price for k in kept):
            kept.append(lvl)
    return sorted(kept, key=lambda l: l.price)
