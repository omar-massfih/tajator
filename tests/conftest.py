from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tajator.models import Bar

ET = ZoneInfo("America/New_York")
DAY = datetime(2026, 7, 6, tzinfo=ET)  # a Monday


def ts(hour: int, minute: int) -> datetime:
    return DAY.replace(hour=hour, minute=minute)


def make_bar(t: datetime, close: float, *, o=None, h=None, lo=None, vol=1000.0) -> Bar:
    return Bar(
        ts=t,
        open=o if o is not None else close,
        high=h if h is not None else max(close, o or close),
        low=lo if lo is not None else min(close, o or close),
        close=close,
        volume=vol,
    )


def walk(start: datetime, closes: list[float], vol: float = 1000.0) -> list[Bar]:
    """One bar per minute following the given close path."""
    bars = []
    prev = closes[0]
    for i, c in enumerate(closes):
        t = start + timedelta(minutes=i)
        bars.append(
            Bar(ts=t, open=prev, high=max(prev, c), low=min(prev, c), close=c, volume=vol)
        )
        prev = c
    return bars


def orb_breakout_day() -> list[Bar]:
    """A scripted ORB day: a tight 500-501 opening range (first 15 min), then a
    clean upside breakout that runs to ~505 and holds. Produces exactly one call
    entry (break above the 501 opening-range high, stop at the 500 low)."""
    opening = [500.0, 501.0] * 7 + [500.0]            # 15 bars, 09:30-09:44 → range [500, 501]
    breakout = [502.0, 503.0, 504.0, 505.0]           # close above 501 at 09:45 → call
    hold = [505.0] * 20                                # ride to the end of the scripted day
    return walk(DAY.replace(hour=9, minute=30), opening + breakout + hold)


@pytest.fixture
def et_day():
    return DAY
