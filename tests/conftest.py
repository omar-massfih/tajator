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


@pytest.fixture
def et_day():
    return DAY
