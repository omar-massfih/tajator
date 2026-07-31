"""Opening-Range Breakout (ORB) detection.

After the first ``window_minutes`` of the regular session the opening range
``[or_low, or_high]`` is locked. A completed 1-minute bar that closes beyond the
range (plus a small buffer) is a momentum breakout: a close above the range buys
calls, a close below it buys puts. The stop is the opposite side of the range.

This replaces the old support/resistance *fade* detector: ORB enters WITH the
move (price already beyond the level) rather than fading INTO a level.
"""

from __future__ import annotations

from ..models import Bar, Level, SetupCandidate, Snapshot
from .indicators import ET, RTH_OPEN

RTH_OPEN_MINUTES = RTH_OPEN.hour * 60 + RTH_OPEN.minute


def _session_minute(ts) -> int:
    """Minutes since the 09:30 ET regular-session open (negative pre-open)."""
    et = ts.astimezone(ET)
    return et.hour * 60 + et.minute - RTH_OPEN_MINUTES


def opening_range(bars: list[Bar], window_minutes: int) -> tuple[float, float] | None:
    """Return ``(or_high, or_low)`` once the opening window is complete, else None.

    Returns ``None`` while the window is still forming (so the detector is a
    no-op until the range is locked) or before the session opens.
    """
    if not bars or window_minutes <= 0:
        return None
    last = bars[-1]
    if _session_minute(last.ts) < window_minutes:
        return None  # opening window still forming
    day = last.ts.astimezone(ET).date()
    window_bars = [
        b for b in bars
        if b.ts.astimezone(ET).date() == day and 0 <= _session_minute(b.ts) < window_minutes
    ]
    if not window_bars:
        return None
    or_high = max(b.high for b in window_bars)
    or_low = min(b.low for b in window_bars)
    return or_high, or_low


def _session_bars(bars: list[Bar]) -> list[Bar]:
    """Regular-session bars of the current day (>= 09:30 ET)."""
    if not bars:
        return []
    day = bars[-1].ts.astimezone(ET).date()
    return [b for b in bars if b.ts.astimezone(ET).date() == day and _session_minute(b.ts) >= 0]


def detect_orb(
    bars: list[Bar],
    snapshot: Snapshot,
    *,
    window_minutes: int,
    breakout_buffer_pct: float,
    min_relative_volume: float = 0.0,
    min_breakout_range_atr: float = 0.0,
) -> list[SetupCandidate]:
    """Emit at most one breakout candidate once a completed bar closes beyond the range.

    Optional entry-quality filters keep only high-conviction breakouts:
    ``min_relative_volume`` requires the breakout bar's volume to exceed that
    multiple of the mean session-bar volume, and ``min_breakout_range_atr``
    requires its high-low range to be at least that multiple of the ATR.

    Re-entry churn is prevented downstream by ``entry_blockers`` (one position
    at a time, ``max_trades_per_day``), so this may fire on every breakout bar.
    """
    rng = opening_range(bars, window_minutes)
    if rng is None:
        return []
    or_high, or_low = rng
    price = snapshot.price  # latest completed-bar close
    buffer = breakout_buffer_pct * price

    breakout_bar = bars[-1]
    if min_relative_volume > 0:
        session = _session_bars(bars)
        vols = [b.volume for b in session if b.volume > 0]
        mean_vol = (sum(vols) / len(vols)) if vols else 0.0
        if mean_vol <= 0 or breakout_bar.volume < min_relative_volume * mean_vol:
            return []
    if min_breakout_range_atr > 0:
        atr = snapshot.atr
        bar_range = breakout_bar.high - breakout_bar.low
        if atr is None or atr <= 0 or bar_range < min_breakout_range_atr * atr:
            return []

    if price > or_high + buffer:
        return [
            SetupCandidate(
                direction="call",
                level=Level(price=round(or_high, 2), kind="support", label="orb_high"),
                distance=round(price - or_high, 2),
                speed=round(price - or_low, 2),
                stop_price=round(or_low, 2),
                note=f"ORB break above {or_high:.2f} (range {or_low:.2f}-{or_high:.2f})",
                regime=snapshot.regime,
            )
        ]
    if price < or_low - buffer:
        return [
            SetupCandidate(
                direction="put",
                level=Level(price=round(or_low, 2), kind="resistance", label="orb_low"),
                distance=round(or_low - price, 2),
                speed=round(or_high - price, 2),
                stop_price=round(or_high, 2),
                note=f"ORB break below {or_low:.2f} (range {or_low:.2f}-{or_high:.2f})",
                regime=snapshot.regime,
            )
        ]
    return []
