"""Mechanical pre-filter: is price approaching a level with speed?

The strategy enters INTO the level (calls while price falls into support,
puts while price rises into resistance) and never chases. Only when this
filter finds a candidate is the LLM consulted.

Not every level is tradable: bare swing points are chart context only, and a
level sitting a few cents from the day's open leaves no room for a move
(strategy notes 08). The risk gate only admits detected candidates, so
filtering here is what keeps the LLM from trading those levels.
"""

from __future__ import annotations

from ..models import Bar, Level, SetupCandidate, Snapshot
from .indicators import bars_to_df, session_df

APPROACH_BAND = 0.003  # price within 0.3% of the level
OVERSHOOT_BAND = 0.001  # slight poke through the level still counts
SPEED_WINDOW = 3  # bars used to measure the move into the level
MIN_SPEED_PCT = 0.0012  # net move over the window, fraction of price
MIN_LEVEL_DIST_FROM_OPEN_PCT = 0.003  # ~$1.50 on a $500 name

NON_TRADABLE_LABELS = frozenset({"swing_high", "swing_low"})


def _day_open(bars: list[Bar]) -> float | None:
    """Open of the first regular-session bar of the current day; None premarket."""
    df = bars_to_df(bars)
    if df.empty:
        return None
    session = session_df(df, bars[-1].ts)
    if session.empty:
        return None
    return float(session.iloc[0]["open"])


def detect_candidates(
    bars: list[Bar],
    levels: list[Level],
    snapshot: Snapshot,
    *,
    min_dist_from_open_pct: float = MIN_LEVEL_DIST_FROM_OPEN_PCT,
    approach_band: float = APPROACH_BAND,
    overshoot_band: float = OVERSHOOT_BAND,
    speed_window: int = SPEED_WINDOW,
    min_speed_pct: float = MIN_SPEED_PCT,
) -> list[SetupCandidate]:
    if len(bars) < speed_window + 1:
        return []

    price = snapshot.price
    net_move = price - bars[-1 - speed_window].close  # + = rising, - = falling
    min_speed = min_speed_pct * price
    band = approach_band * price
    overshoot = overshoot_band * price
    day_open = _day_open(bars)

    candidates: list[SetupCandidate] = []
    for level in levels:
        if level.label in NON_TRADABLE_LABELS:
            continue
        if (
            day_open is not None
            and abs(level.price - day_open) < min_dist_from_open_pct * day_open
        ):
            continue
        diff = price - level.price
        if level.kind == "support":
            # Call setup: price falling into support from above (or a slight poke below).
            if -overshoot <= diff <= band and net_move <= -min_speed:
                candidates.append(
                    SetupCandidate(
                        direction="call",
                        level=level,
                        distance=round(diff, 2),
                        speed=round(net_move, 2),
                        note=f"falling into {level.label} @ {level.price}",
                    )
                )
        else:
            # Put setup: price rising into resistance from below (or a slight poke above).
            if -overshoot <= -diff <= band and net_move >= min_speed:
                candidates.append(
                    SetupCandidate(
                        direction="put",
                        level=level,
                        distance=round(-diff, 2),
                        speed=round(net_move, 2),
                        note=f"rising into {level.label} @ {level.price}",
                    )
                )
    # Closest level first — that's the trade the LLM should judge.
    candidates.sort(key=lambda c: abs(c.distance))
    return candidates
