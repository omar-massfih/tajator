"""Mechanical pre-filter: is price approaching a level with speed?

The strategy enters INTO the level (calls while price falls into support,
puts while price rises into resistance) and never chases. Only when this
filter finds a candidate is the LLM consulted.
"""

from __future__ import annotations

from ..models import Bar, Level, SetupCandidate, Snapshot

APPROACH_BAND = 0.003  # price within 0.3% of the level
OVERSHOOT_BAND = 0.001  # slight poke through the level still counts
SPEED_WINDOW = 3  # bars used to measure the move into the level
MIN_SPEED_PCT = 0.0012  # net move over the window, fraction of price


def detect_candidates(
    bars: list[Bar], levels: list[Level], snapshot: Snapshot
) -> list[SetupCandidate]:
    if len(bars) < SPEED_WINDOW + 1:
        return []

    price = snapshot.price
    net_move = price - bars[-1 - SPEED_WINDOW].close  # + = rising, - = falling
    min_speed = MIN_SPEED_PCT * price
    band = APPROACH_BAND * price
    overshoot = OVERSHOOT_BAND * price

    candidates: list[SetupCandidate] = []
    for level in levels:
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
