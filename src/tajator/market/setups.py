"""Mechanical pre-filter: is price approaching a level with speed?

The strategy enters INTO the level (calls while price falls into support,
puts while price rises into resistance) and never chases. Only when this
filter finds a candidate is the LLM consulted.

Not every level is tradable: bare swing points are chart context only, and a
level sitting a few cents from the day's open leaves no room for a move
(strategy notes 08). The risk gate only admits detected candidates, so
filtering here is what keeps the LLM from trading those levels.

Speed cuts both ways: the approach must be fast enough to mean something, but
fading a *very* fast move with a 20-60 cent stop is an overshoot machine — in
the 2026-07-06..10 backtests every such entry was blown through within two
bars. Above FAST_APPROACH_SPEED_MULT × the minimum speed, the entry bar must
show rejection at the level (a wick back in our direction) before it counts.
"""

from __future__ import annotations

from ..models import Bar, Level, SetupCandidate, Snapshot
from .indicators import bars_to_df, session_df

APPROACH_BAND = 0.003  # price within 0.3% of the level
OVERSHOOT_BAND = 0.001  # slight poke through the level still counts
SPEED_WINDOW = 3  # bars used to measure the move into the level
MIN_SPEED_PCT = 0.0012  # net move over the window, fraction of price
MIN_LEVEL_DIST_FROM_OPEN_PCT = 0.003  # ~$1.50 on a $500 name
FAST_APPROACH_SPEED_MULT = 2.0  # approaches ≥ this × min speed need rejection first
REJECTION_WICK_MIN_FRAC = 0.25  # wick back off the level, fraction of the bar's range

NON_TRADABLE_LABELS = frozenset({"swing_high", "swing_low"})
TRADE_FLIPPED_LEVELS = False  # role-reversed levels (broken support as resistance, ...) don't trade
ENTRY_CONFIRMATION = "immediate"

# The side each level formed on. When its current kind (which follows price)
# disagrees, price has already broken through it — "fading the reclaim" of a
# broken level is a momentum bet, not the notes' S/R fade, and it was the
# worst-performing entry class in the 2026-07-06..10 backtests.
NATURAL_KIND = {
    "prev_day_high": "resistance",
    "premarket_high": "resistance",
    "double_top": "resistance",
    "swing_high": "resistance",
    "prev_day_low": "support",
    "premarket_low": "support",
    "double_bottom": "support",
    "swing_low": "support",
}


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
    fast_approach_mult: float = FAST_APPROACH_SPEED_MULT,
    rejection_wick_frac: float = REJECTION_WICK_MIN_FRAC,
    trade_flipped_levels: bool = TRADE_FLIPPED_LEVELS,
    entry_confirmation: str = ENTRY_CONFIRMATION,
) -> list[SetupCandidate]:
    if len(bars) < speed_window + 1:
        return []

    price = snapshot.price
    net_move = price - bars[-1 - speed_window].close  # + = rising, - = falling
    min_speed = min_speed_pct * price
    band = approach_band * price
    overshoot = overshoot_band * price
    day_open = _day_open(bars)
    # Rejection gate only kicks in above a *measurable* fast threshold —
    # a min_speed_pct=0 override means "admit slow drifts", not "everything is fast".
    fast = fast_approach_mult * min_speed if min_speed > 0 and rejection_wick_frac > 0 else None

    candidates: list[SetupCandidate] = []
    for level in levels:
        if level.label in NON_TRADABLE_LABELS:
            continue
        if not trade_flipped_levels:
            natural = NATURAL_KIND.get(level.label)
            if natural is not None and level.kind != natural:
                continue  # broken level retested from the other side — chart context only
        if (
            day_open is not None
            and abs(level.price - day_open) < min_dist_from_open_pct * day_open
        ):
            continue
        diff = price - level.price
        if level.kind == "support":
            # Call setup: price falling into support from above (or a slight poke below).
            if -overshoot <= diff <= band and net_move <= -min_speed:
                if entry_confirmation == "touch_rejection" and not _touch_rejected(
                    bars[-1], level.price, "call", rejection_wick_frac
                ):
                    continue
                if fast is not None and -net_move >= fast and not _shows_rejection(bars[-1], "call", rejection_wick_frac):
                    continue  # fast flush, entry bar closed on its low — no proof the level held
                candidates.append(
                    SetupCandidate(
                        direction="call",
                        level=level,
                        distance=round(diff, 2),
                        speed=round(net_move, 2),
                        note=f"falling into {level.label} @ {level.price}",
                        regime=snapshot.regime,
                        quality_score=_quality_score(level, bars[-1], "call", snapshot),
                    )
                )
        else:
            # Put setup: price rising into resistance from below (or a slight poke above).
            if -overshoot <= -diff <= band and net_move >= min_speed:
                if entry_confirmation == "touch_rejection" and not _touch_rejected(
                    bars[-1], level.price, "put", rejection_wick_frac
                ):
                    continue
                if fast is not None and net_move >= fast and not _shows_rejection(bars[-1], "put", rejection_wick_frac):
                    continue  # fast squeeze, entry bar closed on its high — no proof the level held
                candidates.append(
                    SetupCandidate(
                        direction="put",
                        level=level,
                        distance=round(-diff, 2),
                        speed=round(net_move, 2),
                        note=f"rising into {level.label} @ {level.price}",
                        regime=snapshot.regime,
                        quality_score=_quality_score(level, bars[-1], "put", snapshot),
                    )
                )
    # Closest level first — that's the trade the LLM should judge.
    candidates.sort(key=lambda c: abs(c.distance))
    return candidates


def _shows_rejection(bar: Bar, direction: str, min_frac: float) -> bool:
    """Did the entry bar wick back off the level in our favor?

    Call at support: price must have closed off the bar's low by at least
    min_frac of the range (a lower wick — sellers rejected). Put at
    resistance: closed off the high. A zero-range bar proves nothing."""
    rng = bar.high - bar.low
    if rng <= 0:
        return False
    wick = bar.close - bar.low if direction == "call" else bar.high - bar.close
    return wick >= min_frac * rng


def _touch_rejected(bar: Bar, level: float, direction: str, min_frac: float) -> bool:
    """Require the level to trade and the bar to close back on the favorable side."""
    touched = bar.low <= level if direction == "call" else bar.high >= level
    reclaimed = bar.close > level if direction == "call" else bar.close < level
    return touched and reclaimed and _shows_rejection(bar, direction, min_frac)


def _quality_score(level: Level, bar: Bar, direction: str, snapshot: Snapshot) -> float:
    """Transparent pre-trade score; recorded before any score filter is applied."""
    source = {
        "prev_day_high": 3.0, "prev_day_low": 3.0,
        "premarket_high": 2.0, "premarket_low": 2.0,
        "double_top": 2.5, "double_bottom": 2.5,
    }.get(level.label, 1.0)
    rejection = 1.0 if _touch_rejected(bar, level.price, direction, 0.25) else 0.0
    vwap_context = 0.0
    if snapshot.vwap is not None:
        favorable = snapshot.price < snapshot.vwap if direction == "call" else snapshot.price > snapshot.vwap
        vwap_context = 0.5 if favorable else 0.0
    return round(source + rejection + vwap_context, 2)
