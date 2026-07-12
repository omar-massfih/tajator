"""Hard veto layer between the LLM and the broker.

Every rule here is deliberately dumb and non-negotiable. Exits and
scale-outs (risk-reducing) always pass; only entries are gated.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from ..models import Decision, OpenPosition, RiskVerdict, SetupCandidate

if TYPE_CHECKING:  # runtime import would be circular: config imports our constants
    from ..config import Settings

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
LEVEL_MATCH_TOL = 0.002  # decision level must be within 0.2% of a detected candidate's
STOP_MIN_CENTS = 20
STOP_MAX_CENTS = 60
STOP_COOLDOWN_MINUTES = 30  # a level that stopped us out is untradable this long


def cooldown_filter(
    candidates: list[SetupCandidate], cooled_levels: list[float]
) -> tuple[list[SetupCandidate], list[SetupCandidate]]:
    """Split candidates into (tradable, cooled).

    A level that just stopped us out may not be re-entered until its cooldown
    lapses: in the 2026-07-06..10 backtests, re-entries at the freshly failed
    level (1-10 minutes after the stop) accounted for nearly all of the losses.
    Filtered here rather than in the risk gate so the LLM is never even asked."""
    if not cooled_levels:
        return candidates, []
    kept, dropped = [], []
    for c in candidates:
        cooled = any(abs(c.level.price - p) <= LEVEL_MATCH_TOL * c.level.price for p in cooled_levels)
        (dropped if cooled else kept).append(c)
    return kept, dropped


def entry_blockers(
    *,
    now: datetime,
    position: OpenPosition | None,
    trades_today: int,
    settings: Settings,
    delayed_data: bool = False,
) -> list[str]:
    """Decision-independent entry vetoes — cheap enough to run before the LLM."""
    violations: list[str] = []
    now_et = now.astimezone(ET)

    if settings.kill_switch_file.exists():
        violations.append("kill switch file present — no new entries")
    if delayed_data and settings.block_entries_on_delayed_data:
        violations.append(
            "market data is DELAYED (no live subscription) — the paper fill engine "
            "cannot fill against data we cannot see; no new entries"
        )
    if now_et.weekday() >= 5:
        violations.append("market closed (weekend)")
    if not (settings.no_new_entries_before <= now_et.time() <= settings.no_new_entries_after):
        violations.append(
            f"entries allowed only {settings.no_new_entries_before:%H:%M}–"
            f"{settings.no_new_entries_after:%H:%M} ET (now {now_et:%H:%M})"
        )
    if trades_today >= settings.max_trades_per_day:
        violations.append(f"max {settings.max_trades_per_day} trades/day reached")
    if position is not None:
        violations.append("a position is already open — one at a time")
    return violations


def check(
    decision: Decision,
    *,
    now: datetime,
    position: OpenPosition | None,
    trades_today: int,
    candidates: list[SetupCandidate],
    settings: Settings,
    delayed_data: bool = False,
    snapshot_price: float | None = None,
) -> RiskVerdict:
    if decision.action in ("wait", "scale_out", "exit"):
        return RiskVerdict(approved=True)

    violations = entry_blockers(
        now=now, position=position, trades_today=trades_today, settings=settings,
        delayed_data=delayed_data,
    )

    direction = "call" if decision.action == "enter_call" else "put"
    matched = _matching_candidate(decision, direction, candidates)
    if matched is None:
        violations.append(
            f"no detected {direction} setup matches level {decision.level_price} — LLM may not invent trades"
        )

    violations.extend(_stop_violations(decision, direction, settings))
    if (
        settings.max_entry_to_stop_cents is not None
        and decision.stop_price is not None
        and snapshot_price is not None
    ):
        # Candidate matching anchors the level; actual risk is measured from
        # the current underlying price, not merely from level to stop.
        matched_risk = abs(snapshot_price - decision.stop_price)
        if matched_risk is not None and matched_risk > settings.max_entry_to_stop_cents / 100:
            violations.append(
                f"actual entry-to-stop risk {matched_risk:.2f} exceeds "
                f"{settings.max_entry_to_stop_cents / 100:.2f} maximum"
            )

    # MAX_PREMIUM_USD is enforced at execution: size_entry sizes the order down
    # to fit the budget and skips the entry if even one contract busts it.

    return RiskVerdict(approved=not violations, violations=violations)


def _matching_candidate(
    decision: Decision, direction: str, candidates: list[SetupCandidate]
) -> SetupCandidate | None:
    if decision.level_price is None:
        return None
    for c in candidates:
        if c.direction == direction and abs(c.level.price - decision.level_price) <= LEVEL_MATCH_TOL * c.level.price:
            return c
    return None


def _stop_violations(decision: Decision, direction: str, settings: Settings) -> list[str]:
    if decision.stop_price is None or decision.level_price is None:
        return ["entry requires both level_price and stop_price (plan before entry)"]
    out: list[str] = []
    dist = decision.stop_price - decision.level_price
    if direction == "call" and dist >= 0:
        out.append("call stop must be BELOW the support level")
    if direction == "put" and dist <= 0:
        out.append("put stop must be ABOVE the resistance level")
    min_cents, max_cents = settings.stop_min_cents, settings.stop_max_cents
    if not (min_cents / 100 <= abs(dist) <= max_cents / 100):
        out.append(
            f"stop distance {abs(dist):.2f} outside the {min_cents}–{max_cents} cent rule"
        )
    return out
