"""Hard veto layer between the LLM and the broker.

Every rule here is deliberately dumb and non-negotiable. Exits and
scale-outs (risk-reducing) always pass; only entries are gated.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from ..config import Settings
from ..models import Decision, OpenPosition, RiskVerdict, SetupCandidate

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
LEVEL_MATCH_TOL = 0.002  # decision level must be within 0.2% of a detected candidate's
STOP_MIN_CENTS = 20
STOP_MAX_CENTS = 60


def entry_blockers(
    *,
    now: datetime,
    position: OpenPosition | None,
    trades_today: int,
    settings: Settings,
) -> list[str]:
    """Decision-independent entry vetoes — cheap enough to run before the LLM."""
    violations: list[str] = []
    now_et = now.astimezone(ET)

    if settings.kill_switch_file.exists():
        violations.append("kill switch file present — no new entries")
    if now_et.weekday() >= 5:
        violations.append("market closed (weekend)")
    if not (RTH_OPEN <= now_et.time() <= settings.no_new_entries_after):
        violations.append(
            f"entries allowed only 09:30–{settings.no_new_entries_after:%H:%M} ET (now {now_et:%H:%M})"
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
) -> RiskVerdict:
    if decision.action in ("wait", "scale_out", "exit"):
        return RiskVerdict(approved=True)

    violations = entry_blockers(
        now=now, position=position, trades_today=trades_today, settings=settings
    )

    direction = "call" if decision.action == "enter_call" else "put"
    matched = _matching_candidate(decision, direction, candidates)
    if matched is None:
        violations.append(
            f"no detected {direction} setup matches level {decision.level_price} — LLM may not invent trades"
        )

    violations.extend(_stop_violations(decision, direction))

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


def _stop_violations(decision: Decision, direction: str) -> list[str]:
    if decision.stop_price is None or decision.level_price is None:
        return ["entry requires both level_price and stop_price (plan before entry)"]
    out: list[str] = []
    dist = decision.stop_price - decision.level_price
    if direction == "call" and dist >= 0:
        out.append("call stop must be BELOW the support level")
    if direction == "put" and dist <= 0:
        out.append("put stop must be ABOVE the resistance level")
    if not (STOP_MIN_CENTS / 100 <= abs(dist) <= STOP_MAX_CENTS / 100):
        out.append(
            f"stop distance {abs(dist):.2f} outside the {STOP_MIN_CENTS}–{STOP_MAX_CENTS} cent rule"
        )
    return out
