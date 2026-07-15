"""Structured OHLCV and pivot analysis for the pattern-data entry mode."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

from ..models import Bar, Decision, Level, PatternAnalysis, SetupCandidate, Snapshot

if TYPE_CHECKING:
    from ..config import Settings

log = logging.getLogger(__name__)

PATTERN_DATA_BARS = 120
PIVOT_WINDOW = 3
MAX_PIVOTS = 24

PATTERN_DIRECTIONS = {
    "double_bottom": "call",
    "inverse_head_and_shoulders": "call",
    "triangle_breakout_up": "call",
    "double_top": "put",
    "head_and_shoulders": "put",
    "triangle_breakout_down": "put",
}

PATTERN_SYSTEM_PROMPT = """You are a conservative numerical chart-pattern classifier.

Inspect only the supplied completed OHLCV rows and objective pivot list. Compare them with this
fixed catalog:
- double top / double bottom
- head and shoulders / inverse head and shoulders
- contracting triangle with an upside / downside breakout

These patterns are hypotheses, not guaranteed edges. Select enter_call or enter_put only when:
1. the whole pattern is present in the supplied completed bars,
2. the most recent completed bars clearly confirm the neckline/boundary break,
3. the current price has not run far beyond that breakout.

Otherwise return wait. Never invent news, order flow, option prices, unseen bars, or a pattern
outside the catalog. breakout_price is the numerical neckline/boundary. invalidation_price is a
structural price on the opposite side of the thesis. Keep evidence observable and short.
"""


@dataclass(frozen=True)
class PatternData:
    text: str
    bar_count: int
    pivot_count: int
    sha256: str


def build_pattern_data(
    symbol: str,
    bars: list[Bar],
    *,
    limit: int = PATTERN_DATA_BARS,
    pivot_window: int = PIVOT_WINDOW,
) -> PatternData:
    """Serialize completed bars and causal swing pivots into compact deterministic text."""
    selected = bars[-limit:]
    if len(selected) < pivot_window * 2 + 1:
        raise ValueError(
            f"pattern data requires at least {pivot_window * 2 + 1} completed bars"
        )

    nonzero_volumes = [bar.volume for bar in selected if bar.volume > 0]
    median_volume = median(nonzero_volumes) if nonzero_volumes else 0.0
    lines = [
        f"symbol={symbol}; timeframe=1m; completed_bars={len(selected)}",
        "bars: index,time,open,high,low,close,volume,volume_vs_median",
    ]
    for index, bar in enumerate(selected):
        relative_volume = bar.volume / median_volume if median_volume else 0.0
        lines.append(
            f"{index},{bar.ts:%Y-%m-%dT%H:%M:%S%z},"
            f"{bar.open:.4f},{bar.high:.4f},{bar.low:.4f},{bar.close:.4f},"
            f"{bar.volume:.0f},{relative_volume:.2f}"
        )

    pivots: list[str] = []
    for index in range(pivot_window, len(selected) - pivot_window):
        bar = selected[index]
        neighbors = (
            selected[index - pivot_window : index]
            + selected[index + 1 : index + pivot_window + 1]
        )
        if (
            all(bar.high >= other.high for other in neighbors)
            and any(bar.high > other.high for other in neighbors)
        ):
            pivots.append(f"{index},high,{bar.high:.4f},{bar.ts:%H:%M}")
        if (
            all(bar.low <= other.low for other in neighbors)
            and any(bar.low < other.low for other in neighbors)
        ):
            pivots.append(f"{index},low,{bar.low:.4f},{bar.ts:%H:%M}")
    pivots = pivots[-MAX_PIVOTS:]
    lines += ["pivots: index,kind,price,time", *(pivots or ["none"])]

    text = "\n".join(lines)
    return PatternData(
        text=text,
        bar_count=len(selected),
        pivot_count=len(pivots),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def pattern_messages(context: str, data: PatternData) -> list[dict]:
    return [
        {"role": "system", "content": PATTERN_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\n{data.text}"},
    ]


def decide_pattern(llm, context: str, data: PatternData) -> PatternAnalysis:
    """Ask the structured classifier; every failure degrades to a non-trading read."""
    try:
        analysis = llm.invoke(pattern_messages(context, data))
        if analysis is None:
            raise ValueError("LLM returned no parseable structured output")
        return analysis
    except Exception as exc:  # noqa: BLE001 - entry must fail closed
        log.warning("pattern-data analysis failed (%s) - defaulting to wait", exc)
        return PatternAnalysis(
            action="wait",
            pattern="none",
            status="none",
            reasoning=f"pattern-data LLM error, defaulting to wait: {exc}",
        )


def validate_pattern_entry(
    analysis: PatternAnalysis,
    bars: list[Bar],
    snapshot: Snapshot,
    settings: "Settings",
) -> tuple[Decision, SetupCandidate | None, list[str]]:
    """Turn a pattern read into an ordinary candidate only after causal checks."""
    if analysis.action == "wait":
        return Decision(action="wait", reasoning=analysis.reasoning), None, []

    violations: list[str] = []
    expected_direction = PATTERN_DIRECTIONS.get(analysis.pattern)
    direction = "call" if analysis.action == "enter_call" else "put"
    if expected_direction is None:
        violations.append(f"pattern {analysis.pattern!r} is not in the executable catalog")
    elif expected_direction != direction:
        violations.append(f"{analysis.pattern} maps to {expected_direction}, not {direction}")
    if analysis.status != "confirmed":
        violations.append(f"pattern status is {analysis.status}, not confirmed")
    if analysis.confidence < settings.pattern_data_min_confidence:
        violations.append(
            f"confidence {analysis.confidence:.2f} below "
            f"{settings.pattern_data_min_confidence:.2f} minimum"
        )
    breakout, invalidation = analysis.breakout_price, analysis.invalidation_price
    if breakout is None or invalidation is None:
        violations.append("confirmed pattern requires breakout_price and invalidation_price")
    else:
        visible_bars = bars[-settings.pattern_data_lookback_bars :]
        visible_low = min(bar.low for bar in visible_bars)
        visible_high = max(bar.high for bar in visible_bars)
        if not visible_low <= breakout <= visible_high:
            violations.append("breakout_price is outside the supplied data range")
        if not visible_low <= invalidation <= visible_high:
            violations.append("invalidation_price is outside the supplied data range")
        if direction == "call" and invalidation >= breakout:
            violations.append("call invalidation must be below the breakout")
        if direction == "put" and invalidation <= breakout:
            violations.append("put invalidation must be above the breakout")

        recent = bars[-4:]
        if direction == "call":
            crossed = any(a.close <= breakout < b.close for a, b in zip(recent, recent[1:]))
            chase = snapshot.price - breakout
            if not crossed or snapshot.price < breakout:
                violations.append("completed bars do not confirm an upside breakout")
        else:
            crossed = any(a.close >= breakout > b.close for a, b in zip(recent, recent[1:]))
            chase = breakout - snapshot.price
            if not crossed or snapshot.price > breakout:
                violations.append("completed bars do not confirm a downside breakout")
        chase_limit = breakout * settings.pattern_data_max_chase_pct
        if chase < -1e-9 or chase > chase_limit + 1e-9:
            violations.append(f"price is {chase:+.2f} from breakout; exceeds no-chase band")

    if violations:
        return (
            Decision(
                action="wait",
                confidence="low",
                reasoning="pattern-data signal rejected: " + "; ".join(violations),
            ),
            None,
            violations,
        )

    assert breakout is not None
    buffer = settings.stop_buffer_cents / 100
    if settings.stop_atr_multiplier is not None and snapshot.atr is not None:
        buffer = min(
            settings.stop_max_cents / 100,
            max(settings.stop_min_cents / 100, snapshot.atr * settings.stop_atr_multiplier),
        )
    stop = breakout - buffer if direction == "call" else breakout + buffer
    confidence = "high" if analysis.confidence >= 0.9 else "medium"
    decision = Decision(
        action=analysis.action,
        level_price=round(breakout, 2),
        stop_price=round(stop, 2),
        confidence=confidence,
        reasoning=f"pattern-data {analysis.pattern}: {analysis.reasoning}",
    )
    level = Level(
        price=round(breakout, 2),
        kind="support" if direction == "call" else "resistance",
        label=f"pattern_{analysis.pattern}",
    )
    speed_window = min(settings.speed_window_bars, len(bars) - 1)
    speed = bars[-1].close - bars[-1 - speed_window].close if speed_window else 0.0
    candidate = SetupCandidate(
        direction=direction,
        level=level,
        distance=snapshot.price - level.price,
        speed=speed,
        note=f"confirmed {analysis.pattern} at {level.price:.2f}",
        regime=snapshot.regime,
        quality_score=round(analysis.confidence * 5, 2),
        ranking_score=round(analysis.confidence * 5, 2),
    )
    return decision, candidate, []
