"""Candlestick image and structured pattern analysis for vision entry mode."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from ..models import Bar, Decision, Level, SetupCandidate, Snapshot, VisionPatternAnalysis

if TYPE_CHECKING:
    from ..config import Settings

log = logging.getLogger(__name__)

CHART_BARS = 120
CHART_WIDTH = 1100
CHART_HEIGHT = 700

PATTERN_DIRECTIONS = {
    "double_bottom": "call",
    "inverse_head_and_shoulders": "call",
    "triangle_breakout_up": "call",
    "double_top": "put",
    "head_and_shoulders": "put",
    "triangle_breakout_down": "put",
}

VISION_SYSTEM_PROMPT = """You are a conservative visual chart-pattern classifier.

Inspect only the supplied candlestick/volume image and compare it with this fixed catalog:
- double top / double bottom
- head and shoulders / inverse head and shoulders
- contracting triangle with an upside / downside breakout

These patterns are hypotheses, not guaranteed edges. Select enter_call or enter_put only when:
1. the whole pattern is visible in completed candles,
2. the most recent completed candles clearly confirm the neckline/boundary break,
3. the current price has not run far beyond that breakout.

Otherwise return wait. Never invent news, order flow, option prices, unseen candles, or a pattern
outside the catalog. breakout_price is the visible neckline/boundary. invalidation_price is a
visible structural level on the opposite side of the thesis. Keep evidence observable and short.
"""


@dataclass(frozen=True)
class ChartImage:
    png: bytes
    bar_count: int
    sha256: str
    width: int = CHART_WIDTH
    height: int = CHART_HEIGHT


def render_bar_chart(symbol: str, bars: list[Bar], *, limit: int = CHART_BARS) -> ChartImage:
    """Render completed OHLCV bars to a deterministic PNG suitable for model input."""
    selected = bars[-limit:]
    if len(selected) < 2:
        raise ValueError("vision chart requires at least two bars")

    image = Image.new("RGB", (CHART_WIDTH, CHART_HEIGHT), "#0b1220")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    left, right = 70, CHART_WIDTH - 74
    price_top, price_bottom = 52, 520
    volume_top, volume_bottom = 555, 650
    prices = [value for bar in selected for value in (bar.low, bar.high)]
    low, high = min(prices), max(prices)
    pad = max((high - low) * 0.06, high * 0.0002, 0.01)
    low, high = low - pad, high + pad
    span = high - low
    max_volume = max((bar.volume for bar in selected), default=0.0) or 1.0

    def y_price(value: float) -> float:
        return price_bottom - ((value - low) / span) * (price_bottom - price_top)

    plot_width = right - left
    step = plot_width / len(selected)
    body_width = max(2, min(8, int(step * 0.62)))

    for index in range(6):
        y = price_top + index * (price_bottom - price_top) / 5
        price = high - index * span / 5
        draw.line((left, y, right, y), fill="#24324a", width=1)
        draw.text((right + 8, y - 6), f"{price:.2f}", fill="#a8b3c7", font=font)

    for index, bar in enumerate(selected):
        x = left + (index + 0.5) * step
        up = bar.close >= bar.open
        color = "#2dd4bf" if up else "#fb7185"
        draw.line((x, y_price(bar.high), x, y_price(bar.low)), fill=color, width=1)
        y_open, y_close = y_price(bar.open), y_price(bar.close)
        top, bottom = min(y_open, y_close), max(y_open, y_close)
        if bottom - top < 1:
            draw.line((x - body_width / 2, top, x + body_width / 2, top), fill=color, width=2)
        else:
            draw.rectangle(
                (x - body_width / 2, top, x + body_width / 2, bottom),
                fill=color,
            )
        volume_height = (bar.volume / max_volume) * (volume_bottom - volume_top)
        draw.rectangle(
            (x - body_width / 2, volume_bottom - volume_height, x + body_width / 2, volume_bottom),
            fill=color,
        )

    first, last = selected[0], selected[-1]
    draw.text(
        (left, 18),
        f"{symbol}  1-minute completed bars  {first.ts:%Y-%m-%d %H:%M} - {last.ts:%H:%M}",
        fill="#e5e7eb",
        font=font,
    )
    draw.text((left, 532), "Volume", fill="#a8b3c7", font=font)
    draw.text((left, 668), "Oldest", fill="#a8b3c7", font=font)
    draw.text((right - 34, 668), "Latest", fill="#a8b3c7", font=font)
    draw.rectangle((left, price_top, right, price_bottom), outline="#52627a")
    draw.rectangle((left, volume_top, right, volume_bottom), outline="#52627a")

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    png = output.getvalue()
    return ChartImage(png=png, bar_count=len(selected), sha256=hashlib.sha256(png).hexdigest())


def vision_messages(context: str, chart: ChartImage) -> list[dict]:
    encoded = base64.b64encode(chart.png).decode("ascii")
    return [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": context},
                {
                    "type": "image",
                    "base64": encoded,
                    "mime_type": "image/png",
                },
            ],
        },
    ]


def decide_vision(llm, context: str, chart: ChartImage) -> VisionPatternAnalysis:
    """Ask the multimodal model; every failure degrades to a non-trading read."""
    try:
        analysis = llm.invoke(vision_messages(context, chart))
        if analysis is None:
            raise ValueError("LLM returned no parseable structured output")
        return analysis
    except Exception as exc:  # noqa: BLE001 - entry must fail closed
        log.warning("vision pattern analysis failed (%s) - defaulting to wait", exc)
        return VisionPatternAnalysis(
            action="wait",
            pattern="none",
            status="none",
            reasoning=f"vision LLM error, defaulting to wait: {exc}",
        )


def validate_vision_entry(
    analysis: VisionPatternAnalysis,
    bars: list[Bar],
    snapshot: Snapshot,
    settings: "Settings",
) -> tuple[Decision, SetupCandidate | None, list[str]]:
    """Turn a chart read into an ordinary candidate only after causal checks."""
    if analysis.action == "wait":
        return Decision(action="wait", reasoning=analysis.reasoning), None, []

    violations: list[str] = []
    expected_direction = PATTERN_DIRECTIONS.get(analysis.pattern)
    direction = "call" if analysis.action == "enter_call" else "put"
    if expected_direction is None:
        violations.append(f"pattern {analysis.pattern!r} is not in the executable catalog")
    elif expected_direction != direction:
        violations.append(
            f"{analysis.pattern} maps to {expected_direction}, not {direction}"
        )
    if analysis.status != "confirmed":
        violations.append(f"pattern status is {analysis.status}, not confirmed")
    if analysis.confidence < settings.vision_pattern_min_confidence:
        violations.append(
            f"confidence {analysis.confidence:.2f} below "
            f"{settings.vision_pattern_min_confidence:.2f} minimum"
        )
    breakout, invalidation = analysis.breakout_price, analysis.invalidation_price
    if breakout is None or invalidation is None:
        violations.append("confirmed pattern requires breakout_price and invalidation_price")
    else:
        chart_bars = bars[-settings.vision_pattern_lookback_bars :]
        chart_low = min(bar.low for bar in chart_bars)
        chart_high = max(bar.high for bar in chart_bars)
        if not chart_low <= breakout <= chart_high:
            violations.append("breakout_price is outside the visible chart range")
        if not chart_low <= invalidation <= chart_high:
            violations.append("invalidation_price is outside the visible chart range")
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
        chase_limit = breakout * settings.vision_pattern_max_chase_pct
        if chase < -1e-9 or chase > chase_limit + 1e-9:
            violations.append(
                f"price is {chase:+.2f} from breakout; exceeds no-chase band"
            )

    if violations:
        return (
            Decision(
                action="wait",
                confidence="low",
                reasoning="vision pattern rejected: " + "; ".join(violations),
            ),
            None,
            violations,
        )

    assert breakout is not None  # established above
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
        reasoning=f"vision {analysis.pattern}: {analysis.reasoning}",
    )
    level = Level(
        price=round(breakout, 2),
        kind="support" if direction == "call" else "resistance",
        label=f"vision_{analysis.pattern}",
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
