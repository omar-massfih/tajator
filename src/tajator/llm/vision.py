"""Candlestick image and structured pattern analysis for vision entry mode."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from ..models import Bar, VisionPatternAnalysis

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
