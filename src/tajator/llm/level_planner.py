"""LLM daily level-planner: once per day, read the daily + premarket picture and
propose the day's support/resistance levels for the fade to trade around.

This is the notes' "pick your own levels every morning," delegated to a model. It
runs ONCE at the open, never per tick. The output is a plain list of `Level`s that
`compute_context` merges ahead of the mechanical detector. On ANY failure (model
down, timeout, bad JSON) it returns an empty list and the bot falls back to
mechanical level detection — the planner can only add information, never break the bot.
"""

from __future__ import annotations

import logging
from datetime import time
from zoneinfo import ZoneInfo

from ..models import Bar, Level

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "bias": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "levels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "role": {"type": "string", "enum": ["support", "resistance"]},
                    "reason": {"type": "string"},
                },
                "required": ["price", "role", "reason"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["bias", "levels", "summary"],
    "additionalProperties": False,
}


def _premarket(bars: list[Bar]) -> list[Bar]:
    return [b for b in bars if b.ts.astimezone(ET).time() < time(9, 30)]


def _fmt_daily(daily: list[Bar], n: int = 30) -> str:
    rows = []
    for b in daily[-n:]:
        rows.append(f"{b.ts.astimezone(ET).date()} O{b.open:.2f} H{b.high:.2f} "
                    f"L{b.low:.2f} C{b.close:.2f}")
    return "\n".join(rows) if rows else "(no daily history)"


def build_prompt(
    symbol: str, daily: list[Bar], prev_high: float | None, prev_low: float | None,
    premarket: list[Bar], spot: float,
) -> str:
    pm = premarket
    pm_line = (
        f"premarket high {max(b.high for b in pm):.2f}, low {min(b.low for b in pm):.2f}, "
        f"last {pm[-1].close:.2f} over {len(pm)} 1-min bars"
        if pm else "premarket: no bars yet"
    )
    return (
        f"You are picking the intraday support/resistance levels for a {symbol} options "
        f"day-trade today. The strategy fades levels: buy calls as price falls INTO support, "
        f"buy puts as price rises INTO resistance, with a tight stop just beyond the level.\n\n"
        f"Current price: {spot:.2f}\n"
        f"Prior day high: {prev_high}  low: {prev_low}\n"
        f"{pm_line}\n\n"
        f"Recent daily bars (oldest→newest):\n{_fmt_daily(daily)}\n\n"
        f"Give 3–6 of the CLEANEST levels price is likely to actually test and react at today "
        f"(prior-day high/low, premarket extremes, obvious multi-day swing highs/lows, round "
        f"numbers only if they coincide). Prefer a few high-quality levels over many marginal "
        f"ones. Each level: price, role (support below / resistance above current price), and a "
        f"one-line reason. Also give an overall bias and a one-line summary."
    )


def plan_levels(
    client, symbol: str, daily: list[Bar], prev_high: float | None,
    prev_low: float | None, bars: list[Bar], spot: float,
) -> list[Level]:
    """Ask the model for today's levels. Returns [] on any failure (caller falls
    back to mechanical detection). `client` has `.complete(prompt, schema) -> dict`."""
    try:
        prompt = build_prompt(symbol, daily, prev_high, prev_low, _premarket(bars), spot)
        data = client.complete(prompt, BRIEFING_SCHEMA)
        out: list[Level] = []
        for item in data.get("levels", []):
            price = float(item["price"])
            if price <= 0:
                continue
            kind = "support" if price <= spot else "resistance"
            out.append(Level(price=round(price, 2), kind=kind, label="llm_level"))
        log.info("LLM planned %d levels for %s (bias=%s): %s",
                 len(out), symbol, data.get("bias"), data.get("summary", "")[:120])
        return out
    except Exception as exc:  # noqa: BLE001 — the planner must never break the session
        log.warning("LLM level planning failed for %s (%s); using mechanical levels", symbol, exc)
        return []
