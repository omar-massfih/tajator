"""LLM decision node: compact snapshot in, structured Decision out.

Any LLM failure degrades safely: entry questions fall back to "wait",
scale questions fall back to "scale one piece now".
"""

from __future__ import annotations

import logging

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from ..models import (
    Bar,
    Decision,
    Level,
    LevelWatch,
    MorningBriefing,
    OpenPosition,
    SetupCandidate,
    Snapshot,
    VisionPatternAnalysis,
)
from ..trade.position import active_stop_price
from .prompts import PREP_SYSTEM_PROMPT, SYSTEM_PROMPT

log = logging.getLogger(__name__)

LLM_TIMEOUT_S = 20
RECENT_BARS = 10


def build_llm(model_string: str, output_model: type[BaseModel] = Decision):
    """`model_string` is an init_chat_model id like 'openai:gpt-5.1', or
    'codex' / 'codex:<model>' to use the Codex CLI (ChatGPT subscription)."""
    if model_string == "codex" or model_string.startswith("codex:"):
        from .codex import BRIEFING_SCHEMA, DECISION_SCHEMA, CodexDecider

        if output_model not in (Decision, MorningBriefing):
            raise ValueError("Codex CLI models do not support vision-pattern image input")
        _, _, model = model_string.partition(":")
        schema = BRIEFING_SCHEMA if output_model is MorningBriefing else DECISION_SCHEMA
        return CodexDecider(model=model or None, output_model=output_model, schema=schema)
    llm = init_chat_model(model_string, timeout=LLM_TIMEOUT_S)
    return llm.with_structured_output(output_model)


def build_vision_llm(model_string: str):
    """Build the structured multimodal classifier used only by vision mode."""
    return build_llm(model_string, output_model=VisionPatternAnalysis)


def format_snapshot(
    bars: list[Bar],
    snapshot: Snapshot,
    levels: list[Level],
    candidates: list[SetupCandidate],
    trades_today: int,
    position: OpenPosition | None = None,
    manage_note: str | None = None,
) -> str:
    def fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "n/a"

    lines = [
        f"{snapshot.symbol} @ {snapshot.ts:%H:%M} ET — price {snapshot.price:.2f}",
        f"ema9 {fmt(snapshot.ema9)} | ema50 {fmt(snapshot.ema50)} | vwap {fmt(snapshot.vwap)}"
        f" | HOD {fmt(snapshot.hod)} | LOD {fmt(snapshot.lod)}",
        f"trades taken today: {trades_today}",
        "",
        "last bars (open/high/low/close):",
    ]
    for b in bars[-RECENT_BARS:]:
        lines.append(f"  {b.ts:%H:%M}  {b.open:.2f} {b.high:.2f} {b.low:.2f} {b.close:.2f}")

    multi = snapshot.multi_timeframe
    if multi is not None and multi.enabled:
        daily = multi.daily
        five = multi.five_minute
        lines += [
            "",
            "HIGHER TIMEFRAMES (context/ranking only; never independent trade authorization):",
            f"  daily bias {daily.bias} | EMA20 {fmt(daily.ema20)} | EMA50 {fmt(daily.ema50)} "
            f"| EMA20 slope(5) {fmt(daily.ema20_slope_5)} | ATR14 {fmt(daily.atr14)}",
        ]
        if daily.reference_levels:
            refs = ", ".join(
                f"{level.price:.2f} ({level.label})" for level in daily.reference_levels
            )
            lines.append(f"  daily reference-only levels: {refs}")
        lines.append(
            f"  5m trend {five.trend} | EMA9 {fmt(five.ema9)} | "
            f"EMA20 {fmt(five.ema20)} | ATR14 {fmt(five.atr14)}"
        )
        lines.append("  completed 5m candles (open/high/low/close):")
        for bar in five.completed_bars:
            lines.append(
                f"    {bar.ts:%H:%M}  {bar.open:.2f} {bar.high:.2f} {bar.low:.2f} {bar.close:.2f}"
            )
        if five.forming_bar is not None:
            bar = five.forming_bar
            lines.append(
                f"  FORMING 5m candle {bar.ts:%H:%M}: "
                f"{bar.open:.2f} {bar.high:.2f} {bar.low:.2f} {bar.close:.2f}"
            )

    lines += ["", "levels:"]
    for l in levels:
        lines.append(f"  {l.price:.2f}  {l.kind:<10} ({l.label})")

    if candidates:
        lines += ["", "DETECTED SETUP CANDIDATES (only these are tradeable):"]
        for c in candidates:
            pa = c.price_action
            labels = ", ".join(pa.reaction_labels) if pa.reaction_labels else "none"
            range_atr = f"{pa.range_atr:.2f}" if pa.range_atr is not None else "n/a"
            rel_volume = f"{pa.relative_volume:.2f}" if pa.relative_volume is not None else "n/a"
            lines.append(
                f"  {c.direction.upper()} — {c.note}, distance {c.distance:+.2f}, "
                f"3-bar move {c.speed:+.2f}, quality {c.quality_score:.2f}, "
                f"HTF {c.higher_timeframe_score.total:+.2f}, rank {c.ranking_score:.2f}"
            )
            htf = c.higher_timeframe_score
            lines.append(
                "    HTF score: "
                f"daily-bias {htf.daily_bias:+.2f}, confluence {htf.daily_confluence:+.2f}, "
                f"5m-trend {htf.five_minute_trend:+.2f}, reaction {htf.five_minute_reaction:+.2f}"
            )
            lines.append(
                "    price action: "
                f"body {pa.body_fraction:.2f}, upper wick {pa.upper_wick_fraction:.2f}, "
                f"lower wick {pa.lower_wick_fraction:.2f}, close location {pa.close_location:.2f}, "
                f"close-off-extreme {pa.close_rejection_fraction:.2f}, "
                f"range/ATR {range_atr}, relative volume {rel_volume}"
            )
            lines.append(
                "    level reaction: "
                f"touched={pa.touched}, reclaimed={pa.reclaimed}, "
                f"break-and-reclaim={pa.break_and_reclaim}, penetration={pa.penetration:.2f}, "
                f"rejections={pa.rejection_count}, clean-slice={pa.clean_slice}; labels: {labels}"
            )
    else:
        lines += ["", "no setup candidates detected this tick"]

    if position is not None:
        p = position.plan
        lines += [
            "",
            f"OPEN POSITION: {position.qty_remaining}x {position.contract.local_name} "
            f"({p.direction}), entry equity {p.entry_equity_price:.2f}, "
            f"stop {active_stop_price(position):.2f}, pieces sold {position.pieces_sold}/{len(p.pieces)}",
        ]
    if manage_note:
        lines += ["", f"QUESTION: {manage_note}"]
    return "\n".join(lines)


def _ask(llm, user_text: str) -> Decision:
    decision = llm.invoke(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
    )
    if decision is None:  # with_structured_output returns None on an unparseable answer
        raise ValueError("LLM returned no parseable structured output")
    return decision


def decide_entry(llm, user_text: str) -> Decision:
    try:
        return _ask(llm, user_text)
    except Exception as exc:  # noqa: BLE001 — any LLM failure must not stop the loop
        log.warning("LLM entry decision failed (%s) — defaulting to wait", exc)
        return Decision(action="wait", reasoning=f"LLM error, defaulting to wait: {exc}")


def decide_scale(llm, user_text: str) -> Decision:
    try:
        d = _ask(llm, user_text)
        if d.action not in ("scale_out", "wait", "exit"):
            return Decision(action="scale_out", reasoning=f"LLM answered {d.action}; scaling by default")
        return d
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM scale decision failed (%s) — scaling one piece", exc)
        return Decision(action="scale_out", reasoning=f"LLM error, scaling by default: {exc}")


def format_prep_snapshot(symbol: str, snapshot: Snapshot, levels: list[Level]) -> str:
    lines = [
        f"{symbol} pre-market prep @ {snapshot.ts:%H:%M} ET — price {snapshot.price:.2f}",
        "",
        "levels (price, kind, label, distance from current price):",
    ]
    for l in levels:
        distance = l.price - snapshot.price
        lines.append(f"  {l.price:.2f}  {l.kind:<10} ({l.label})  distance {distance:+.2f}")
    multi = snapshot.multi_timeframe
    if multi is not None and multi.enabled:
        daily = multi.daily
        lines += [
            "",
            f"daily context: bias {daily.bias}, EMA20 {daily.ema20}, "
            f"EMA50 {daily.ema50}, ATR14 {daily.atr14}",
            "daily reference-only levels: " + ", ".join(
                f"{level.price:.2f} ({level.label})" for level in daily.reference_levels
            ),
        ]
    return "\n".join(lines)


def no_llm_briefing(symbol: str, levels: list[Level], reason: str) -> MorningBriefing:
    """Deterministic fallback: raw levels only, no judgment invented for the LLM."""
    return MorningBriefing(
        symbol=symbol,
        bias="neutral",
        watch_levels=[LevelWatch(level=l, tradable=False, note=reason) for l in levels],
        summary=reason,
    )


def decide_prep(llm, symbol: str, levels: list[Level], user_text: str) -> MorningBriefing:
    try:
        briefing = llm.invoke(
            [{"role": "system", "content": PREP_SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
        )
        if briefing is None:  # with_structured_output returns None on an unparseable answer
            raise ValueError("LLM returned no parseable structured output")
        return briefing
    except Exception as exc:  # noqa: BLE001 — any LLM failure must not stop prep
        log.warning("LLM prep briefing failed (%s) — deterministic levels only", exc)
        return no_llm_briefing(symbol, levels, f"LLM unavailable ({exc}); deterministic levels only")
