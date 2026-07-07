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
)
from .prompts import PREP_SYSTEM_PROMPT, SYSTEM_PROMPT

log = logging.getLogger(__name__)

LLM_TIMEOUT_S = 20
RECENT_BARS = 10


def build_llm(model_string: str, output_model: type[BaseModel] = Decision):
    """`model_string` is an init_chat_model id like 'openai:gpt-5.1', or
    'codex' / 'codex:<model>' to use the Codex CLI (ChatGPT subscription)."""
    if model_string == "codex" or model_string.startswith("codex:"):
        from .codex import BRIEFING_SCHEMA, DECISION_SCHEMA, CodexDecider

        _, _, model = model_string.partition(":")
        schema = BRIEFING_SCHEMA if output_model is MorningBriefing else DECISION_SCHEMA
        return CodexDecider(model=model or None, output_model=output_model, schema=schema)
    llm = init_chat_model(model_string, timeout=LLM_TIMEOUT_S)
    return llm.with_structured_output(output_model)


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

    lines += ["", "levels:"]
    for l in levels:
        lines.append(f"  {l.price:.2f}  {l.kind:<10} ({l.label})")

    if candidates:
        lines += ["", "DETECTED SETUP CANDIDATES (only these are tradeable):"]
        for c in candidates:
            lines.append(
                f"  {c.direction.upper()} — {c.note}, distance {c.distance:+.2f}, "
                f"3-bar move {c.speed:+.2f}"
            )
    else:
        lines += ["", "no setup candidates detected this tick"]

    if position is not None:
        p = position.plan
        lines += [
            "",
            f"OPEN POSITION: {position.qty_remaining}x {position.contract.local_name} "
            f"({p.direction}), entry equity {p.entry_equity_price:.2f}, "
            f"stop {p.stop_price:.2f}, pieces sold {position.pieces_sold}/{len(p.pieces)}",
        ]
    if manage_note:
        lines += ["", f"QUESTION: {manage_note}"]
    return "\n".join(lines)


def _ask(llm, user_text: str) -> Decision:
    return llm.invoke(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
    )


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
        return llm.invoke(
            [{"role": "system", "content": PREP_SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
        )
    except Exception as exc:  # noqa: BLE001 — any LLM failure must not stop prep
        log.warning("LLM prep briefing failed (%s) — deterministic levels only", exc)
        return no_llm_briefing(symbol, levels, f"LLM unavailable ({exc}); deterministic levels only")
