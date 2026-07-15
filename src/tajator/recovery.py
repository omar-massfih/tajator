"""Rebuild session state from journal fill records after a crash.

state.json is written after a tick completes, but orders fill mid-tick, so a
crash in between leaves the broker holding a position the state file does not
know about. Every `fill` journal record carries the full post-fill
OpenPosition (contract, plan, qty_remaining), so the LAST fill record for a
symbol is a complete snapshot of the position at that instant — replaying it
closes the crash window. Adoption still requires an EXACT match with the live
broker position (con_id and qty); anything unexplained keeps refusing.

Residual gaps this cannot close (refusal correctly stands):
- a crash between the broker fill and the journal write (~milliseconds) —
  no record exists;
- an `entry_order_failed` whose reconciled partial fill left contracts in the
  account — no OpenPosition was ever built, so there is no plan to adopt.
  Such events count only toward trades_today here. The broker halts entries in
  the running process and startup account reconciliation refuses unexplained
  positions; only the operator may create the KILL file.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from .models import OpenPosition

ET = ZoneInfo("America/New_York")

# journal day-files to scan: covers a crash Friday 15:59 -> restart after a
# long weekend, with slack for market holidays
RECOVERY_LOOKBACK_DAYS = 5


class RecoveredSession(BaseModel):
    """Journal-derived candidate state for one symbol. position is None when
    the journal has no usable fill or proves the position was fully exited."""

    position: OpenPosition | None = None
    last_fill_ts: str | None = None  # provenance, journaled on adoption
    trades_today: int = 0


def read_journal_events(
    log_dir: Path, today: date, lookback_days: int = RECOVERY_LOOKBACK_DAYS
) -> tuple[list[dict], list[str]]:
    """All events from journal-<day>.jsonl for the lookback window, oldest
    first in file/line order. Unparseable lines (a crash mid-append truncates
    the last one) are skipped and reported as warnings; missing day files are
    normal (weekends, holidays)."""
    events: list[dict] = []
    warnings: list[str] = []
    for offset in range(lookback_days, -1, -1):
        day = today - timedelta(days=offset)
        path = log_dir / f"journal-{day.isoformat()}.jsonl"
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                warnings.append(f"{path.name} line {lineno}: unparseable ({exc}) — skipped")
    return events, warnings


def last_fill_position(
    events: list[dict], symbol: str
) -> tuple[OpenPosition | None, str | None]:
    """The position snapshot from the LAST `fill` event for the symbol.
    (None, None) when no fill exists or the record does not validate — schema
    drift or a hand-edited line must degrade to refusal, never crash startup
    or fall back to an older (stale) fill."""
    for ev in reversed(events):
        if ev.get("type") != "fill" or ev.get("symbol") != symbol:
            continue
        try:
            return OpenPosition.model_validate(ev["position"]), ev.get("ts")
        except Exception:  # noqa: BLE001 — unusable record, see docstring
            return None, None
    return None, None


def journal_trades_today(events: list[dict], symbol: str, today: date) -> int:
    """Entry attempts for the symbol on `today` (ET): entry fills plus
    entry_order_failed events — both consume a trade in the live graph.
    Scale-outs and exits are not entries."""
    count = 0
    for ev in events:
        if ev.get("symbol") != symbol:
            continue
        try:
            ev_date = datetime.fromisoformat(ev["ts"]).astimezone(ET).date()
        except (KeyError, TypeError, ValueError):
            continue
        if ev_date != today:
            continue
        if ev.get("type") == "entry_order_failed":
            count += 1
        elif ev.get("type") == "fill" and (ev.get("action") or {}).get("kind") == "entry":
            count += 1
    return count


def recover_sessions(
    log_dir: Path,
    symbols: list[str],
    today: date,
    lookback_days: int = RECOVERY_LOOKBACK_DAYS,
) -> tuple[dict[str, RecoveredSession], list[str]]:
    """Per-symbol journal-derived candidate state, plus reader warnings."""
    events, warnings = read_journal_events(log_dir, today, lookback_days)
    recovered: dict[str, RecoveredSession] = {}
    for symbol in symbols:
        position, ts = last_fill_position(events, symbol)
        if position is not None and position.qty_remaining == 0:
            position = None  # the journal proves the position was fully exited
        recovered[symbol] = RecoveredSession(
            position=position,
            last_fill_ts=ts,
            trades_today=journal_trades_today(events, symbol, today),
        )
    return recovered, warnings
