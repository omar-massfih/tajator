"""Append-only JSONL trade journal — one file per trading day."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel

ET = ZoneInfo("America/New_York")


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


class Journal:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, ts: datetime | None = None, **payload: object) -> None:
        ts = ts or datetime.now(ET)
        record = {"ts": ts.isoformat(), "type": event_type, **{k: _jsonable(v) for k, v in payload.items()}}
        path = self.log_dir / f"journal-{ts.astimezone(ET).date().isoformat()}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
