"""Durable per-symbol session state, so a restart can adopt its own positions.

One JSON file, rewritten atomically on every change. Like the KILL file this
is operator-visible runtime state, not a journal: it holds only the current
position and today's trade count per symbol. Only the live runner writes it —
replay and backtest never get a store.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from .models import OpenPosition

ET = ZoneInfo("America/New_York")


class PersistedSession(BaseModel):
    position: OpenPosition | None = None
    trades_today: int = 0


class PersistedState(BaseModel):
    version: int = 1
    updated_at: datetime
    trading_day: date  # trades_today is only valid for this ET date
    sessions: dict[str, PersistedSession] = Field(default_factory=dict)


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self._state: PersistedState | None = None

    def load(self) -> PersistedState | None:
        """The persisted state, or None when no file exists yet. A corrupt
        file raises — the caller must refuse to trade rather than guess."""
        if not self.path.exists():
            return None
        self._state = PersistedState.model_validate_json(self.path.read_text())
        return self._state

    def update(
        self,
        symbol: str,
        position: OpenPosition | None,
        trades_today: int,
        trading_day: date,
    ) -> None:
        if self._state is None:
            self._state = PersistedState(
                updated_at=datetime.now(ET), trading_day=trading_day
            )
        self._state.trading_day = trading_day
        self._state.updated_at = datetime.now(ET)
        self._state.sessions[symbol] = PersistedSession(
            position=position, trades_today=trades_today
        )
        self._write()

    def _write(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(self._state.model_dump_json(indent=2))
        os.replace(tmp, self.path)  # atomic — a crash never leaves half a file
