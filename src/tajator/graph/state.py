"""Graph state: pure data, rebuilt fresh on every 1-minute tick.

Durable things (broker connection, open position, trades-today counter,
journal) live in the TradingSession / RuntimeContext outside the graph.
"""

from __future__ import annotations

from typing import TypedDict

from ..models import (
    Bar,
    Decision,
    ExecutedAction,
    Level,
    OpenPosition,
    RiskVerdict,
    SetupCandidate,
    Snapshot,
)
from ..trade.position import ManageAction


class AgentState(TypedDict, total=False):
    # injected each tick from the session
    position: OpenPosition | None
    trades_today: int
    # built by the graph
    bars: list[Bar]
    prev_day_high: float | None
    prev_day_low: float | None
    snapshot: Snapshot
    levels: list[Level]
    candidates: list[SetupCandidate]
    manage_action: ManageAction
    decision: Decision
    risk: RiskVerdict
    actions: list[ExecutedAction]
    skip_reason: str
