"""Turn a backtest's raw fills into round-trip trades, an equity curve, and summary stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from ..broker.base import Fill
from ..models import SelectedContract


@dataclass
class TradeResult:
    day: date
    symbol: str
    contract: str
    direction: str
    qty: int
    entry_ts: datetime
    exit_ts: datetime | None
    entry_premium: float
    exit_premium: float | None
    pnl: float
    closed: bool


@dataclass
class BacktestReport:
    symbol: str
    start: date
    end: date
    trades: list[TradeResult] = field(default_factory=list)
    daily_pnl: dict[date, float] = field(default_factory=dict)
    equity_curve: list[tuple[date, float]] = field(default_factory=list)
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0


def _trades_for_day(day: date, fills: list[tuple[str, SelectedContract, Fill]]) -> list[TradeResult]:
    trades = []
    i = 0
    while i < len(fills):
        side, contract, fill = fills[i]
        if side != "BUY":  # a stray SELL with no matching BUY shouldn't happen; skip defensively
            i += 1
            continue
        total_qty = fill.qty
        entry_cost = fill.premium * fill.qty
        entry_ts = fill.ts
        remaining = fill.qty
        proceeds = 0.0
        exit_ts = None
        j = i + 1
        while remaining > 0 and j < len(fills) and fills[j][0] == "SELL":
            _, _, s_fill = fills[j]
            proceeds += s_fill.premium * s_fill.qty
            remaining -= s_fill.qty
            exit_ts = s_fill.ts
            j += 1
        closed_qty = total_qty - remaining
        pnl = (proceeds - entry_cost * (closed_qty / total_qty if total_qty else 1)) * 100
        trades.append(
            TradeResult(
                day=day,
                symbol=contract.symbol,
                contract=contract.local_name,
                direction="call" if contract.right == "C" else "put",
                qty=total_qty,
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                entry_premium=round(entry_cost / total_qty, 2) if total_qty else 0.0,
                exit_premium=round(proceeds / closed_qty, 2) if closed_qty else None,
                pnl=round(pnl, 2),
                closed=remaining == 0,
            )
        )
        i = j
    return trades


def build_report(
    symbol: str, start: date, end: date, fills_by_day: dict[date, list[tuple[str, SelectedContract, Fill]]]
) -> BacktestReport:
    report = BacktestReport(symbol=symbol, start=start, end=end)
    for day in sorted(fills_by_day):
        day_trades = _trades_for_day(day, fills_by_day[day])
        report.trades.extend(day_trades)
        report.daily_pnl[day] = round(sum(t.pnl for t in day_trades if t.closed), 2)

    closed = [t for t in report.trades if t.closed]
    report.total_trades = len(closed)
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    report.wins = len(wins)
    report.losses = len(losses)
    report.win_rate = round(len(wins) / len(closed), 4) if closed else 0.0
    report.total_pnl = round(sum(t.pnl for t in closed), 2)
    report.avg_win = round(sum(t.pnl for t in wins) / len(wins), 2) if wins else 0.0
    report.avg_loss = round(sum(t.pnl for t in losses) / len(losses), 2) if losses else 0.0

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for day in sorted(report.daily_pnl):
        cumulative += report.daily_pnl[day]
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
        report.equity_curve.append((day, round(cumulative, 2)))
    report.max_drawdown = round(max_dd, 2)
    return report
