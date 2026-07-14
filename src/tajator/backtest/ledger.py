"""Turn a backtest's raw fills into round-trip trades, an equity curve, and summary stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from ..broker.base import Fill
from ..models import Bar, SelectedContract


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
    gross_pnl: float = 0.0
    fees: float = 0.0
    return_on_premium: float | None = None
    planned_equity_risk: float | None = None
    favorable_equity_move: float | None = None
    adverse_equity_move: float | None = None
    exit_reason: str = ""
    underlying_points: float | None = None
    regime: str = "unknown"
    level_quality_score: float = 0.0


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
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    profit_factor: float = 0.0
    metadata: dict = field(default_factory=dict)
    total_underlying_points: float = 0.0
    avg_underlying_win: float = 0.0
    avg_underlying_loss: float = 0.0
    underlying_wins: int = 0
    underlying_losses: int = 0
    underlying_win_rate: float = 0.0
    underlying_expectancy: float = 0.0
    underlying_equity_curve: list[tuple[date, float]] = field(default_factory=list)
    max_underlying_drawdown: float = 0.0
    # Counterfactual option contracts priced at the base strategy's exact fill
    # timestamps. Values are JSON-ready variant ledgers + summaries.
    option_panel: dict[str, dict] = field(default_factory=dict)


def _trades_for_day(
    day: date,
    fills: list[tuple[str, SelectedContract, Fill]],
    bars: list[Bar] | None = None,
) -> list[TradeResult]:
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
        fees = fill.fee
        exit_prices: list[float] = []
        reasons: list[str] = []
        exit_ts = None
        j = i + 1
        while remaining > 0 and j < len(fills) and fills[j][0] == "SELL":
            _, _, s_fill = fills[j]
            proceeds += s_fill.premium * s_fill.qty
            fees += s_fill.fee
            if s_fill.equity_price is not None:
                exit_prices.append(s_fill.equity_price)
            if s_fill.exit_reason:
                reasons.append(s_fill.exit_reason)
            remaining -= s_fill.qty
            exit_ts = s_fill.ts
            j += 1
        closed_qty = total_qty - remaining
        gross_pnl = (proceeds - entry_cost * (closed_qty / total_qty if total_qty else 1)) * 100
        pnl = gross_pnl - fees
        equity_risk = (
            abs(fill.equity_price - fill.stop_price)
            if fill.equity_price is not None and fill.stop_price is not None else None
        )
        favorable = adverse = None
        underlying_points = None
        if fill.equity_price is not None and closed_qty:
            weighted_exit = sum(
                s.equity_price * s.qty
                for _, _, s in fills[i + 1:j]
                if s.equity_price is not None
            )
            priced_qty = sum(s.qty for _, _, s in fills[i + 1:j] if s.equity_price is not None)
            if priced_qty:
                raw_move = weighted_exit / priced_qty - fill.equity_price
                underlying_points = raw_move if contract.right == "C" else -raw_move
        if fill.equity_price is not None and exit_ts is not None:
            trade_bars = [b for b in (bars or []) if entry_ts <= b.ts <= exit_ts]
            if trade_bars:
                if contract.right == "C":
                    favorable = max(0.0, max(b.high for b in trade_bars) - fill.equity_price)
                    adverse = max(0.0, fill.equity_price - min(b.low for b in trade_bars))
                else:
                    favorable = max(0.0, fill.equity_price - min(b.low for b in trade_bars))
                    adverse = max(0.0, max(b.high for b in trade_bars) - fill.equity_price)
            elif exit_prices:
                moves = [p - fill.equity_price for p in exit_prices]
                if contract.right == "P":
                    moves = [-m for m in moves]
                favorable, adverse = max(0.0, max(moves)), max(0.0, max(-m for m in moves))
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
                gross_pnl=round(gross_pnl, 2),
                fees=round(fees, 2),
                return_on_premium=round(pnl / (entry_cost * 100), 4) if entry_cost else None,
                planned_equity_risk=round(equity_risk, 4) if equity_risk is not None else None,
                favorable_equity_move=round(favorable, 4) if favorable is not None else None,
                adverse_equity_move=round(adverse, 4) if adverse is not None else None,
                exit_reason="; ".join(dict.fromkeys(reasons)),
                underlying_points=round(underlying_points, 4) if underlying_points is not None else None,
                regime=fill.regime,
                level_quality_score=fill.level_quality_score,
            )
        )
        i = j
    return trades


def build_report(
    symbol: str, start: date, end: date, fills_by_day: dict[date, list[tuple[str, SelectedContract, Fill]]],
    *, metadata: dict | None = None, bars_by_day: dict[date, list[Bar]] | None = None,
) -> BacktestReport:
    report = BacktestReport(symbol=symbol, start=start, end=end, metadata=metadata or {})
    for day in sorted(fills_by_day):
        day_trades = _trades_for_day(day, fills_by_day[day], (bars_by_day or {}).get(day))
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
    report.gross_pnl = round(sum(t.gross_pnl for t in closed), 2)
    report.total_fees = round(sum(t.fees for t in closed), 2)
    report.avg_win = round(sum(t.pnl for t in wins) / len(wins), 2) if wins else 0.0
    report.avg_loss = round(sum(t.pnl for t in losses) / len(losses), 2) if losses else 0.0
    gross_wins = sum(t.pnl for t in wins)
    gross_losses = abs(sum(t.pnl for t in losses))
    report.profit_factor = round(gross_wins / gross_losses, 4) if gross_losses else 0.0
    point_results = [t.underlying_points for t in closed if t.underlying_points is not None]
    point_wins = [p for p in point_results if p > 0]
    point_losses = [p for p in point_results if p <= 0]
    report.total_underlying_points = round(sum(point_results), 4)
    report.underlying_wins = len(point_wins)
    report.underlying_losses = len(point_losses)
    report.underlying_win_rate = round(len(point_wins) / len(point_results), 4) if point_results else 0.0
    report.avg_underlying_win = round(sum(point_wins) / len(point_wins), 4) if point_wins else 0.0
    report.avg_underlying_loss = round(sum(point_losses) / len(point_losses), 4) if point_losses else 0.0
    report.underlying_expectancy = (
        round(report.total_underlying_points / len(point_results), 4) if point_results else 0.0
    )

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for day in sorted(report.daily_pnl):
        cumulative += report.daily_pnl[day]
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
        report.equity_curve.append((day, round(cumulative, 2)))
    report.max_drawdown = round(max_dd, 2)

    point_cumulative = 0.0
    point_peak = 0.0
    point_max_dd = 0.0
    for day in sorted(fills_by_day):
        day_points = sum(
            trade.underlying_points or 0.0
            for trade in report.trades
            if trade.day == day and trade.closed
        )
        point_cumulative += day_points
        point_peak = max(point_peak, point_cumulative)
        point_max_dd = max(point_max_dd, point_peak - point_cumulative)
        report.underlying_equity_curve.append((day, round(point_cumulative, 4)))
    report.max_underlying_drawdown = round(point_max_dd, 4)
    return report
