"""Multi-day backtest driver: the exact live/replay graph, stepped over a date range."""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import date
from pathlib import Path

from ..broker.backtest import BacktestBroker
from ..config import Settings
from ..graph.nodes import RuntimeContext
from ..journal import Journal
from ..llm.decide import build_llm
from ..runner import TradingSession
from .data import ensure_underlying_bars, fetch_daily_series, prev_day_range_for, trading_days
from .ledger import BacktestReport, build_report

log = logging.getLogger(__name__)


def run_backtest(
    symbol: str, start: date, end: date, settings: Settings, use_llm: bool, ib, cache_dir: Path
) -> BacktestReport:
    days = trading_days(start, end)
    daily_series = fetch_daily_series(ib, symbol, start, end) if ib is not None else []
    journal = Journal(settings.log_dir / "backtests" / f"{symbol}_{start.isoformat()}_{end.isoformat()}")
    llm = build_llm(settings.llm_model) if use_llm else None

    fills_by_day = {}
    for day in days:
        bars = ensure_underlying_bars(ib, symbol, day, cache_dir)
        if not bars:
            log.info("no bars for %s %s — skipping (holiday or no data)", symbol, day)
            continue
        prev_high, prev_low = prev_day_range_for(daily_series, day)
        broker = BacktestBroker(bars, prev_high, prev_low, ib=ib, cache_dir=cache_dir)
        ctx = RuntimeContext(
            settings=settings, broker=broker, journal=journal, symbol=symbol, use_llm=use_llm, _llm=llm,
        )
        TradingSession(ctx).run_replay(broker, verbose=False)
        if broker.fills:
            fills_by_day[day] = broker.fills

    report = build_report(symbol, start, end, fills_by_day)
    _persist_report(report, settings.log_dir)
    return report


def _persist_report(report: BacktestReport, log_dir: Path) -> Path:
    out_dir = log_dir / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.symbol}_{report.start.isoformat()}_{report.end.isoformat()}.json"
    payload = dataclasses.asdict(report)
    payload["daily_pnl"] = {d.isoformat(): pnl for d, pnl in report.daily_pnl.items()}
    payload["equity_curve"] = [(d.isoformat(), cum) for d, cum in report.equity_curve]
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def print_summary(report: BacktestReport) -> None:
    print(f"\n--- backtest summary: {report.symbol} {report.start} → {report.end} ---")
    if report.total_trades == 0:
        print("no closed trades.")
        return
    print(
        f"trades: {report.total_trades}  win rate: {report.win_rate:.0%}  "
        f"total PnL: ${report.total_pnl:,.0f}  max drawdown: ${report.max_drawdown:,.0f}"
    )
    print(f"avg win: ${report.avg_win:,.0f}  avg loss: ${report.avg_loss:,.0f}")
