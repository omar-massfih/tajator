"""Multi-day backtest driver: the exact live/replay graph, stepped over a date range."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import subprocess
from datetime import date
from pathlib import Path

from ..broker.backtest import BacktestBroker
from ..broker.stub import StubBroker
from ..config import Settings
from ..graph.nodes import RuntimeContext
from ..journal import Journal
from ..llm.decide import build_llm
from ..runner import TradingSession
from .data import ET, ensure_underlying_bars, fetch_daily_series, prev_day_range_for, trading_days
from .ledger import BacktestReport, build_report

log = logging.getLogger(__name__)


def run_backtest(
    symbol: str, start: date, end: date, settings: Settings, use_llm: bool, ib, cache_dir: Path,
    *, skip_missing_option_data: bool = False,
    underlying_only: bool = False,
    cached_only: bool = False,
    experiment: str = "baseline",
) -> BacktestReport:
    days = trading_days(start, end)
    # Underlying research already loads every minute of every session. Derive
    # prior-day levels from those bars instead of making a second HMDS request;
    # this also keeps cached research runnable during a daily-data farm outage.
    daily_series = (
        fetch_daily_series(ib, symbol, start, end)
        if ib is not None and not underlying_only else []
    )
    rolling_prev_range: tuple[float | None, float | None] = (None, None)
    experiment = _safe_experiment(experiment)
    journal = Journal(
        settings.log_dir / "backtests" /
        f"{symbol}_{start.isoformat()}_{end.isoformat()}_{experiment}"
    )
    llm = build_llm(settings.llm_model) if use_llm else None

    fills_by_day = {}
    bars_by_day = {}
    skipped_days: list[dict[str, str]] = []
    metrics: dict[str, int] = {}
    for day in days:
        bars = ensure_underlying_bars(None if cached_only else ib, symbol, day, cache_dir)
        if not bars:
            log.info("no bars for %s %s — skipping (holiday or no data)", symbol, day)
            continue
        bars_by_day[day] = bars
        prev_high, prev_low = (
            prev_day_range_for(daily_series, day) if daily_series else rolling_prev_range
        )
        broker = (
            StubBroker(bars, prev_high, prev_low)
            if underlying_only else
            BacktestBroker(
                bars, prev_high, prev_low, ib=ib, cache_dir=cache_dir,
                half_spread_pct=settings.backtest_half_spread_pct,
                slippage_cents=settings.backtest_slippage_cents,
                commission_per_contract=settings.backtest_commission_per_contract,
                min_commission_per_order=settings.backtest_min_commission_per_order,
            )
        )
        ctx = RuntimeContext(
            settings=settings, broker=broker, journal=journal, symbol=symbol,
            use_llm=use_llm, _llm=llm, metrics=metrics,
        )
        try:
            TradingSession(ctx).run_replay(broker, verbose=False)
        except RuntimeError as exc:
            if not skip_missing_option_data or "no historical option data for" not in str(exc):
                raise
            # Discard every fill from this day. Keeping an entry or scale-out
            # before a later missing quote would create an incomplete trade and
            # biased PnL. Coverage loss is disclosed in report metadata.
            skipped_days.append({"day": day.isoformat(), "reason": str(exc)})
            log.warning("skipping all of %s %s: %s", symbol, day, exc)
            continue
        if broker.fills:
            fills_by_day[day] = broker.fills
        session_bars = [
            b for b in bars
            if b.ts.astimezone(ET).date() == day
            and (b.ts.astimezone(ET).hour, b.ts.astimezone(ET).minute) >= (9, 30)
            and (b.ts.astimezone(ET).hour, b.ts.astimezone(ET).minute) <= (16, 0)
        ]
        if session_bars:
            rolling_prev_range = (
                max(b.high for b in session_bars), min(b.low for b in session_bars)
            )

    resolved_config = _strategy_config(settings.for_symbol(symbol))
    metadata = {
        "use_llm": use_llm,
        "llm_model": settings.llm_model if use_llm else None,
        "code_revision": _code_revision(),
        "execution_model": {
            "price_source": "next option bar open (last bar close at EOD)",
            "modeled_half_spread_pct": settings.backtest_half_spread_pct,
            "slippage_cents_per_contract": settings.backtest_slippage_cents,
            "commission_per_contract": settings.backtest_commission_per_contract,
            "minimum_commission_per_order": settings.backtest_min_commission_per_order,
        },
        "strategy_config": resolved_config,
        "config_fingerprint": hashlib.sha256(
            json.dumps(resolved_config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
        "experiment": experiment,
        "data_coverage": {
            "requested_weekdays": len(days),
            "days_with_underlying_bars": len(bars_by_day),
            "skipped_missing_option_days": skipped_days,
            "skip_missing_option_data": skip_missing_option_data,
            "cached_only": cached_only,
        },
        "research_mode": "underlying_only" if underlying_only else "historical_options",
        "veto_counts": metrics,
    }
    report = build_report(
        symbol, start, end, fills_by_day, metadata=metadata, bars_by_day=bars_by_day
    )
    _persist_report(report, settings.log_dir, experiment)
    return report


def _code_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _strategy_config(settings: Settings) -> dict:
    names = (
        "max_trades_per_day", "max_contracts", "max_premium_usd", "stop_buffer_cents",
        "no_new_entries_after", "double_min_touch_separation_bars", "double_min_pullback_pct",
        "min_level_dist_from_open_pct", "swing_window_bars", "level_cluster_tol_pct",
        "approach_band_pct", "overshoot_band_pct", "speed_window_bars", "min_speed_pct",
        "fast_approach_speed_mult", "rejection_wick_min_frac", "trade_flipped_levels",
        "entry_confirmation", "max_entry_to_stop_cents", "no_new_entries_before",
        "opening_confirmation_until", "stop_atr_multiplier", "atr_window_bars",
        "allowed_regimes", "blocked_direction_regimes", "min_level_quality_score",
        "stop_min_cents", "stop_max_cents", "stop_cooldown_minutes", "runner_stop",
    )
    return {name: getattr(settings, name) for name in names}


def _safe_experiment(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    if not safe:
        raise ValueError("experiment name must contain a letter or number")
    return safe


def _persist_report(report: BacktestReport, log_dir: Path, experiment: str = "baseline") -> Path:
    out_dir = log_dir / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (
        f"{report.symbol}_{report.start.isoformat()}_{report.end.isoformat()}_{experiment}.json"
    )
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
    if report.metadata.get("research_mode") == "underlying_only":
        print(
            f"trades: {report.total_trades}  stock-direction win rate: "
            f"{report.underlying_win_rate:.0%}  "
            f"direction-adjusted underlying points: {report.total_underlying_points:+.2f}"
        )
        print(
            f"avg underlying win: {report.avg_underlying_win:+.2f}  "
            f"avg underlying loss: {report.avg_underlying_loss:+.2f}"
        )
        return
    print(
        f"trades: {report.total_trades}  win rate: {report.win_rate:.0%}  "
        f"total PnL: ${report.total_pnl:,.0f}  max drawdown: ${report.max_drawdown:,.0f}"
    )
    print(f"avg win: ${report.avg_win:,.0f}  avg loss: ${report.avg_loss:,.0f}")
