"""Causal intraday signal tournament on cached TWS historical stock bars.

This is deliberately separate from the production trading graph.  It tests a
small preregistered family of directional signals and opens validation only for
the development winner.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Callable, Literal

from ..models import Bar
from .data import ET, _read_csv, trading_days

Direction = Literal["call", "put"]
DEVELOPMENT_START = date(2024, 7, 1)
DEVELOPMENT_END = date(2025, 6, 30)
VALIDATION_START = date(2025, 7, 1)
VALIDATION_END = date(2026, 6, 30)
DEVELOPMENT_SYMBOLS = ("IWM", "META")
VALIDATION_SYMBOLS = ("AAPL", "QQQ", "IWM", "META")
BASE_COST_BPS_PER_SIDE = 1.0
STRESS_COST_BPS_PER_SIDE = 2.0
HORIZON_BARS = 60


@dataclass(frozen=True)
class SessionContext:
    symbol: str
    day: date
    bars: list[Bar]
    previous_close: float | None
    atr20: float | None


@dataclass(frozen=True)
class SignalTrade:
    strategy: str
    symbol: str
    day: date
    direction: Direction
    signal_ts: datetime
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    gross_points: float
    net_points: float
    stress_net_points: float


def _rth(bars: list[Bar]) -> list[Bar]:
    return [
        bar for bar in bars
        if time(9, 30) <= bar.ts.astimezone(ET).time() <= time(15, 59)
    ]


def _complete_session(bars: list[Bar]) -> bool:
    return bool(
        len(bars) >= 370
        and bars[0].ts.astimezone(ET).time() <= time(9, 31)
        and bars[-1].ts.astimezone(ET).time() >= time(15, 59)
    )


def _true_range(high: float, low: float, previous_close: float) -> float:
    return max(high - low, abs(high - previous_close), abs(low - previous_close))


def load_symbol_sessions(
    cache_dir: Path, symbol: str, start: date, end: date
) -> tuple[list[SessionContext], int]:
    """Load complete sessions and construct prior-close/ATR context causally."""
    symbol_dir = cache_dir / symbol
    raw: list[tuple[date, list[Bar]]] = []
    for path in sorted(symbol_dir.glob("????-??-??.csv")):
        day = date.fromisoformat(path.stem)
        if day > end or day < start:
            continue
        bars = _rth(_read_csv(path))
        if _complete_session(bars):
            raw.append((day, bars))

    contexts: list[SessionContext] = []
    daily: list[tuple[float, float, float]] = []
    previous_close = None
    for day, bars in raw:
        atr20 = None
        if previous_close is not None and len(daily) >= 21:
            ranges = []
            prior = None
            for high, low, close in daily[-21:]:
                if prior is not None:
                    ranges.append(_true_range(high, low, prior))
                prior = close
            if len(ranges) >= 20:
                atr20 = sum(ranges[-20:]) / 20
        contexts.append(SessionContext(symbol, day, bars, previous_close, atr20))
        high = max(bar.high for bar in bars)
        low = min(bar.low for bar in bars)
        close = bars[-1].close
        daily.append((high, low, close))
        previous_close = close
    return contexts, len(trading_days(start, end))


def _trade(
    ctx: SessionContext,
    strategy: str,
    signal_index: int,
    direction: Direction,
) -> SignalTrade | None:
    entry_index = signal_index + 1
    exit_index = entry_index + HORIZON_BARS - 1
    if entry_index >= len(ctx.bars) or exit_index >= len(ctx.bars):
        return None
    signal = ctx.bars[signal_index]
    entry = ctx.bars[entry_index]
    exit_bar = ctx.bars[exit_index]
    sign = 1.0 if direction == "call" else -1.0
    gross = sign * (exit_bar.close - entry.open)

    def net(cost_bps: float) -> float:
        return gross - (entry.open + exit_bar.close) * cost_bps / 10_000

    return SignalTrade(
        strategy=strategy,
        symbol=ctx.symbol,
        day=ctx.day,
        direction=direction,
        signal_ts=signal.ts,
        entry_ts=entry.ts,
        exit_ts=exit_bar.ts,
        entry_price=round(entry.open, 4),
        exit_price=round(exit_bar.close, 4),
        gross_points=round(gross, 4),
        net_points=round(net(BASE_COST_BPS_PER_SIDE), 4),
        stress_net_points=round(net(STRESS_COST_BPS_PER_SIDE), 4),
    )


def orb15_breakout(ctx: SessionContext) -> SignalTrade | None:
    bars = ctx.bars
    opening = bars[:15]
    if len(opening) < 15:
        return None
    high = max(bar.high for bar in opening)
    low = min(bar.low for bar in opening)
    previous = opening[-1].close
    for index in range(15, len(bars) - HORIZON_BARS):
        if bars[index].ts.astimezone(ET).time() > time(11, 0):
            break
        close = bars[index].close
        if close > high and previous <= high:
            return _trade(ctx, "orb15_breakout", index, "call")
        if close < low and previous >= low:
            return _trade(ctx, "orb15_breakout", index, "put")
        previous = close
    return None


def _gap(ctx: SessionContext, *, fade: bool) -> SignalTrade | None:
    if ctx.previous_close is None or ctx.atr20 is None or ctx.atr20 <= 0 or len(ctx.bars) < 6:
        return None
    open_price = ctx.bars[0].open
    gap = open_price - ctx.previous_close
    if abs(gap) < 0.35 * ctx.atr20:
        return None
    five_minute_close = ctx.bars[4].close
    up = gap > 0
    moved_with_gap = five_minute_close > open_price if up else five_minute_close < open_price
    moved_toward_close = five_minute_close < open_price if up else five_minute_close > open_price
    if fade and moved_toward_close:
        return _trade(ctx, "gap_fade", 4, "put" if up else "call")
    if not fade and moved_with_gap:
        return _trade(ctx, "gap_continuation", 4, "call" if up else "put")
    return None


def gap_continuation(ctx: SessionContext) -> SignalTrade | None:
    return _gap(ctx, fade=False)


def gap_fade(ctx: SessionContext) -> SignalTrade | None:
    return _gap(ctx, fade=True)


def opening_drive(ctx: SessionContext) -> SignalTrade | None:
    return _opening_drive(ctx, fade=False)


def opening_drive_fade(ctx: SessionContext) -> SignalTrade | None:
    return _opening_drive(ctx, fade=True)


def _opening_drive(ctx: SessionContext, *, fade: bool) -> SignalTrade | None:
    if ctx.atr20 is None or ctx.atr20 <= 0 or len(ctx.bars) < 16:
        return None
    opening = ctx.bars[:15]
    high = max(bar.high for bar in opening)
    low = min(bar.low for bar in opening)
    width = high - low
    if width < 0.20 * ctx.atr20 or width <= 0:
        return None
    location = (opening[-1].close - low) / width
    if location >= 0.80:
        return _trade(
            ctx, "opening_drive_fade" if fade else "opening_drive", 14,
            "put" if fade else "call",
        )
    if location <= 0.20:
        return _trade(
            ctx, "opening_drive_fade" if fade else "opening_drive", 14,
            "call" if fade else "put",
        )
    return None


def vwap_trend_pullback(ctx: SessionContext) -> SignalTrade | None:
    cumulative_value = 0.0
    cumulative_volume = 0.0
    vwaps: list[float | None] = []
    for bar in ctx.bars:
        cumulative_value += ((bar.high + bar.low + bar.close) / 3) * bar.volume
        cumulative_volume += bar.volume
        vwaps.append(cumulative_value / cumulative_volume if cumulative_volume > 0 else None)
    for index in range(30, len(ctx.bars) - HORIZON_BARS):
        tod = ctx.bars[index].ts.astimezone(ET).time()
        if tod > time(13, 30):
            break
        current, prior = vwaps[index], vwaps[index - 15]
        if current is None or prior is None:
            continue
        bar = ctx.bars[index]
        touched = bar.low <= current <= bar.high
        if not touched:
            continue
        if current > prior and bar.close > current:
            return _trade(ctx, "vwap_trend_pullback", index, "call")
        if current < prior and bar.close < current:
            return _trade(ctx, "vwap_trend_pullback", index, "put")
    return None


CANDIDATES: dict[str, Callable[[SessionContext], SignalTrade | None]] = {
    "orb15_breakout": orb15_breakout,
    "gap_continuation": gap_continuation,
    "gap_fade": gap_fade,
    "opening_drive": opening_drive,
    "vwap_trend_pullback": vwap_trend_pullback,
}


def _stats(trades: list[SignalTrade], metric: str = "net_points") -> dict:
    values = [float(getattr(trade, metric)) for trade in trades]
    count = len(values)
    mean = sum(values) / count if count else 0.0
    by_day: dict[date, list[float]] = defaultdict(list)
    by_month: dict[str, float] = defaultdict(float)
    by_symbol: dict[str, list[float]] = defaultdict(list)
    by_symbol_month: dict[tuple[str, str], float] = defaultdict(float)
    for trade, value in zip(trades, values, strict=True):
        by_day[trade.day].append(value)
        month = trade.day.strftime("%Y-%m")
        by_month[month] += value
        by_symbol[trade.symbol].append(value)
        by_symbol_month[(trade.symbol, month)] += value
    groups = len(by_day)
    if count and groups > 1:
        cluster_sums = [sum(value - mean for value in day_values) for day_values in by_day.values()]
        standard_error = math.sqrt(
            groups / (groups - 1) * sum(value**2 for value in cluster_sums) / count**2
        )
    else:
        standard_error = 0.0
    return {
        "trades": count,
        "trading_days": groups,
        "wins": sum(value > 0 for value in values),
        "win_rate": round(sum(value > 0 for value in values) / count, 4) if count else 0.0,
        "total": round(sum(values), 4),
        "expectancy": round(mean, 6),
        "clustered_standard_error": round(standard_error, 6),
        "ci95_low": round(mean - 1.96 * standard_error, 6),
        "ci95_high": round(mean + 1.96 * standard_error, 6),
        "active_months": len(by_month),
        "positive_months": sum(value > 0 for value in by_month.values()),
        "positive_month_ratio": (
            round(sum(value > 0 for value in by_month.values()) / len(by_month), 4)
            if by_month else 0.0
        ),
        "positive_symbol_month_ratio": (
            round(sum(value > 0 for value in by_symbol_month.values()) / len(by_symbol_month), 4)
            if by_symbol_month else 0.0
        ),
        "symbols": {
            symbol: {
                "trades": len(symbol_values),
                "total": round(sum(symbol_values), 4),
                "expectancy": round(sum(symbol_values) / len(symbol_values), 6),
            }
            for symbol, symbol_values in sorted(by_symbol.items())
        },
    }


def _collect(
    sessions: dict[str, list[SessionContext]],
    candidate: Callable[[SessionContext], SignalTrade | None],
) -> list[SignalTrade]:
    trades = []
    for symbol in sorted(sessions):
        for ctx in sessions[symbol]:
            trade = candidate(ctx)
            if trade is not None:
                trades.append(trade)
    return trades


def _load_split(
    cache_dir: Path, symbols: tuple[str, ...], start: date, end: date
) -> tuple[dict[str, list[SessionContext]], dict]:
    sessions = {}
    requested = covered = 0
    for symbol in symbols:
        symbol_sessions, symbol_requested = load_symbol_sessions(cache_dir, symbol, start, end)
        sessions[symbol] = symbol_sessions
        requested += symbol_requested
        covered += len(symbol_sessions)
    return sessions, {
        "requested_symbol_weekdays": requested,
        "complete_symbol_sessions": covered,
        "coverage_ratio": round(covered / requested, 4) if requested else 0.0,
    }


def run_tournament(cache_dir: Path) -> dict:
    development_sessions, development_coverage = _load_split(
        cache_dir, DEVELOPMENT_SYMBOLS, DEVELOPMENT_START, DEVELOPMENT_END
    )
    development: dict[str, dict] = {}
    development_trades: dict[str, list[SignalTrade]] = {}
    eligible = []
    for name, candidate in CANDIDATES.items():
        trades = _collect(development_sessions, candidate)
        development_trades[name] = trades
        stats = _stats(trades)
        symbol_positive = all(
            stats["symbols"].get(symbol, {}).get("expectancy", 0.0) > 0
            for symbol in DEVELOPMENT_SYMBOLS
        )
        qualifies = (
            stats["trades"] >= 50
            and symbol_positive
            and stats["positive_symbol_month_ratio"] >= 0.5
        )
        development[name] = {**stats, "eligible": qualifies}
        if qualifies:
            eligible.append(name)

    selected = max(
        eligible, key=lambda name: (development[name]["expectancy"], name), default=None
    )
    report = {
        "protocol": "historical_signal_tournament_v1",
        "data_source": "cached TWS historical one-minute stock trades",
        "costs": {
            "base_bps_per_side": BASE_COST_BPS_PER_SIDE,
            "stress_bps_per_side": STRESS_COST_BPS_PER_SIDE,
            "holding_period_bars": HORIZON_BARS,
        },
        "development_period": [DEVELOPMENT_START.isoformat(), DEVELOPMENT_END.isoformat()],
        "development_symbols": list(DEVELOPMENT_SYMBOLS),
        "development_coverage": development_coverage,
        "development": development,
        "selected": selected,
        "validation_opened": selected is not None,
        "validation": None,
        "verdict": "no_development_candidate" if selected is None else "pending",
    }
    if selected is None:
        return report

    # The validation split is not loaded or evaluated until development has
    # mechanically selected exactly one candidate.
    validation_sessions, validation_coverage = _load_split(
        cache_dir, VALIDATION_SYMBOLS, VALIDATION_START, VALIDATION_END
    )
    trades = _collect(validation_sessions, CANDIDATES[selected])
    stats = _stats(trades)
    stress = _stats(trades, "stress_net_points")
    sufficiently_sampled = {
        symbol: values for symbol, values in stats["symbols"].items()
        if values["trades"] >= 10
    }
    gates = {
        "minimum_trades": stats["trades"] >= 100,
        "positive_each_sampled_symbol": bool(sufficiently_sampled) and all(
            values["expectancy"] > 0 for values in sufficiently_sampled.values()
        ),
        "day_clustered_ci_above_zero": stats["ci95_low"] > 0,
        "positive_month_stability": (
            stats["active_months"] >= 3 and stats["positive_month_ratio"] >= 0.6
        ),
        "positive_stress_expectancy": stress["expectancy"] > 0,
        "data_coverage": validation_coverage["coverage_ratio"] >= 0.9,
    }
    report["validation"] = {
        **stats,
        "stress_expectancy": stress["expectancy"],
        "coverage": validation_coverage,
        "gates": gates,
        "trades_ledger": [dataclasses.asdict(trade) for trade in trades],
    }
    report["verdict"] = (
        "historically_supported_stock_signal"
        if all(gates.values()) else "validation_failed"
    )
    return report


def run_opening_drive_fade_followup(cache_dir: Path) -> dict:
    """One development-generated follow-up; validation stays closed on ineligibility."""
    development_sessions, development_coverage = _load_split(
        cache_dir, DEVELOPMENT_SYMBOLS, DEVELOPMENT_START, DEVELOPMENT_END
    )
    development_trades = _collect(development_sessions, opening_drive_fade)
    development = _stats(development_trades)
    symbol_positive = all(
        development["symbols"].get(symbol, {}).get("expectancy", 0.0) > 0
        for symbol in DEVELOPMENT_SYMBOLS
    )
    eligible = (
        development["trades"] >= 50
        and symbol_positive
        and development["positive_symbol_month_ratio"] >= 0.5
    )
    report = {
        "protocol": "opening_drive_fade_followup_v1",
        "data_source": "cached TWS historical one-minute stock trades",
        "candidate": "opening_drive_fade",
        "costs": {
            "base_bps_per_side": BASE_COST_BPS_PER_SIDE,
            "stress_bps_per_side": STRESS_COST_BPS_PER_SIDE,
            "holding_period_bars": HORIZON_BARS,
        },
        "development_period": [DEVELOPMENT_START.isoformat(), DEVELOPMENT_END.isoformat()],
        "development_symbols": list(DEVELOPMENT_SYMBOLS),
        "development_coverage": development_coverage,
        "development": {**development, "eligible": eligible},
        "validation_opened": eligible,
        "validation": None,
        "verdict": "development_ineligible" if not eligible else "pending",
    }
    if not eligible:
        return report

    validation_sessions, validation_coverage = _load_split(
        cache_dir, VALIDATION_SYMBOLS, VALIDATION_START, VALIDATION_END
    )
    trades = _collect(validation_sessions, opening_drive_fade)
    stats = _stats(trades)
    stress = _stats(trades, "stress_net_points")
    sufficiently_sampled = {
        symbol: values for symbol, values in stats["symbols"].items()
        if values["trades"] >= 10
    }
    gates = {
        "minimum_trades": stats["trades"] >= 100,
        "positive_each_sampled_symbol": bool(sufficiently_sampled) and all(
            values["expectancy"] > 0 for values in sufficiently_sampled.values()
        ),
        "day_clustered_ci_above_zero": stats["ci95_low"] > 0,
        "positive_month_stability": (
            stats["active_months"] >= 3 and stats["positive_month_ratio"] >= 0.6
        ),
        "positive_stress_expectancy": stress["expectancy"] > 0,
        "data_coverage": validation_coverage["coverage_ratio"] >= 0.9,
    }
    report["validation"] = {
        **stats,
        "stress_expectancy": stress["expectancy"],
        "coverage": validation_coverage,
        "gates": gates,
        "trades_ledger": [dataclasses.asdict(trade) for trade in trades],
    }
    report["verdict"] = (
        "historically_supported_stock_signal"
        if all(gates.values()) else "validation_failed"
    )
    return report


def write_tournament(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))


def print_tournament(report: dict) -> None:
    print("\n--- historical stock signal tournament ---")
    print("development candidate       trades  expectancy  +symbol-months  eligible")
    for name, stats in report["development"].items():
        print(
            f"{name:<27} {stats['trades']:>6} {stats['expectancy']:>+11.4f} "
            f"{stats['positive_symbol_month_ratio']:>13.1%}  "
            f"{'yes' if stats['eligible'] else 'no'}"
        )
    print(f"selected using development only: {report['selected'] or 'NONE'}")
    if report["validation"] is None:
        print("validation was not opened.")
        print(f"verdict: {report['verdict']}")
        return
    validation = report["validation"]
    print(
        f"validation: {validation['trades']} trades, expectancy "
        f"{validation['expectancy']:+.4f} points "
        f"(day-clustered 95% CI {validation['ci95_low']:+.4f}.."
        f"{validation['ci95_high']:+.4f})"
    )
    print(
        f"positive months: {validation['positive_months']}/{validation['active_months']}  "
        f"stress expectancy: {validation['stress_expectancy']:+.4f}"
    )
    for gate, passed in validation["gates"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {gate.replace('_', ' ')}")
    print(f"verdict: {report['verdict']}")


def print_followup(report: dict) -> None:
    print("\n--- opening-drive fade historical follow-up ---")
    development = report["development"]
    print(
        f"development: {development['trades']} trades, expectancy "
        f"{development['expectancy']:+.4f}, positive symbol-months "
        f"{development['positive_symbol_month_ratio']:.1%}"
    )
    for symbol, values in development["symbols"].items():
        print(
            f"  {symbol}: {values['trades']} trades, expectancy "
            f"{values['expectancy']:+.4f}"
        )
    print(f"development eligible: {'yes' if development['eligible'] else 'NO'}")
    if report["validation"] is None:
        print("validation was not opened.")
        print(f"verdict: {report['verdict']}")
        return
    validation = report["validation"]
    print(
        f"validation: {validation['trades']} trades, expectancy "
        f"{validation['expectancy']:+.4f} points "
        f"(day-clustered 95% CI {validation['ci95_low']:+.4f}.."
        f"{validation['ci95_high']:+.4f})"
    )
    print(
        f"positive months: {validation['positive_months']}/{validation['active_months']}  "
        f"stress expectancy: {validation['stress_expectancy']:+.4f}"
    )
    for gate, passed in validation["gates"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {gate.replace('_', ' ')}")
    print(f"verdict: {report['verdict']}")
