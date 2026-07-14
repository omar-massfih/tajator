"""Sequential swing-signal research on historical daily stock bars from TWS."""

from __future__ import annotations

import csv
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
DEVELOPMENT_SYMBOLS = ("AAPL", "META", "MSFT", "SPY")
VALIDATION_SYMBOLS = ("AMZN", "GOOGL", "NVDA", "QQQ")
ALL_SYMBOLS = DEVELOPMENT_SYMBOLS + VALIDATION_SYMBOLS
DEVELOPMENT_START = date(2018, 1, 1)
DEVELOPMENT_END = date(2022, 12, 30)
VALIDATION_START = DEVELOPMENT_START
VALIDATION_END = DEVELOPMENT_END
FINAL_START = date(2023, 1, 3)
FINAL_END = date(2026, 6, 30)
FOCUSED_START = FINAL_START
FOCUSED_END = FINAL_END
FETCH_START = date(2017, 1, 1)
BASE_COST_BPS_PER_SIDE = 2.0
STRESS_COST_BPS_PER_SIDE = 5.0


@dataclass(frozen=True)
class DailyTrade:
    strategy: str
    symbol: str
    direction: Direction
    signal_day: date
    entry_day: date
    exit_day: date
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    stress_net_return: float


def daily_cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / "daily" / f"{symbol.upper()}.csv"


def fetch_daily_history(ib, symbol: str, start: date, end: date, cache_dir: Path) -> Path:
    """Fetch one long RTH daily series directly from TWS and cache it as CSV."""
    stop = datetime.combine(end, time(20, 0), tzinfo=ET)
    years = max(1, math.ceil((end - start).days / 365) + 1)
    raw = ib.ib.reqHistoricalData(
        ib._underlying(symbol),
        endDateTime=stop,
        durationStr=f"{years} Y",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=2,
    )
    bars = []
    for item in raw:
        item_day = item.date.date() if isinstance(item.date, datetime) else item.date
        if not start <= item_day <= end:
            continue
        bars.append(
            Bar(
                ts=datetime.combine(item_day, time(), tzinfo=ET),
                open=float(item.open), high=float(item.high), low=float(item.low),
                close=float(item.close), volume=float(item.volume or 0),
            )
        )
    if not bars:
        raise RuntimeError(f"TWS returned no daily historical bars for {symbol}")
    path = daily_cache_path(cache_dir, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])
        for bar in sorted(bars, key=lambda value: value.ts):
            writer.writerow([
                bar.ts.isoformat(), bar.open, bar.high, bar.low, bar.close, bar.volume
            ])
    return path


def load_daily_history(cache_dir: Path, symbol: str) -> list[Bar]:
    path = daily_cache_path(cache_dir, symbol)
    if not path.exists():
        raise ValueError(f"missing TWS daily cache for {symbol}: {path}")
    bars = sorted(_read_csv(path), key=lambda value: value.ts)
    for previous, current in zip(bars, bars[1:], strict=False):
        if previous.close <= 0 or abs(current.close / previous.close - 1) > 0.80:
            raise ValueError(
                f"{symbol} daily series has an >80% adjacent close discontinuity "
                f"at {current.ts.date()}; adjustment continuity is unverified"
            )
    return bars


def _rsi2(bars: list[Bar], index: int) -> float | None:
    if index < 2:
        return None
    changes = [bars[offset].close - bars[offset - 1].close for offset in (index - 1, index)]
    gains = sum(max(change, 0.0) for change in changes) / 2
    losses = sum(max(-change, 0.0) for change in changes) / 2
    if gains + losses == 0:
        return 50.0
    return 100 * gains / (gains + losses)


def _true_ranges(bars: list[Bar]) -> list[float | None]:
    values: list[float | None] = [None]
    for previous, current in zip(bars, bars[1:], strict=False):
        values.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return values


def _direction_donchian(bars: list[Bar], index: int) -> Direction | None:
    if index < 20:
        return None
    prior = bars[index - 20:index]
    if bars[index].close > max(bar.high for bar in prior):
        return "call"
    if bars[index].close < min(bar.low for bar in prior):
        return "put"
    return None


def _direction_momentum60(bars: list[Bar], index: int) -> Direction | None:
    if index < 60 or bars[index - 60].close <= 0:
        return None
    momentum = bars[index].close / bars[index - 60].close - 1
    if momentum >= 0.10:
        return "call"
    if momentum <= -0.10:
        return "put"
    return None


def _direction_trend_rsi2(bars: list[Bar], index: int) -> Direction | None:
    if index < 100:
        return None
    rsi = _rsi2(bars, index)
    average = sum(bar.close for bar in bars[index - 99:index + 1]) / 100
    if bars[index].close > average and rsi is not None and rsi <= 10:
        return "call"
    if bars[index].close < average and rsi is not None and rsi >= 90:
        return "put"
    return None


def _direction_rsi2_reversal(bars: list[Bar], index: int) -> Direction | None:
    rsi = _rsi2(bars, index)
    if rsi is not None and rsi <= 5:
        return "call"
    if rsi is not None and rsi >= 95:
        return "put"
    return None


def _direction_compression_breakout(bars: list[Bar], index: int) -> Direction | None:
    if index < 20:
        return None
    ranges = _true_ranges(bars)
    prior20 = [value for value in ranges[index - 20:index] if value is not None]
    prior5 = [value for value in ranges[index - 5:index] if value is not None]
    if len(prior20) < 20 or len(prior5) < 5:
        return None
    if sum(prior5) / 5 > 0.60 * (sum(prior20) / 20):
        return None
    prior10 = bars[index - 10:index]
    if bars[index].close > max(bar.high for bar in prior10):
        return "call"
    if bars[index].close < min(bar.low for bar in prior10):
        return "put"
    return None


STRATEGIES: dict[str, tuple[int, Callable[[list[Bar], int], Direction | None]]] = {
    "donchian20_breakout": (10, _direction_donchian),
    "momentum60": (10, _direction_momentum60),
    "trend_rsi2_pullback": (5, _direction_trend_rsi2),
    "rsi2_reversal": (3, _direction_rsi2_reversal),
    "compression_breakout": (5, _direction_compression_breakout),
}


def generate_trades(
    symbol: str,
    bars: list[Bar],
    strategy: str,
    start: date,
    end: date,
) -> list[DailyTrade]:
    hold, direction_fn = STRATEGIES[strategy]
    trades = []
    blocked_through = -1
    for index, bar in enumerate(bars):
        signal_day = bar.ts.date()
        if signal_day < start or signal_day > end or index < blocked_through:
            continue
        direction = direction_fn(bars, index)
        entry_index, exit_index = index + 1, index + hold
        if direction is None or exit_index >= len(bars) or bars[exit_index].ts.date() > end:
            continue
        entry, exit_bar = bars[entry_index], bars[exit_index]
        if entry.open <= 0:
            continue
        sign = 1.0 if direction == "call" else -1.0
        ratio = exit_bar.close / entry.open
        gross = sign * (ratio - 1)

        def net(cost_bps: float) -> float:
            return gross - cost_bps / 10_000 * (1 + ratio)

        trades.append(
            DailyTrade(
                strategy=strategy, symbol=symbol, direction=direction,
                signal_day=signal_day, entry_day=entry.ts.date(), exit_day=exit_bar.ts.date(),
                entry_price=round(entry.open, 4), exit_price=round(exit_bar.close, 4),
                gross_return=round(gross, 8),
                net_return=round(net(BASE_COST_BPS_PER_SIDE), 8),
                stress_net_return=round(net(STRESS_COST_BPS_PER_SIDE), 8),
            )
        )
        blocked_through = exit_index
    return trades


def _stats(trades: list[DailyTrade], metric: str = "net_return") -> dict:
    values = [float(getattr(trade, metric)) for trade in trades]
    count = len(values)
    mean = sum(values) / count if count else 0.0
    by_day: dict[date, list[float]] = defaultdict(list)
    by_month: dict[str, float] = defaultdict(float)
    by_symbol: dict[str, list[float]] = defaultdict(list)
    by_symbol_year: dict[tuple[str, int], float] = defaultdict(float)
    by_year: dict[int, float] = defaultdict(float)
    for trade, value in zip(trades, values, strict=True):
        by_day[trade.signal_day].append(value)
        by_month[trade.signal_day.strftime("%Y-%m")] += value
        by_symbol[trade.symbol].append(value)
        by_symbol_year[(trade.symbol, trade.signal_day.year)] += value
        by_year[trade.signal_day.year] += value
    groups = len(by_day)
    if count and groups > 1:
        cluster_sums = [sum(value - mean for value in group) for group in by_day.values()]
        se = math.sqrt(groups / (groups - 1) * sum(value**2 for value in cluster_sums) / count**2)
    else:
        se = 0.0
    return {
        "trades": count,
        "trading_days": groups,
        "wins": sum(value > 0 for value in values),
        "win_rate": round(sum(value > 0 for value in values) / count, 4) if count else 0.0,
        "total_return_units": round(sum(values), 8),
        "expectancy": round(mean, 8),
        "clustered_standard_error": round(se, 8),
        "ci95_low": round(mean - 1.96 * se, 8),
        "ci95_high": round(mean + 1.96 * se, 8),
        "active_months": len(by_month),
        "positive_months": sum(value > 0 for value in by_month.values()),
        "positive_month_ratio": (
            round(sum(value > 0 for value in by_month.values()) / len(by_month), 4)
            if by_month else 0.0
        ),
        "positive_symbol_year_ratio": (
            round(sum(value > 0 for value in by_symbol_year.values()) / len(by_symbol_year), 4)
            if by_symbol_year else 0.0
        ),
        "symbols": {
            symbol: {
                "trades": len(symbol_values),
                "total_return_units": round(sum(symbol_values), 8),
                "expectancy": round(sum(symbol_values) / len(symbol_values), 8),
            }
            for symbol, symbol_values in sorted(by_symbol.items())
        },
        "years": {str(year): round(value, 8) for year, value in sorted(by_year.items())},
    }


def _coverage(histories: dict[str, list[Bar]], symbols: tuple[str, ...], start: date, end: date) -> dict:
    requested = len(trading_days(start, end)) * len(symbols)
    covered = sum(
        sum(start <= bar.ts.date() <= end for bar in histories[symbol]) for symbol in symbols
    )
    return {
        "requested_symbol_weekdays": requested,
        "covered_symbol_sessions": covered,
        "coverage_ratio": round(covered / requested, 4) if requested else 0.0,
    }


def _collect(
    histories: dict[str, list[Bar]], symbols: tuple[str, ...], strategy: str,
    start: date, end: date,
) -> list[DailyTrade]:
    return [
        trade
        for symbol in symbols
        for trade in generate_trades(symbol, histories[symbol], strategy, start, end)
    ]


def _stage_gates(stats: dict, stress: dict, coverage: dict, *, minimum: int = 100) -> dict:
    sampled = {symbol: row for symbol, row in stats["symbols"].items() if row["trades"] >= 10}
    return {
        "minimum_trades": stats["trades"] >= minimum,
        "positive_each_sampled_symbol": bool(sampled) and all(
            row["expectancy"] > 0 for row in sampled.values()
        ),
        "day_clustered_ci_above_zero": stats["ci95_low"] > 0,
        "positive_month_stability": (
            stats["active_months"] >= 3 and stats["positive_month_ratio"] >= 0.6
        ),
        "positive_stress_expectancy": stress["expectancy"] > 0,
        "data_coverage": coverage["coverage_ratio"] >= 0.9,
    }


def run_daily_tournament(cache_dir: Path) -> dict:
    histories = {symbol: load_daily_history(cache_dir, symbol) for symbol in ALL_SYMBOLS}
    development = {}
    eligible = []
    for strategy in STRATEGIES:
        trades = _collect(
            histories, DEVELOPMENT_SYMBOLS, strategy, DEVELOPMENT_START, DEVELOPMENT_END
        )
        stats = _stats(trades)
        sampled = {
            symbol: row for symbol, row in stats["symbols"].items() if row["trades"] >= 10
        }
        qualifies = (
            stats["trades"] >= 100
            and len(sampled) == len(DEVELOPMENT_SYMBOLS)
            and all(row["expectancy"] > 0 for row in sampled.values())
            and stats["positive_symbol_year_ratio"] >= 0.55
        )
        development[strategy] = {**stats, "eligible": qualifies}
        if qualifies:
            eligible.append(strategy)
    selected = max(
        eligible, key=lambda strategy: (development[strategy]["expectancy"], strategy),
        default=None,
    )
    report = {
        "protocol": "historical_daily_swing_tournament_v1",
        "data_source": "TWS RTH daily stock TRADES bars",
        "costs": {
            "base_bps_per_side": BASE_COST_BPS_PER_SIDE,
            "stress_bps_per_side": STRESS_COST_BPS_PER_SIDE,
        },
        "development": development,
        "development_coverage": _coverage(
            histories, DEVELOPMENT_SYMBOLS, DEVELOPMENT_START, DEVELOPMENT_END
        ),
        "selected": selected,
        "validation_opened": selected is not None,
        "validation": None,
        "final_opened": False,
        "final": None,
        "verdict": "no_development_candidate" if selected is None else "pending",
    }
    if selected is None:
        return report

    validation_trades = _collect(
        histories, VALIDATION_SYMBOLS, selected, VALIDATION_START, VALIDATION_END
    )
    validation_stats = _stats(validation_trades)
    validation_stress = _stats(validation_trades, "stress_net_return")
    validation_coverage = _coverage(
        histories, VALIDATION_SYMBOLS, VALIDATION_START, VALIDATION_END
    )
    validation_gates = _stage_gates(
        validation_stats, validation_stress, validation_coverage
    )
    report["validation"] = {
        **validation_stats,
        "stress_expectancy": validation_stress["expectancy"],
        "coverage": validation_coverage,
        "gates": validation_gates,
        "trades_ledger": [dataclasses.asdict(trade) for trade in validation_trades],
    }
    if not all(validation_gates.values()):
        report["verdict"] = "validation_failed"
        return report

    report["final_opened"] = True
    final_trades = _collect(histories, ALL_SYMBOLS, selected, FINAL_START, FINAL_END)
    final_stats = _stats(final_trades)
    final_stress = _stats(final_trades, "stress_net_return")
    final_coverage = _coverage(histories, ALL_SYMBOLS, FINAL_START, FINAL_END)
    final_gates = _stage_gates(final_stats, final_stress, final_coverage)
    final_gates["positive_each_calendar_year"] = bool(final_stats["years"]) and all(
        value > 0 for value in final_stats["years"].values()
    )
    report["final"] = {
        **final_stats,
        "stress_expectancy": final_stress["expectancy"],
        "coverage": final_coverage,
        "gates": final_gates,
        "trades_ledger": [dataclasses.asdict(trade) for trade in final_trades],
    }
    report["verdict"] = (
        "historically_supported_swing_signal"
        if all(final_gates.values()) else "final_holdout_failed"
    )
    return report


def _focused_stage(
    histories: dict[str, list[Bar]], symbol: str
) -> tuple[dict, bool]:
    trades = generate_trades(
        symbol, histories[symbol], "trend_rsi2_pullback", FOCUSED_START, FOCUSED_END
    )
    stats = _stats(trades)
    stress = _stats(trades, "stress_net_return")
    coverage = _coverage(histories, (symbol,), FOCUSED_START, FOCUSED_END)
    gates = {
        "minimum_trades": stats["trades"] >= 75,
        "positive_expectancy": stats["expectancy"] > 0,
        "day_clustered_ci_above_zero": stats["ci95_low"] > 0,
        "positive_month_stability": (
            stats["active_months"] >= 3 and stats["positive_month_ratio"] >= 0.6
        ),
        "positive_stress_expectancy": stress["expectancy"] > 0,
        "positive_each_calendar_year": bool(stats["years"]) and all(
            value > 0 for value in stats["years"].values()
        ),
        "data_coverage": coverage["coverage_ratio"] >= 0.9,
    }
    return {
        **stats,
        "stress_expectancy": stress["expectancy"],
        "coverage": coverage,
        "gates": gates,
        "trades_ledger": [dataclasses.asdict(trade) for trade in trades],
    }, all(gates.values())


def run_aapl_focused_holdout(cache_dir: Path) -> dict:
    """Open MSFT replication only after every preregistered AAPL gate passes."""
    aapl_history = load_daily_history(cache_dir, "AAPL")
    primary, primary_passed = _focused_stage({"AAPL": aapl_history}, "AAPL")
    report = {
        "protocol": "aapl_focused_holdout_v1",
        "data_source": "fresh isolated TWS RTH daily stock TRADES bars",
        "strategy": "trend_rsi2_pullback",
        "period": [FOCUSED_START.isoformat(), FOCUSED_END.isoformat()],
        "primary_symbol": "AAPL",
        "primary": primary,
        "primary_passed": primary_passed,
        "replication_symbol": "MSFT",
        "replication_opened": False,
        "replication": None,
        "options_stage_eligible": primary_passed,
        "verdict": "aapl_historical_signal_not_supported",
    }
    if not primary_passed:
        return report

    # Conditional boundary: no MSFT file is loaded before an AAPL pass.
    msft_history = load_daily_history(cache_dir, "MSFT")
    replication, replication_passed = _focused_stage({"MSFT": msft_history}, "MSFT")
    report["replication_opened"] = True
    report["replication"] = replication
    report["replication_passed"] = replication_passed
    report["verdict"] = (
        "aapl_signal_supported_with_msft_replication"
        if replication_passed else "aapl_signal_supported_replication_failed"
    )
    return report


def write_daily_tournament(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))


def print_daily_tournament(report: dict) -> None:
    print("\n--- historical daily swing-signal tournament ---")
    print("development candidate       trades  expectancy   +symbol-years  eligible")
    for name, stats in report["development"].items():
        print(
            f"{name:<27} {stats['trades']:>6} {stats['expectancy']:>+11.3%} "
            f"{stats['positive_symbol_year_ratio']:>14.1%}  "
            f"{'yes' if stats['eligible'] else 'no'}"
        )
    print(f"selected using development only: {report['selected'] or 'NONE'}")
    for label in ("validation", "final"):
        stage = report[label]
        if stage is None:
            print(f"{label}: not opened")
            continue
        print(
            f"{label}: {stage['trades']} trades, expectancy {stage['expectancy']:+.3%}, "
            f"day-clustered 95% CI {stage['ci95_low']:+.3%}..{stage['ci95_high']:+.3%}, "
            f"positive months {stage['positive_months']}/{stage['active_months']}"
        )
        for gate, passed in stage["gates"].items():
            print(f"  {'PASS' if passed else 'FAIL'}  {gate.replace('_', ' ')}")
    print(f"verdict: {report['verdict']}")


def print_aapl_focused(report: dict) -> None:
    print("\n--- AAPL-focused historical TWS holdout ---")
    primary = report["primary"]
    print(
        f"AAPL: {primary['trades']} trades, expectancy {primary['expectancy']:+.3%}, "
        f"95% CI {primary['ci95_low']:+.3%}..{primary['ci95_high']:+.3%}, "
        f"positive months {primary['positive_months']}/{primary['active_months']}"
    )
    for gate, passed in primary["gates"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {gate.replace('_', ' ')}")
    print(f"MSFT replication opened: {'yes' if report['replication_opened'] else 'NO'}")
    if report["replication"] is not None:
        replication = report["replication"]
        print(
            f"MSFT: {replication['trades']} trades, expectancy "
            f"{replication['expectancy']:+.3%}, 95% CI "
            f"{replication['ci95_low']:+.3%}..{replication['ci95_high']:+.3%}"
        )
        for gate, passed in replication["gates"].items():
            print(f"  {'PASS' if passed else 'FAIL'}  {gate.replace('_', ' ')}")
    print(f"options stage eligible: {'yes' if report['options_stage_eligible'] else 'NO'}")
    print(f"verdict: {report['verdict']}")
