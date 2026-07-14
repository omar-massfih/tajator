"""Evidence gates for deciding whether a backtest supports a tradeable edge.

The ordinary backtest summary answers "what happened?".  This module answers
the stricter research question: "is the sample large, stable, and genuinely
out-of-sample enough to justify calling it an edge?"
"""

from __future__ import annotations

import json
import math
import dataclasses
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from ..broker.backtest import BacktestBroker
from ..broker.base import Fill
from ..models import SelectedContract
from .data import ET, ensure_option_bars
from .ledger import BacktestReport, build_report


def _shadow_journal_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("journal-*.jsonl"))
    raise ValueError(f"shadow journal path does not exist: {path}")


def build_shadow_report(path: Path, *, symbol: str) -> BacktestReport:
    """Convert quote-side shadow fills into the ordinary options ledger schema."""
    symbol = symbol.upper()
    paths = _shadow_journal_paths(path)
    if not paths:
        raise ValueError(f"no journal-*.jsonl files found in {path}")
    fills_by_day: dict[date, list[tuple[str, SelectedContract, Fill]]] = defaultdict(list)
    observed_days: set[date] = set()
    covered_days: set[date] = set()
    for journal_path in paths:
        for line_number, line in enumerate(journal_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {journal_path}:{line_number}: {exc}") from exc
            event_type = event.get("type")
            if event_type == "shadow_started" and str(event.get("symbol", "")).upper() == symbol:
                observed_days.add(datetime.fromisoformat(event["ts"]).astimezone(ET).date())
            if (
                event_type == "shadow_session_covered"
                and str(event.get("symbol", "")).upper() == symbol
                and event.get("no_order_placed") is True
            ):
                covered_days.add(datetime.fromisoformat(event["ts"]).astimezone(ET).date())
            if event_type != "shadow_execution":
                continue
            contract = SelectedContract.model_validate(event.get("contract"))
            if contract.symbol.upper() != symbol:
                continue
            if event.get("no_order_placed") is not True:
                raise ValueError(
                    f"shadow event in {journal_path}:{line_number} lacks no_order_placed=true"
                )
            fill = Fill.model_validate(event.get("fill"))
            day = fill.ts.astimezone(ET).date()
            observed_days.add(day)
            fills_by_day[day].append((str(event.get("side", "")).upper(), contract, fill))
    if not observed_days:
        raise ValueError(f"no {symbol} shadow sessions found in {path}")
    for fills in fills_by_day.values():
        fills.sort(key=lambda row: row[2].ts)
    metadata = {
        "research_mode": "historical_options",
        "validation_protocol": {
            "kind": "deterministic_live_quote_shadow",
            "decision_policy": "same deterministic graph as no-LLM backtest",
            "no_orders": True,
        },
        "execution_model": {
            "entry": "live TWS ask",
            "exit": "live TWS bid",
            "commissions": "journaled per simulated fill",
        },
        "data_coverage": {
            "requested_weekdays": len(observed_days),
            "days_with_underlying_bars": len(covered_days),
            "skipped_missing_option_days": [],
        },
        "source": "TWS live underlying bars and option bid/ask; no orders placed",
    }
    return build_report(
        symbol, min(observed_days), max(observed_days), fills_by_day, metadata=metadata
    )


def _report_payload(report: BacktestReport) -> dict:
    payload = dataclasses.asdict(report)
    payload["daily_pnl"] = {day.isoformat(): pnl for day, pnl in report.daily_pnl.items()}
    payload["equity_curve"] = [(day.isoformat(), value) for day, value in report.equity_curve]
    payload["underlying_equity_curve"] = [
        (day.isoformat(), value) for day, value in report.underlying_equity_curve
    ]
    return payload


def write_report(report: BacktestReport, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_report_payload(report), indent=2, default=str))


def print_shadow_report(report: BacktestReport) -> None:
    print(f"\n--- no-order shadow report: {report.symbol} {report.start} → {report.end} ---")
    print(
        f"closed trades: {report.total_trades}  win rate: {report.win_rate:.1%}  "
        f"net PnL: ${report.total_pnl:+.2f}  fees: ${report.total_fees:.2f}"
    )
    print("execution: BUY at live ask / SELL at live bid; NO ORDERS PLACED")


@dataclass(frozen=True)
class SampleStats:
    start: str | None
    end: str | None
    trades: int
    wins: int
    win_rate: float
    total: float
    expectancy: float
    standard_deviation: float
    ci95_low: float
    ci95_high: float
    profit_factor: float
    active_months: int
    positive_months: int
    positive_month_ratio: float
    max_drawdown: float


@dataclass(frozen=True)
class PairedPanelStats:
    pairs: int
    base_avg_return: float
    variant_avg_return: float
    mean_return_improvement: float
    ci95_familywise_low: float
    ci95_familywise_high: float
    variant_better_pairs: int
    complete: bool
    verdict: str


def _sample_stats(trades: list[dict], metric: str) -> SampleStats:
    observations = [
        (str(trade["day"]), float(trade[metric]))
        for trade in trades
        if trade.get("closed", True) and trade.get(metric) is not None
    ]
    if not observations:
        return SampleStats(
            None, None, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0
        )

    values = [value for _, value in observations]
    count = len(values)
    total = sum(values)
    mean = total / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1) if count > 1 else 0.0
    standard_deviation = math.sqrt(variance)
    margin = 1.96 * standard_deviation / math.sqrt(count) if count > 1 else 0.0
    gains = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value <= 0))

    monthly: dict[str, float] = defaultdict(float)
    daily: dict[str, float] = defaultdict(float)
    for day, value in observations:
        monthly[day[:7]] += value
        daily[day] += value

    cumulative = peak = max_drawdown = 0.0
    for day in sorted(daily):
        cumulative += daily[day]
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    positive_months = sum(value > 0 for value in monthly.values())
    return SampleStats(
        start=min(day for day, _ in observations),
        end=max(day for day, _ in observations),
        trades=count,
        wins=sum(value > 0 for value in values),
        win_rate=round(sum(value > 0 for value in values) / count, 4),
        total=round(total, 4),
        expectancy=round(mean, 4),
        standard_deviation=round(standard_deviation, 4),
        ci95_low=round(mean - margin, 4),
        ci95_high=round(mean + margin, 4),
        profit_factor=round(gains / losses, 4) if losses else (math.inf if gains else 0.0),
        active_months=len(monthly),
        positive_months=positive_months,
        positive_month_ratio=round(positive_months / len(monthly), 4),
        max_drawdown=round(max_drawdown, 4),
    )


def audit_report(
    report: dict,
    *,
    validation_start: date | None = None,
    validation_only: bool = False,
    min_trades: int = 50,
    min_profit_factor: float = 1.2,
    min_positive_month_ratio: float = 0.6,
) -> dict:
    """Audit one persisted report without rerunning or retuning the strategy."""
    if validation_start is not None and validation_only:
        raise ValueError("validation_start and validation_only are mutually exclusive")
    if min_trades <= 0:
        raise ValueError("min_trades must be positive")

    metadata = report.get("metadata") or {}
    mode = metadata.get("research_mode", "historical_options")
    metric = "underlying_points" if mode == "underlying_only" else "pnl"
    trades = report.get("trades") or []
    split = validation_start.isoformat() if validation_start else None
    development = _sample_stats(
        [trade for trade in trades if split is not None and str(trade.get("day", "")) < split],
        metric,
    ) if split else None
    validation_trades = (
        [trade for trade in trades if str(trade.get("day", "")) >= split]
        if split else trades
    )
    validation = _sample_stats(validation_trades, metric)
    is_validation = validation_only or validation_start is not None

    coverage = metadata.get("data_coverage") or {}
    requested = int(coverage.get("requested_weekdays") or 0)
    covered = int(coverage.get("days_with_underlying_bars") or 0)
    coverage_ratio = covered / requested if requested else 0.0
    missing_option_days = coverage.get("skipped_missing_option_days") or []
    coverage_ok = coverage_ratio >= 0.9 and not missing_option_days

    gates = {
        "declared_out_of_sample": is_validation,
        "minimum_trades": validation.trades >= min_trades,
        "positive_expectancy": validation.expectancy > 0,
        "ci95_above_zero": validation.ci95_low > 0,
        "positive_month_stability": (
            validation.active_months >= 3
            and validation.positive_month_ratio >= min_positive_month_ratio
        ),
        "data_coverage": coverage_ok,
    }
    if mode == "historical_options":
        gates["profit_factor"] = validation.profit_factor >= min_profit_factor
        gates["cost_model_disclosed"] = bool(metadata.get("execution_model"))

    if not is_validation:
        verdict = "exploratory_only"
    elif validation.trades < min_trades:
        verdict = "insufficient_sample"
    elif mode == "underlying_only":
        verdict = (
            "statistically_supported_stock_signal"
            if all(gates.values()) else
            "stock_signal_not_yet_supported"
        )
    else:
        verdict = "confirmed_options_edge" if all(gates.values()) else "options_edge_not_confirmed"

    return {
        "symbol": report.get("symbol", ""),
        "mode": mode,
        "metric": "underlying points" if metric == "underlying_points" else "net dollars",
        "verdict": verdict,
        "development": asdict(development) if development else None,
        "validation": asdict(validation),
        "gates": gates,
        "thresholds": {
            "min_trades": min_trades,
            "min_profit_factor": min_profit_factor if mode == "historical_options" else None,
            "min_positive_month_ratio": min_positive_month_ratio,
        },
        "coverage": {
            "underlying_session_ratio": round(coverage_ratio, 4),
            "skipped_missing_option_days": len(missing_option_days),
        },
    }


def load_and_audit(
    path: Path,
    **kwargs,
) -> dict:
    return audit_report(json.loads(path.read_text()), **kwargs)


def print_audit(audit: dict) -> None:
    sample = audit["validation"]
    unit = "pts" if audit["metric"] == "underlying points" else "$"
    print(f"\n--- edge audit: {audit['symbol']} ({audit['mode']}) ---")
    print(f"verdict: {audit['verdict']}")
    print(
        f"validation: {sample['trades']} trades, {sample['win_rate']:.1%} wins, "
        f"expectancy {sample['expectancy']:+.3f}{unit} "
        f"(95% CI {sample['ci95_low']:+.3f}..{sample['ci95_high']:+.3f})"
    )
    print(
        f"profit factor {sample['profit_factor']:.2f}, "
        f"positive months {sample['positive_months']}/{sample['active_months']}, "
        f"max drawdown {sample['max_drawdown']:.2f}{unit}"
    )
    for name, passed in audit["gates"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name.replace('_', ' ')}")


def print_option_panel(report: dict, min_pairs: int = 50) -> None:
    """Compare contract variants; this is exploratory contract research, not an edge gate."""
    base = _sample_stats(report.get("trades") or [], "pnl")
    print(f"\n--- option panel: {report.get('symbol', '')} ---")
    base_audit = audit_report(report, validation_only=True)
    base_confirmed = base_audit["verdict"] == "confirmed_options_edge"
    print(f"base options edge confirmed: {'yes' if base_confirmed else 'NO'}")
    print(
        "variant             pairs   net-pnl   avg RoP   delta RoP      adjusted CI   verdict"
    )

    def row(name: str, trades: list[dict], complete: bool, paired: PairedPanelStats | None) -> None:
        stats = _sample_stats(trades, "pnl")
        returns = [
            float(trade["return_on_premium"])
            for trade in trades if trade.get("return_on_premium") is not None
        ]
        average_return = sum(returns) / len(returns) if returns else 0.0
        if paired is None:
            print(
                f"{name:<19} {stats.trades:>5} {stats.total:>9.2f} {average_return:>9.2%} "
                f"{'-':>11} {'-':>16}   base"
            )
            return
        interval = f"{paired.ci95_familywise_low:+.2%}..{paired.ci95_familywise_high:+.2%}"
        print(
            f"{name:<19} {paired.pairs:>5} {stats.total:>9.2f} {average_return:>9.2%} "
            f"{paired.mean_return_improvement:>+10.2%} {interval:>16}   {paired.verdict}"
        )

    row("base_atm_near", report.get("trades") or [], True, None)
    paired_rows = paired_panel_rows(report, min_pairs=min_pairs)
    for name, data in sorted((report.get("option_panel") or {}).items()):
        row(
            name,
            data.get("trades") or [],
            bool(data.get("complete", not data.get("missing_contracts"))),
            paired_rows[name],
        )
    if not report.get("option_panel"):
        print("no option panel was captured in this report")
    elif not base_confirmed:
        print("no contract variant is promotable until the base options edge passes its audit")


def _indexed_trades(trades: list[dict]) -> dict[tuple, dict]:
    """Stable keys allow exact pairing even when a variant is partially missing."""
    indexed = {}
    occurrences: dict[tuple, int] = defaultdict(int)
    for trade in trades:
        core = (
            str(trade.get("day")), str(trade.get("entry_ts")), str(trade.get("exit_ts")),
            str(trade.get("direction")), int(trade.get("qty") or 0),
        )
        occurrence = occurrences[core]
        occurrences[core] += 1
        indexed[(*core, occurrence)] = trade
    return indexed


def paired_panel_rows(report: dict, min_pairs: int = 50) -> dict[str, PairedPanelStats]:
    """Paired contract-efficiency tests with Bonferroni-adjusted 95% familywise CIs."""
    if min_pairs <= 0:
        raise ValueError("min_pairs must be positive")
    base = _indexed_trades(report.get("trades") or [])
    rows = {}
    # Three variants are predeclared. z=2.394 gives 98.33% per-comparison CIs,
    # approximately preserving a 95% familywise error rate.
    adjusted_z = 2.394
    for name, data in sorted((report.get("option_panel") or {}).items()):
        variant = _indexed_trades(data.get("trades") or [])
        keys = sorted(set(base) & set(variant))
        paired = [
            (float(base[key]["return_on_premium"]), float(variant[key]["return_on_premium"]))
            for key in keys
            if base[key].get("return_on_premium") is not None
            and variant[key].get("return_on_premium") is not None
        ]
        deltas = [alternative - base_return for base_return, alternative in paired]
        count = len(deltas)
        mean = sum(deltas) / count if count else 0.0
        variance = (
            sum((value - mean) ** 2 for value in deltas) / (count - 1)
            if count > 1 else 0.0
        )
        margin = adjusted_z * math.sqrt(variance) / math.sqrt(count) if count > 1 else 0.0
        complete = bool(data.get("complete", not data.get("missing_contracts")))
        if not complete:
            verdict = "incomplete"
        elif count < min_pairs:
            verdict = "insufficient_pairs"
        elif mean - margin > 0:
            verdict = "positive_paired_advantage"
        else:
            verdict = "no_proven_advantage"
        rows[name] = PairedPanelStats(
            pairs=count,
            base_avg_return=round(sum(pair[0] for pair in paired) / count, 6) if count else 0.0,
            variant_avg_return=round(sum(pair[1] for pair in paired) / count, 6) if count else 0.0,
            mean_return_improvement=round(mean, 6),
            ci95_familywise_low=round(mean - margin, 6),
            ci95_familywise_high=round(mean + margin, 6),
            variant_better_pairs=sum(alternative > base_return for base_return, alternative in paired),
            complete=complete,
            verdict=verdict,
        )
    return rows


def calibrate_execution_journal(
    path: Path,
    *,
    symbol: str,
    ib,
    cache_dir: Path,
    settings,
) -> dict:
    """Compare paper fills with the historical-bar execution model at each signal time."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    fill_records = [
        record for record in records
        if record.get("type") == "fill" and record.get("symbol") == symbol.upper()
    ]
    policy_modes = sorted({
        record.get("policy_mode")
        for record in records
        if record.get("type") == "policy_start"
        and (
            not record.get("symbols")
            or symbol.upper() in record.get("symbols", [])
        )
        and record.get("policy_mode") in {"deterministic", "llm"}
    })
    if len(policy_modes) == 1:
        decision_mode = policy_modes[0]
    elif len(policy_modes) > 1:
        decision_mode = "mixed"
    else:
        # Legacy journals used the event name ``llm_decision`` for both policy
        # modes. Only explicit non-rule-follower reasoning proves LLM use.
        decisions = [
            record.get("decision") or {}
            for record in records
            if record.get("type") == "llm_decision"
            and record.get("symbol") == symbol.upper()
        ]
        reasonings = [
            str(decision.get("reasoning") or "")
            for decision in decisions
            if decision.get("action") not in (None, "wait")
        ]
        if reasonings and all(reason.startswith("rule-follower:") for reason in reasonings):
            decision_mode = "deterministic_inferred"
        elif decisions:
            decision_mode = "llm"
        else:
            decision_mode = "unknown"

    actual_commissions: dict[int, float] = {}
    for record in records:
        if record.get("type") != "commission_report" or record.get("symbol") != symbol.upper():
            continue
        order_id = record.get("order_id")
        commission = record.get("commission")
        if order_id is not None and commission is not None:
            actual_commissions[int(order_id)] = (
                actual_commissions.get(int(order_id), 0.0) + float(commission)
            )
    series_cache: dict[tuple, list] = {}
    rows = []
    trades = []
    active: dict | None = None

    for record in fill_records:
        action = record.get("action") or {}
        position = record.get("position") or {}
        contract_data = position.get("contract") or {}
        if not contract_data:
            continue
        contract = SelectedContract.model_validate(contract_data)
        signal_ts = datetime.fromisoformat(record["ts"])
        actual_ts = datetime.fromisoformat(action["ts"])
        day = signal_ts.astimezone(ET).date()
        key = (contract.symbol, contract.expiry, contract.strike, contract.right, day)
        if key not in series_cache:
            series = ensure_option_bars(ib, contract, day, cache_dir)
            if not series:
                raise ValueError(f"no historical bars for journal contract {contract.local_name}")
            series_cache[key] = series
        reference = BacktestBroker._fill_price(series_cache[key], signal_ts)
        if reference is None:
            raise ValueError(
                f"no timely option bar for {contract.local_name} at {signal_ts.isoformat()}"
            )
        side = "BUY" if action.get("kind") == "entry" else "SELL"
        qty = int(action.get("qty") or 0)
        actual = float(action["premium"])
        modeled_adverse = reference * settings.backtest_half_spread_pct + (
            settings.backtest_slippage_cents / 100
        )
        modeled = (
            reference + modeled_adverse if side == "BUY"
            else max(0.01, reference - modeled_adverse)
        )
        actual_adverse = actual - reference if side == "BUY" else reference - actual
        estimated_fee = max(
            settings.backtest_min_commission_per_order,
            qty * settings.backtest_commission_per_contract,
        )
        execution_quality = action.get("execution_quality") or {}
        order_id = execution_quality.get("order_id")
        actual_fee = actual_commissions.get(int(order_id)) if order_id is not None else None
        applied_fee = actual_fee if actual_fee is not None else estimated_fee
        signal_to_fill_s = max(0.0, (actual_ts - signal_ts).total_seconds())
        # Historical replay intentionally prices the next one-minute bar open;
        # this is the extra wall-clock delay beyond that modeled minute.
        next_bar_delay_s = max(0.0, signal_to_fill_s - 60.0)
        row = {
            "side": side,
            "kind": action.get("kind"),
            "contract": contract.local_name,
            "qty": qty,
            "signal_ts": signal_ts.isoformat(),
            "actual_fill_ts": actual_ts.isoformat(),
            "latency_s": round(signal_to_fill_s, 3),
            "next_bar_delay_s": round(next_bar_delay_s, 3),
            "reference_price": round(reference, 4),
            "modeled_fill": round(modeled, 4),
            "actual_fill": actual,
            "modeled_adverse_cents": round(modeled_adverse * 100, 3),
            "actual_adverse_cents": round(actual_adverse * 100, 3),
            "estimated_fee": round(estimated_fee, 2),
            "actual_commission": round(actual_fee, 6) if actual_fee is not None else None,
            "applied_fee": round(applied_fee, 6),
            "fee_source": "ib_commission_report" if actual_fee is not None else "configured_estimate",
        }
        rows.append(row)

        if side == "BUY":
            if active is not None:
                raise ValueError("journal has a new entry before the prior trade closed")
            active = {"entry": row, "exits": []}
        elif active is not None:
            active["exits"].append(row)
            if int(position.get("qty_remaining") or 0) == 0:
                entry = active["entry"]
                exits = active["exits"]
                closed_qty = sum(exit_row["qty"] for exit_row in exits)
                actual_gross = (
                    sum(exit_row["actual_fill"] * exit_row["qty"] for exit_row in exits)
                    - entry["actual_fill"] * closed_qty
                ) * 100
                modeled_gross = (
                    sum(exit_row["modeled_fill"] * exit_row["qty"] for exit_row in exits)
                    - entry["modeled_fill"] * closed_qty
                ) * 100
                fees = entry["applied_fee"] + sum(
                    exit_row["applied_fee"] for exit_row in exits
                )
                trades.append({
                    "contract": entry["contract"],
                    "entry_signal_ts": entry["signal_ts"],
                    "qty": closed_qty,
                    "actual_gross_pnl": round(actual_gross, 2),
                    "modeled_gross_pnl": round(modeled_gross, 2),
                    "estimated_fees": round(fees, 2),
                    "actual_net_estimate": round(actual_gross - fees, 2),
                    "modeled_net_pnl": round(modeled_gross - fees, 2),
                    "actual_minus_modeled": round(actual_gross - modeled_gross, 2),
                })
                active = None

    actual_adverse = [row["actual_adverse_cents"] for row in rows]
    modeled_adverse = [row["modeled_adverse_cents"] for row in rows]
    latencies = [row["latency_s"] for row in rows]
    next_bar_delays = [row["next_bar_delay_s"] for row in rows]
    return {
        "symbol": symbol.upper(),
        "journal": str(path),
        "fills": rows,
        "trades": trades,
        "summary": {
            "fills": len(rows),
            "closed_trades": len(trades),
            "decision_mode": decision_mode,
            "mean_signal_to_fill_latency_s": round(sum(latencies) / len(latencies), 3)
            if latencies else 0.0,
            "mean_next_bar_delay_s": round(
                sum(next_bar_delays) / len(next_bar_delays), 3
            ) if next_bar_delays else 0.0,
            "mean_actual_adverse_cents": round(sum(actual_adverse) / len(actual_adverse), 3)
            if actual_adverse else 0.0,
            "mean_modeled_adverse_cents": round(sum(modeled_adverse) / len(modeled_adverse), 3)
            if modeled_adverse else 0.0,
            "actual_net_estimate": round(sum(trade["actual_net_estimate"] for trade in trades), 2),
            "modeled_net_pnl": round(sum(trade["modeled_net_pnl"] for trade in trades), 2),
            "actual_minus_modeled": round(
                sum(trade["actual_minus_modeled"] for trade in trades), 2
            ),
            "open_trade_omitted": active is not None,
            "commission_note": (
                "IB commission reports applied where matched by order ID; configured fees fill gaps"
                if actual_commissions else
                "actual commissions absent from journal; configured fees applied"
            ),
            "transferability_note": (
                "LLM fill adversity includes decision latency and must not replace deterministic "
                "assumptions without deterministic paper-fill evidence"
                if decision_mode in {"llm", "mixed"} else
                "sample is explicitly deterministic and comparable to frozen cohorts"
                if decision_mode == "deterministic" else
                "decision mode was inferred or could not be proven from legacy journal events"
            ),
        },
    }


def print_execution_calibration(calibration: dict) -> None:
    summary = calibration["summary"]
    print(f"\n--- execution calibration: {calibration['symbol']} ---")
    print(
        f"fills {summary['fills']}  closed trades {summary['closed_trades']}  "
        f"actual net estimate ${summary['actual_net_estimate']:+.2f}  "
        f"modeled net ${summary['modeled_net_pnl']:+.2f}"
    )
    print(
        f"mean adverse fill: actual {summary['mean_actual_adverse_cents']:+.2f}c  "
        f"model {summary['mean_modeled_adverse_cents']:+.2f}c  "
        f"actual-minus-modeled PnL ${summary['actual_minus_modeled']:+.2f}"
    )
    print(
        f"decision mode {summary['decision_mode']}  mean signal-to-fill latency "
        f"{summary['mean_signal_to_fill_latency_s']:.1f}s  delay beyond next bar "
        f"{summary['mean_next_bar_delay_s']:.1f}s"
    )
    for row in calibration["fills"]:
        print(
            f"  {row['side']:<4} {row['signal_ts'][11:19]} {row['contract']}  "
            f"ref {row['reference_price']:.2f} model {row['modeled_fill']:.2f} "
            f"actual {row['actual_fill']:.2f}  adverse {row['actual_adverse_cents']:+.1f}c"
        )
    print(f"warning: {summary['transferability_note']}")
