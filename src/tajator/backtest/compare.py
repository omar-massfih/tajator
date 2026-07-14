"""Compact comparison table for experiment-safe backtest JSON reports."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist


def comparison_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        meta = data.get("metadata", {})
        mode = meta.get("research_mode", "historical_options")
        metric = "underlying_points" if mode == "underlying_only" else "pnl"
        observations = [
            (str(trade.get("day", "")), float(trade[metric]))
            for trade in data.get("trades", [])
            if trade.get("closed", True) and trade.get(metric) is not None
        ]
        values = [value for _, value in observations]
        expectancy = sum(values) / len(values) if values else 0.0
        variance = (
            sum((value - expectancy) ** 2 for value in values) / (len(values) - 1)
            if len(values) > 1 else 0.0
        )
        monthly: dict[str, float] = defaultdict(float)
        for day, value in observations:
            monthly[day[:7]] += value
        rows.append({
            "symbol": data["symbol"],
            "period": f"{data['start']}..{data['end']}",
            "mode": mode,
            "experiment": meta.get("experiment", "legacy"),
            "fingerprint": meta.get("config_fingerprint", "unknown"),
            "trades": len(values),
            "win_rate": data.get("underlying_win_rate", data.get("win_rate", 0.0)),
            "points": round(sum(values), 4),
            "avg_win": data.get("avg_underlying_win", data.get("avg_win", 0.0)),
            "avg_loss": data.get("avg_underlying_loss", data.get("avg_loss", 0.0)),
            "expectancy": expectancy,
            "standard_error": math.sqrt(variance / len(values)) if values else 0.0,
            "positive_months": sum(value > 0 for value in monthly.values()),
            "active_months": len(monthly),
            "max_drawdown": data.get("max_underlying_drawdown", data.get("max_drawdown", 0.0)),
        })
    # Bonferroni controls the chance of blessing any one of the compared
    # variants merely because many configurations were tried.
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * len(rows))) if rows else 1.96
    for row in rows:
        margin = z * row["standard_error"]
        row["familywise_ci_low"] = row["expectancy"] - margin
        row["familywise_ci_high"] = row["expectancy"] + margin
        row["candidate_supported"] = (
            row["trades"] >= 50
            and row["active_months"] >= 3
            and row["familywise_ci_low"] > 0
        )
    return rows


def print_comparison(paths: list[Path]) -> None:
    rows = comparison_rows(paths)
    print(
        "symbol  period                  experiment          trades  total   expectancy  "
        "familywise CI       +months  supported  config"
    )
    for row in rows:
        print(
            f"{row['symbol']:<7} {row['period']:<23} {row['experiment']:<19} "
            f"{row['trades']:>6} {row['points']:>7.2f} {row['expectancy']:>11.3f} "
            f"{row['familywise_ci_low']:>+7.3f}..{row['familywise_ci_high']:<+7.3f} "
            f"{row['positive_months']:>2}/{row['active_months']:<2}    "
            f"{'YES' if row['candidate_supported'] else 'no ':>3}      {row['fingerprint']}"
        )
    if rows:
        print(
            "Familywise CI uses a Bonferroni-adjusted 95% confidence level across "
            f"all {len(rows)} supplied variants. Results remain exploratory until a frozen holdout passes."
        )
