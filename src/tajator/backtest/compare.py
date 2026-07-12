"""Compact comparison table for experiment-safe backtest JSON reports."""

from __future__ import annotations

import json
from pathlib import Path


def comparison_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        meta = data.get("metadata", {})
        rows.append({
            "symbol": data["symbol"],
            "period": f"{data['start']}..{data['end']}",
            "experiment": meta.get("experiment", "legacy"),
            "fingerprint": meta.get("config_fingerprint", "unknown"),
            "trades": data.get("total_trades", 0),
            "win_rate": data.get("underlying_win_rate", data.get("win_rate", 0.0)),
            "points": data.get("total_underlying_points", 0.0),
            "avg_win": data.get("avg_underlying_win", data.get("avg_win", 0.0)),
            "avg_loss": data.get("avg_underlying_loss", data.get("avg_loss", 0.0)),
        })
    return rows


def print_comparison(paths: list[Path]) -> None:
    rows = comparison_rows(paths)
    print("symbol  period                  experiment          trades  win%    points  avg-win  avg-loss  config")
    for row in rows:
        print(
            f"{row['symbol']:<7} {row['period']:<23} {row['experiment']:<19} "
            f"{row['trades']:>6} {row['win_rate']:>6.1%} {row['points']:>8.2f} "
            f"{row['avg_win']:>8.2f} {row['avg_loss']:>9.2f}  {row['fingerprint']}"
        )
