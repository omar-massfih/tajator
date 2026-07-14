#!/usr/bin/env python3
"""Diagnose whether one-bar follow-through separates Tajator entry outcomes.

This is a read-only research helper.  It joins persisted backtest trades to the
underlying one-minute cache and reports rules that could have been known on the
bar after the original rejection.  It intentionally does not claim delayed
entry PnL; a candidate still needs a full graph replay at its true timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float


def _bars(path: Path) -> list[Bar]:
    with path.open(newline="") as handle:
        return [
            Bar(
                ts=datetime.fromisoformat(row["ts"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
            for row in csv.DictReader(handle)
        ]


def _rules(direction: str, entry: Bar, next_bar: Bar) -> dict[str, bool]:
    favorable_close = (
        next_bar.close > entry.close if direction == "call" else next_bar.close < entry.close
    )
    closes_beyond_extreme = (
        next_bar.close > entry.high if direction == "call" else next_bar.close < entry.low
    )
    breaks_extreme = (
        next_bar.high > entry.high if direction == "call" else next_bar.low < entry.low
    )
    favorable_body = (
        next_bar.close > next_bar.open if direction == "call" else next_bar.close < next_bar.open
    )
    return {
        "next_close_favorable": favorable_close,
        "next_close_beyond_rejection": closes_beyond_extreme,
        "next_break_and_favorable_close": breaks_extreme and favorable_close,
        "next_favorable_body": favorable_body,
    }


def analyze(report_path: Path, cache_dir: Path) -> None:
    report = json.loads(report_path.read_text())
    rows: list[dict[str, object]] = []
    day_cache: dict[str, list[Bar]] = {}
    for trade in report["trades"]:
        day = trade["day"]
        bars = day_cache.setdefault(day, _bars(cache_dir / report["symbol"] / f"{day}.csv"))
        entry_ts = datetime.fromisoformat(trade["entry_ts"])
        index = next((i for i, bar in enumerate(bars) if bar.ts == entry_ts), None)
        if index is None or index + 1 >= len(bars):
            continue
        entry, next_bar = bars[index], bars[index + 1]
        rows.append(
            {
                "points": float(trade["underlying_points"]),
                "stopped": "mental stop" in trade["exit_reason"],
                **_rules(trade["direction"], entry, next_bar),
            }
        )

    print(f"{report_path.name}: joined {len(rows)}/{len(report['trades'])} trades")
    for rule in (
        "next_close_favorable",
        "next_close_beyond_rejection",
        "next_break_and_favorable_close",
        "next_favorable_body",
    ):
        for value in (True, False):
            selected = [row for row in rows if row[rule] is value]
            points = sum(float(row["points"]) for row in selected)
            stops = sum(bool(row["stopped"]) for row in selected)
            wins = sum(float(row["points"]) > 0 for row in selected)
            count = len(selected)
            print(
                f"  {rule}={str(value).lower():5} n={count:3} "
                f"total={points:+8.3f} avg={points / count:+.4f} "
                f"wins={wins / count:5.1%} initial_stops={stops / count:5.1%}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="REPORT=CACHE_DIR")
    args = parser.parse_args()
    for pair in args.pairs:
        report, separator, cache = pair.partition("=")
        if not separator:
            parser.error(f"expected REPORT=CACHE_DIR, got {pair!r}")
        analyze(Path(report), Path(cache))


if __name__ == "__main__":
    main()
