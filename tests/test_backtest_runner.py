"""Integration test: run_backtest over two cached days (no IB, no LLM connection).

Day 1 reuses the bundled scripted CSV (proven by test_graph_full.py to produce
exactly one clean call setup); day 2 is a flat no-setup day. Option pricing is
faked via a monkeypatched `ensure_option_bars` so the test doesn't need real IB
historical option data — that lookup/fallback behavior is covered separately by
test_backtest_broker.py.
"""

import csv
import shutil
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from conftest import walk

from tajator.backtest.runner import run_backtest
from tajator.config import Settings
from tajator.models import Bar

ET = ZoneInfo("America/New_York")
SAMPLE_CSV = Path(__file__).parent / "data" / "spy_sample_day.csv"
DAY_WITH_TRADE = date(2026, 6, 15)  # the sample CSV's own date
DAY_FLAT = date(2026, 6, 16)


def _write_bars_csv(path: Path, bars: list[Bar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b.ts.isoformat(), b.open, b.high, b.low, b.close, b.volume])


def _seed_cache(cache_dir: Path) -> None:
    (cache_dir / "SPY").mkdir(parents=True, exist_ok=True)
    shutil.copy(SAMPLE_CSV, cache_dir / "SPY" / f"{DAY_WITH_TRADE.isoformat()}.csv")
    flat_bars = walk(datetime(2026, 6, 16, 9, 30, tzinfo=ET), [500.0] * 390)
    _write_bars_csv(cache_dir / "SPY" / f"{DAY_FLAT.isoformat()}.csv", flat_bars)


def test_run_backtest_across_days(tmp_path, monkeypatch):
    _seed_cache(tmp_path)

    def fake_ensure_option_bars(ib, contract, day, cache_dir):
        return walk(datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET), [3.0] * 390)

    monkeypatch.setattr("tajator.broker.backtest.ensure_option_bars", fake_ensure_option_bars)

    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path / "logs")
    report = run_backtest(
        "SPY", DAY_WITH_TRADE, DAY_FLAT, settings, use_llm=False, ib=None, cache_dir=tmp_path
    )

    assert report.symbol == "SPY"
    assert len(report.trades) == 1, "only the scripted day should produce a trade"
    assert report.trades[0].day == DAY_WITH_TRADE
    assert report.trades[0].closed
    assert report.total_trades == 1
    assert len(report.equity_curve) == 1  # only the day with a closed trade contributes

    report_path = tmp_path / "logs" / "backtests" / f"SPY_{DAY_WITH_TRADE.isoformat()}_{DAY_FLAT.isoformat()}.json"
    assert report_path.exists()
