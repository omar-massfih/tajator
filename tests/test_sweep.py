"""Smoke test for the ORB parameter sweep against a tiny synthetic cache."""

import csv
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tajator.backtest.sweep import DEFAULT_GRID, _variants, run_sweep
from tajator.config import Settings

ET = ZoneInfo("America/New_York")


def _breakout_day_csv(path, day):
    """Full RTH day: tight 500-501 opening range, then a break to 505 that holds."""
    rows = []
    prev = 500.0
    t = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    end = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET)
    while t <= end:
        m = (t.hour * 60 + t.minute) - (9 * 60 + 30)
        if m < 15:
            c = 501.0 if m % 2 == 1 else 500.0
        elif m <= 18:
            c = 500.0 + (m - 14)  # ramp 502..505
        else:
            c = 505.0
        rows.append((t.isoformat(), prev, max(prev, c), min(prev, c), c, 1000))
        prev = c
        t += timedelta(minutes=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        w.writerows(rows)


def test_variants_expand_the_grid():
    variants = list(_variants(DEFAULT_GRID))
    expected = 1
    for values in DEFAULT_GRID.values():
        expected *= len(values)
    assert len(variants) == expected
    assert all("exit_mode" in v and "runner_target_r" in v for v in variants)


def test_run_sweep_ranks_and_validates_offline(tmp_path):
    cache = tmp_path / "cache"
    # Seed a handful of identical breakout days across the dev and holdout windows.
    days = [date(2026, 6, d) for d in (1, 2, 3)] + [date(2026, 6, d) for d in (8, 9, 10)]
    for d in days:
        _breakout_day_csv(cache / "SPY" / f"{d.isoformat()}.csv", d)

    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    tiny_grid = {
        "orb_window_minutes": [15],
        "orb_breakout_buffer_pct": [0.0005],
        "orb_min_relative_volume": [0.0],
        "orb_min_breakout_range_atr": [0.0],
        "exit": [("scale", 3.0), ("let_run", 3.0)],
    }
    result = run_sweep(
        ["SPY"],
        date(2026, 6, 1), date(2026, 6, 3),
        date(2026, 6, 8), date(2026, 6, 10),
        settings, cache, min_trades=1, top_k=3, grid=tiny_grid, workers=1,
    )

    assert result["variants_tested"] == 2
    assert result["ranked"], "at least one variant should clear the min-trades floor"
    row = result["ranked"][0]
    assert {"overrides", "dev", "holdout", "survived"} <= row.keys()
    assert "expectancy" in row["dev"] and "expectancy" in row["holdout"]
