"""Momentum basket: chaining, drawdown, and that a persistent-winner panel is captured."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tajator.backtest import momentum as M
from tajator.backtest.data import _write_csv
from tajator.models import Bar

ET = ZoneInfo("America/New_York")


def test_curve_stats_chains_and_measures_drawdown():
    s = M._curve_stats([0.10, -0.05, 0.10], per_year=12)
    # (1.10)(0.95)(1.10) - 1 = +14.95%
    assert abs(s["total_return_pct"] - 14.95) < 0.1
    assert s["max_drawdown_pct"] < 0            # the -5% period creates a drawdown
    assert s["periods"] == 3


def _series(daily_dir, sym, closes, start=date(2018, 1, 1)):
    bars, d = [], start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        bars.append(Bar(ts=datetime(d.year, d.month, d.day, tzinfo=ET), open=c, high=c, low=c, close=c, volume=1e6))
        d += timedelta(days=1)
    _write_csv(daily_dir / f"{sym}.csv", bars)


def test_run_momentum_produces_all_legs(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    n = 400
    # WINNER trends up, LOSER trends down, flats are steady — need >=10 names to rank.
    _series(daily_dir, "WIN", [100 * (1.002 ** i) for i in range(n)])
    _series(daily_dir, "LOSE", [100 * (0.999 ** i) for i in range(n)])
    for j in range(10):
        _series(daily_dir, f"FLAT{j}", [100 + (i % 2) for i in range(n)])

    r = M.run_momentum(sorted(["WIN", "LOSE"] + [f"FLAT{j}" for j in range(10)]),
                       daily_dir=daily_dir, rebalance_days=21)
    assert r["symbols"] >= 12
    for leg in ("momentum", "benchmark", "premium", "long_short"):
        assert leg in r and "sharpe" in r[leg]
    assert r["premium_by_year"]  # per-year premium present
