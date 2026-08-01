"""Tests for the volatility-risk-premium edge module and the Index contract branch."""

import csv
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tajator.backtest.vol_edge import (
    crash_pnl,
    filtered_short_vol,
    perf_stats,
    run_vol_edge,
)

ET = ZoneInfo("America/New_York")


def _write(path, closes, start=datetime(2018, 1, 2, tzinfo=ET)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for i, c in enumerate(closes):
            ts = start + timedelta(days=i)
            w.writerow([ts.isoformat(), c, c, c, c, 0.0])


# --------------------------------------------------------------------------- #
# perf_stats
# --------------------------------------------------------------------------- #

def test_perf_stats_flat_series_is_zero():
    s = perf_stats([0.0] * 30)
    assert s.ann_return == 0.0 and s.sharpe == 0.0 and s.max_drawdown == 0.0


def test_perf_stats_captures_drawdown_and_worst_day():
    returns = [0.01] * 25 + [-0.40] + [0.01] * 5
    s = perf_stats(returns)
    assert s.worst_day == -0.40
    assert s.max_drawdown <= -0.39  # the -40% day dominates the drawdown


# --------------------------------------------------------------------------- #
# filtered_short_vol — the term-structure gate
# --------------------------------------------------------------------------- #

def test_filter_goes_flat_after_backwardation(tmp_path):
    # SVXY: +10%, +10%, -17.35%, +10%
    _write(tmp_path / "SVXY.csv", [100, 110, 121, 100, 110])
    # contango (VIX<VIX3M) on d0,d1,d2; backwardation (VIX>VIX3M) on d3.
    _write(tmp_path / "VIX.csv", [15, 15, 15, 25, 15])
    _write(tmp_path / "VIX3M.csv", [18, 18, 18, 18, 18])

    rows = filtered_short_vol(tmp_path, cost_bps=0.0)
    by_date = {d: (net, always) for d, net, always in rows}
    dates = sorted(by_date)

    # d1 held (prior day was contango) -> earns the +10%
    assert abs(by_date[dates[0]][0] - 0.10) < 1e-9
    # last day is FLAT: prior day (d3) was backwardation -> position 0 -> net 0
    assert by_date[dates[-1]][0] == 0.0
    # always-on column always carries the raw SVXY return
    assert abs(by_date[dates[-1]][1] - 0.10) < 1e-9


def test_flip_incurs_cost(tmp_path):
    _write(tmp_path / "SVXY.csv", [100, 110, 121])
    _write(tmp_path / "VIX.csv", [25, 15, 15])      # d0 backwardation, then contango
    _write(tmp_path / "VIX3M.csv", [18, 18, 18])
    rows = filtered_short_vol(tmp_path, cost_bps=10.0)
    # d1: prior day d0 was backwardation -> flat -> net 0 (no position, no flip from 0)
    # d2: prior day d1 contango -> enter (0->1 flip) -> return 0.10 minus 10bps
    net_d2 = rows[1][1]
    assert abs(net_d2 - (0.10 - 0.001)) < 1e-9


# --------------------------------------------------------------------------- #
# crash_pnl + run_vol_edge integration
# --------------------------------------------------------------------------- #

def test_crash_pnl_compounds_within_window():
    returns = [
        (date(2018, 1, 26), -0.10),
        (date(2018, 2, 5), -0.20),
        (date(2019, 1, 1), 0.50),  # outside every window
    ]
    out = crash_pnl(returns)
    # (1-.1)(1-.2)-1 = -0.28
    assert abs(out["Feb-2018 Volmageddon"] - (-0.28)) < 1e-9


def test_run_vol_edge_smoke(tmp_path):
    n = 60
    _write(tmp_path / "SVXY.csv", [100 * (1.001 ** i) for i in range(n)])
    _write(tmp_path / "VIX.csv", [15] * n)
    _write(tmp_path / "VIX3M.csv", [18] * n)  # always contango -> always in market
    result = run_vol_edge(tmp_path, spy_dir=None)
    assert result["in_market_frac"] == 1.0
    assert result["filtered"]["full"].n == n - 1
    assert set(result["crash_filtered"]) == {
        "Feb-2018 Volmageddon", "Mar-2020 COVID", "2022 bear"
    }


# --------------------------------------------------------------------------- #
# Index contract branch in the live broker
# --------------------------------------------------------------------------- #

def test_underlying_builds_index_for_vix(monkeypatch):
    from ib_async import Index, Stock

    from tajator.broker.ib import IBBroker
    from tajator.config import Settings

    broker = IBBroker(Settings(_env_file=None))
    monkeypatch.setattr(broker.ib, "qualifyContracts", lambda c: [c])

    vix = broker._underlying("VIX")
    assert isinstance(vix, Index)
    assert vix.exchange == "CBOE"

    aapl = broker._underlying("AAPL")
    assert isinstance(aapl, Stock)
    assert aapl.exchange == "SMART"
