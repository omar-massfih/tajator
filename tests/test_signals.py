"""Signal engine: causality (no lookahead), correct returns, and validation splits."""

from datetime import date, datetime, timedelta
from statistics import NormalDist
from zoneinfo import ZoneInfo

from tajator.backtest import signals as S
from tajator.models import Bar

ET = ZoneInfo("America/New_York")


def daily(closes, opens=None, start=date(2020, 1, 1)):
    """Synthetic daily bars stepping one weekday at a time."""
    bars, d = [], start
    for i, c in enumerate(closes):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        o = opens[i] if opens else c
        bars.append(Bar(ts=datetime(d.year, d.month, d.day, tzinfo=ET),
                        open=o, high=max(o, c), low=min(o, c), close=c, volume=1e6))
        d += timedelta(days=1)
    return bars


def test_overnight_uses_close_to_next_open():
    bars = daily([100, 100], opens=[100, 101])  # day2 opens at 101
    trades = S.sig_overnight(bars, "X", cost_bps=0.0)
    assert len(trades) == 1
    assert abs(trades[0].ret - (101 / 100 - 1)) < 1e-9


def test_reversal_1d_only_enters_after_a_down_day():
    # closes: up, DOWN(99), up(101). Only i=1 (99<100) qualifies.
    bars = daily([100, 99, 101])
    trades = S.sig_reversal_1d(bars, "X", cost_bps=0.0)
    assert len(trades) == 1
    assert trades[0].day == bars[1].ts.date()
    assert abs(trades[0].ret - (101 / 99 - 1)) < 1e-9


def test_momentum_ma20_needs_enough_history_and_is_above_ma():
    bars = daily([100] * 19 + [110, 111])  # MA20 only computable from index 19
    trades = S.sig_momentum_ma20(bars, "X", cost_bps=0.0)
    # index 19: close 110 > MA of prior (mostly 100) -> a trade to index 20
    assert trades and all(t.day >= bars[19].ts.date() for t in trades)


def test_cost_is_deducted():
    bars = daily([100, 100], opens=[100, 100])  # flat overnight
    trades = S.sig_overnight(bars, "X", cost_bps=5.0)
    assert abs(trades[0].ret - (-5.0 / 1e4)) < 1e-12


def test_clustered_stats_treats_a_day_as_one_observation():
    z = 1.96
    trades = [
        S.Trade(date(2020, 1, 1), "A", 0.01), S.Trade(date(2020, 1, 1), "B", 0.03),
        S.Trade(date(2020, 1, 2), "A", 0.00),
    ]
    st = S.clustered_stats(trades, z=z)
    assert st["trades"] == 3 and st["days"] == 2
    # daily means [0.02, 0.00] -> mean 0.01 = 100 bps
    assert abs(st["mean_bps"] - 100.0) < 1e-6


def test_run_edge_search_splits_dev_holdout_and_cross_symbol(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    # Two symbols, 60 weekdays spanning the dev/holdout boundary.
    for sym in ("AAA", "BBB"):
        bars = daily([100 + (i % 3) for i in range(60)], start=date(2023, 11, 1))
        from tajator.backtest.data import _write_csv
        _write_csv(daily_dir / f"{sym}.csv", bars)

    result = S.run_edge_search(
        ["AAA", "BBB"], daily_dir=daily_dir, intraday_dir=tmp_path,
        dev_end=date(2023, 12, 31), holdout_start=date(2024, 1, 1), horizon="daily",
    )
    assert result["signals_tested"] == len(S.DAILY_SIGNALS)
    assert result["dev_symbols"] == ["AAA"] and result["holdout_symbols"] == ["BBB"]
    row = result["results"][0]
    assert {"dev", "holdout", "cross_symbol", "survives"} <= row.keys()
    # Bonferroni z must exceed the plain 95% z given >1 signal.
    assert result["bonferroni_z"] > NormalDist().inv_cdf(0.975)
