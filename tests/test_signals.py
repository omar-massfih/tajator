"""Signal engine: causality (no lookahead), stop-loss realization, cross-sectional
ranking, and the validation splits."""

from datetime import date, datetime, timedelta
from statistics import NormalDist
from zoneinfo import ZoneInfo

from tajator.backtest import signals as S
from tajator.backtest.data import _write_csv
from tajator.models import Bar

ET = ZoneInfo("America/New_York")


def daily(closes, opens=None, highs=None, lows=None, start=date(2020, 1, 1)):
    bars, d = [], start
    for i, c in enumerate(closes):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        o = opens[i] if opens else c
        h = highs[i] if highs else max(o, c)
        lo = lows[i] if lows else min(o, c)
        bars.append(Bar(ts=datetime(d.year, d.month, d.day, tzinfo=ET), open=o, high=h, low=lo, close=c, volume=1e6))
        d += timedelta(days=1)
    return bars


def bar(high, low, close=None):
    return Bar(ts=datetime(2020, 1, 1, tzinfo=ET), open=high, high=high, low=low, close=close or low, volume=1e6)


# ---- signals produce causal Positions --------------------------------------

def test_overnight_position_is_close_to_next_open():
    bars = daily([100, 100], opens=[100, 101])
    pos = S.sig_overnight(bars, "X")
    assert len(pos) == 1 and pos[0].entry == 100 and pos[0].horizon_exit == 101 and pos[0].hold == ()


def test_reversal_1d_only_after_a_down_day():
    bars = daily([100, 99, 101])
    pos = S.sig_reversal_1d(bars, "X")
    assert len(pos) == 1 and pos[0].entry_day == bars[1].ts.date() and pos[0].entry == 99


# ---- stop-loss realization -------------------------------------------------

def test_fixed_stop_exits_at_the_stop_on_a_breach():
    p = S.Position(date(2020, 1, 1), "X", 1, 100.0, 105.0, (bar(high=101, low=96),))
    t = S.realize(p, "fixed", stop_pct=0.03, trail_pct=0.03, cost_bps=0.0)
    assert abs(t.ret - (97 / 100 - 1)) < 1e-9   # stopped at 97, not the 105 horizon


def test_no_stop_takes_the_horizon_exit():
    p = S.Position(date(2020, 1, 1), "X", 1, 100.0, 105.0, (bar(high=101, low=96),))
    t = S.realize(p, "none", 0.03, 0.03, 0.0)
    assert abs(t.ret - 0.05) < 1e-9


def test_trailing_stop_locks_a_profit():
    # runs to 110 then pulls back through the 3% trail (110*0.97=106.7)
    hold = (bar(high=110, low=105), bar(high=107, low=106))
    p = S.Position(date(2020, 1, 1), "X", 1, 100.0, 100.0, hold)  # horizon exit would be flat
    t = S.realize(p, "trailing", 0.03, 0.03, 0.0)
    assert abs(t.ret - (106.7 / 100 - 1)) < 1e-6   # profit locked, not round-tripped to 0


# ---- cross-sectional -------------------------------------------------------

def test_xs_reversal_longs_the_biggest_loser():
    # day index 1: A fell 5%, B rose, C flat -> A is the sole bottom-decile loser.
    series = {
        "A": daily([100, 95, 96]),
        "B": daily([100, 105, 106]),
        "C": daily([100, 100, 101]),
    }
    pos = S.xs_reversal_1d(series)
    entries = {(p.symbol, p.entry_day) for p in pos}
    assert ("A", series["A"][1].ts.date()) in entries
    assert not any(p.symbol in ("B", "C") and p.entry_day == series["A"][1].ts.date() for p in pos)


# ---- stats + engine --------------------------------------------------------

def test_non_overlapping_drops_overlapping_holds_per_symbol():
    b = daily([100, 101, 102, 103, 104])  # 5 sessions
    # three 3-day-hold positions entered on consecutive days overlap heavily
    def pos(i):
        return S.Position(b[i].ts.date(), "X", 1, b[i].close, b[i + 2].close, tuple(b[i + 1:i + 3]))
    kept = S._non_overlapping([pos(0), pos(1), pos(2)])
    assert len(kept) == 2  # entry day0 (exit day2) then next allowed is day2, not day1
    assert [p.entry_day for p in kept] == [b[0].ts.date(), b[2].ts.date()]


def test_clustered_stats_treats_a_day_as_one_observation():
    trades = [S.Trade(date(2020, 1, 1), "A", 0.01), S.Trade(date(2020, 1, 1), "B", 0.03),
              S.Trade(date(2020, 1, 2), "A", 0.0)]
    st = S.clustered_stats(trades, z=1.96)
    assert st["trades"] == 3 and st["days"] == 2 and abs(st["mean_bps"] - 100.0) < 1e-6


def test_run_edge_search_splits_and_picks_a_stop_mode(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    for sym in ("AAA", "BBB"):
        _write_csv(daily_dir / f"{sym}.csv", daily([100 + (i % 3) for i in range(80)], start=date(2023, 10, 1)))
    result = S.run_edge_search(
        ["AAA", "BBB"], daily_dir=daily_dir, intraday_dir=tmp_path,
        dev_end=date(2023, 12, 31), holdout_start=date(2024, 1, 1), horizon="daily",
    )
    assert result["signals_tested"] == len(S.DAILY_SIGNALS) + len(S.XS_SIGNALS)
    assert result["dev_symbols"] == ["AAA"] and result["holdout_symbols"] == ["BBB"]
    row = result["results"][0]
    assert row["best_mode"] in S.STOP_MODES
    assert set(row["per_mode"]) == set(S.STOP_MODES)
    assert result["bonferroni_z"] > NormalDist().inv_cdf(0.975)
