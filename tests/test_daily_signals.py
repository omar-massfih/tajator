from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from tajator.backtest import daily_signals as ds
from tajator.backtest.data import _write_csv
from tajator.models import Bar

ET = ZoneInfo("America/New_York")


def daily_bars(count=180, *, start=date(2017, 1, 2), step=0.1):
    bars = []
    value = 100.0
    day = start
    while len(bars) < count:
        if day.weekday() < 5:
            bars.append(
                Bar(
                    ts=datetime.combine(day, datetime.min.time(), tzinfo=ET),
                    open=value,
                    high=value + 0.2,
                    low=value - 0.2,
                    close=value + step,
                    volume=1000,
                )
            )
            value += step
        day += timedelta(days=1)
    return bars


def test_donchian_trades_enter_next_session_and_do_not_overlap():
    bars = daily_bars(step=1.0)
    trades = ds.generate_trades(
        "TEST", bars, "donchian20_breakout", bars[20].ts.date(), bars[-1].ts.date()
    )

    assert trades
    first = trades[0]
    assert first.direction == "call"
    assert first.entry_day > first.signal_day
    assert first.exit_day == bars[30].ts.date()
    assert first.net_return < first.gross_return
    assert first.stress_net_return < first.net_return
    assert all(right.entry_day > left.exit_day for left, right in zip(trades, trades[1:]))


def test_simple_rsi2_extremes_are_causal():
    bars = daily_bars(count=5, step=0.0)
    bars[1] = bars[1].model_copy(update={"close": 99.0})
    bars[2] = bars[2].model_copy(update={"close": 98.0})
    assert ds._rsi2(bars, 2) == 0.0
    assert ds._direction_rsi2_reversal(bars, 2) == "call"

    bars[3] = bars[3].model_copy(update={"close": 100.0})
    bars[4] = bars[4].model_copy(update={"close": 102.0})
    assert ds._rsi2(bars, 4) == 100.0
    assert ds._direction_rsi2_reversal(bars, 4) == "put"


def test_daily_loader_refuses_unverified_split_discontinuity(tmp_path):
    bars = daily_bars(count=3)
    bars[2] = bars[2].model_copy(update={"close": bars[1].close * 2})
    path = ds.daily_cache_path(tmp_path, "TEST")
    _write_csv(path, bars)

    with pytest.raises(ValueError, match="discontinuity"):
        ds.load_daily_history(tmp_path, "TEST")


def test_daily_fetch_writes_only_requested_tws_dates(tmp_path):
    items = [
        SimpleNamespace(
            date=date(2016, 12, 30), open=90, high=91, low=89, close=90, volume=1
        ),
        SimpleNamespace(
            date=date(2017, 1, 3), open=100, high=101, low=99, close=100.5, volume=10
        ),
    ]

    class FakeIB:
        def __init__(self):
            self.ib = self

        def _underlying(self, symbol):
            return symbol

        def reqHistoricalData(self, *args, **kwargs):
            return items

    path = ds.fetch_daily_history(
        FakeIB(), "TEST", date(2017, 1, 1), date(2017, 12, 31), tmp_path
    )
    loaded = ds.load_daily_history(tmp_path, "TEST")
    assert path.exists()
    assert len(loaded) == 1
    assert loaded[0].ts.date() == date(2017, 1, 3)


def test_no_daily_development_candidate_keeps_later_stages_closed(monkeypatch, tmp_path):
    flat = daily_bars(count=3000, step=0.0)
    monkeypatch.setattr(ds, "load_daily_history", lambda cache, symbol: flat)
    calls = []
    original = ds._collect

    def recording_collect(histories, symbols, strategy, start, end):
        calls.append((symbols, start, end))
        return original(histories, symbols, strategy, start, end)

    monkeypatch.setattr(ds, "_collect", recording_collect)
    report = ds.run_daily_tournament(tmp_path)

    assert report["selected"] is None
    assert report["validation_opened"] is False
    assert report["validation"] is None
    assert report["final_opened"] is False
    assert report["final"] is None
    assert all(symbols == ds.DEVELOPMENT_SYMBOLS for symbols, _, _ in calls)


def test_failed_aapl_primary_does_not_load_msft(monkeypatch, tmp_path):
    flat = daily_bars(count=3000, step=0.0)
    loaded = []

    def load(cache, symbol):
        loaded.append(symbol)
        return flat

    monkeypatch.setattr(ds, "load_daily_history", load)
    report = ds.run_aapl_focused_holdout(tmp_path)

    assert loaded == ["AAPL"]
    assert report["primary_passed"] is False
    assert report["replication_opened"] is False
    assert report["replication"] is None
    assert report["options_stage_eligible"] is False
