import pytest
from conftest import ts, walk

from tajator.broker.backtest import BacktestBroker
from tajator.models import SelectedContract

CONTRACT = SelectedContract(symbol="SPY", expiry="20260710", strike=500.0, right="C")


def _bars():
    return walk(ts(9, 30), [500.0] * 5)


def test_fails_fast_when_no_real_option_data(tmp_path):
    broker = BacktestBroker(_bars(), prev_day_high=505.0, prev_day_low=495.0, ib=None, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="no historical option data"):
        broker.get_option_premium(CONTRACT)


def test_fills_at_next_bar_open_no_lookahead(tmp_path):
    cache_dir = tmp_path
    option_csv = cache_dir / "SPY" / "options" / "20260710_500C_2026-07-06.csv"
    option_csv.parent.mkdir(parents=True)
    option_csv.write_text(
        "ts,open,high,low,close,volume\n"
        "2026-07-06T09:30:00-04:00,3.00,3.20,2.80,3.10,10\n"
        "2026-07-06T09:31:00-04:00,3.15,3.30,3.00,3.25,10\n"
    )
    broker = BacktestBroker(_bars(), prev_day_high=505.0, prev_day_low=495.0, ib=None, cache_dir=cache_dir)
    # decision on the 09:30 close -> the order fills at the NEXT bar's open
    premium = broker.get_option_premium(CONTRACT)
    assert premium == 3.15

    fill = broker.sell_option(CONTRACT, 1)
    assert fill.premium == 3.15


def test_fills_at_decision_bar_close_at_end_of_series(tmp_path):
    """Last bar of the day (forced flatten): no next bar, use the decision bar's close."""
    cache_dir = tmp_path
    option_csv = cache_dir / "SPY" / "options" / "20260710_500C_2026-07-06.csv"
    option_csv.parent.mkdir(parents=True)
    option_csv.write_text(
        "ts,open,high,low,close,volume\n"
        "2026-07-06T09:30:00-04:00,3.00,3.20,2.80,3.10,10\n"
    )
    broker = BacktestBroker(_bars(), prev_day_high=505.0, prev_day_low=495.0, ib=None, cache_dir=cache_dir)
    assert broker.get_option_premium(CONTRACT) == 3.10


def test_stale_option_bars_count_as_no_data(tmp_path):
    """Nearest option bar is far from the decision time -> abort, don't fill off a stale print."""
    cache_dir = tmp_path
    option_csv = cache_dir / "SPY" / "options" / "20260710_500C_2026-07-06.csv"
    option_csv.parent.mkdir(parents=True)
    option_csv.write_text(
        "ts,open,high,low,close,volume\n"
        "2026-07-06T10:30:00-04:00,3.00,3.20,2.80,3.10,10\n"
    )
    broker = BacktestBroker(_bars(), prev_day_high=505.0, prev_day_low=495.0, ib=None, cache_dir=cache_dir)
    with pytest.raises(RuntimeError, match="no historical option data"):
        broker.get_option_premium(CONTRACT)


def test_cached_data_miss_is_remembered_and_still_fails(tmp_path):
    """ensure_option_bars writes an empty cache file on a miss so reruns don't re-hit IB;
    BacktestBroker must still fail fast on that cached miss."""
    cache_dir = tmp_path
    option_csv = cache_dir / "SPY" / "options" / "20260710_500C_2026-07-06.csv"
    option_csv.parent.mkdir(parents=True)
    option_csv.write_text("ts,open,high,low,close,volume\n")
    broker = BacktestBroker(_bars(), prev_day_high=505.0, prev_day_low=495.0, ib=None, cache_dir=cache_dir)
    with pytest.raises(RuntimeError, match="no historical option data"):
        broker.get_option_premium(CONTRACT)
