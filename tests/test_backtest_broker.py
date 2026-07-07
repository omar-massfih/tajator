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


def test_uses_real_option_bars_when_cached(tmp_path):
    cache_dir = tmp_path
    option_csv = cache_dir / "SPY" / "options" / "20260710_500C_2026-07-06.csv"
    option_csv.parent.mkdir(parents=True)
    option_csv.write_text(
        "ts,open,high,low,close,volume\n"
        "2026-07-06T09:30:00-04:00,3.00,3.20,2.80,3.00,10\n"
    )
    broker = BacktestBroker(_bars(), prev_day_high=505.0, prev_day_low=495.0, ib=None, cache_dir=cache_dir)
    premium = broker.get_option_premium(CONTRACT)
    assert premium == 3.00  # midpoint of the cached bar's high/low

    fill = broker.sell_option(CONTRACT, 1)
    assert fill.premium == 3.00


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
