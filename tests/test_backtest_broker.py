import pytest
from conftest import ts, walk

from tajator.broker.backtest import BacktestBroker
from tajator.broker.base import ChainParams, Fill
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


def test_execution_model_is_adverse_on_both_sides_and_charges_fees(tmp_path):
    option_csv = tmp_path / "SPY" / "options" / "20260710_500C_2026-07-06.csv"
    option_csv.parent.mkdir(parents=True)
    option_csv.write_text(
        "ts,open,high,low,close,volume\n"
        "2026-07-06T09:31:00-04:00,2.00,2.00,2.00,2.00,10\n"
    )
    broker = BacktestBroker(
        _bars(), ib=None, cache_dir=tmp_path, half_spread_pct=0.01,
        slippage_cents=1.0, commission_per_contract=0.65, min_commission_per_order=1.0,
    )
    buy = broker.buy_option(CONTRACT, 2)
    sell = broker.sell_option(CONTRACT, 1)
    assert buy.premium == 2.03 and sell.premium == 1.97
    assert buy.fee == 1.30 and sell.fee == 1.0


def test_option_panel_reprices_adjacent_strikes_and_next_expiry_at_same_times(tmp_path):
    contracts = [
        ("20260710", 499, "C", 4.0, 4.5),
        ("20260710", 501, "C", 2.0, 2.2),
        ("20260717", 500, "C", 5.0, 5.3),
    ]
    for expiry, strike, right, entry, exit_ in contracts:
        path = tmp_path / "SPY" / "options" / f"{expiry}_{strike}{right}_2026-07-06.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "ts,open,high,low,close,volume\n"
            f"2026-07-06T09:31:00-04:00,{entry},{entry},{entry},{entry},10\n"
            f"2026-07-06T09:32:00-04:00,{exit_},{exit_},{exit_},{exit_},10\n"
        )
    broker = BacktestBroker(_bars(), ib=None, cache_dir=tmp_path)
    fills = [
        ("BUY", CONTRACT, Fill(premium=3.0, qty=1, ts=ts(9, 30))),
        ("SELL", CONTRACT, Fill(premium=3.2, qty=1, ts=ts(9, 31))),
    ]
    chain = ChainParams(
        expirations=["20260710", "20260717"], strikes=[499.0, 500.0, 501.0]
    )
    panel, missing = broker.reprice_option_panel(fills, chain)

    assert [fill.premium for _, _, fill in panel["itm_1_near"]] == [4.0, 4.5]
    assert [fill.premium for _, _, fill in panel["otm_1_near"]] == [2.0, 2.2]
    assert [fill.premium for _, _, fill in panel["atm_next_expiry"]] == [5.0, 5.3]
    assert all(not items for items in missing.values())
