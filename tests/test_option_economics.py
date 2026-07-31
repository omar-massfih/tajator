"""Black-Scholes sanity and the cost-drag economics of expressing a move as options."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tajator.backtest.option_economics import bs_price, compare_expressions, realized_vol
from tajator.models import Bar

ET = ZoneInfo("America/New_York")


def test_bs_price_basic_sanity():
    atm = bs_price(100, 100, 0.1, 0.2, call=True)
    assert atm > 0
    # deep ITM call is worth at least its intrinsic value
    assert bs_price(120, 100, 0.1, 0.2, call=True) >= 20 - 1e-6
    # more volatility -> more expensive option (positive vega)
    assert bs_price(100, 100, 0.1, 0.3, call=True) > bs_price(100, 100, 0.1, 0.2, call=True)
    # put-call parity-ish: at the money, call ~ put for small rates
    assert abs(bs_price(100, 100, 0.1, 0.2, call=True) - bs_price(100, 100, 0.1, 0.2, call=False)) < 0.5


def test_zero_move_loses_on_options_but_not_stock():
    econ = compare_expressions(100.0, 0.0, 1, 0.25)
    assert econ["stock"]["net_pct"] == 0.0            # stock is flat on no move
    assert econ["atm_option"]["net_pct"] < 0          # option bleeds spread+theta
    assert econ["itm_option"]["net_pct"] < 0
    # you need a positive underlying move just to break even on the option
    assert econ["atm_option"]["breakeven_move_pct"] > 0


def test_small_edge_may_not_beat_option_cost():
    # a +0.1% expected move: stock is +0.1%, but the option breakeven is far higher
    econ = compare_expressions(100.0, 0.001, 1, 0.25)
    assert econ["stock"]["net_pct"] == 0.1
    assert econ["atm_option"]["breakeven_move_pct"] > 0.1  # 0.1% move doesn't clear the option cost


def test_realized_vol_positive_for_moving_series():
    bars, d = [], datetime(2020, 1, 1, tzinfo=ET)
    price = 100.0
    for i in range(25):
        price *= 1.01 if i % 2 == 0 else 0.99
        bars.append(Bar(ts=d, open=price, high=price, low=price, close=price, volume=1e6))
        d += timedelta(days=1)
    assert realized_vol(bars) > 0
