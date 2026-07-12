from datetime import date

from conftest import ts

from tajator.backtest.ledger import build_report
from tajator.broker.base import Fill
from tajator.models import SelectedContract

DAY1 = date(2026, 7, 6)
DAY2 = date(2026, 7, 7)
CONTRACT = SelectedContract(symbol="SPY", expiry="20260710", strike=500.0, right="C")


def fill(qty: float, premium: float, minute: int) -> Fill:
    return Fill(premium=premium, qty=qty, ts=ts(10, minute))


def test_winning_multi_leg_trade():
    fills_by_day = {
        DAY1: [
            ("BUY", CONTRACT, fill(4, 2.00, 0)),
            ("SELL", CONTRACT, fill(1, 2.50, 5)),
            ("SELL", CONTRACT, fill(1, 2.20, 10)),
            ("SELL", CONTRACT, fill(1, 1.90, 15)),
            ("SELL", CONTRACT, fill(1, 2.10, 20)),
        ]
    }
    report = build_report("SPY", DAY1, DAY1, fills_by_day)
    assert report.total_trades == 1
    assert report.wins == 1 and report.losses == 0
    assert report.trades[0].closed
    assert report.trades[0].pnl == 70.0
    assert report.total_pnl == 70.0
    assert report.win_rate == 1.0


def test_losing_trade_and_aggregate_stats_across_days():
    fills_by_day = {
        DAY1: [
            ("BUY", CONTRACT, fill(4, 2.00, 0)),
            ("SELL", CONTRACT, fill(1, 2.50, 5)),
            ("SELL", CONTRACT, fill(1, 2.20, 10)),
            ("SELL", CONTRACT, fill(1, 1.90, 15)),
            ("SELL", CONTRACT, fill(1, 2.10, 20)),
        ],
        DAY2: [
            ("BUY", CONTRACT, fill(2, 3.00, 0)),
            ("SELL", CONTRACT, fill(2, 2.50, 5)),
        ],
    }
    report = build_report("SPY", DAY1, DAY2, fills_by_day)
    assert report.total_trades == 2
    assert report.wins == 1 and report.losses == 1
    assert report.win_rate == 0.5
    assert report.total_pnl == -30.0
    assert report.avg_win == 70.0
    assert report.avg_loss == -100.0
    assert report.daily_pnl[DAY1] == 70.0
    assert report.daily_pnl[DAY2] == -100.0
    assert report.equity_curve == [(DAY1, 70.0), (DAY2, -30.0)]
    assert report.max_drawdown == 100.0


def test_unclosed_position_excluded_from_stats_but_kept_in_ledger():
    fills_by_day = {
        DAY1: [
            ("BUY", CONTRACT, fill(4, 2.00, 0)),
            ("SELL", CONTRACT, fill(1, 2.50, 5)),  # only 1 of 4 exited before day-end
        ]
    }
    report = build_report("SPY", DAY1, DAY1, fills_by_day)
    assert report.total_trades == 0, "an open-at-EOD position isn't a realized trade"
    assert len(report.trades) == 1
    assert not report.trades[0].closed
    assert report.daily_pnl[DAY1] == 0.0


def test_fees_and_trade_context_are_reported():
    buy = fill(2, 2.00, 0)
    buy.fee = 1.30
    buy.equity_price = 500.0
    buy.stop_price = 499.6
    sell = fill(2, 2.50, 5)
    sell.fee = 1.30
    sell.equity_price = 500.8
    sell.exit_reason = "first target"
    report = build_report(
        "SPY", DAY1, DAY1, {DAY1: [("BUY", CONTRACT, buy), ("SELL", CONTRACT, sell)]},
        metadata={"execution_model": {"modeled": True}},
    )
    trade = report.trades[0]
    assert trade.gross_pnl == 100.0
    assert trade.fees == 2.6
    assert trade.pnl == 97.4
    assert trade.return_on_premium == 0.2435
    assert trade.planned_equity_risk == 0.4
    assert trade.favorable_equity_move == 0.8
    assert trade.adverse_equity_move == 0.0
    assert trade.exit_reason == "first target"
    assert trade.underlying_points == 0.8
    assert report.gross_pnl == 100.0
    assert report.total_fees == 2.6
    assert report.metadata["execution_model"]["modeled"] is True
    assert report.total_underlying_points == 0.8
    assert report.underlying_win_rate == 1.0
