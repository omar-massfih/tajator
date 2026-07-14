from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tajator.backtest.audit import build_shadow_report
from tajator.broker.base import ChainParams
from tajator.broker.shadow import ShadowBroker
from tajator.config import Settings
from tajator.journal import Journal
from tajator.models import Bar, OptionQuote, SelectedContract

ET = ZoneInfo("America/New_York")


class MarketDataOnly:
    def __init__(self, quote, now):
        self.quote = quote
        self.current = now
        self.order_calls = 0

    def ensure_connected(self):
        return False

    def now(self):
        return self.current

    def get_bars(self, symbol, lookback_minutes=390):
        return []

    def get_prev_day_range(self, symbol):
        return 99.0, 98.0

    def get_daily_bars(self, symbol, lookback_days=90):
        return []

    def get_option_chain(self, symbol):
        return ChainParams(expirations=["20260717"], strikes=[100.0])

    def get_option_premium(self, contract):
        return self.quote.last

    def get_option_quote(self, contract):
        return self.quote

    def get_underlying_price(self, symbol):
        return 100.0

    @property
    def is_delayed_data(self):
        return self.quote.delayed

    def buy_option(self, *args, **kwargs):
        self.order_calls += 1
        raise AssertionError("shadow broker reached the market order method")

    def sell_option(self, *args, **kwargs):
        self.order_calls += 1
        raise AssertionError("shadow broker reached the market order method")


def test_shadow_fills_at_executable_sides_without_order_calls(tmp_path):
    now = datetime(2026, 7, 14, 10, 0, tzinfo=ET)
    market = MarketDataOnly(
        OptionQuote(bid=2.0, ask=2.2, last=2.1, ts=now), now
    )
    broker = ShadowBroker(
        market, Settings(_env_file=None), Journal(tmp_path)
    )
    contract = SelectedContract(
        symbol="MSFT", expiry="20260717", strike=500, right="C", con_id=123
    )

    buy = broker.buy_option(contract, 2)
    market.current += timedelta(minutes=5)
    market.quote = market.quote.model_copy(update={"ts": market.current})
    sell = broker.sell_option(contract, 2)

    assert buy.premium == 2.2
    assert sell.premium == 2.0
    assert buy.fee == sell.fee == 1.3
    assert market.order_calls == 0
    assert all(row[2].execution_quality.order_type == "SHADOW_BID_ASK" for row in broker.fills)
    text = next(tmp_path.glob("journal-*.jsonl")).read_text()
    assert '"no_order_placed": true' in text


def test_shadow_rejects_delayed_quote(tmp_path):
    now = datetime(2026, 7, 14, 10, 0, tzinfo=ET)
    quote = OptionQuote(bid=2.0, ask=2.2, last=2.1, ts=now, delayed=True)
    market = MarketDataOnly(quote, now)
    broker = ShadowBroker(market, Settings(_env_file=None), Journal(tmp_path))
    contract = SelectedContract(symbol="MSFT", expiry="20260717", strike=500, right="C")
    with pytest.raises(RuntimeError, match="no executable bid/ask"):
        broker.buy_option(contract, 1)
    assert market.order_calls == 0


def test_shadow_report_builds_net_options_ledger_and_coverage(tmp_path):
    now = datetime(2026, 7, 14, 10, 0, tzinfo=ET)
    market = MarketDataOnly(
        OptionQuote(bid=2.0, ask=2.2, last=2.1, ts=now), now
    )
    journal = Journal(tmp_path)
    journal.write(
        "shadow_started", ts=now, symbol="MSFT", deterministic=True,
        no_order_placed=True,
    )
    broker = ShadowBroker(market, Settings(_env_file=None), journal)
    contract = SelectedContract(symbol="MSFT", expiry="20260717", strike=500, right="C")
    broker.buy_option(contract, 2)
    market.current += timedelta(minutes=5)
    market.quote = market.quote.model_copy(update={"ts": market.current})
    broker.sell_option(contract, 2)
    journal.write(
        "shadow_session_covered", ts=now.replace(hour=15, minute=54),
        symbol="MSFT", regular_bars=385, no_order_placed=True,
    )

    report = build_shadow_report(tmp_path, symbol="msft")

    assert report.total_trades == 1
    assert report.gross_pnl == -40.0
    assert report.total_fees == 2.6
    assert report.total_pnl == -42.6
    assert report.metadata["validation_protocol"]["no_orders"] is True
    assert report.metadata["data_coverage"]["days_with_underlying_bars"] == 1


def test_shadow_marks_only_nearly_complete_regular_session(tmp_path):
    now = datetime(2026, 7, 14, 15, 54, tzinfo=ET)
    market = MarketDataOnly(
        OptionQuote(bid=2.0, ask=2.2, last=2.1, ts=now), now
    )
    start = now.replace(hour=9, minute=30)
    market.get_bars = lambda symbol, lookback_minutes=390: [
        Bar(ts=start + timedelta(minutes=i), open=100, high=101, low=99, close=100, volume=1)
        for i in range(385)
    ]
    broker = ShadowBroker(market, Settings(_env_file=None), Journal(tmp_path))

    broker.get_bars("MSFT")

    text = next(tmp_path.glob("journal-*.jsonl")).read_text()
    assert '"type": "shadow_session_covered"' in text
    assert market.order_calls == 0
