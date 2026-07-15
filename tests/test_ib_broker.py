"""IBBroker safety mechanics that don't need a live Gateway."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from tajator.broker.base import OrderFailed
from tajator.broker.ib import IBBroker
from tajator.config import Settings
from tajator.journal import Journal
from tajator.models import OptionQuote, ProtectiveStop, SelectedContract

ET = ZoneInfo("America/New_York")


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path,
        fill_grace_s=0,  # no post-cancel wait in tests
    )


def test_partial_fill_halt_is_internal_and_does_not_create_kill(settings):
    broker = IBBroker(settings)
    broker._halt_new_entries("partial fill: BUY 1/4 SPY 20260710 500C")
    assert "partial fill" in broker.entry_halt_reason
    assert not settings.kill_switch_file.exists()


def test_partial_fill_halt_notifies(settings):
    class RecordingNotifier:
        def __init__(self):
            self.statuses = []

        def notify_fill(self, symbol, action, position):
            pass

        def notify_status(self, text):
            self.statuses.append(text)

    notifier = RecordingNotifier()
    broker = IBBroker(settings, notifier=notifier)
    broker._halt_new_entries("partial fill: BUY 1/4 SPY 20260710 500C")
    assert notifier.statuses
    assert "halted new entries" in notifier.statuses[0]
    assert "partial fill" in notifier.statuses[0]
    assert "KILL file was not modified" in notifier.statuses[0]


def test_open_option_positions_reports_configured_symbols_only(settings):
    broker = IBBroker(settings)

    def fake_positions():
        return [
            SimpleNamespace(  # option in a configured symbol — must be reported
                contract=SimpleNamespace(secType="OPT", symbol="SPY", localSymbol="SPY 260710C00500000"),
                position=2.0, avgCost=210.0,
            ),
            SimpleNamespace(  # option in an unrelated symbol — ignored
                contract=SimpleNamespace(secType="OPT", symbol="TSLA", localSymbol="TSLA ..."),
                position=1.0, avgCost=500.0,
            ),
            SimpleNamespace(  # stock position — ignored
                contract=SimpleNamespace(secType="STK", symbol="SPY", localSymbol="SPY"),
                position=100.0, avgCost=500.0,
            ),
            SimpleNamespace(  # closed-out option (qty 0) — ignored
                contract=SimpleNamespace(secType="OPT", symbol="SPY", localSymbol="SPY ..."),
                position=0.0, avgCost=0.0,
            ),
        ]

    broker.ib = SimpleNamespace(positions=fake_positions)
    found = broker.open_option_positions(["SPY", "AAPL"])
    assert len(found) == 1
    assert "SPY 260710C00500000" in found[0]


def test_entry_market_snapshot_requests_option_and_stock_together(settings, monkeypatch):
    broker = IBBroker(settings)
    contract = SelectedContract(
        symbol="AAPL", expiry="20260715", strike=315.0, right="P"
    )
    option = object()
    stock = object()
    captured_at = datetime(2026, 7, 14, 13, 11, 7, tzinfo=ET)
    option_ticker = SimpleNamespace(
        bid=1.98,
        ask=2.04,
        last=2.00,
        time=captured_at - timedelta(seconds=30),
    )
    stock_ticker = SimpleNamespace(marketPrice=lambda: 314.705)
    requests = []

    def req_tickers(*contracts):
        requests.append(contracts)
        return [option_ticker, stock_ticker]

    broker.ib = SimpleNamespace(reqTickers=req_tickers)
    monkeypatch.setattr(broker, "_option", lambda selected: option)
    monkeypatch.setattr(broker, "_underlying", lambda symbol: stock)
    monkeypatch.setattr(broker, "now", lambda: captured_at)

    quote, underlying = broker.get_entry_market_snapshot(contract)

    assert requests == [(option, stock)]
    assert quote.bid == 1.98 and quote.ask == 2.04
    assert quote.ts == captured_at
    assert underlying == 314.705


def test_buy_uses_accepted_snapshot_without_another_market_request(settings, monkeypatch):
    trade = make_trade(status="Filled", status_filled=1, avg_price=2.04)
    broker = IBBroker(settings)
    broker.ib = FakeIB(trade, fill_effect=1)
    monkeypatch.setattr(broker, "_option", lambda contract: SimpleNamespace(conId=CON_ID))
    monkeypatch.setattr(
        broker, "get_entry_market_snapshot",
        lambda contract: pytest.fail("duplicate market snapshot requested"),
    )
    quote = OptionQuote(bid=1.98, ask=2.04, ts=broker.now())
    contract = SelectedContract(
        symbol="AAPL", expiry="20260715", strike=315.0, right="P"
    )

    fill = broker.buy_option_from_snapshot(contract, 1, quote, 314.705)

    assert fill.execution_quality.quote == quote
    assert fill.execution_quality.underlying_submit == 314.705
    assert fill.execution_quality.underlying_fill == 314.705


def test_sell_submits_without_waiting_for_market_snapshot(settings, monkeypatch):
    trade = make_trade(status="Filled", status_filled=1, avg_price=1.80)
    broker = IBBroker(settings)
    broker.ib = FakeIB(trade, qty_before=1, fill_effect=1)
    monkeypatch.setattr(broker, "_option", lambda contract: SimpleNamespace(conId=CON_ID))
    monkeypatch.setattr(
        broker, "get_entry_market_snapshot",
        lambda contract: pytest.fail("risk-reducing exit waited for market data"),
    )
    contract = SelectedContract(
        symbol="AAPL", expiry="20260715", strike=315.0, right="P"
    )

    fill = broker.sell_option(contract, 1)

    assert fill.qty == 1
    assert fill.execution_quality.quote is None


# -- _place reconciliation ------------------------------------------------------
#
# Regression for the 2026-07-08 incident: market orders timed out, were
# cancelled, and IB reported "Cancelled (filled 0/N)" — but the orders had
# actually filled. Trusting orderStatus.filled left untracked contracts in
# the account while the session retried a fresh order every minute.

CON_ID = 111


class FakeIB:
    """Just enough of ib_async.IB for _place: order placement returns a
    scripted trade, and the account position moves by `fill_effect`."""

    def __init__(self, trade, qty_before=0, fill_effect=0, positions_fail=False):
        self.trade = trade
        self.qty = qty_before
        self.fill_effect = fill_effect
        self.positions_fail = positions_fail
        self.placed = []

    def placeOrder(self, contract, order):
        self.placed.append((order.action, int(order.totalQuantity), order))
        self.qty += self.fill_effect if order.action == "BUY" else -self.fill_effect
        return self.trade

    def cancelOrder(self, order):
        pass

    def waitOnUpdate(self, timeout=1.0):
        pass

    def positions(self):
        if self.positions_fail:
            raise RuntimeError("positions unavailable")
        return [SimpleNamespace(contract=SimpleNamespace(conId=CON_ID), position=self.qty)]

    def reqPositions(self):
        if self.positions_fail:
            raise RuntimeError("positions unavailable")


def make_trade(status="Cancelled", status_filled=0, avg_price=0.0, fills=(), order=None, done=True):
    return SimpleNamespace(
        orderStatus=SimpleNamespace(status=status, filled=status_filled, avgFillPrice=avg_price),
        fills=[SimpleNamespace(execution=SimpleNamespace(shares=s, price=p)) for s, p in fills],
        isDone=lambda: done,
        order=order or SimpleNamespace(orderId=0, permId=0, orderRef=""),
    )


def place(settings, monkeypatch, trade, side="BUY", qty=1, **fake_ib_kwargs):
    broker = IBBroker(settings)
    broker.ib = FakeIB(trade, **fake_ib_kwargs)
    contract = SelectedContract(symbol="NVDA", expiry="20260710", strike=200.0, right="P")
    monkeypatch.setattr(broker, "_option", lambda c: SimpleNamespace(conId=CON_ID))
    return broker._place(contract, side, qty, snapshot_supplied=True)


def test_cancelled_but_actually_filled_adopts_the_fill(settings, monkeypatch):
    """A full fill confirmed by executions and account position is safe even if
    IB briefly reported Cancelled/0 filled during preset normalization."""
    trade = make_trade(fills=[(1, 2.97)])
    fill = place(settings, monkeypatch, trade, qty=1, fill_effect=1)
    assert fill.qty == 1
    assert fill.premium == pytest.approx(2.97)
    assert not settings.kill_switch_file.exists()


def test_partial_fill_returns_partial_and_halts(settings, monkeypatch):
    trade = make_trade(fills=[(2, 2.20)])
    fill = place(settings, monkeypatch, trade, qty=4, fill_effect=2)
    assert fill.qty == 2
    assert fill.premium == pytest.approx(2.20)
    assert not settings.kill_switch_file.exists()


def test_confirmed_zero_fill_raises_and_halts_buys(settings, monkeypatch):
    with pytest.raises(OrderFailed) as exc_info:
        place(settings, monkeypatch, make_trade(), qty=2)
    assert exc_info.value.filled == 0
    assert exc_info.value.suspect is False
    assert not settings.kill_switch_file.exists()


def test_confirmed_zero_fill_sell_raises_without_halting(settings, monkeypatch):
    """An unfilled exit must keep retrying next tick — no kill switch needed."""
    with pytest.raises(OrderFailed) as exc_info:
        place(settings, monkeypatch, make_trade(), side="SELL", qty=2)
    assert exc_info.value.filled == 0
    assert not settings.kill_switch_file.exists()


def test_confirmed_full_sell_fill_does_not_halt_entries(settings, monkeypatch):
    """A fully filled exit removes risk; IB's cancel/fill weirdness should be
    journaled, but it should not leave a kill switch behind."""
    trade = make_trade(status="Cancelled", fills=[(1, 1.48), (1, 1.38)])
    fill = place(settings, monkeypatch, trade, side="SELL", qty=2, qty_before=2, fill_effect=2)
    assert fill.qty == 2
    assert fill.premium == pytest.approx(1.43)
    assert not settings.kill_switch_file.exists()


def test_position_moved_without_execution_reports_is_suspect(settings, monkeypatch):
    """Account gained a contract but no executions/price arrived: cannot build a
    Fill, must raise suspect and halt so the operator reconciles."""
    with pytest.raises(OrderFailed) as exc_info:
        place(settings, monkeypatch, make_trade(), qty=1, fill_effect=1)
    assert exc_info.value.suspect is True
    assert exc_info.value.filled == 1
    assert "unconfirmed" in str(exc_info.value).lower()
    assert not settings.kill_switch_file.exists()


def test_full_fill_per_reports_with_positions_down_still_adopts(settings, monkeypatch):
    """Execution reports say fully filled but the position cross-check is
    unavailable: adopt what the reports say (tracked beats orphaned), halted."""
    trade = make_trade(fills=[(1, 3.10)])
    fill = place(settings, monkeypatch, trade, qty=1, positions_fail=True)
    assert fill.qty == 1
    assert not settings.kill_switch_file.exists()


def test_clean_filled_path_untouched(settings, monkeypatch):
    trade = make_trade(status="Filled", status_filled=3, avg_price=1.50)
    fill = place(settings, monkeypatch, trade, qty=3, fill_effect=3)
    assert fill.qty == 3
    assert fill.premium == pytest.approx(1.50)
    assert not settings.kill_switch_file.exists()


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_normal_market_orders_are_explicit_day(settings, monkeypatch, side):
    trade = make_trade(status="Filled", status_filled=1, avg_price=2.50)
    broker = IBBroker(settings)
    fake_ib = FakeIB(trade, qty_before=0 if side == "BUY" else 1, fill_effect=1)
    broker.ib = fake_ib
    monkeypatch.setattr(broker, "_option", lambda c: SimpleNamespace(conId=CON_ID))
    contract = SelectedContract(symbol="NVDA", expiry="20260710", strike=200.0, right="P")

    broker._place(contract, side, 1, snapshot_supplied=True)

    _, _, order = fake_ib.placed[0]
    assert order.orderType == "MKT"
    assert order.tif == "DAY"


def quality_broker(settings, monkeypatch, *, fill_price, quote, side="BUY", latency=1.0):
    trade = make_trade(status="Filled", status_filled=1, avg_price=fill_price)
    trade.contract = SimpleNamespace(symbol="NVDA")
    trade.log = []
    broker = IBBroker(settings)
    broker.journal = Journal(settings.log_dir)
    broker.ib = FakeIB(trade, qty_before=0 if side == "BUY" else 1, fill_effect=1)
    monkeypatch.setattr(broker, "_option", lambda c: SimpleNamespace(conId=CON_ID))
    monkeypatch.setattr(broker, "get_entry_market_snapshot", lambda c: (quote, 200.0))
    times = iter([100.0, 100.0, 100.0 + latency])
    monkeypatch.setattr("tajator.broker.ib.time.monotonic", lambda: next(times))
    contract = SelectedContract(symbol="NVDA", expiry="20260710", strike=200.0, right="P")
    return broker, contract


def test_clean_fill_journals_timeline_and_execution_quality(settings, monkeypatch):
    quote = OptionQuote(bid=2.0, ask=2.05, ts=IBBroker(settings).now())
    broker, contract = quality_broker(
        settings, monkeypatch, fill_price=2.05, quote=quote,
    )
    fill = broker._place(contract, "BUY", 1)
    assert fill.execution_quality.latency_s == 1.0
    assert fill.execution_quality.breaches == []
    events = [line for line in settings.log_dir.glob("journal-*.jsonl")]
    text = "\n".join(path.read_text() for path in events)
    assert '"type": "order_timeline"' in text
    assert '"type": "execution_quality"' in text


def test_slow_slipped_over_budget_fill_is_adopted_and_halts(settings, monkeypatch):
    quote = OptionQuote(bid=4.4, ask=4.5, ts=IBBroker(settings).now())
    broker, contract = quality_broker(
        settings, monkeypatch, fill_price=5.4, quote=quote, latency=12.0,
    )
    fill = broker._place(contract, "BUY", 1)
    assert fill.qty == 1
    assert len(fill.execution_quality.breaches) == 3
    assert "latency" in fill.execution_quality.breaches[0]
    assert "slippage" in fill.execution_quality.breaches[1]
    assert "budget" in fill.execution_quality.breaches[2]
    assert broker.entry_halt_reason is not None
    assert not settings.kill_switch_file.exists()


def test_market_exit_proceeds_without_valid_quote(settings, monkeypatch):
    quote = OptionQuote(bid=None, ask=None, ts=IBBroker(settings).now())
    broker, contract = quality_broker(
        settings, monkeypatch, fill_price=1.5, quote=quote, side="SELL",
    )
    fill = broker._place(contract, "SELL", 1)
    assert fill.qty == 1
    assert fill.execution_quality.reference_price is None
    assert fill.execution_quality.breaches == []


# -- protective stop ------------------------------------------------------------


class FakeStopIB(FakeIB):
    """FakeIB extended with the protective-stop surface."""

    def __init__(self, trade=None, qty_before=0, all_fills=()):
        super().__init__(trade, qty_before=qty_before)
        self.all_fills = list(all_fills)
        self.last_order = None

    def placeOrder(self, contract, order):
        self.last_order = order
        return self.trade

    def reqAllOpenOrders(self):
        pass

    def trades(self):
        return [self.trade] if self.trade is not None else []

    def fills(self):
        return self.all_fills


STOP_ORDER = dict(orderId=42, permId=990, orderRef="tajator-stop:NVDA")


def stop_broker(settings, monkeypatch, fake_ib, underlying_con_id=555):
    broker = IBBroker(settings)
    broker.ib = fake_ib
    monkeypatch.setattr(broker, "_option", lambda c: SimpleNamespace(conId=CON_ID))
    monkeypatch.setattr(broker, "_underlying", lambda s: SimpleNamespace(conId=underlying_con_id))
    return broker


def make_stop(qty=2):
    return ProtectiveStop(order_id=42, perm_id=990, order_ref="tajator-stop:NVDA",
                          qty=qty, stop_price=200.9)


CONTRACT = SelectedContract(symbol="NVDA", expiry="20260710", strike=200.0, right="P")


def test_place_protective_stop_builds_gtc_conditional_on_underlying(settings, monkeypatch):
    trade = make_trade(status="Submitted", order=SimpleNamespace(**STOP_ORDER))
    fake = FakeStopIB(trade)
    broker = stop_broker(settings, monkeypatch, fake)
    stop = broker.place_protective_stop(CONTRACT, 2, 200.9, "put", "tajator-stop:NVDA")

    order = fake.last_order
    assert order.action == "SELL" and int(order.totalQuantity) == 2
    assert order.tif == "GTC"
    assert order.orderRef == "tajator-stop:NVDA"
    assert order.conditionsIgnoreRth is False
    [cond] = order.conditions
    assert cond.conId == 555, "the trigger watches the UNDERLYING, not the option"
    assert cond.price == 200.9
    assert cond.isMore is True, "a put position stops out when the stock RISES through the stop"
    assert stop.order_id == 42 and stop.perm_id == 990 and stop.qty == 2


def test_place_protective_stop_call_direction_triggers_below(settings, monkeypatch):
    trade = make_trade(status="Submitted", order=SimpleNamespace(**STOP_ORDER))
    fake = FakeStopIB(trade)
    broker = stop_broker(settings, monkeypatch, fake)
    broker.place_protective_stop(CONTRACT, 1, 198.5, "call", "tajator-stop:NVDA")
    [cond] = fake.last_order.conditions
    assert cond.isMore is False, "a call position stops out when the stock FALLS through the stop"


def test_place_protective_stop_rejection_raises(settings, monkeypatch):
    trade = make_trade(status="Inactive", order=SimpleNamespace(**STOP_ORDER))
    broker = stop_broker(settings, monkeypatch, FakeStopIB(trade))
    with pytest.raises(RuntimeError, match="rejected"):
        broker.place_protective_stop(CONTRACT, 2, 200.9, "put", "tajator-stop:NVDA")


def test_cancel_stop_adopts_race_fill(settings, monkeypatch):
    """The cancel raced the stop's execution: reports show 1 sold, the account
    dropped to 1 — the fill is adopted, not resold."""
    trade = make_trade(status="Cancelled", fills=[(1, 2.50)], order=SimpleNamespace(**STOP_ORDER))
    fake = FakeStopIB(trade, qty_before=1)  # account already reflects the sold contract
    broker = stop_broker(settings, monkeypatch, fake)
    result = broker.cancel_protective_stop(CONTRACT, make_stop(qty=2), expected_held=2)
    assert result.cancelled and result.filled_qty == 1
    assert result.avg_price == pytest.approx(2.50)


def test_cancel_stop_position_mismatch_halts_and_raises(settings, monkeypatch):
    """Reports explain 1 sold but the account is flat: an execution no session
    saw — kill switch, refuse to sell."""
    trade = make_trade(status="Cancelled", fills=[(1, 2.50)], order=SimpleNamespace(**STOP_ORDER))
    fake = FakeStopIB(trade, qty_before=0)
    broker = stop_broker(settings, monkeypatch, fake)
    with pytest.raises(OrderFailed) as exc_info:
        broker.cancel_protective_stop(CONTRACT, make_stop(qty=2), expected_held=2)
    assert exc_info.value.suspect is True
    assert broker.entry_halt_reason is not None
    assert not settings.kill_switch_file.exists()


def test_cancel_stop_unconfirmed_terminal_state_halts_and_raises(settings, monkeypatch):
    monkeypatch.setattr("tajator.broker.ib.CANCEL_TIMEOUT_S", 0)
    trade = make_trade(status="Submitted", order=SimpleNamespace(**STOP_ORDER), done=False)
    broker = stop_broker(settings, monkeypatch, FakeStopIB(trade))
    with pytest.raises(OrderFailed) as exc_info:
        broker.cancel_protective_stop(CONTRACT, make_stop(qty=2), expected_held=2)
    assert exc_info.value.suspect is True
    assert broker.entry_halt_reason is not None
    assert not settings.kill_switch_file.exists()


def test_poll_stop_working(settings, monkeypatch):
    trade = make_trade(status="Submitted", order=SimpleNamespace(**STOP_ORDER), done=False)
    broker = stop_broker(settings, monkeypatch, FakeStopIB(trade))
    status = broker.poll_protective_stop(CONTRACT, make_stop(qty=2))
    assert status.state == "working" and status.working_qty == 2


def test_poll_stop_filled(settings, monkeypatch):
    trade = make_trade(status="Filled", fills=[(2, 2.10)], order=SimpleNamespace(**STOP_ORDER))
    broker = stop_broker(settings, monkeypatch, FakeStopIB(trade))
    status = broker.poll_protective_stop(CONTRACT, make_stop(qty=2))
    assert status.state == "filled" and status.filled_qty == 2
    assert status.avg_price == pytest.approx(2.10)


def test_poll_stop_gone_when_order_unknown_and_no_fills(settings, monkeypatch):
    broker = stop_broker(settings, monkeypatch, FakeStopIB(None))
    assert broker.poll_protective_stop(CONTRACT, make_stop(qty=2)).state == "gone"


def test_poll_stop_filled_via_executions_when_order_object_lost(settings, monkeypatch):
    fills = [SimpleNamespace(execution=SimpleNamespace(orderId=42, permId=990, shares=2, price=2.30))]
    broker = stop_broker(settings, monkeypatch, FakeStopIB(None, all_fills=fills))
    status = broker.poll_protective_stop(CONTRACT, make_stop(qty=2))
    assert status.state == "filled" and status.filled_qty == 2


def test_ensure_connected_noop_when_connected(settings, monkeypatch):
    broker = IBBroker(settings)
    monkeypatch.setattr(broker.ib, "isConnected", lambda: True)
    assert broker.ensure_connected() is False


def test_ensure_connected_redials_after_drop(settings, monkeypatch):
    broker = IBBroker(settings)
    calls = []
    monkeypatch.setattr(broker.ib, "isConnected", lambda: False)
    monkeypatch.setattr(broker.ib, "disconnect", lambda: calls.append("disconnect"))
    monkeypatch.setattr(broker, "connect", lambda: calls.append("connect"))
    assert broker.ensure_connected() is True
    assert calls == ["disconnect", "connect"]
