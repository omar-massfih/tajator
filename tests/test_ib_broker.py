"""IBBroker safety mechanics that don't need a live Gateway."""

from types import SimpleNamespace

import pytest

from tajator.broker.base import OrderFailed
from tajator.broker.ib import IBBroker
from tajator.config import Settings
from tajator.models import SelectedContract


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)


def test_partial_fill_halt_activates_kill_switch(settings):
    broker = IBBroker(settings)
    broker._halt_new_entries("partial fill: BUY 1/4 SPY 20260710 500C")
    text = settings.kill_switch_file.read_text()
    assert "partial fill" in text
    assert "delete this file" in text


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
        self.placed.append((order.action, int(order.totalQuantity)))
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


def make_trade(status="Cancelled", status_filled=0, avg_price=0.0, fills=()):
    return SimpleNamespace(
        orderStatus=SimpleNamespace(status=status, filled=status_filled, avgFillPrice=avg_price),
        fills=[SimpleNamespace(execution=SimpleNamespace(shares=s, price=p)) for s, p in fills],
        isDone=lambda: True,
        order=SimpleNamespace(),
    )


def place(settings, monkeypatch, trade, side="BUY", qty=1, **fake_ib_kwargs):
    monkeypatch.setattr("tajator.broker.ib.FILL_GRACE_S", 0)
    broker = IBBroker(settings)
    broker.ib = FakeIB(trade, **fake_ib_kwargs)
    contract = SelectedContract(symbol="NVDA", expiry="20260710", strike=200.0, right="P")
    monkeypatch.setattr(broker, "_option", lambda c: SimpleNamespace(conId=CON_ID))
    return broker._place(contract, side, qty)


def test_cancelled_but_actually_filled_adopts_the_fill(settings, monkeypatch):
    """The incident case: status says Cancelled/0 filled, executions say otherwise."""
    trade = make_trade(fills=[(1, 2.97)])
    fill = place(settings, monkeypatch, trade, qty=1, fill_effect=1)
    assert fill.qty == 1
    assert fill.premium == pytest.approx(2.97)
    assert "adopted" in settings.kill_switch_file.read_text()


def test_partial_fill_returns_partial_and_halts(settings, monkeypatch):
    trade = make_trade(fills=[(2, 2.20)])
    fill = place(settings, monkeypatch, trade, qty=4, fill_effect=2)
    assert fill.qty == 2
    assert fill.premium == pytest.approx(2.20)
    assert settings.kill_switch_file.exists()


def test_confirmed_zero_fill_raises_and_halts_buys(settings, monkeypatch):
    with pytest.raises(OrderFailed) as exc_info:
        place(settings, monkeypatch, make_trade(), qty=2)
    assert exc_info.value.filled == 0
    assert exc_info.value.suspect is False
    assert settings.kill_switch_file.exists(), "failing entry orders must halt new entries"


def test_confirmed_zero_fill_sell_raises_without_halting(settings, monkeypatch):
    """An unfilled exit must keep retrying next tick — no kill switch needed."""
    with pytest.raises(OrderFailed) as exc_info:
        place(settings, monkeypatch, make_trade(), side="SELL", qty=2)
    assert exc_info.value.filled == 0
    assert not settings.kill_switch_file.exists()


def test_position_moved_without_execution_reports_is_suspect(settings, monkeypatch):
    """Account gained a contract but no executions/price arrived: cannot build a
    Fill, must raise suspect and halt so the operator reconciles."""
    with pytest.raises(OrderFailed) as exc_info:
        place(settings, monkeypatch, make_trade(), qty=1, fill_effect=1)
    assert exc_info.value.suspect is True
    assert exc_info.value.filled == 1
    assert "UNCONFIRMED" in settings.kill_switch_file.read_text()


def test_full_fill_per_reports_with_positions_down_still_adopts(settings, monkeypatch):
    """Execution reports say fully filled but the position cross-check is
    unavailable: adopt what the reports say (tracked beats orphaned), halted."""
    trade = make_trade(fills=[(1, 3.10)])
    fill = place(settings, monkeypatch, trade, qty=1, positions_fail=True)
    assert fill.qty == 1
    assert settings.kill_switch_file.exists()


def test_clean_filled_path_untouched(settings, monkeypatch):
    trade = make_trade(status="Filled", status_filled=3, avg_price=1.50)
    fill = place(settings, monkeypatch, trade, qty=3, fill_effect=3)
    assert fill.qty == 3
    assert fill.premium == pytest.approx(1.50)
    assert not settings.kill_switch_file.exists()


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
