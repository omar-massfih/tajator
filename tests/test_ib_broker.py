"""IBBroker safety mechanics that don't need a live Gateway."""

from types import SimpleNamespace

import pytest

from tajator.broker.ib import IBBroker
from tajator.config import Settings


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
