"""fetch_daily_series must omit endDateTime for ADJUSTED_LAST (IB Error 321)."""

from datetime import date

import tajator.backtest.data as data
from tajator.backtest.data import fetch_daily_series


class _FakeIB:
    """Captures the reqHistoricalData kwargs; returns no bars."""

    def __init__(self):
        self.calls = []
        self.ib = self

    def _underlying(self, symbol):
        return f"stock:{symbol}"

    def reqHistoricalData(self, contract, **kwargs):
        self.calls.append(kwargs)
        return []


def _fetch(monkeypatch, what):
    monkeypatch.setattr(data.time_mod, "sleep", lambda *_: None)
    ib = _FakeIB()
    fetch_daily_series(ib, "AAPL", date(2017, 1, 1), date(2026, 8, 1), what_to_show=what)
    return ib.calls[0]


def test_adjusted_last_omits_end_datetime(monkeypatch):
    call = _fetch(monkeypatch, "ADJUSTED_LAST")
    assert call["whatToShow"] == "ADJUSTED_LAST"
    assert call["endDateTime"] == ""  # IB rejects an explicit end with adjusted bars


def test_trades_anchors_at_end_datetime(monkeypatch):
    call = _fetch(monkeypatch, "TRADES")
    assert call["whatToShow"] == "TRADES"
    assert call["endDateTime"] != ""  # raw trades are anchored at the window end
