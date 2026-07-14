from datetime import date, datetime
from types import SimpleNamespace

from tajator.backtest.data import ET, ensure_underlying_bars


def test_refresh_replaces_an_existing_partial_underlying_cache(tmp_path, monkeypatch):
    day = date(2026, 7, 13)
    path = tmp_path / "AAPL" / f"{day}.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "ts,open,high,low,close,volume\n"
        "2026-07-13T09:30:00-04:00,100,100,100,100,1\n"
    )
    fresh = SimpleNamespace(
        date=datetime(2026, 7, 13, 15, 59, tzinfo=ET),
        open=110.0, high=111.0, low=109.0, close=110.5, volume=10,
    )

    class Client:
        def reqHistoricalData(self, *args, **kwargs):
            return [fresh]

    broker = SimpleNamespace(ib=Client(), _underlying=lambda symbol: symbol)
    monkeypatch.setattr("tajator.backtest.data.time_mod.sleep", lambda seconds: None)

    cached = ensure_underlying_bars(broker, "AAPL", day, tmp_path)
    refreshed = ensure_underlying_bars(broker, "AAPL", day, tmp_path, refresh=True)

    assert cached[0].close == 100.0
    assert refreshed[0].close == 110.5
    assert "110.5" in path.read_text()
