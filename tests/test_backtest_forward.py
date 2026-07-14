import json
from datetime import date, datetime

import pytest
from conftest import walk

from tajator.backtest.data import ET
from tajator.backtest.forward import capture_forward_day, latest_completed_session, session_quality
from tajator.backtest.ledger import BacktestReport
from tajator.broker.base import ChainParams
from tajator.config import Settings


DAY = date(2026, 1, 2)


class FakeIB:
    def get_option_chain(self, symbol):
        return ChainParams(expirations=["20260105", "20260107"], strikes=[99.0, 100.0, 101.0])


@pytest.fixture(autouse=True)
def complete_underlying(monkeypatch):
    def bars(ib, symbol, day, cache_dir, **kwargs):
        return walk(
            datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET), [100.0] * 390
        )

    monkeypatch.setattr("tajator.backtest.forward.ensure_underlying_bars", bars)


def _fake_report(settings):
    report = BacktestReport(
        symbol="AAPL",
        start=DAY,
        end=DAY,
        metadata={
            "research_mode": "historical_options",
            "execution_model": {"price_source": "next bar"},
            "option_chain_model": {
                "source": "captured_tws_snapshot",
                "expirations": ["20260105"],
                "strikes": [99.0, 100.0, 101.0],
            },
            "data_coverage": {
                "requested_weekdays": 1,
                "days_with_underlying_bars": 1,
                "skipped_missing_option_days": [],
            },
        },
    )
    report.daily_pnl[DAY] = 12.5
    report.equity_curve.append((DAY, 12.5))
    report.option_panel["itm_1_near"] = {
        "trades": [{
            "day": DAY.isoformat(), "closed": True, "pnl": 8.0,
            "return_on_premium": 0.04,
        }],
        "missing_contracts": [],
    }
    return report


def test_forward_capture_locks_definition_and_builds_cumulative(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None,
        log_dir=tmp_path / "logs",
        kill_switch_file=tmp_path / "KILL",
    )
    monkeypatch.setattr(
        "tajator.backtest.forward.run_backtest",
        lambda *args, **kwargs: _fake_report(settings),
    )

    record, cumulative = capture_forward_day(
        name="aapl v1", symbol="AAPL", day=DAY, settings=settings, ib=FakeIB(),
        cache_dir=tmp_path / "cache",
    )

    assert record.exists()
    payload = json.loads(cumulative.read_text())
    assert payload["metadata"]["validation_protocol"]["kind"] == "frozen_forward"
    assert payload["metadata"]["data_coverage"]["days_with_underlying_bars"] == 1
    assert payload["metadata"]["option_chain_snapshots"][DAY.isoformat()]["source"] == (
        "captured_tws_snapshot"
    )
    assert payload["option_panel"]["itm_1_near"]["stats"]["trades"] == 1
    assert payload["option_panel"]["itm_1_near"]["complete"] is True
    assert payload["metadata"]["capture_protocol"]["version"] == 1
    assert payload["metadata"]["underlying_session_quality"][DAY.isoformat()]["complete"] is True
    day_record = json.loads(record.read_text())
    assert day_record["report"]["daily_pnl"] == {DAY.isoformat(): 12.5}
    manifest = json.loads((tmp_path / "logs" / "forward" / "aapl-v1" / "manifest.json").read_text())
    assert manifest["captured_days"] == [DAY.isoformat()]


def test_forward_capture_refuses_changed_strategy_in_same_cohort(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, log_dir=tmp_path / "logs")
    monkeypatch.setattr(
        "tajator.backtest.forward.run_backtest",
        lambda *args, **kwargs: _fake_report(settings),
    )
    kwargs = dict(
        name="locked", symbol="AAPL", day=DAY, settings=settings, ib=FakeIB(),
        cache_dir=tmp_path / "cache",
    )
    capture_forward_day(**kwargs)

    changed = settings.model_copy(update={"stop_buffer_cents": 55})
    with pytest.raises(ValueError, match="Use a new cohort name"):
        capture_forward_day(**{**kwargs, "settings": changed})


def test_forward_capture_requires_a_completed_day(tmp_path):
    settings = Settings(_env_file=None, log_dir=tmp_path / "logs")
    with pytest.raises(ValueError, match="completed session"):
        capture_forward_day(
            name="future", symbol="AAPL", day=date(2099, 1, 1), settings=settings,
            ib=object(), cache_dir=tmp_path / "cache",
        )


def test_latest_completed_session_skips_weekend_and_empty_holiday(tmp_path, monkeypatch):
    calls = []

    def bars(ib, symbol, day, cache_dir, **kwargs):
        calls.append(day)
        return (
            walk(datetime(2026, 7, 2, 9, 30, tzinfo=ET), [100.0] * 390)
            if day == date(2026, 7, 2) else []
        )

    monkeypatch.setattr("tajator.backtest.forward.ensure_underlying_bars", bars)
    found = latest_completed_session(
        symbol="AAPL", ib=object(), cache_dir=tmp_path, today=date(2026, 7, 7)
    )
    assert found == date(2026, 7, 2)
    assert date(2026, 7, 4) not in calls and date(2026, 7, 5) not in calls


def test_latest_completed_session_fails_when_lookback_has_no_bars(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tajator.backtest.forward.ensure_underlying_bars", lambda *args, **kwargs: []
    )
    with pytest.raises(ValueError, match="no completed AAPL session"):
        latest_completed_session(
            symbol="AAPL", ib=object(), cache_dir=tmp_path,
            today=date(2026, 7, 7), lookback_calendar_days=3,
        )


def test_latest_completed_session_refuses_a_truncated_latest_day(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tajator.backtest.forward.ensure_underlying_bars",
        lambda *args, **kwargs: walk(
            datetime(2026, 7, 6, 9, 30, tzinfo=ET), [100.0] * 270
        ),
    )
    with pytest.raises(ValueError, match="is incomplete"):
        latest_completed_session(
            symbol="AAPL", ib=object(), cache_dir=tmp_path, today=date(2026, 7, 7)
        )


def test_session_quality_accepts_full_and_standard_early_close():
    full = walk(datetime(2026, 7, 6, 9, 30, tzinfo=ET), [100.0] * 390)
    early = walk(datetime(2026, 7, 3, 9, 30, tzinfo=ET), [100.0] * 210)
    assert session_quality(full, date(2026, 7, 6))["kind"] == "full"
    quality = session_quality(early, date(2026, 7, 3))
    assert quality["complete"] is True
    assert quality["kind"] == "early_close"


def test_session_quality_rejects_midday_truncation():
    partial = walk(datetime(2026, 7, 6, 9, 30, tzinfo=ET), [100.0] * 270)
    quality = session_quality(partial, date(2026, 7, 6))
    assert quality["complete"] is False
    assert quality["kind"] == "truncated"
