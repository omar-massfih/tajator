import json
from datetime import datetime

from tajator.backtest.audit import calibrate_execution_journal
from tajator.backtest.data import ET
from tajator.config import Settings
from tajator.models import Bar


def test_execution_calibration_compares_actual_and_modeled_round_trip(tmp_path, monkeypatch):
    journal = tmp_path / "journal.jsonl"
    contract = {"symbol": "MSFT", "expiry": "20260715", "strike": 392.5, "right": "P"}
    records = [
        {
            "ts": "2026-07-13T09:59:00-04:00", "type": "llm_decision", "symbol": "MSFT",
            "decision": {"action": "enter_put"},
        },
        {
            "ts": "2026-07-13T10:00:00-04:00", "type": "fill", "symbol": "MSFT",
            "action": {
                "kind": "entry", "qty": 1, "premium": 3.05,
                "ts": "2026-07-13T10:01:10-04:00",
            },
            "position": {"contract": contract, "qty_remaining": 1},
        },
        {
            "ts": "2026-07-13T10:05:00-04:00", "type": "fill", "symbol": "MSFT",
            "action": {
                "kind": "stop_exit", "qty": 1, "premium": 3.45,
                "ts": "2026-07-13T10:06:05-04:00",
            },
            "position": {"contract": contract, "qty_remaining": 0},
        },
    ]
    journal.write_text("\n".join(json.dumps(record) for record in records))
    series = [
        Bar(ts=datetime(2026, 7, 13, 10, 1, tzinfo=ET), open=3.0, high=3, low=3, close=3),
        Bar(ts=datetime(2026, 7, 13, 10, 6, tzinfo=ET), open=3.5, high=3.5, low=3.5, close=3.5),
    ]
    monkeypatch.setattr(
        "tajator.backtest.audit.ensure_option_bars", lambda *args, **kwargs: series
    )
    settings = Settings(_env_file=None)

    result = calibrate_execution_journal(
        journal, symbol="MSFT", ib=object(), cache_dir=tmp_path, settings=settings
    )

    assert result["summary"]["fills"] == 2
    assert result["summary"]["closed_trades"] == 1
    assert result["summary"]["mean_actual_adverse_cents"] == 5.0
    assert result["summary"]["mean_modeled_adverse_cents"] == 4.25
    assert result["summary"]["decision_mode"] == "llm"
    assert result["summary"]["mean_signal_to_fill_latency_s"] == 67.5
    assert result["summary"]["mean_next_bar_delay_s"] == 7.5
    assert result["trades"][0]["actual_net_estimate"] == 38.0
    assert result["trades"][0]["modeled_net_pnl"] == 39.5
    assert result["trades"][0]["actual_minus_modeled"] == -1.5


def test_execution_calibration_uses_policy_marker_and_actual_commissions(
    tmp_path, monkeypatch
):
    journal = tmp_path / "journal.jsonl"
    contract = {"symbol": "MSFT", "expiry": "20260715", "strike": 392.5, "right": "P"}
    records = [
        {
            "ts": "2026-07-13T09:29:00-04:00", "type": "policy_start",
            "policy_mode": "deterministic", "symbols": ["MSFT"],
        },
        {
            "ts": "2026-07-13T10:00:00-04:00", "type": "fill", "symbol": "MSFT",
            "action": {
                "kind": "entry", "qty": 1, "premium": 3.05,
                "ts": "2026-07-13T10:00:02-04:00",
                "execution_quality": {"order_id": 10},
            },
            "position": {"contract": contract, "qty_remaining": 1},
        },
        {
            "ts": "2026-07-13T10:05:00-04:00", "type": "fill", "symbol": "MSFT",
            "action": {
                "kind": "stop_exit", "qty": 1, "premium": 3.45,
                "ts": "2026-07-13T10:05:01-04:00",
                "execution_quality": {"order_id": 11},
            },
            "position": {"contract": contract, "qty_remaining": 0},
        },
        {"type": "commission_report", "symbol": "MSFT", "order_id": 10, "commission": 0.75},
        {"type": "commission_report", "symbol": "MSFT", "order_id": 11, "commission": 0.55},
    ]
    journal.write_text("\n".join(json.dumps(record) for record in records))
    series = [
        Bar(ts=datetime(2026, 7, 13, 10, 1, tzinfo=ET), open=3.0, high=3, low=3, close=3),
        Bar(ts=datetime(2026, 7, 13, 10, 6, tzinfo=ET), open=3.5, high=3.5, low=3.5, close=3.5),
    ]
    monkeypatch.setattr(
        "tajator.backtest.audit.ensure_option_bars", lambda *args, **kwargs: series
    )

    result = calibrate_execution_journal(
        journal, symbol="MSFT", ib=object(), cache_dir=tmp_path,
        settings=Settings(_env_file=None),
    )

    assert result["summary"]["decision_mode"] == "deterministic"
    assert result["summary"]["mean_next_bar_delay_s"] == 0.0
    assert result["fills"][0]["fee_source"] == "ib_commission_report"
    assert result["trades"][0]["estimated_fees"] == 1.3
    assert result["trades"][0]["actual_net_estimate"] == 38.7
