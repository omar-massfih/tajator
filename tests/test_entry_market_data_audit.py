import json
from datetime import datetime, timedelta

import pytest

from tajator.backtest.audit import audit_entry_market_data


def _event(
    diagnostic_id,
    symbol,
    method,
    ts,
    *,
    elapsed,
    window=True,
    complete=True,
    underlying=200.0,
    liquidity_reason=None,
    error=None,
):
    return {
        "ts": ts.isoformat(),
        "type": "entry_market_data_diagnostic",
        "diagnostic_id": diagnostic_id,
        "regular_entry_window": window,
        "symbol": symbol,
        "method": method,
        "elapsed_seconds": elapsed,
        "complete_bid_ask": complete,
        "underlying_price": underlying,
        "liquidity_reason": liquidity_reason,
        "error": error,
        "no_order_placed": True,
    }


def _write(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def _supported_events():
    start = datetime.fromisoformat("2026-07-15T10:00:00-04:00")
    events = []
    for symbol in ("AAPL", "MSFT"):
        for index in range(5):
            ts = start + timedelta(days=index // 3, minutes=index)
            diagnostic_id = f"{symbol}:{index}"
            events.extend([
                _event(
                    diagnostic_id, symbol, "temporary_streams", ts,
                    elapsed=0.8 + index * 0.1,
                ),
                _event(
                    diagnostic_id, symbol, "production_snapshot", ts,
                    elapsed=12.0 + index * 0.1,
                ),
            ])
    return events


def test_entry_market_data_audit_supports_only_full_two_symbol_gate(tmp_path):
    path = tmp_path / "journal-2026-07-15.jsonl"
    _write(path, _supported_events())

    result = audit_entry_market_data(path, symbols=("AAPL", "MSFT"))

    assert result["verdict"] == "candidate_supported"
    assert result["production_changed"] is False
    assert result["symbols"]["AAPL"]["paired_checks"] == 5
    assert result["symbols"]["AAPL"]["regular_sessions"] == 2
    assert result["symbols"]["MSFT"]["supported"] is True


def test_entry_market_data_audit_excludes_after_hours_plumbing(tmp_path):
    ts = datetime.fromisoformat("2026-07-14T16:47:00-04:00")
    events = [
        _event(None, "AAPL", "temporary_streams", ts, elapsed=5.0, window=False),
        _event(None, "AAPL", "production_snapshot", ts, elapsed=12.2, window=False),
    ]
    path = tmp_path / "journal-2026-07-14.jsonl"
    _write(path, events)

    result = audit_entry_market_data(path, symbols=("AAPL", "MSFT"))

    assert result["verdict"] == "insufficient_regular_session_evidence"
    assert result["excluded_outside_entry_window"] == 2
    assert result["symbols"]["AAPL"]["paired_checks"] == 0


@pytest.mark.parametrize(
    ("field", "value", "failed_gate"),
    [
        ("complete_bid_ask", False, "all_stream_quotes_complete"),
        ("error", "temporary market-data cleanup failed", "no_stream_cleanup_failures"),
        ("elapsed_seconds", 5.0, "stream_max_below_5s"),
        ("liquidity_reason", "option spread too wide", "no_quote_guard_regressions"),
    ],
)
def test_entry_market_data_audit_rejects_frozen_gate_failures(
    tmp_path, field, value, failed_gate,
):
    events = _supported_events()
    stream = next(
        event for event in events
        if event["symbol"] == "AAPL" and event["method"] == "temporary_streams"
    )
    stream[field] = value
    if field == "complete_bid_ask":
        stream["underlying_price"] = None
    path = tmp_path / "journal-2026-07-15.jsonl"
    _write(path, events)

    result = audit_entry_market_data(path, symbols=("AAPL", "MSFT"))

    assert result["verdict"] == "candidate_not_supported"
    assert result["symbols"]["AAPL"]["gates"][failed_gate] is False


def test_entry_market_data_audit_requires_complete_pairs(tmp_path):
    events = _supported_events()
    events.pop()
    path = tmp_path / "journal-2026-07-15.jsonl"
    _write(path, events)

    result = audit_entry_market_data(path, symbols=("AAPL", "MSFT"))

    assert result["symbols"]["MSFT"]["unpaired_checks"] == 1
    assert result["symbols"]["MSFT"]["gates"]["all_checks_paired"] is False
    assert result["verdict"] == "insufficient_regular_session_evidence"


def test_entry_market_data_audit_refuses_any_order_capable_record(tmp_path):
    event = _supported_events()[0]
    event["no_order_placed"] = False
    path = tmp_path / "journal-2026-07-15.jsonl"
    _write(path, [event])

    with pytest.raises(ValueError, match="lacks no_order_placed=true"):
        audit_entry_market_data(path, symbols=("AAPL", "MSFT"))
