from datetime import date

import pytest

from tajator.backtest.audit import audit_report


def _report(values, *, mode="underlying_only", skipped=None):
    trades = []
    for index, value in enumerate(values):
        month = index % 6 + 1
        day = index % 27 + 1
        trades.append({
            "day": f"2026-{month:02d}-{day:02d}",
            "closed": True,
            "underlying_points": value,
            "pnl": value * 100,
        })
    return {
        "symbol": "AAPL",
        "trades": trades,
        "metadata": {
            "research_mode": mode,
            "execution_model": {"price_source": "next bar"},
            "data_coverage": {
                "requested_weekdays": 100,
                "days_with_underlying_bars": 96,
                "skipped_missing_option_days": skipped or [],
            },
        },
    }


def test_audit_refuses_to_call_exploratory_results_an_edge():
    result = audit_report(_report([1.0] * 60))
    assert result["verdict"] == "exploratory_only"
    assert result["gates"]["declared_out_of_sample"] is False


def test_underlying_holdout_can_support_signal_but_not_options_edge():
    result = audit_report(
        _report([1.0] * 60), validation_start=date(2026, 1, 1)
    )
    assert result["verdict"] == "statistically_supported_stock_signal"
    assert result["validation"]["ci95_low"] == 1.0


def test_options_edge_requires_sample_profit_factor_and_complete_data():
    values = [2.0, 2.0, -1.0] * 20
    report = _report(values, mode="historical_options")
    result = audit_report(report, validation_only=True)
    assert result["verdict"] == "confirmed_options_edge"
    assert result["gates"]["profit_factor"] is True

    report["metadata"]["data_coverage"]["skipped_missing_option_days"] = [{"day": "2026-01-02"}]
    result = audit_report(report, validation_only=True)
    assert result["verdict"] == "options_edge_not_confirmed"
    assert result["gates"]["data_coverage"] is False


def test_small_forward_sample_is_insufficient_even_when_profitable():
    result = audit_report(
        _report([3.0] * 10, mode="historical_options"), validation_only=True
    )
    assert result["verdict"] == "insufficient_sample"
    assert result["gates"]["minimum_trades"] is False


def test_validation_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        audit_report(
            _report([1.0]), validation_start=date(2026, 1, 1), validation_only=True
        )
