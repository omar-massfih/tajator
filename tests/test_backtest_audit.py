from datetime import date

import pytest

from tajator.backtest.audit import audit_report, compare_strategy_reports


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


def test_paired_strategy_comparison_uses_daily_causal_deltas():
    def report(trades, fingerprint, risk_cap):
        return {
            "symbol": "MSFT",
            "start": "2024-07-01",
            "end": "2025-12-31",
            "trades": trades,
            "metadata": {
                "research_mode": "underlying_only",
                "config_fingerprint": fingerprint,
                "code_revision": "same-revision",
                "execution_model": {"price_source": "next minute"},
                "strategy_config": {"max_entry_to_stop_cents": risk_cap},
                "data_coverage": {
                    "requested_weekdays": 100,
                    "days_with_underlying_bars": 95,
                },
            },
        }

    baseline = report([
        {"day": "2024-07-01", "underlying_points": -1.0, "closed": True},
        {"day": "2025-01-02", "underlying_points": 0.5, "closed": True},
    ], "base", None)
    candidate = report([
        {"day": "2024-07-01", "underlying_points": 1.0, "closed": True},
        {"day": "2025-07-01", "underlying_points": 1.0, "closed": True},
    ], "candidate", 100)

    result = compare_strategy_reports(
        baseline,
        candidate,
        min_trades=2,
        expected_config_changes=("max_entry_to_stop_cents",),
    )

    # Candidate-minus-baseline by union day: +2.0, -0.5, +1.0.
    assert result["paired_daily"]["days"] == 3
    assert result["paired_daily"]["total_improvement"] == 2.5
    assert result["coverage"]["matching"] is True
    assert result["strategy_config_changes"] == {
        "max_entry_to_stop_cents": {"baseline": None, "candidate": 100}
    }
    assert result["candidate_half_year_totals"] == {
        "2024-H2": 1.0,
        "2025-H1": 0.0,
        "2025-H2": 1.0,
    }
    assert result["gates"]["positive_each_half_year"] is False
    assert result["gates"]["same_code_revision"] is True
    assert result["gates"]["only_expected_config_changes"] is True


def test_paired_strategy_comparison_refuses_scope_mismatch():
    baseline = {
        "symbol": "MSFT", "start": "2024-07-01", "end": "2025-12-31",
        "trades": [], "metadata": {"research_mode": "underlying_only"},
    }
    candidate = {**baseline, "symbol": "AAPL"}
    with pytest.raises(ValueError, match="scopes differ"):
        compare_strategy_reports(baseline, candidate)
