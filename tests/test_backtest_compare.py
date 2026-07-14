import json

from tajator.backtest.compare import comparison_rows


def _write_report(path, *, experiment, values, persisted_expectancy=0.0):
    trades = [
        {
            "day": f"2026-{index % 6 + 1:02d}-{index % 27 + 1:02d}",
            "closed": True,
            "underlying_points": value,
            "pnl": value * 100,
        }
        for index, value in enumerate(values)
    ]
    path.write_text(json.dumps({
        "symbol": "AAPL",
        "start": "2026-01-01",
        "end": "2026-06-30",
        "total_trades": len(trades),
        "total_underlying_points": sum(values),
        "underlying_expectancy": persisted_expectancy,
        "trades": trades,
        "metadata": {
            "research_mode": "underlying_only",
            "experiment": experiment,
            "config_fingerprint": experiment,
        },
    }))


def test_comparison_recomputes_stale_expectancy_from_trade_ledger(tmp_path):
    path = tmp_path / "candidate.json"
    _write_report(path, experiment="candidate", values=[1.0] * 60, persisted_expectancy=0.0)

    row = comparison_rows([path])[0]

    assert row["expectancy"] == 1.0
    assert row["trades"] == 60
    assert row["positive_months"] == 6
    assert row["candidate_supported"] is True


def test_familywise_interval_penalizes_multiple_variants(tmp_path):
    paths = []
    alternating = [1.0, -0.8] * 30
    for index in range(10):
        path = tmp_path / f"candidate-{index}.json"
        _write_report(path, experiment=str(index), values=alternating)
        paths.append(path)

    rows = comparison_rows(paths)

    assert all(row["familywise_ci_low"] < row["expectancy"] for row in rows)
    assert all(row["candidate_supported"] is False for row in rows)
