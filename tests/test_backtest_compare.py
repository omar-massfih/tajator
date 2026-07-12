import json

from tajator.backtest.compare import comparison_rows


def test_comparison_rows_reads_experiment_metrics(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "symbol": "AAPL", "start": "2026-01-01", "end": "2026-03-31",
        "total_trades": 10, "underlying_win_rate": 0.6,
        "total_underlying_points": 2.5, "avg_underlying_win": 0.8,
        "avg_underlying_loss": -0.5,
        "metadata": {"experiment": "risk-cap", "config_fingerprint": "abc123"},
    }))
    [row] = comparison_rows([path])
    assert row["experiment"] == "risk-cap"
    assert row["points"] == 2.5
    assert row["fingerprint"] == "abc123"
