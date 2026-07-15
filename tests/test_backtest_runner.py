"""Integration test: run_backtest over two cached days (no IB, no LLM connection).

Day 1 reuses the bundled scripted CSV (proven by test_graph_full.py to produce
exactly one clean call setup); day 2 is a flat no-setup day. Option pricing is
faked via a monkeypatched `ensure_option_bars` so the test doesn't need real IB
historical option data — that lookup/fallback behavior is covered separately by
test_backtest_broker.py.
"""

import csv
import json
import shutil
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from conftest import walk

from tajator.backtest.runner import run_backtest
from tajator.broker.base import ChainParams
from tajator.config import Settings
from tajator.models import Bar

ET = ZoneInfo("America/New_York")
SAMPLE_CSV = Path(__file__).parent / "data" / "spy_sample_day.csv"
DAY_WITH_TRADE = date(2026, 6, 15)  # the sample CSV's own date
DAY_FLAT = date(2026, 6, 16)


def _write_bars_csv(path: Path, bars: list[Bar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b.ts.isoformat(), b.open, b.high, b.low, b.close, b.volume])


def _seed_cache(cache_dir: Path) -> None:
    (cache_dir / "SPY").mkdir(parents=True, exist_ok=True)
    shutil.copy(SAMPLE_CSV, cache_dir / "SPY" / f"{DAY_WITH_TRADE.isoformat()}.csv")
    flat_bars = walk(datetime(2026, 6, 16, 9, 30, tzinfo=ET), [500.0] * 390)
    _write_bars_csv(cache_dir / "SPY" / f"{DAY_FLAT.isoformat()}.csv", flat_bars)


def test_run_backtest_across_days(tmp_path, monkeypatch):
    _seed_cache(tmp_path)

    def fake_ensure_option_bars(ib, contract, day, cache_dir):
        return walk(datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET), [3.0] * 390)

    monkeypatch.setattr("tajator.broker.backtest.ensure_option_bars", fake_ensure_option_bars)

    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path / "logs")
    report = run_backtest(
        "SPY", DAY_WITH_TRADE, DAY_FLAT, settings, use_llm=False, ib=None, cache_dir=tmp_path
    )

    assert report.symbol == "SPY"
    assert len(report.trades) == 1, "only the scripted day should produce a trade"
    assert report.trades[0].day == DAY_WITH_TRADE
    assert report.trades[0].closed
    assert report.total_trades == 1
    assert len(report.equity_curve) == 1  # only the day with a closed trade contributes

    report_path = tmp_path / "logs" / "backtests" / f"SPY_{DAY_WITH_TRADE.isoformat()}_{DAY_FLAT.isoformat()}_baseline.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["metadata"]["experiment"] == "baseline"
    assert payload["metadata"]["use_llm"] is False
    assert payload["metadata"]["pattern_data"] is False
    assert payload["metadata"]["execution_model"]["modeled_half_spread_pct"] == 0.01
    assert "approach_band_pct" in payload["metadata"]["strategy_config"]
    assert payload["metadata"]["strategy_config"]["reaction_lookback_bars"] == 5
    assert payload["metadata"]["strategy_config"]["long_wick_min_frac"] == 0.25


def test_experiment_name_is_sanitized_in_output_path(tmp_path, monkeypatch):
    _seed_cache(tmp_path)
    monkeypatch.setattr(
        "tajator.broker.backtest.ensure_option_bars",
        lambda ib, contract, day, cache_dir: walk(
            datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET), [3.0] * 390
        ),
    )
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path / "logs")
    run_backtest(
        "SPY", DAY_WITH_TRADE, DAY_FLAT, settings, False, None, tmp_path,
        experiment="risk cap / 100c",
    )
    assert (tmp_path / "logs" / "backtests" /
            f"SPY_{DAY_WITH_TRADE}_{DAY_FLAT}_risk-cap-100c.json").exists()


def test_skip_missing_option_data_discards_whole_day_and_records_coverage(tmp_path, monkeypatch):
    _seed_cache(tmp_path)

    def missing_option_day(self, broker, verbose=False):
        broker.fills.append(("BUY", None, None))  # proves partial day state is discarded
        raise RuntimeError("no historical option data for SPY test contract")

    monkeypatch.setattr("tajator.backtest.runner.TradingSession.run_replay", missing_option_day)
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path / "logs")
    report = run_backtest(
        "SPY", DAY_WITH_TRADE, DAY_FLAT, settings, use_llm=False, ib=None,
        cache_dir=tmp_path, skip_missing_option_data=True,
    )
    assert report.total_trades == 0
    coverage = report.metadata["data_coverage"]
    assert coverage["skip_missing_option_data"] is True
    assert len(coverage["skipped_missing_option_days"]) == 2
    assert "test contract" in coverage["skipped_missing_option_days"][0]["reason"]


def test_underlying_only_does_not_request_option_history(tmp_path, monkeypatch):
    _seed_cache(tmp_path)

    def option_history_must_not_run(*args, **kwargs):
        raise AssertionError("underlying-only mode requested option history")

    monkeypatch.setattr("tajator.broker.backtest.ensure_option_bars", option_history_must_not_run)
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path / "logs")
    report = run_backtest(
        "SPY", DAY_WITH_TRADE, DAY_FLAT, settings, use_llm=False, ib=None,
        cache_dir=tmp_path, underlying_only=True,
    )
    assert report.metadata["research_mode"] == "underlying_only"
    assert report.total_trades == 1
    assert report.trades[0].underlying_points is not None


def test_forward_chain_snapshot_is_disclosed_and_passed_to_broker(tmp_path, monkeypatch):
    _seed_cache(tmp_path)
    observed = []
    chain = ChainParams(expirations=["20260617"], strikes=[499.0, 500.0, 501.0])

    def capture_chain(self, symbol):
        observed.append(self._chain_override)
        return self._chain_override

    monkeypatch.setattr("tajator.broker.backtest.BacktestBroker.get_option_chain", capture_chain)
    monkeypatch.setattr(
        "tajator.broker.backtest.ensure_option_bars",
        lambda ib, contract, day, cache_dir: walk(
            datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET), [3.0] * 390
        ),
    )
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path / "logs")
    report = run_backtest(
        "SPY", DAY_WITH_TRADE, DAY_WITH_TRADE, settings, False, None, tmp_path,
        chain_override=chain,
    )

    assert observed and all(value == chain for value in observed)
    assert report.metadata["option_chain_model"]["source"] == "captured_tws_snapshot"
    assert report.metadata["option_chain_model"]["expirations"] == ["20260617"]


def test_option_panel_is_persisted_with_counterfactual_trades(tmp_path, monkeypatch):
    _seed_cache(tmp_path)
    chain = ChainParams(
        expirations=["20260617", "20260619"], strikes=[499.0, 500.0, 501.0]
    )
    monkeypatch.setattr(
        "tajator.broker.backtest.ensure_option_bars",
        lambda ib, contract, day, cache_dir: walk(
            datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET), [3.0] * 390
        ),
    )
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path / "logs")
    report = run_backtest(
        "SPY", DAY_WITH_TRADE, DAY_WITH_TRADE, settings, False, None, tmp_path,
        chain_override=chain, option_panel=True,
    )

    assert set(report.option_panel) == {"atm_next_expiry", "itm_1_near", "otm_1_near"}
    assert all(data["total_trades"] == 1 for data in report.option_panel.values())
    assert all(not data["missing_contracts"] for data in report.option_panel.values())
    payload = json.loads(
        (tmp_path / "logs" / "backtests" /
         f"SPY_{DAY_WITH_TRADE}_{DAY_WITH_TRADE}_baseline.json").read_text()
    )
    assert payload["option_panel"]["itm_1_near"]["total_trades"] == 1
