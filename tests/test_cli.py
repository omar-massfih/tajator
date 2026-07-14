import sys
from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from tajator import cli
from tajator.broker.ib import IBBroker
from tajator.config import Settings
from tajator.models import OptionQuote, SelectedContract


def test_run_deterministic_flag_is_forwarded(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_run", lambda args: seen.append(args.deterministic))
    monkeypatch.setattr(sys, "argv", ["tajator", "run", "--deterministic"])
    cli.main()
    assert seen == [True]


def test_run_defaults_to_deterministic_policy(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_run", lambda args: seen.append(args.deterministic))
    monkeypatch.setattr(sys, "argv", ["tajator", "run"])
    cli.main()
    assert seen == [True]


def test_run_llm_flag_is_an_explicit_opt_in(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_run", lambda args: seen.append(args.deterministic))
    monkeypatch.setattr(sys, "argv", ["tajator", "run", "--llm"])
    cli.main()
    assert seen == [False]


def test_runtime_policy_metadata_labels_validation_compatibility():
    settings = Settings(_env_file=None, symbols=["AAPL"])
    deterministic = cli._runtime_policy_metadata(settings, True)
    llm = cli._runtime_policy_metadata(settings, False)
    assert deterministic["policy_mode"] == "deterministic"
    assert deterministic["validation_compatible"] is True
    assert deterministic["cohort_fingerprints"]["AAPL"]
    assert llm["policy_mode"] == "llm"
    assert llm["validation_compatible"] is False
    assert llm["cohort_fingerprints"] == {}


def test_forward_latest_has_dedicated_default_client_id(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_forward_latest", lambda args: seen.append(args.client_id))
    monkeypatch.setattr(
        sys, "argv", ["tajator", "forward-latest", "--name", "cohort", "--symbol", "AAPL"]
    )
    cli.main()
    assert seen == [117]


def test_shadow_has_dedicated_default_client_id(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_shadow", lambda args: seen.append(args.client_id))
    monkeypatch.setattr(sys, "argv", ["tajator", "shadow", "--symbol", "MSFT"])
    cli.main()
    assert seen == [116]


def test_check_ib_has_dedicated_default_client_id(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_check_ib", lambda args: seen.append(args))
    monkeypatch.setattr(sys, "argv", ["tajator", "check-ib"])
    cli.main()
    assert seen[0].client_id == 118
    assert seen[0].entry_samples == 1


def test_check_ib_accepts_bounded_multi_sample_collection(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_check_ib", lambda args: seen.append(args))
    monkeypatch.setattr(sys, "argv", ["tajator", "check-ib", "--entry-samples", "3"])
    cli.main()
    assert seen[0].entry_samples == 3


def test_check_ib_refuses_more_than_five_entry_samples(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tajator", "check-ib", "--entry-samples", "6"])
    with pytest.raises(SystemExit):
        cli.main()


def test_strategy_compare_forwards_locked_report_paths(monkeypatch, tmp_path):
    seen = []
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    monkeypatch.setattr(cli, "cmd_strategy_compare", lambda args: seen.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tajator", "strategy-compare", str(baseline), str(candidate),
            "--min-trades", "250", "--only-change", "max_entry_to_stop_cents",
            "--output", str(tmp_path / "comparison.json"),
        ],
    )
    cli.main()
    assert seen[0].baseline == baseline
    assert seen[0].candidate == candidate
    assert seen[0].min_trades == 250
    assert seen[0].only_change == ["max_entry_to_stop_cents"]
    assert seen[0].output == tmp_path / "comparison.json"


def test_entry_data_report_forwards_fixed_evidence_scope(monkeypatch, tmp_path):
    seen = []
    journal = tmp_path / "diagnostics"
    output = tmp_path / "audit.json"
    monkeypatch.setattr(cli, "cmd_entry_data_report", lambda args: seen.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tajator", "entry-data-report", "--path", str(journal),
            "--symbols", "AAPL,MSFT", "--output", str(output),
        ],
    )
    cli.main()
    assert seen[0].path == journal
    assert seen[0].symbols == "AAPL,MSFT"
    assert seen[0].output == output


def test_forward_init_forwards_manifest_identity_without_tws(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_forward_init", lambda args: seen.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        ["tajator", "forward-init", "--name", "aapl-panel-v4", "--symbol", "AAPL"],
    )
    cli.main()
    assert seen[0].name == "aapl-panel-v4"
    assert seen[0].symbol == "AAPL"


def test_backtest_accepts_current_tws_chain_diagnostic_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_backtest", lambda args: seen.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tajator", "backtest", "--symbol", "AAPL", "--start", "2026-07-14",
            "--end", "2026-07-14", "--no-llm", "--tws-chain-snapshot",
        ],
    )
    cli.main()
    assert seen[0].tws_chain_snapshot is True


@pytest.mark.parametrize(
    ("cached_only", "underlying_only"),
    [(True, False), (False, True)],
)
def test_tws_chain_snapshot_requires_online_exact_option_mode(
    cached_only, underlying_only,
):
    args = SimpleNamespace(cached_only=cached_only, underlying_only=underlying_only)
    now = datetime(2026, 7, 14, 16, 5, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(ValueError, match="online exact-option"):
        cli._validate_tws_chain_snapshot(args, date(2026, 7, 14), date(2026, 7, 14), now)


def test_tws_chain_snapshot_requires_single_current_day():
    args = SimpleNamespace(cached_only=False, underlying_only=False)
    now = datetime(2026, 7, 14, 16, 5, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(ValueError, match="single current-day"):
        cli._validate_tws_chain_snapshot(args, date(2026, 7, 13), date(2026, 7, 14), now)


def test_tws_chain_snapshot_waits_until_regular_session_close():
    args = SimpleNamespace(cached_only=False, underlying_only=False)
    now = datetime(2026, 7, 14, 13, 5, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(ValueError, match="session to be complete"):
        cli._validate_tws_chain_snapshot(args, date(2026, 7, 14), date(2026, 7, 14), now)


def test_streaming_entry_diagnostic_returns_first_complete_quote_and_cancels(
    monkeypatch, tmp_path,
):
    broker = IBBroker(Settings(_env_file=None, log_dir=tmp_path))
    contract = SelectedContract(
        symbol="AAPL", expiry="20260715", strike=315.0, right="P"
    )
    option, stock = object(), object()
    option_ticker = SimpleNamespace(bid=float("nan"), ask=float("nan"), last=2.0)
    stock_ticker = SimpleNamespace(price=None)
    stock_ticker.marketPrice = lambda: stock_ticker.price
    requests, cancelled = [], []

    def request(requested, snapshot):
        requests.append((requested, snapshot))
        return option_ticker if requested is option else stock_ticker

    def update(timeout):
        option_ticker.bid, option_ticker.ask = 1.98, 2.04
        stock_ticker.price = 314.705

    broker.ib = SimpleNamespace(
        reqMktData=request, waitOnUpdate=update, cancelMktData=cancelled.append,
    )
    monkeypatch.setattr(broker, "_option", lambda selected: option)
    monkeypatch.setattr(broker, "_underlying", lambda symbol: stock)
    times = iter([100.0, 100.0, 100.2])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(times))

    quote, underlying, elapsed = cli._streaming_entry_market_diagnostic(broker, contract)

    assert requests == [(option, False), (stock, False)]
    assert cancelled == [option, stock]
    assert quote.bid == 1.98 and quote.ask == 2.04
    assert underlying == 314.705
    assert elapsed == pytest.approx(0.2)


def test_streaming_entry_diagnostic_timeout_reports_partial_state_and_cancels(
    monkeypatch, tmp_path,
):
    broker = IBBroker(Settings(_env_file=None, log_dir=tmp_path))
    contract = SelectedContract(
        symbol="AAPL", expiry="20260715", strike=315.0, right="P"
    )
    option, stock = object(), object()
    option_ticker = SimpleNamespace(bid=None, ask=None, last=None)
    stock_ticker = SimpleNamespace(marketPrice=lambda: 314.70)
    cancelled = []
    broker.ib = SimpleNamespace(
        reqMktData=lambda requested, snapshot: (
            option_ticker if requested is option else stock_ticker
        ),
        waitOnUpdate=lambda timeout: None,
        cancelMktData=cancelled.append,
    )
    monkeypatch.setattr(broker, "_option", lambda selected: option)
    monkeypatch.setattr(broker, "_underlying", lambda symbol: stock)
    times = iter([100.0, 105.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(times))

    with pytest.raises(TimeoutError, match=r"bid=None, ask=None, underlying=314.7"):
        cli._streaming_entry_market_diagnostic(broker, contract)

    assert cancelled == [option, stock]


def test_streaming_entry_diagnostic_reports_cleanup_failure(monkeypatch, tmp_path):
    broker = IBBroker(Settings(_env_file=None, log_dir=tmp_path))
    contract = SelectedContract(
        symbol="AAPL", expiry="20260715", strike=315.0, right="P"
    )
    option, stock = object(), object()
    option_ticker = SimpleNamespace(bid=1.98, ask=2.04, last=2.0)
    stock_ticker = SimpleNamespace(marketPrice=lambda: 314.70)

    def cancel(requested):
        if requested is option:
            raise RuntimeError("option cancel rejected")

    broker.ib = SimpleNamespace(
        reqMktData=lambda requested, snapshot: (
            option_ticker if requested is option else stock_ticker
        ),
        waitOnUpdate=lambda timeout: None,
        cancelMktData=cancel,
    )
    monkeypatch.setattr(broker, "_option", lambda selected: option)
    monkeypatch.setattr(broker, "_underlying", lambda symbol: stock)
    times = iter([100.0, 100.1])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(times))

    with pytest.raises(RuntimeError, match="cleanup failed.*option cancel rejected"):
        cli._streaming_entry_market_diagnostic(broker, contract)


def test_entry_market_data_pair_persists_matching_no_order_records(monkeypatch):
    now = datetime(2026, 7, 15, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    contract = SelectedContract(
        symbol="AAPL", expiry="20260717", strike=315.0, right="C"
    )
    quote = OptionQuote(bid=1.98, ask=2.04, last=2.0, ts=now)
    broker = SimpleNamespace(
        now=lambda: now,
        get_entry_market_snapshot=lambda selected: (quote, 314.70),
    )
    records = []
    diagnostics = SimpleNamespace(
        write=lambda event_type, **payload: records.append(
            {"type": event_type, **payload}
        )
    )
    monkeypatch.setattr(
        cli,
        "_streaming_entry_market_diagnostic",
        lambda current_broker, selected: (quote, 314.70, 0.8),
    )
    times = iter([100.0, 200.0, 212.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(times))

    cli._run_entry_market_data_pair(
        broker, Settings(_env_file=None), diagnostics, contract, "AAPL", 2,
    )

    assert [record["method"] for record in records] == [
        "temporary_streams", "production_snapshot",
    ]
    assert records[0]["diagnostic_id"] == records[1]["diagnostic_id"]
    assert records[0]["diagnostic_id"].endswith(":sample-2")
    assert all(record["regular_entry_window"] is True for record in records)
    assert all(record["no_order_placed"] is True for record in records)
    assert all(record["liquidity_reason"] is None for record in records)
    assert records[0]["elapsed_seconds"] == 0.8
    assert records[1]["elapsed_seconds"] == 12.0
