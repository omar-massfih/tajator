import sys
from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from tajator import cli
from tajator.config import Settings


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
    monkeypatch.setattr(cli, "cmd_check_ib", lambda args: seen.append(args.client_id))
    monkeypatch.setattr(sys, "argv", ["tajator", "check-ib"])
    cli.main()
    assert seen == [118]


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
