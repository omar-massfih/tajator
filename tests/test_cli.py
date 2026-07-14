import sys

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
