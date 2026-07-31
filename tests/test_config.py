import pytest
from pydantic import ValidationError

from tajator.config import Settings


def test_symbols_defaults_to_spy():
    settings = Settings(_env_file=None)
    assert settings.symbols == ["SPY"]


def test_no_symbol_overrides_by_default():
    settings = Settings(_env_file=None)
    # ORB is symbol-agnostic — every symbol resolves to the global settings.
    assert settings.symbol_strategy_overrides == {}
    assert settings.for_symbol("AAPL") is settings
    assert settings.for_symbol("MSFT") is settings


def test_symbols_parses_comma_separated_env_string():
    settings = Settings(_env_file=None, symbols="SPY,AAPL,MSFT,NVDA")
    assert settings.symbols == ["SPY", "AAPL", "MSFT", "NVDA"]


def test_symbols_uppercases_and_strips_whitespace():
    settings = Settings(_env_file=None, symbols=" spy, aapl ,msft")
    assert settings.symbols == ["SPY", "AAPL", "MSFT"]


def test_live_mode_rejects_paper_port():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, trading_mode="live", ib_port=4002)


def test_paper_mode_rejects_gateway_live_port():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, trading_mode="paper", ib_port=4001)


def test_paper_mode_rejects_tws_live_port():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, trading_mode="paper", ib_port=7496)


def test_live_mode_accepts_tws_live_port():
    settings = Settings(_env_file=None, trading_mode="live", ib_port=7496)
    assert settings.trading_mode == "live"


def test_paper_mode_accepts_tws_paper_port():
    settings = Settings(_env_file=None, trading_mode="paper", ib_port=7497)
    assert settings.ib_port == 7497


def test_orb_defaults():
    settings = Settings(_env_file=None)
    assert settings.orb_window_minutes == 15
    assert settings.orb_breakout_buffer_pct == 0.0005
    assert settings.orb_min_relative_volume == 0.0
    assert settings.orb_min_breakout_range_atr == 0.0
    assert settings.exit_mode == "scale"
    assert settings.runner_target_r == 3.0
    assert settings.runner_trail_atr_mult == 1.5


def test_let_run_exit_fields_parse_env_strings():
    settings = Settings(
        _env_file=None,
        exit_mode="let_run",
        runner_target_r="5",
        runner_trail_atr_mult="2.0",
        orb_min_relative_volume="1.3",
        orb_min_breakout_range_atr="1.0",
    )
    assert settings.exit_mode == "let_run"
    assert settings.runner_target_r == 5.0
    assert settings.runner_trail_atr_mult == 2.0
    assert settings.orb_min_relative_volume == 1.3
    assert settings.orb_min_breakout_range_atr == 1.0


def test_stop_band_defaults_match_the_guardrail_constants():
    from tajator.risk.guardrails import STOP_MAX_CENTS, STOP_MIN_CENTS

    settings = Settings(_env_file=None)
    assert settings.stop_min_cents == STOP_MIN_CENTS
    assert settings.stop_max_cents == STOP_MAX_CENTS


def test_raised_but_capped_sizing_defaults():
    settings = Settings(_env_file=None)
    assert settings.max_contracts == 10
    assert settings.max_premium_usd == 2000.0
    assert settings.max_trades_per_day == 2


def test_guarded_execution_defaults():
    settings = Settings(_env_file=None)
    assert settings.max_option_spread_pct == 0.08
    assert settings.max_option_spread_cents == 30
    assert settings.entry_budget_reserve_pct == 0.05
    assert settings.max_entry_drift_atr == 0.5
    assert settings.max_execution_slippage_pct == 0.03
    assert settings.max_execution_slippage_cents == 10
    assert settings.max_acceptable_fill_latency_s == 10
    assert settings.execution_diagnostic_max_age_days == 7
    assert settings.execution_live_confirmed is False


def test_orb_fields_parse_env_strings():
    settings = Settings(
        _env_file=None,
        orb_window_minutes="30",
        orb_breakout_buffer_pct="0.001",
        stop_min_cents="10",
        stop_max_cents="200",
    )
    assert settings.orb_window_minutes == 30
    assert settings.orb_breakout_buffer_pct == 0.001
    assert settings.stop_min_cents == 10
    assert settings.stop_max_cents == 200


def test_backtest_execution_costs_must_be_nonnegative():
    with pytest.raises(ValidationError, match="cannot be negative"):
        Settings(_env_file=None, backtest_slippage_cents=-0.01)


def test_symbol_time_window_override_resolves_without_mutating_global():
    settings = Settings(
        _env_file=None,
        symbol_strategy_overrides={"aapl": {"no_new_entries_after": "14:00"}},
    )
    aapl = settings.for_symbol("AAPL")
    assert aapl.no_new_entries_after.hour == 14
    assert settings.no_new_entries_after.hour == 15  # global unchanged
    assert settings.for_symbol("MSFT") is settings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"orb_window_minutes": 0},
        {"orb_breakout_buffer_pct": -0.001},
        {"stop_min_cents": 0},
        {"stop_max_cents": 0},
        {"max_contracts": 0},
        {"max_premium_usd": 0},
        {"stop_min_cents": 100, "stop_max_cents": 50},
        {"orb_min_relative_volume": -0.1},
        {"orb_min_breakout_range_atr": -0.1},
        {"runner_target_r": 0},
        {"runner_trail_atr_mult": 0},
    ],
)
def test_invalid_orb_settings_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)
