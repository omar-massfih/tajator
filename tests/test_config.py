import pytest
from pydantic import ValidationError

from tajator.config import Settings


def test_symbols_defaults_to_spy():
    settings = Settings(_env_file=None)
    assert settings.symbols == ["SPY"]


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


def test_level_quality_defaults_match_the_algorithm_constants():
    from tajator.market.levels import (
        CLUSTER_TOL,
        DOUBLE_MIN_PULLBACK_PCT,
        DOUBLE_MIN_TOUCH_SEPARATION_BARS,
        SWING_WINDOW,
    )
    from tajator.market.setups import MIN_LEVEL_DIST_FROM_OPEN_PCT

    settings = Settings(_env_file=None)
    assert settings.double_min_touch_separation_bars == DOUBLE_MIN_TOUCH_SEPARATION_BARS
    assert settings.double_min_pullback_pct == DOUBLE_MIN_PULLBACK_PCT
    assert settings.min_level_dist_from_open_pct == MIN_LEVEL_DIST_FROM_OPEN_PCT
    assert settings.swing_window_bars == SWING_WINDOW
    assert settings.level_cluster_tol_pct == CLUSTER_TOL


def test_setup_and_stop_defaults_match_the_algorithm_constants():
    from tajator.market.setups import (
        APPROACH_BAND,
        MIN_SPEED_PCT,
        OVERSHOOT_BAND,
        SPEED_WINDOW,
    )
    from tajator.risk.guardrails import STOP_MAX_CENTS, STOP_MIN_CENTS

    settings = Settings(_env_file=None)
    assert settings.approach_band_pct == APPROACH_BAND
    assert settings.overshoot_band_pct == OVERSHOOT_BAND
    assert settings.speed_window_bars == SPEED_WINDOW
    assert settings.min_speed_pct == MIN_SPEED_PCT
    assert settings.stop_min_cents == STOP_MIN_CENTS
    assert settings.stop_max_cents == STOP_MAX_CENTS


def test_level_quality_fields_parse_env_strings():
    settings = Settings(
        _env_file=None,
        double_min_touch_separation_bars="15",
        double_min_pullback_pct="0.003",
        min_level_dist_from_open_pct="0.005",
    )
    assert settings.double_min_touch_separation_bars == 15
    assert settings.double_min_pullback_pct == 0.003
    assert settings.min_level_dist_from_open_pct == 0.005


def test_setup_and_stop_fields_parse_env_strings():
    settings = Settings(
        _env_file=None,
        approach_band_pct="0.005",
        speed_window_bars="5",
        stop_min_cents="10",
        stop_max_cents="80",
    )
    assert settings.approach_band_pct == 0.005
    assert settings.speed_window_bars == 5
    assert settings.stop_min_cents == 10
    assert settings.stop_max_cents == 80


def test_backtest_execution_costs_must_be_nonnegative():
    with pytest.raises(ValidationError, match="cannot be negative"):
        Settings(_env_file=None, backtest_slippage_cents=-0.01)


def test_symbol_strategy_override_resolves_without_mutating_global():
    settings = Settings(
        _env_file=None,
        entry_confirmation="immediate",
        symbol_strategy_overrides={
            "aapl": {
                "entry_confirmation": "touch_rejection",
                "max_entry_to_stop_cents": 90,
                "no_new_entries_after": "14:00",
                "blocked_direction_regimes": ["put:trend_up"],
            }
        },
    )
    aapl = settings.for_symbol("AAPL")
    assert aapl.entry_confirmation == "touch_rejection"
    assert aapl.max_entry_to_stop_cents == 90
    assert aapl.no_new_entries_after.hour == 14
    assert aapl.blocked_direction_regimes == ["put:trend_up"]
    assert settings.entry_confirmation == "immediate"
    assert settings.for_symbol("MSFT") is settings


def test_invalid_direction_regime_block_is_rejected():
    with pytest.raises(ValidationError, match="invalid direction/regime"):
        Settings(_env_file=None, blocked_direction_regimes=["put:sideways"])
