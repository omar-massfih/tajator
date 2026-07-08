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
    from tajator.market.levels import DOUBLE_MIN_PULLBACK_PCT, DOUBLE_MIN_TOUCH_SEPARATION_BARS
    from tajator.market.setups import MIN_LEVEL_DIST_FROM_OPEN_PCT

    settings = Settings(_env_file=None)
    assert settings.double_min_touch_separation_bars == DOUBLE_MIN_TOUCH_SEPARATION_BARS
    assert settings.double_min_pullback_pct == DOUBLE_MIN_PULLBACK_PCT
    assert settings.min_level_dist_from_open_pct == MIN_LEVEL_DIST_FROM_OPEN_PCT


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
