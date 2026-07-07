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
