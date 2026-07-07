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
