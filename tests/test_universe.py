from tajator.backtest.universe import DEFAULT_UNIVERSE


def test_universe_is_broad_clean_and_unique():
    assert len(DEFAULT_UNIVERSE) >= 90
    assert len(set(DEFAULT_UNIVERSE)) == len(DEFAULT_UNIVERSE)          # no dups
    assert DEFAULT_UNIVERSE == sorted(DEFAULT_UNIVERSE)                  # sorted
    for t in DEFAULT_UNIVERSE:
        assert t.isupper() and "." not in t and 1 <= len(t) <= 5        # clean IB tickers
    for core in ("AAPL", "SPY", "QQQ", "JPM", "XOM"):
        assert core in DEFAULT_UNIVERSE
