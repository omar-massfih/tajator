from tajator.market.indicators import build_snapshot, bars_to_df, ema, session_df, session_vwap

from conftest import make_bar, ts, walk


def manual_ema(values: list[float], span: int) -> float:
    alpha = 2.0 / (span + 1)
    e = values[0]
    for v in values[1:]:
        e = e + alpha * (v - e)
    return e


def test_ema_matches_recursive_definition():
    closes = [100.0, 101.0, 102.5, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5]
    bars = walk(ts(9, 30), closes)
    df = bars_to_df(bars)
    result = ema(df["close"], 9)
    assert result is not None
    assert abs(result - manual_ema(closes, 9)) < 1e-9


def test_ema_needs_enough_bars():
    bars = walk(ts(9, 30), [100.0, 101.0, 102.0])
    df = bars_to_df(bars)
    assert ema(df["close"], 9) is None


def test_session_vwap_exact():
    bars = [
        make_bar(ts(9, 30), 101.0, o=100.0, h=102.0, lo=100.0, vol=100),  # typical 101
        make_bar(ts(9, 31), 103.0, o=101.0, h=104.0, lo=102.0, vol=300),  # typical 103
    ]
    df = bars_to_df(bars)
    vwap = session_vwap(session_df(df, bars[-1].ts))
    expected = (101.0 * 100 + 103.0 * 300) / 400
    assert abs(vwap - expected) < 1e-9


def test_snapshot_excludes_premarket_from_session_stats():
    pre = [make_bar(ts(9, 0), 490.0, h=495.0, lo=485.0)]  # extreme premarket range
    rth = walk(ts(9, 30), [500.0 + 0.1 * i for i in range(60)])
    snap = build_snapshot("SPY", pre + rth)
    assert snap.hod == max(b.high for b in rth)
    assert snap.lod == min(b.low for b in rth)
    assert snap.price == rth[-1].close
    assert snap.ema9 is not None and snap.ema50 is not None and snap.vwap is not None
