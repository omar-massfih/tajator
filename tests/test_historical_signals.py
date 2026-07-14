from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tajator.backtest import historical_signals as hs
from tajator.models import Bar

ET = ZoneInfo("America/New_York")


def session_bars(day=date(2026, 1, 5), price=100.0):
    start = datetime.combine(day, datetime.min.time(), tzinfo=ET).replace(hour=9, minute=30)
    return [
        Bar(
            ts=start + timedelta(minutes=index),
            open=price,
            high=price + 0.1,
            low=price - 0.1,
            close=price,
            volume=1000,
        )
        for index in range(390)
    ]


def context(bars, *, previous_close=100.0, atr20=2.0, symbol="TEST"):
    return hs.SessionContext(symbol, bars[0].ts.date(), bars, previous_close, atr20)


def test_orb_enters_next_bar_and_deducts_two_sided_costs():
    bars = session_bars()
    # Opening range is 99.9..100.1; the 09:45 completed bar breaks it.
    bars[15] = bars[15].model_copy(update={"high": 100.6, "close": 100.5})
    bars[16] = bars[16].model_copy(update={"open": 100.55})
    bars[75] = bars[75].model_copy(update={"close": 101.55})

    trade = hs.orb15_breakout(context(bars))

    assert trade.direction == "call"
    assert trade.signal_ts == bars[15].ts
    assert trade.entry_ts == bars[16].ts
    assert trade.exit_ts == bars[75].ts
    assert trade.gross_points == 1.0
    assert trade.net_points == round(1 - (100.55 + 101.55) / 10_000, 4)
    assert trade.stress_net_points < trade.net_points


def test_gap_candidates_are_mutually_conditioned_by_first_five_minutes():
    continuation = session_bars(price=101.0)
    for index in range(5):
        continuation[index] = continuation[index].model_copy(
            update={"close": 101.0 + index * 0.1}
        )
    ctx = context(continuation, previous_close=100.0, atr20=2.0)
    assert hs.gap_continuation(ctx).direction == "call"
    assert hs.gap_fade(ctx) is None

    fading = session_bars(price=101.0)
    for index in range(5):
        fading[index] = fading[index].model_copy(update={"close": 101.0 - index * 0.1})
    ctx = context(fading, previous_close=100.0, atr20=2.0)
    assert hs.gap_fade(ctx).direction == "put"
    assert hs.gap_continuation(ctx) is None


def test_opening_drive_uses_prior_atr_and_close_location():
    bars = session_bars()
    for index in range(15):
        value = 100 + index * 0.1
        bars[index] = bars[index].model_copy(
            update={"open": value, "high": value + 0.1, "low": value - 0.1, "close": value}
        )
    trade = hs.opening_drive(context(bars, atr20=4.0))
    assert trade is not None
    assert trade.direction == "call"

    fade = hs.opening_drive_fade(context(bars, atr20=4.0))
    assert fade is not None
    assert fade.direction == "put"
    assert fade.gross_points == -trade.gross_points


def test_no_development_candidate_keeps_validation_closed(monkeypatch, tmp_path):
    calls = []

    def empty_sessions(cache_dir, symbol, start, end):
        calls.append((symbol, start, end))
        return [], 100

    monkeypatch.setattr(hs, "load_symbol_sessions", empty_sessions)
    monkeypatch.setattr(hs, "CANDIDATES", {"empty": lambda ctx: None})

    report = hs.run_tournament(tmp_path)

    assert report["selected"] is None
    assert report["validation_opened"] is False
    assert report["validation"] is None
    assert {symbol for symbol, _, _ in calls} == set(hs.DEVELOPMENT_SYMBOLS)
    assert all(end == hs.DEVELOPMENT_END for _, _, end in calls)


def test_ineligible_followup_keeps_validation_closed(monkeypatch, tmp_path):
    calls = []

    def empty_sessions(cache_dir, symbol, start, end):
        calls.append((symbol, start, end))
        return [], 100

    monkeypatch.setattr(hs, "load_symbol_sessions", empty_sessions)

    report = hs.run_opening_drive_fade_followup(tmp_path)

    assert report["development"]["eligible"] is False
    assert report["validation_opened"] is False
    assert report["validation"] is None
    assert {symbol for symbol, _, _ in calls} == set(hs.DEVELOPMENT_SYMBOLS)


def test_day_clustered_stats_do_not_treat_same_day_symbols_as_independent():
    now = datetime(2026, 1, 5, 10, 0, tzinfo=ET)

    def trade(symbol, day, value):
        ts = now.replace(day=day)
        return hs.SignalTrade(
            strategy="test", symbol=symbol, day=ts.date(), direction="call",
            signal_ts=ts, entry_ts=ts, exit_ts=ts,
            entry_price=100, exit_price=100 + value,
            gross_points=value, net_points=value, stress_net_points=value,
        )

    trades = [
        trade("A", 5, 1.0), trade("B", 5, 1.0),
        trade("A", 6, -0.5), trade("B", 6, -0.5),
    ]
    stats = hs._stats(trades)
    assert stats["trades"] == 4
    assert stats["trading_days"] == 2
    assert stats["expectancy"] == 0.25
    assert stats["clustered_standard_error"] > 0
