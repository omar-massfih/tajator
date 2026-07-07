"""Daily lifecycle: trade-counter reset, next-session scheduling, EOD flatten."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tajator.broker.stub import StubBroker
from tajator.config import Settings
from tajator.graph.nodes import RuntimeContext
from tajator.journal import Journal
from tajator.runner import TradingSession, _next_session_prep

ET = ZoneInfo("America/New_York")
CSV = Path(__file__).parent / "data" / "spy_sample_day.csv"


def make_session(tmp_path, broker):
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY", use_llm=False
    )
    return TradingSession(ctx)


def test_start_new_day_resets_trade_counter(tmp_path):
    broker = StubBroker.from_csv(CSV, prev_day_high=503.5, prev_day_low=497.0)
    sess = make_session(tmp_path, broker)
    sess.trades_today = 2
    sess.start_new_day()
    assert sess.trades_today == 0


def test_next_session_prep_same_day_before_prep():
    monday_early = datetime(2026, 7, 6, 5, 0, tzinfo=ET)  # Monday 05:00
    assert _next_session_prep(monday_early) == datetime(2026, 7, 6, 9, 0, tzinfo=ET)


def test_next_session_prep_after_close_is_next_day():
    monday_evening = datetime(2026, 7, 6, 17, 0, tzinfo=ET)
    assert _next_session_prep(monday_evening) == datetime(2026, 7, 7, 9, 0, tzinfo=ET)


def test_next_session_prep_skips_weekend():
    friday_evening = datetime(2026, 7, 10, 17, 0, tzinfo=ET)  # Friday
    assert _next_session_prep(friday_evening) == datetime(2026, 7, 13, 9, 0, tzinfo=ET)  # Monday


def test_position_left_open_is_flattened_at_end_of_replay(tmp_path):
    """A day truncated right after the entry must still end flat, with the
    forced exit in the fills — otherwise backtest PnL silently drops the trade."""
    full = StubBroker.from_csv(CSV, prev_day_high=503.5, prev_day_low=497.0)
    cutoff = [
        b for b in full.bars
        if b.ts.astimezone(ET).hour < 11
        or (b.ts.astimezone(ET).hour == 11 and b.ts.astimezone(ET).minute <= 16)
    ]
    broker = StubBroker(cutoff, prev_day_high=503.5, prev_day_low=497.0)
    sess = make_session(tmp_path, broker)
    sess.run_replay(broker, verbose=False)

    assert sess.position is None, "EOD flatten must close any open position"
    buys = sum(f[2].qty for f in broker.fills if f[0] == "BUY")
    sells = sum(f[2].qty for f in broker.fills if f[0] == "SELL")
    assert buys > 0, "the scripted entry should have fired before the cutoff"
    assert sells == buys, "every bought contract must be sold by the end of the day"

    content = "\n".join(f.read_text() for f in tmp_path.glob("journal-*.jsonl"))
    assert "end of replay day" in content
