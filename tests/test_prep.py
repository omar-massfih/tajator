import json
from datetime import datetime, time
from zoneinfo import ZoneInfo

from tajator.broker.stub import StubBroker
from tajator.config import Settings
from tajator.graph.nodes import RuntimeContext
from tajator.journal import Journal
from tajator.runner import PREP_TIME, RTH_OPEN, TradingSession, _todays_prep_and_open

from conftest import make_bar, ts

ET = ZoneInfo("America/New_York")


def test_todays_prep_and_open_before_prep():
    now = datetime(2026, 7, 6, 7, 0, tzinfo=ET)
    prep_at, open_at = _todays_prep_and_open(now)
    assert prep_at == datetime(2026, 7, 6, 9, 0, tzinfo=ET)
    assert open_at == datetime(2026, 7, 6, 9, 30, tzinfo=ET)


def test_todays_prep_and_open_uses_constants():
    now = datetime(2026, 7, 6, 12, 0, tzinfo=ET)
    prep_at, open_at = _todays_prep_and_open(now)
    assert prep_at.time() == PREP_TIME
    assert open_at.time() == RTH_OPEN


def test_prep_writes_journal_event_and_no_llm_fallback(tmp_path):
    # Premarket-only bars (before 09:30), matching when prep actually runs
    # (30 min before the open) — well separated from the prev-day range so
    # dedupe keeps both premarket and prev-day levels distinct.
    bars = [
        make_bar(ts(8, 0), 501.0, h=501.8, lo=500.5),
        make_bar(ts(8, 30), 499.0, h=499.5, lo=498.2),
    ]
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    broker = StubBroker(bars, prev_day_high=503.5, prev_day_low=497.0)
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY", use_llm=False
    )
    broker.cursor = len(bars) - 1

    TradingSession(ctx).prep()

    path = tmp_path / f"journal-{bars[-1].ts.astimezone(ET).date().isoformat()}.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    events = [l for l in lines if l["type"] == "pre_market_prep"]
    assert len(events) == 1
    event = events[0]
    assert event["symbol"] == "SPY"
    labels = {lvl["label"] for lvl in event["levels"]}
    assert {"prev_day_high", "prev_day_low", "premarket_high", "premarket_low"} <= labels
    briefing = event["briefing"]
    assert briefing["bias"] == "neutral"
    assert all(w["tradable"] is False for w in briefing["watch_levels"])


def test_prep_skips_when_broker_has_no_bars(tmp_path):
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    broker = StubBroker([], prev_day_high=503.5, prev_day_low=497.0)
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY", use_llm=False
    )

    TradingSession(ctx).prep()  # must not raise

    assert not any(tmp_path.glob("journal-*.jsonl"))
