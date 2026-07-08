"""StateStore persistence and the TradingSession hooks that feed it."""

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tajator.broker.stub import StubBroker
from tajator.config import Settings
from tajator.graph.nodes import RuntimeContext
from tajator.journal import Journal
from tajator.models import OpenPosition, SelectedContract
from tajator.runner import TradingSession
from tajator.state_store import PersistedSession, StateStore
from tajator.trade.position import build_plan

ET = ZoneInfo("America/New_York")
CSV = Path(__file__).parent / "data" / "spy_sample_day.csv"
NOW = datetime(2026, 7, 8, 11, 0, tzinfo=ET)
TODAY = date(2026, 7, 8)


def make_position(qty=2):
    contract = SelectedContract(symbol="SPY", expiry="20260710", strike=500.0, right="C")
    plan = build_plan(
        direction="call", level_price=499.2, stop_price=498.8,
        entry_equity_price=499.5, entry_premium=1.5, qty=qty,
    )
    return OpenPosition(contract=contract, plan=plan, qty_remaining=qty, opened_at=NOW)


def test_round_trip_preserves_the_position(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.update("SPY", make_position(qty=3), 1, TODAY)

    reloaded = StateStore(tmp_path / "state.json").load()
    sess = reloaded.sessions["SPY"]
    assert sess.trades_today == 1
    assert sess.position.qty_remaining == 3
    assert sess.position.plan.stop_price == 498.8
    assert sess.position.contract.local_name == "SPY 20260710 500C"
    assert reloaded.trading_day == TODAY


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.update("SPY", None, 0, TODAY)
    assert (tmp_path / "state.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_file_raises_on_load(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{definitely not json")
    with pytest.raises(Exception):
        StateStore(path).load()


def test_missing_file_loads_as_none(tmp_path):
    assert StateStore(tmp_path / "state.json").load() is None


def test_update_preserves_other_symbols(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.update("SPY", make_position(), 1, TODAY)
    store.update("NVDA", None, 2, TODAY)

    reloaded = StateStore(tmp_path / "state.json").load()
    assert reloaded.sessions["SPY"].position is not None
    assert reloaded.sessions["NVDA"].trades_today == 2


# -- TradingSession hooks ----------------------------------------------------------


def make_session(tmp_path, store=None, restored=None, day=None):
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    broker = StubBroker.from_csv(CSV, prev_day_high=503.5, prev_day_low=497.0)
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY", use_llm=False
    )
    return TradingSession(ctx, store=store, restored=restored, day=day), broker


def test_session_persists_through_a_replay_day(tmp_path):
    store = StateStore(tmp_path / "state.json")
    sess, broker = make_session(tmp_path, store=store)
    sess.run_replay(broker, verbose=False)

    assert sess.trades_today == 1, "the scripted day contains exactly one trade"
    reloaded = StateStore(tmp_path / "state.json").load()
    assert reloaded.sessions["SPY"].trades_today == 1
    assert reloaded.sessions["SPY"].position is None, "the day ends flat"


def test_session_without_store_never_writes(tmp_path):
    sess, broker = make_session(tmp_path)  # store=None — the replay/backtest path
    sess.run_replay(broker, verbose=False)
    assert list(tmp_path.glob("state.json*")) == []


def test_restored_state_survives_same_day_start_new_day(tmp_path):
    restored = PersistedSession(position=make_position(), trades_today=2)
    sess, broker = make_session(tmp_path, restored=restored, day=TODAY)
    assert sess.trades_today == 2
    assert sess.position is not None

    sess.start_new_day(TODAY)  # same-day restart: the runner's first pass
    assert sess.trades_today == 2, "a same-day reset would reopen the daily trade limit"

    sess.start_new_day(date(2026, 7, 9))  # actual new day
    assert sess.trades_today == 0
    assert sess.position is not None, "an overnight position keeps being managed"
