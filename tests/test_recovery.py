"""Journal replay: rebuilding session state from fill records after a crash."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tajator.journal import Journal
from tajator.models import ExecutedAction, OpenPosition, SelectedContract
from tajator.recovery import (
    journal_trades_today,
    last_fill_position,
    read_journal_events,
    recover_sessions,
)
from tajator.trade.position import build_plan

ET = ZoneInfo("America/New_York")
TODAY = date(2026, 7, 8)
NOW = datetime(2026, 7, 8, 10, 30, tzinfo=ET)


def make_position(qty=2, qty_remaining=None, pieces_sold=0, con_id=222, symbol="NVDA"):
    contract = SelectedContract(
        symbol=symbol, expiry="20260710", strike=200.0, right="P", con_id=con_id
    )
    plan = build_plan(
        direction="put", level_price=200.5, stop_price=200.9,
        entry_equity_price=200.2, entry_premium=2.97, qty=qty,
    )
    return OpenPosition(
        contract=contract, plan=plan,
        qty_remaining=qty if qty_remaining is None else qty_remaining,
        pieces_sold=pieces_sold, opened_at=NOW,
    )


def make_action(kind="entry", qty=1, ts=NOW):
    return ExecutedAction(kind=kind, qty=qty, premium=2.97, equity_price=200.2, ts=ts)


def write_fill(journal, position, kind="entry", ts=NOW, symbol="NVDA", qty=1):
    journal.write(
        "fill", ts=ts, symbol=symbol, action=make_action(kind, qty, ts), position=position
    )


# -- read_journal_events ------------------------------------------------------------


def test_truncated_last_line_is_skipped_with_warning(tmp_path):
    write_fill(Journal(tmp_path), make_position())
    path = tmp_path / f"journal-{TODAY.isoformat()}.jsonl"
    with path.open("a") as f:
        f.write('{"ts": "2026-07-')  # crash mid-append
    events, warnings = read_journal_events(tmp_path, TODAY)
    assert len(events) == 1
    assert len(warnings) == 1
    assert "line 2" in warnings[0] and "skipped" in warnings[0]


def test_events_span_day_files_oldest_first(tmp_path):
    journal = Journal(tmp_path)
    journal.write("startup_warning", ts=NOW - timedelta(days=3), marker="old")
    journal.write("startup_warning", ts=NOW, marker="new")
    events, warnings = read_journal_events(tmp_path, TODAY)
    assert [e["marker"] for e in events] == ["old", "new"]
    assert warnings == []  # missing day files in between are normal


def test_no_journal_files_is_empty_not_an_error(tmp_path):
    assert read_journal_events(tmp_path, TODAY) == ([], [])


def test_lookback_window_excludes_older_files(tmp_path):
    journal = Journal(tmp_path)
    journal.write("startup_warning", ts=NOW - timedelta(days=6), marker="too_old")
    journal.write("startup_warning", ts=NOW - timedelta(days=5), marker="oldest_kept")
    events, _ = read_journal_events(tmp_path, TODAY, lookback_days=5)
    assert [e["marker"] for e in events] == ["oldest_kept"]


# -- last_fill_position -------------------------------------------------------------


def test_last_fill_wins_and_round_trips_the_plan(tmp_path):
    journal = Journal(tmp_path)
    write_fill(journal, make_position(qty=3), kind="entry", ts=NOW, qty=3)
    later = NOW + timedelta(minutes=5)
    write_fill(
        journal, make_position(qty=3, qty_remaining=2, pieces_sold=1),
        kind="scale_out", ts=later,
    )
    events, _ = read_journal_events(tmp_path, TODAY)
    pos, ts = last_fill_position(events, "NVDA")
    assert pos.qty_remaining == 2 and pos.pieces_sold == 1
    assert pos.contract.con_id == 222
    assert pos.plan.stop_price == 200.9
    assert pos.plan.pieces and pos.plan.target_refs
    assert ts == later.isoformat()


def test_other_symbols_fills_are_ignored(tmp_path):
    journal = Journal(tmp_path)
    write_fill(journal, make_position(symbol="SPY"), symbol="SPY")
    events, _ = read_journal_events(tmp_path, TODAY)
    assert last_fill_position(events, "NVDA") == (None, None)


def test_unvalidatable_last_fill_degrades_to_none(tmp_path):
    journal = Journal(tmp_path)
    write_fill(journal, make_position())  # a good older fill must NOT be used
    journal.write(
        "fill", ts=NOW + timedelta(minutes=1), symbol="NVDA",
        action=make_action(), position={"garbage": True},
    )
    events, _ = read_journal_events(tmp_path, TODAY)
    assert last_fill_position(events, "NVDA") == (None, None)


# -- journal_trades_today -----------------------------------------------------------


def test_counts_todays_entries_and_failed_orders_only(tmp_path):
    journal = Journal(tmp_path)
    write_fill(journal, make_position(), kind="entry", ts=NOW - timedelta(days=1))  # yesterday
    write_fill(journal, make_position(), kind="entry", ts=NOW)
    write_fill(journal, make_position(), kind="entry", ts=NOW + timedelta(minutes=30))
    journal.write(
        "entry_order_failed", ts=NOW + timedelta(hours=1), symbol="NVDA", error="timeout"
    )
    write_fill(
        journal, make_position(qty_remaining=1, pieces_sold=1),
        kind="scale_out", ts=NOW + timedelta(hours=2),
    )
    write_fill(journal, make_position(symbol="SPY"), kind="entry", symbol="SPY")
    events, _ = read_journal_events(tmp_path, TODAY)
    assert journal_trades_today(events, "NVDA", TODAY) == 3


# -- recover_sessions ---------------------------------------------------------------


def test_fully_exited_position_recovers_as_flat(tmp_path):
    journal = Journal(tmp_path)
    write_fill(journal, make_position(qty=2), kind="entry", ts=NOW, qty=2)
    write_fill(
        journal, make_position(qty=2, qty_remaining=0, pieces_sold=0),
        kind="stop_exit", ts=NOW + timedelta(minutes=10), qty=2,
    )
    recovered, warnings = recover_sessions(tmp_path, ["NVDA"], TODAY)
    assert recovered["NVDA"].position is None
    assert recovered["NVDA"].trades_today == 1
    assert warnings == []


def test_recover_sessions_returns_position_and_provenance(tmp_path):
    write_fill(Journal(tmp_path), make_position(qty=2), kind="entry", ts=NOW, qty=2)
    recovered, _ = recover_sessions(tmp_path, ["NVDA"], TODAY)
    rec = recovered["NVDA"]
    assert rec.position.qty_remaining == 2
    assert rec.last_fill_ts == NOW.isoformat()
    assert rec.trades_today == 1


# -- replay isolation ----------------------------------------------------------------


def test_replay_journals_into_its_own_directory(tmp_path, monkeypatch):
    """Recovery trusts logs/journal-*.jsonl to be live-only, so `tajator
    replay` must journal its synthetic fills elsewhere."""
    import argparse

    import tajator.cli as cli
    import tajator.runner as runner
    from tajator.config import Settings

    settings = Settings(
        _env_file=None, kill_switch_file=tmp_path / "KILL",
        state_file=tmp_path / "state.json", log_dir=tmp_path / "logs", symbols=["NVDA"],
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    captured = {}

    class FakeSession:
        def __init__(self, ctx):
            captured["ctx"] = ctx

        def run_replay(self, stub):
            pass

    monkeypatch.setattr(runner, "TradingSession", FakeSession)
    csv_path = tmp_path / "bars.csv"
    csv_path.write_text(
        "ts,open,high,low,close,volume\n2026-07-08 09:30:00-04:00,100,100.5,99.5,100,1000\n"
    )
    args = argparse.Namespace(
        csv=csv_path, date=None, symbol=None, no_llm=True, prev_high=101.0, prev_low=99.0
    )
    cli.cmd_replay(args)
    assert captured["ctx"].journal.log_dir == settings.log_dir / "replays"
