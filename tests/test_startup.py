"""Startup preflight: kill switch, resting orders, position reconciliation."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tajator.broker.base import BrokerOpenOrder, BrokerOptionPosition, StopCancelResult
from tajator.config import Settings
from tajator.journal import Journal
from tajator.models import ExecutedAction, OpenPosition, ProtectiveStop, SelectedContract
from tajator.notify import NullNotifier
from tajator.startup import (
    check_execution_diagnostics,
    check_kill_switch,
    reconcile_positions,
    run_startup_checks,
)
from tajator.state_store import PersistedSession, PersistedState, StateStore
from tajator.trade.position import build_plan

ET = ZoneInfo("America/New_York")
TODAY = date(2026, 7, 8)
NOW = datetime(2026, 7, 8, 8, 0, tzinfo=ET)


def make_settings(tmp_path, **kwargs):
    kwargs.setdefault("symbols", ["NVDA"])
    return Settings(
        _env_file=None,
        kill_switch_file=tmp_path / "KILL",
        state_file=tmp_path / "state.json",
        log_dir=tmp_path,
        **kwargs,
    )


def test_execution_diagnostic_gate_is_live_only(tmp_path):
    check_execution_diagnostics(make_settings(tmp_path), now=NOW)


def test_live_execution_requires_explicit_confirmation(tmp_path):
    settings = make_settings(tmp_path, trading_mode="live", ib_port=4001)
    with pytest.raises(SystemExit, match="EXECUTION_LIVE_CONFIRMED"):
        check_execution_diagnostics(settings, now=NOW)


def test_live_execution_requires_recent_pass_for_every_symbol(tmp_path):
    settings = make_settings(
        tmp_path, symbols=["NVDA", "AAPL"], trading_mode="live", ib_port=4001,
        execution_live_confirmed=True,
    )
    journal = Journal(tmp_path)
    journal.write("execution_diagnostic", ts=NOW, symbol="NVDA", passed=True)
    with pytest.raises(SystemExit, match="AAPL: no paper execution diagnostic"):
        check_execution_diagnostics(settings, now=NOW + timedelta(hours=1))
    journal.write("execution_diagnostic", ts=NOW, symbol="AAPL", passed=True)
    check_execution_diagnostics(settings, now=NOW + timedelta(hours=1))


def test_latest_failed_or_stale_execution_diagnostic_blocks_live(tmp_path):
    settings = make_settings(
        tmp_path, trading_mode="live", ib_port=4001, execution_live_confirmed=True,
    )
    journal = Journal(tmp_path)
    journal.write("execution_diagnostic", ts=NOW, symbol="NVDA", passed=True)
    journal.write(
        "execution_diagnostic", ts=NOW + timedelta(minutes=1), symbol="NVDA", passed=False,
    )
    with pytest.raises(SystemExit, match="latest diagnostic failed"):
        check_execution_diagnostics(settings, now=NOW + timedelta(hours=1))
    with pytest.raises(SystemExit, match="older than"):
        check_execution_diagnostics(settings, now=NOW + timedelta(days=8))


def make_position(qty=2, con_id=None, symbol="NVDA", strike=200.0):
    contract = SelectedContract(
        symbol=symbol, expiry="20260710", strike=strike, right="P", con_id=con_id
    )
    plan = build_plan(
        direction="put", level_price=200.5, stop_price=200.9,
        entry_equity_price=200.2, entry_premium=2.97, qty=qty,
    )
    return OpenPosition(contract=contract, plan=plan, qty_remaining=qty, opened_at=NOW)


def broker_pos(qty=2, con_id=222, symbol="NVDA", strike=200.0, right="P"):
    return BrokerOptionPosition(
        symbol=symbol, expiry="20260710", strike=strike, right=right,
        con_id=con_id, local_symbol=f"{symbol} 260710{right}00{strike:.0f}000",
        qty=qty, avg_cost=297.0,
    )


def persisted(position=None, trades_today=0, trading_day=TODAY, symbol="NVDA"):
    return PersistedState(
        updated_at=NOW, trading_day=trading_day,
        sessions={symbol: PersistedSession(position=position, trades_today=trades_today)},
    )


# -- kill switch -----------------------------------------------------------------


def test_kill_switch_file_refuses_launch(tmp_path):
    settings = make_settings(tmp_path)
    settings.kill_switch_file.write_text("partial fill: reconcile first")
    with pytest.raises(SystemExit) as exc_info:
        check_kill_switch(settings)
    assert "partial fill: reconcile first" in str(exc_info.value)


def test_kill_switch_file_notifies_on_launch_refusal(tmp_path):
    class RecordingNotifier:
        def __init__(self):
            self.statuses = []

        def notify_fill(self, symbol, action, position):
            pass

        def notify_status(self, text):
            self.statuses.append(text)

    settings = make_settings(tmp_path)
    settings.kill_switch_file.write_text("partial fill: reconcile first")
    notifier = RecordingNotifier()
    with pytest.raises(SystemExit):
        check_kill_switch(settings, notifier)
    assert notifier.statuses
    assert "KILL switch is ON" in notifier.statuses[0]
    assert "partial fill: reconcile first" in notifier.statuses[0]


def test_no_kill_switch_passes(tmp_path):
    check_kill_switch(make_settings(tmp_path))


# -- reconcile_positions (pure) ----------------------------------------------------


def test_exact_match_is_adopted():
    pos = make_position(qty=2)
    adopt, warnings, refusals = reconcile_positions(
        persisted(pos, trades_today=1), [broker_pos(qty=2)], ["NVDA"], TODAY
    )
    assert refusals == [] and warnings == []
    assert adopt["NVDA"].position is not None
    assert adopt["NVDA"].position.plan.stop_price == 200.9
    assert adopt["NVDA"].trades_today == 1


def test_broker_position_without_state_is_refused():
    adopt, _, refusals = reconcile_positions(None, [broker_pos(qty=5)], ["NVDA"], TODAY)
    assert len(refusals) == 1
    assert "+5x" in refusals[0]


def test_position_closed_externally_warns_and_starts_flat():
    pos = make_position(qty=2)
    adopt, warnings, refusals = reconcile_positions(persisted(pos), [], ["NVDA"], TODAY)
    assert refusals == []
    assert len(warnings) == 1 and "closed externally" in warnings[0]
    assert adopt["NVDA"].position is None


def test_qty_mismatch_is_refused():
    pos = make_position(qty=4)
    adopt, _, refusals = reconcile_positions(
        persisted(pos), [broker_pos(qty=2)], ["NVDA"], TODAY
    )
    assert len(refusals) == 1
    assert "changed externally" in refusals[0]


def test_different_contract_same_symbol_is_refused_and_state_warned():
    """Broker holds a different strike than the persisted plan: the held one is
    unexplained (refuse) and the persisted one is gone (warn)."""
    pos = make_position(qty=2)
    _, warnings, refusals = reconcile_positions(
        persisted(pos), [broker_pos(qty=2, strike=197.5)], ["NVDA"], TODAY
    )
    assert len(refusals) == 1
    assert len(warnings) == 1


def test_con_id_mismatch_is_refused():
    pos = make_position(qty=2, con_id=111)
    _, _, refusals = reconcile_positions(
        persisted(pos), [broker_pos(qty=2, con_id=222)], ["NVDA"], TODAY
    )
    assert len(refusals) == 1


def test_trades_today_survives_same_day_restart_only():
    same_day, _, _ = reconcile_positions(
        persisted(trades_today=2, trading_day=TODAY), [], ["NVDA"], TODAY
    )
    other_day, _, _ = reconcile_positions(
        persisted(trades_today=2, trading_day=date(2026, 7, 7)), [], ["NVDA"], TODAY
    )
    assert same_day["NVDA"].trades_today == 2
    assert other_day["NVDA"].trades_today == 0


# -- run_startup_checks -------------------------------------------------------------


def broker_order(
    order_id=901, order_ref="", action="SELL", qty=2, status="Submitted",
    con_id=222, symbol="NVDA", strike=200.0, right="P",
):
    return BrokerOpenOrder(
        order_id=order_id, order_ref=order_ref, action=action, qty=qty,
        order_type="MKT", status=status, con_id=con_id,
        local_symbol=f"{symbol} 260710{right}00{strike:.0f}000", symbol=symbol,
        expiry="20260710", strike=strike, right=right,
    )


class FakeBroker:
    def __init__(self, orders=(), positions=(), other=()):
        self.orders = list(orders)
        self.positions_ = list(positions)
        self.other = list(other)
        self.cancelled: list[int] = []
        self.placed_stops: list[ProtectiveStop] = []
        self.cancel_result = StopCancelResult(cancelled=True)

    def now(self):
        return NOW

    def open_option_orders_detailed(self, symbols):
        return self.orders

    def option_positions(self, symbols):
        return self.positions_

    def other_positions_summary(self, symbols):
        return self.other

    def place_protective_stop(self, contract, qty, stop_price, direction, order_ref):
        stop = ProtectiveStop(
            order_id=900 + len(self.placed_stops), order_ref=order_ref,
            qty=qty, stop_price=stop_price,
        )
        self.placed_stops.append(stop)
        return stop

    def cancel_protective_stop(self, contract, stop, expected_held=None):
        self.cancelled.append(stop.order_id)
        return self.cancel_result

    def poll_protective_stop(self, contract, stop):
        raise AssertionError("startup never polls")


def run_checks(tmp_path, broker, state=None, **settings_kwargs):
    settings = make_settings(tmp_path, **settings_kwargs)
    store = StateStore(settings.state_file)
    if state is not None:
        settings.state_file.write_text(state if isinstance(state, str) else state.model_dump_json())
    journal = Journal(tmp_path)
    return run_startup_checks(settings, broker, store, journal, NullNotifier()), settings


def test_resting_orders_refuse_startup(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        run_checks(tmp_path, FakeBroker(orders=[broker_order(action="BUY", order_ref="")]))
    assert "resting orders" in str(exc_info.value)
    assert "NVDA" in str(exc_info.value)


def test_corrupt_state_file_refuses_startup(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        run_checks(tmp_path, FakeBroker(), state="{not json")
    assert "state file" in str(exc_info.value)


def test_unexplained_position_refuses_startup(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        run_checks(tmp_path, FakeBroker(positions=[broker_pos(qty=5)]))
    assert "cannot explain" in str(exc_info.value)


def test_unrelated_positions_warn_but_do_not_block(tmp_path):
    broker = FakeBroker(other=["+100x STK TSLA"])
    adopted, settings = run_checks(tmp_path, broker)
    assert adopted["NVDA"].position is None
    content = "\n".join(f.read_text() for f in tmp_path.glob("journal-*.jsonl"))
    assert "startup_warning" in content and "TSLA" in content


def test_matching_position_is_adopted_and_reseeded(tmp_path):
    pos = make_position(qty=2)
    adopted, settings = run_checks(
        tmp_path, FakeBroker(positions=[broker_pos(qty=2)]), state=persisted(pos, trades_today=1)
    )
    assert adopted["NVDA"].position is not None
    assert adopted["NVDA"].trades_today == 1
    content = "\n".join(f.read_text() for f in tmp_path.glob("journal-*.jsonl"))
    assert "position_adopted" in content
    # the store must be seeded immediately, even before the first tick
    reloaded = StateStore(settings.state_file).load()
    assert reloaded.sessions["NVDA"].position.qty_remaining == 2


# -- protective stop reconciliation ---------------------------------------------------

STOP_REF = "tajator-stop:NVDA"


def test_own_stop_matching_position_is_adopted(tmp_path):
    pos = make_position(qty=2, con_id=222)
    broker = FakeBroker(
        orders=[broker_order(order_ref=STOP_REF, qty=2, con_id=222)],
        positions=[broker_pos(qty=2, con_id=222)],
    )
    adopted, _ = run_checks(
        tmp_path, broker, state=persisted(pos), protective_stop_enabled=True
    )
    stop = adopted["NVDA"].position.protective_stop
    assert stop is not None and stop.order_id == 901
    assert stop.stop_price == 200.9, "the adopted stop carries the plan's price"
    assert broker.cancelled == [] and broker.placed_stops == []


def test_adopted_position_without_stop_gets_one_replaced(tmp_path):
    pos = make_position(qty=2, con_id=222)
    broker = FakeBroker(positions=[broker_pos(qty=2, con_id=222)])
    adopted, _ = run_checks(
        tmp_path, broker, state=persisted(pos), protective_stop_enabled=True
    )
    assert adopted["NVDA"].position.protective_stop is not None
    assert len(broker.placed_stops) == 1
    assert "re-placed" in journal_content(tmp_path)


def test_orphan_own_stop_is_cancelled(tmp_path):
    broker = FakeBroker(orders=[broker_order(order_ref=STOP_REF)])
    adopted, _ = run_checks(tmp_path, broker, protective_stop_enabled=True)
    assert broker.cancelled == [901]
    assert adopted["NVDA"].position is None
    assert "stale protective stop" in journal_content(tmp_path)


def test_stop_qty_mismatch_is_cancelled_and_replaced(tmp_path):
    pos = make_position(qty=2, con_id=222)
    broker = FakeBroker(
        orders=[broker_order(order_ref=STOP_REF, qty=1, con_id=222)],
        positions=[broker_pos(qty=2, con_id=222)],
    )
    adopted, _ = run_checks(
        tmp_path, broker, state=persisted(pos), protective_stop_enabled=True
    )
    assert broker.cancelled == [901]
    assert len(broker.placed_stops) == 1
    assert adopted["NVDA"].position.protective_stop.qty == 2


def test_orphan_stop_cancel_revealing_fills_refuses_startup(tmp_path):
    broker = FakeBroker(orders=[broker_order(order_ref=STOP_REF)])
    broker.cancel_result = StopCancelResult(cancelled=True, filled_qty=1, avg_price=2.5)
    with pytest.raises(SystemExit) as exc_info:
        run_checks(tmp_path, broker, protective_stop_enabled=True)
    assert "protective stop reconciliation failed" in str(exc_info.value)


def test_disabled_still_cancels_own_stale_stops(tmp_path):
    broker = FakeBroker(orders=[broker_order(order_ref=STOP_REF)])
    run_checks(tmp_path, broker)  # PROTECTIVE_STOP off (default)
    assert broker.cancelled == [901], "toggling the feature off must not strand orders"


def test_position_gone_with_persisted_stop_warns_stop_fired_offline(tmp_path):
    pos = make_position(qty=2, con_id=222)
    pos.protective_stop = ProtectiveStop(
        order_id=901, order_ref=STOP_REF, qty=2, stop_price=200.9
    )
    adopted, _ = run_checks(
        tmp_path, FakeBroker(), state=persisted(pos), protective_stop_enabled=True
    )
    assert adopted["NVDA"].position is None
    assert "protective stop likely fired" in journal_content(tmp_path)


# -- journal-replay crash recovery ---------------------------------------------------


def make_action(kind="entry", qty=1, ts=NOW):
    return ExecutedAction(kind=kind, qty=qty, premium=2.97, equity_price=200.2, ts=ts)


def write_fill(tmp_path, position, kind="entry", ts=NOW, symbol="NVDA", qty=1):
    Journal(tmp_path).write(
        "fill", ts=ts, symbol=symbol, action=make_action(kind, qty, ts), position=position
    )


def journal_content(tmp_path):
    return "\n".join(f.read_text() for f in tmp_path.glob("journal-*.jsonl"))


def test_crash_window_position_recovered_from_journal(tmp_path):
    # crash between the entry fill (journaled) and the state.json write:
    # no persisted state at all, but today's journal explains the position
    write_fill(tmp_path, make_position(qty=2, con_id=222), qty=2)
    adopted, settings = run_checks(tmp_path, FakeBroker(positions=[broker_pos(qty=2, con_id=222)]))
    assert adopted["NVDA"].position is not None
    assert adopted["NVDA"].position.plan.stop_price == 200.9
    assert adopted["NVDA"].trades_today == 1
    content = journal_content(tmp_path)
    assert "position_recovered" in content and "position_adopted" not in content
    reloaded = StateStore(settings.state_file).load()
    assert reloaded.sessions["NVDA"].position.qty_remaining == 2


def test_stale_state_qty_mismatch_recovered_from_scale_out_fill(tmp_path):
    # crash between a scale-out fill and the persist: state.json still says 3
    pos3 = make_position(qty=3, con_id=222)
    write_fill(tmp_path, pos3, kind="entry", ts=NOW, qty=3)
    after = pos3.model_copy(update={"qty_remaining": 2, "pieces_sold": 1})
    write_fill(tmp_path, after, kind="scale_out", ts=NOW + timedelta(minutes=5))
    adopted, _ = run_checks(
        tmp_path,
        FakeBroker(positions=[broker_pos(qty=2, con_id=222)]),
        state=persisted(pos3, trades_today=1),
    )
    assert adopted["NVDA"].position.qty_remaining == 2
    assert adopted["NVDA"].position.pieces_sold == 1
    assert adopted["NVDA"].trades_today == 1
    assert "position_recovered" in journal_content(tmp_path)


def test_journal_flat_but_broker_holds_still_refuses(tmp_path):
    pos = make_position(qty=2, con_id=222)
    write_fill(tmp_path, pos, kind="entry", ts=NOW, qty=2)
    write_fill(
        tmp_path, pos.model_copy(update={"qty_remaining": 0}),
        kind="stop_exit", ts=NOW + timedelta(minutes=5), qty=2,
    )
    with pytest.raises(SystemExit) as exc_info:
        run_checks(tmp_path, FakeBroker(positions=[broker_pos(qty=2, con_id=222)]))
    assert "cannot explain" in str(exc_info.value)


def test_journal_broker_qty_mismatch_still_refuses(tmp_path):
    write_fill(tmp_path, make_position(qty=3, con_id=222), qty=3)
    with pytest.raises(SystemExit):
        run_checks(tmp_path, FakeBroker(positions=[broker_pos(qty=2, con_id=222)]))


def test_journal_fill_without_con_id_still_refuses(tmp_path):
    # replay/stub fills carry con_id null — the strict match must reject them
    write_fill(tmp_path, make_position(qty=2, con_id=None), qty=2)
    with pytest.raises(SystemExit):
        run_checks(tmp_path, FakeBroker(positions=[broker_pos(qty=2, con_id=222)]))


def test_journal_contract_mismatch_still_refuses(tmp_path):
    write_fill(tmp_path, make_position(qty=2, con_id=222, strike=197.5), qty=2)
    with pytest.raises(SystemExit):
        run_checks(tmp_path, FakeBroker(positions=[broker_pos(qty=2, con_id=222)]))


def test_overnight_crash_recovers_from_yesterdays_journal(tmp_path):
    yesterday = NOW - timedelta(days=1)
    write_fill(tmp_path, make_position(qty=2, con_id=222), ts=yesterday, qty=2)
    adopted, _ = run_checks(tmp_path, FakeBroker(positions=[broker_pos(qty=2, con_id=222)]))
    assert adopted["NVDA"].position is not None
    assert adopted["NVDA"].trades_today == 0  # yesterday's entry does not count


def test_recovered_trades_today_takes_the_max(tmp_path):
    # persisted (stale) knows 1 trade; the journal shows a failed order plus
    # the entry the broker holds — the conservative count wins
    Journal(tmp_path).write("entry_order_failed", ts=NOW, symbol="NVDA", error="timeout")
    pos = make_position(qty=2, con_id=222)
    write_fill(tmp_path, pos, kind="entry", ts=NOW + timedelta(minutes=5), qty=2)
    stale = make_position(qty=3, con_id=222)
    adopted, _ = run_checks(
        tmp_path,
        FakeBroker(positions=[broker_pos(qty=2, con_id=222)]),
        state=persisted(stale, trades_today=1),
    )
    assert adopted["NVDA"].trades_today == 2


def test_corrupt_trailing_journal_line_does_not_block_recovery(tmp_path):
    write_fill(tmp_path, make_position(qty=2, con_id=222), qty=2)
    path = tmp_path / f"journal-{TODAY.isoformat()}.jsonl"
    with path.open("a") as f:
        f.write('{"ts": "2026-07-')  # crash mid-append
    adopted, _ = run_checks(tmp_path, FakeBroker(positions=[broker_pos(qty=2, con_id=222)]))
    assert adopted["NVDA"].position is not None
    content = journal_content(tmp_path)
    assert "position_recovered" in content
    assert "startup_warning" in content and "unparseable" in content


def test_recovery_is_all_or_nothing_across_symbols(tmp_path):
    # NVDA is journal-recoverable, SPY is not — the launch must still refuse,
    # showing the first pass's diagnostics for both
    write_fill(tmp_path, make_position(qty=2, con_id=222), qty=2)
    spy = broker_pos(qty=1, con_id=333, symbol="SPY")
    with pytest.raises(SystemExit) as exc_info:
        run_checks(
            tmp_path,
            FakeBroker(positions=[broker_pos(qty=2, con_id=222), spy]),
            symbols=["NVDA", "SPY"],
        )
    assert "NVDA" in str(exc_info.value) and "SPY" in str(exc_info.value)
    assert "position_recovered" not in journal_content(tmp_path)
