"""Startup preflight: kill switch, resting orders, position reconciliation."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from tajator.broker.base import BrokerOptionPosition
from tajator.config import Settings
from tajator.journal import Journal
from tajator.models import OpenPosition, SelectedContract
from tajator.notify import NullNotifier
from tajator.startup import check_kill_switch, reconcile_positions, run_startup_checks
from tajator.state_store import PersistedSession, PersistedState, StateStore
from tajator.trade.position import build_plan

ET = ZoneInfo("America/New_York")
TODAY = date(2026, 7, 8)
NOW = datetime(2026, 7, 8, 8, 0, tzinfo=ET)


def make_settings(tmp_path, **kwargs):
    return Settings(
        _env_file=None,
        kill_switch_file=tmp_path / "KILL",
        state_file=tmp_path / "state.json",
        log_dir=tmp_path,
        symbols=["NVDA"],
        **kwargs,
    )


def make_position(qty=2, con_id=None, symbol="NVDA"):
    contract = SelectedContract(
        symbol=symbol, expiry="20260710", strike=200.0, right="P", con_id=con_id
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


class FakeBroker:
    def __init__(self, orders=(), positions=(), other=()):
        self.orders = list(orders)
        self.positions_ = list(positions)
        self.other = list(other)

    def now(self):
        return NOW

    def open_option_orders(self, symbols):
        return self.orders

    def option_positions(self, symbols):
        return self.positions_

    def other_positions_summary(self, symbols):
        return self.other


def run_checks(tmp_path, broker, state=None):
    settings = make_settings(tmp_path)
    store = StateStore(settings.state_file)
    if state is not None:
        settings.state_file.write_text(state if isinstance(state, str) else state.model_dump_json())
    journal = Journal(tmp_path)
    return run_startup_checks(settings, broker, store, journal, NullNotifier()), settings


def test_resting_orders_refuse_startup(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        run_checks(tmp_path, FakeBroker(orders=["BUY 1x NVDA 260710P00200000 (MKT, status Submitted)"]))
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
