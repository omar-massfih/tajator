from datetime import datetime
from zoneinfo import ZoneInfo

from tajator.broker.base import ChainParams
from tajator.broker.stub import StubBroker
from tajator.config import Settings
from tajator.market.indicators import build_snapshot
from tajator.models import Decision
from tajator.trade.contracts import select_contract
from tajator.trade.execution import execute_entry, execute_exit, execute_scale_out, size_entry

from conftest import ts, walk

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 7, 6, 11, 0, tzinfo=ET)  # Monday

CHAIN = ChainParams(
    expirations=["20260706", "20260710", "20260717"],  # today, this Friday, next Friday
    strikes=[498.0, 498.5, 499.0, 499.5, 500.0, 500.5],
)


def test_nearest_strike_and_no_0dte():
    c = select_contract(CHAIN, "SPY", 499.2, "call", NOW)
    assert c.strike == 499.0 and c.right == "C"
    assert c.expiry == "20260710"  # skips today's 0DTE expiry


def test_put_right_and_exact_midpoint_strike():
    c = select_contract(CHAIN, "SPY", 500.1, "put", NOW)
    assert c.right == "P" and c.strike == 500.0


def test_no_future_expiry_returns_none():
    chain = ChainParams(expirations=["20260706"], strikes=[499.0])
    assert select_contract(chain, "SPY", 499.0, "call", NOW) is None


def test_size_entry_respects_budget_and_max_contracts():
    settings = Settings(_env_file=None)  # $500 budget, 4 contracts max
    assert size_entry(1.0, settings) == 4  # $100 each -> capped by MAX_CONTRACTS
    assert size_entry(2.0, settings) == 2  # $200 each -> budget allows 2
    assert size_entry(6.0, settings) == 0  # $600 each -> unaffordable


def entry_setup():
    settings = Settings(_env_file=None)
    bars = walk(ts(9, 30), [500.0, 499.8, 499.5, 499.2])
    broker = StubBroker(bars)
    broker.cursor = len(bars) - 1
    snap = build_snapshot("SPY", bars)
    decision = Decision(action="enter_call", level_price=499.0, stop_price=498.6, reasoning="test")
    return settings, broker, snap, decision


def test_execute_entry_builds_position_and_plan():
    settings, broker, snap, decision = entry_setup()
    position, action, skip = execute_entry(broker, settings, decision, "call", snap)
    assert skip is None
    assert action.kind == "entry"
    assert position.contract.right == "C"
    assert abs(position.contract.strike - snap.price) <= 0.5
    assert position.plan.stop_price == 498.6
    assert position.qty_remaining == action.qty == sum(position.plan.pieces)
    assert position.plan.hod_at_entry == snap.hod
    assert position.plan.lod_at_entry == snap.lod
    assert broker.fills[0][0] == "BUY"


def test_scale_and_exit_bookkeeping():
    settings, broker, snap, decision = entry_setup()
    position, _, _ = execute_entry(broker, settings, decision, "call", snap)
    start_qty = position.qty_remaining

    [a] = execute_scale_out(broker, settings, position, snap, "ema9 target")
    assert a.kind == "scale_out"
    assert position.pieces_sold == 1
    assert position.qty_remaining == start_qty - a.qty
    assert position.profit_taken is True
    assert position.profit_lock_price is None  # RUNNER_STOP=breakeven default

    [a] = execute_exit(broker, settings, position, snap, "stop_exit", "stop hit")
    assert position.qty_remaining == 0
    assert a.qty == start_qty - position.plan.pieces[0]
    sells = [f for f in broker.fills if f[0] == "SELL"]
    assert sum(f[2].qty for f in sells) == start_qty


def test_first_target_runner_stop_locks_the_scale_price():
    _, broker, snap, decision = entry_setup()
    settings = Settings(_env_file=None, runner_stop="first_target")
    position, _, _ = execute_entry(broker, settings, decision, "call", snap)
    execute_scale_out(broker, settings, position, snap, "ema50 target")
    assert position.profit_lock_price == snap.price


class PartialFillBroker(StubBroker):
    """Fills at most `cap` contracts per order, like a reconciled partial fill."""

    def __init__(self, *args, cap=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap = cap

    def buy_option(self, contract, qty):
        return super().buy_option(contract, min(qty, self.cap))

    def sell_option(self, contract, qty):
        return super().sell_option(contract, min(qty, self.cap))


def test_partial_entry_fill_sizes_position_to_what_filled():
    settings, broker, snap, decision = entry_setup()
    partial = PartialFillBroker(broker.bars, cap=2)
    partial.cursor = broker.cursor
    position, action, skip = execute_entry(partial, settings, decision, "call", snap)
    assert skip is None
    assert action.qty == 2
    assert position.qty_remaining == 2
    assert position.plan.total_qty == 2, "the plan must cover the actual holding, not the request"
    assert sum(position.plan.pieces) == 2


def test_partial_exit_leaves_remainder_tracked():
    settings, broker, snap, decision = entry_setup()
    position, _, _ = execute_entry(broker, settings, decision, "call", snap)
    start_qty = position.qty_remaining

    partial = PartialFillBroker(broker.bars, cap=1)
    partial.cursor = broker.cursor
    [a] = execute_exit(partial, settings, position, snap, "stop_exit", "stop hit")
    assert a.qty == 1
    assert position.qty_remaining == start_qty - 1, "unsold contracts must stay tracked for retry"


def test_partial_scale_out_does_not_advance_piece_schedule():
    from tajator.models import OpenPosition
    from tajator.trade.position import build_plan

    settings, broker, snap, _ = entry_setup()
    contract = select_contract(CHAIN, "SPY", snap.price, "call", NOW)
    plan = build_plan(
        direction="call", level_price=499.0, stop_price=498.6,
        entry_equity_price=snap.price, entry_premium=1.7, qty=5,
    )
    assert plan.pieces[0] == 2, "5 contracts must front-load a 2-lot first piece"
    position = OpenPosition(contract=contract, plan=plan, qty_remaining=5, opened_at=NOW)

    partial = PartialFillBroker(broker.bars, cap=1)
    partial.cursor = broker.cursor
    [a] = execute_scale_out(partial, settings, position, snap, "ema9 target")
    assert a.qty == 1
    assert position.pieces_sold == 0, "a partially sold piece must be retried, not skipped"
    assert position.qty_remaining == 4
