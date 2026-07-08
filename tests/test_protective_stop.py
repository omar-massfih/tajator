"""Broker-side protective stop: execution lifecycle, graph sync, persistence.

The invariant under test throughout: the agent never transmits a SELL while a
protective stop might still be working (cancel-and-confirm first), and fills
revealed by that cancel shrink the agent's own sell (the double-sell guard).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tajator.broker.base import StopCancelResult, StopStatus
from tajator.broker.stub import StubBroker
from tajator.config import Settings
from tajator.graph.nodes import RuntimeContext, make_nodes
from tajator.journal import Journal
from tajator.market.indicators import build_snapshot
from tajator.models import Decision, OpenPosition, ProtectiveStop, SelectedContract
from tajator.trade.execution import execute_entry, execute_exit, execute_scale_out
from tajator.trade.position import build_plan

from conftest import ts, walk

ET = ZoneInfo("America/New_York")


def make_settings(tmp_path, **kwargs):
    kwargs.setdefault("protective_stop_enabled", True)
    return Settings(
        _env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path, **kwargs
    )


def entry_setup(tmp_path, **settings_kwargs):
    settings = make_settings(tmp_path, **settings_kwargs)
    bars = walk(ts(9, 30), [500.0, 499.8, 499.5, 499.2])
    broker = StubBroker(bars)
    broker.cursor = len(bars) - 1
    snap = build_snapshot("SPY", bars)
    decision = Decision(action="enter_call", level_price=499.0, stop_price=498.6, reasoning="test")
    return settings, broker, snap, decision


# -- execution lifecycle -----------------------------------------------------------


def test_entry_places_stop_with_plan_price_and_ref(tmp_path):
    settings, broker, snap, decision = entry_setup(tmp_path)
    position, _, skip = execute_entry(broker, settings, decision, "call", snap)
    assert skip is None
    stop = position.protective_stop
    assert stop is not None
    assert stop.qty == position.qty_remaining
    assert stop.stop_price == 498.6
    assert stop.order_ref == "tajator-stop:SPY"
    assert broker.protective_stops[stop.order_id] == stop


def test_entry_survives_stop_placement_failure(tmp_path):
    class RejectingBroker(StubBroker):
        def place_protective_stop(self, *a, **kw):
            raise RuntimeError("IB rejected the order")

    settings, broker, snap, decision = entry_setup(tmp_path)
    rejecting = RejectingBroker(broker.bars)
    rejecting.cursor = broker.cursor
    position, _, skip = execute_entry(rejecting, settings, decision, "call", snap)
    assert skip is None
    assert position is not None, "the fill is real — a failed backstop must not orphan it"
    assert position.protective_stop is None


def test_disabled_places_no_stop(tmp_path):
    settings, broker, snap, decision = entry_setup(tmp_path, protective_stop_enabled=False)
    position, _, _ = execute_entry(broker, settings, decision, "call", snap)
    assert position.protective_stop is None
    assert broker.protective_stops == {}


def test_scale_out_is_cancel_then_sell_then_replace(tmp_path):
    settings, broker, snap, decision = entry_setup(tmp_path)
    position, _, _ = execute_entry(broker, settings, decision, "call", snap)
    first = position.protective_stop.order_id

    actions = execute_scale_out(broker, settings, position, snap, "ema9 target")
    assert [a.kind for a in actions] == ["scale_out"]
    assert broker.stop_calls == [("place", first), ("cancel", first), ("place", first + 1)]
    assert position.protective_stop.order_id == first + 1
    assert position.protective_stop.qty == position.qty_remaining, "resized to the remainder"


def test_exit_cancels_and_does_not_replace_at_zero(tmp_path):
    settings, broker, snap, decision = entry_setup(tmp_path)
    position, _, _ = execute_entry(broker, settings, decision, "call", snap)

    execute_exit(broker, settings, position, snap, "runner_exit", "done")
    assert position.qty_remaining == 0
    assert position.protective_stop is None
    assert broker.protective_stops == {}, "no stop may outlive the position"


def test_cancel_race_fill_shrinks_the_agents_sell(tmp_path):
    """The stop fired between ticks; its fill surfaces during the cancel. The
    agent must adopt it and sell only what is left — never the full count."""

    class RaceBroker(StubBroker):
        def cancel_protective_stop(self, contract, stop, expected_held=None):
            self.protective_stops.pop(stop.order_id, None)
            self.stop_calls.append(("cancel", stop.order_id))
            return StopCancelResult(cancelled=True, filled_qty=1, avg_price=2.50)

    settings, broker, snap, decision = entry_setup(tmp_path)
    racing = RaceBroker(broker.bars)
    racing.cursor = broker.cursor
    position, _, _ = execute_entry(racing, settings, decision, "call", snap)
    start_qty = position.qty_remaining

    actions = execute_exit(racing, settings, position, snap, "runner_exit", "target hit")
    assert [a.kind for a in actions] == ["stop_exit", "runner_exit"]
    assert actions[0].qty == 1 and actions[0].premium == 2.50
    assert actions[1].qty == start_qty - 1
    sells = [f for f in racing.fills if f[0] == "SELL"]
    assert sum(f[2].qty for f in sells) == start_qty - 1, "the raced contract must not be re-sold"
    assert position.qty_remaining == 0


def test_cancel_race_covering_everything_skips_the_sell(tmp_path):
    class FullRaceBroker(StubBroker):
        def cancel_protective_stop(self, contract, stop, expected_held=None):
            self.protective_stops.pop(stop.order_id, None)
            return StopCancelResult(cancelled=True, filled_qty=stop.qty, avg_price=2.10)

    settings, broker, snap, decision = entry_setup(tmp_path)
    racing = FullRaceBroker(broker.bars)
    racing.cursor = broker.cursor
    position, _, _ = execute_entry(racing, settings, decision, "call", snap)

    actions = execute_exit(racing, settings, position, snap, "runner_exit", "target hit")
    assert [a.kind for a in actions] == ["stop_exit"]
    assert not [f for f in racing.fills if f[0] == "SELL"]
    assert position.qty_remaining == 0


# -- graph: sync_protective_stop preamble ------------------------------------------


class PollBroker(StubBroker):
    """StubBroker whose poll/cancel results are scripted."""

    def __init__(self, *args, poll=None, cancel=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._poll = poll
        self._cancel = cancel

    def poll_protective_stop(self, contract, stop):
        return self._poll or super().poll_protective_stop(contract, stop)

    def cancel_protective_stop(self, contract, stop, expected_held=None):
        if self._cancel is not None:
            self.protective_stops.pop(stop.order_id, None)
            return self._cancel
        return super().cancel_protective_stop(contract, stop, expected_held)


def graph_setup(tmp_path, broker, qty=2, **settings_kwargs):
    settings = make_settings(tmp_path, **settings_kwargs)
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY", use_llm=False
    )
    nodes = make_nodes(ctx)
    contract = SelectedContract(symbol="SPY", expiry="20260710", strike=499.0, right="C")
    plan = build_plan(
        direction="call", level_price=499.0, stop_price=490.0,  # far below — never hit here
        entry_equity_price=499.2, entry_premium=1.7, qty=qty,
    )
    position = OpenPosition(
        contract=contract, plan=plan, qty_remaining=qty,
        opened_at=datetime(2026, 7, 6, 11, 0, tzinfo=ET),
        protective_stop=ProtectiveStop(
            order_id=77, order_ref="tajator-stop:SPY", qty=qty, stop_price=490.0
        ),
    )
    snap = build_snapshot("SPY", broker.get_bars("SPY"))
    state = {"position": position, "snapshot": snap, "bars": broker.get_bars("SPY"), "levels": []}
    return nodes, position, state, tmp_path


def make_poll_broker(**kwargs):
    bars = walk(ts(11, 0), [499.2, 499.4, 499.3])
    broker = PollBroker(bars, **kwargs)
    broker.cursor = len(bars) - 1
    return broker


def test_stop_fired_between_ticks_closes_the_position(tmp_path):
    broker = make_poll_broker(poll=StopStatus(state="filled", filled_qty=2, avg_price=1.10))
    nodes, position, state, tmp_path = graph_setup(tmp_path, broker)
    out = nodes["manage_position"](state)
    assert out["position"] is None
    assert out["manage_action"].kind == "broker_stop_filled"
    [action] = out["actions"]
    assert action.kind == "stop_exit" and action.qty == 2 and action.premium == 1.10
    content = "\n".join(f.read_text() for f in tmp_path.glob("journal-*.jsonl"))
    assert '"fill"' in content and "broker protective stop fired" in content


def test_partial_stop_fill_adopts_and_replaces_via_cancel(tmp_path):
    broker = make_poll_broker(
        poll=StopStatus(state="partial", filled_qty=1, avg_price=1.20, working_qty=1),
        cancel=StopCancelResult(cancelled=True, filled_qty=1, avg_price=1.20),
    )
    nodes, position, state, _ = graph_setup(tmp_path, broker)
    out = nodes["manage_position"](state)
    assert "position" not in out or out.get("position") is not None
    assert position.qty_remaining == 1
    assert position.protective_stop is not None, "the remainder must be re-protected"
    assert position.protective_stop.qty == 1
    assert position.protective_stop.order_id != 77, "old order was cancelled, a fresh one placed"


def test_stop_gone_externally_is_replaced(tmp_path):
    broker = make_poll_broker(poll=StopStatus(state="gone"))
    nodes, position, state, tmp_path = graph_setup(tmp_path, broker)
    out = nodes["manage_position"](state)
    assert out["manage_action"].kind == "hold"
    assert position.protective_stop is not None
    assert position.protective_stop.order_id != 77
    content = "\n".join(f.read_text() for f in tmp_path.glob("journal-*.jsonl"))
    assert "disappeared at the broker" in content


def test_missing_stop_is_placed_on_the_next_tick(tmp_path):
    broker = make_poll_broker()
    nodes, position, state, _ = graph_setup(tmp_path, broker)
    position.protective_stop = None  # e.g. placement failed at entry
    nodes["manage_position"](state)
    assert position.protective_stop is not None, "self-heals every tick"


# -- full replay ---------------------------------------------------------------------


def test_full_replay_with_stops_enabled_leaves_none_outstanding(tmp_path):
    """The bundled scripted day (entry → scale-outs → runner exit) must place
    and retire stops in lockstep and end with nothing resting."""
    from pathlib import Path

    from tajator.runner import TradingSession

    csv = Path(__file__).parent / "data" / "spy_sample_day.csv"
    settings = make_settings(tmp_path)
    broker = StubBroker.from_csv(csv, prev_day_high=503.5, prev_day_low=497.0)
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY", use_llm=False
    )
    sess = TradingSession(ctx)
    sess.run_replay(broker, verbose=False)

    assert sess.position is None
    assert broker.protective_stops == {}, "no stop may outlive the day"
    places = [c for c in broker.stop_calls if c[0] == "place"]
    cancels = [c for c in broker.stop_calls if c[0] == "cancel"]
    assert places and len(places) == len(cancels)


# -- persistence ---------------------------------------------------------------------


def test_old_state_format_without_protective_stop_loads():
    old = {
        "contract": {"symbol": "SPY", "expiry": "20260710", "strike": 499.0, "right": "C"},
        "plan": {
            "direction": "call", "level_price": 499.0, "stop_price": 498.6,
            "entry_equity_price": 499.2, "entry_premium": 1.7, "total_qty": 2,
            "pieces": [1, 1], "target_refs": ["ema9", "runner"],
        },
        "qty_remaining": 2,
        "opened_at": "2026-07-06T11:00:00-04:00",
    }
    position = OpenPosition.model_validate(old)
    assert position.protective_stop is None


def test_protective_stop_round_trips_through_state_json():
    stop = ProtectiveStop(order_id=9, perm_id=123, order_ref="tajator-stop:SPY", qty=2, stop_price=498.6)
    position = OpenPosition(
        contract=SelectedContract(symbol="SPY", expiry="20260710", strike=499.0, right="C"),
        plan=build_plan(
            direction="call", level_price=499.0, stop_price=498.6,
            entry_equity_price=499.2, entry_premium=1.7, qty=2,
        ),
        qty_remaining=2,
        opened_at=datetime(2026, 7, 6, 11, 0, tzinfo=ET),
        protective_stop=stop,
    )
    restored = OpenPosition.model_validate_json(position.model_dump_json())
    assert restored.protective_stop == stop
