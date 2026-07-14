from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tajator.broker.stub import StubBroker
from tajator.broker.base import Fill
from tajator.config import Settings
from tajator.market.indicators import build_snapshot
from tajator.models import Decision, ExecutedAction, Level, OptionQuote, SetupCandidate
from tajator.trade.execution import execute_entry, validate_entry_preflight

from conftest import ts, walk

ET = ZoneInfo("America/New_York")


class GuardedStub(StubBroker):
    def __init__(self, bars, quote: OptionQuote, underlying: float):
        super().__init__(bars)
        self.quote = quote
        self.underlying = underlying
        self.preflights = []

    @property
    def uses_live_execution_guards(self):
        return True

    def get_option_quote(self, contract):
        return self.quote

    def get_underlying_price(self, symbol):
        return self.underlying

    def record_execution_preflight(self, **payload):
        self.preflights.append(payload)


def current_quote(bid=1.95, ask=2.0, *, delayed=False, age_s=0):
    return OptionQuote(
        bid=bid,
        ask=ask,
        last=1.98,
        ts=datetime.now(ET) - timedelta(seconds=age_s),
        delayed=delayed,
    )


def support_setup(underlying=499.2, quote=None):
    bars = walk(ts(9, 30), [500.2, 500.0, 499.6, 499.2])
    snapshot = build_snapshot("SPY", bars)
    level = Level(price=499.0, kind="support", label="prev_day_low")
    candidate = SetupCandidate(direction="call", level=level, distance=0.2, speed=-0.8)
    decision = Decision(
        action="enter_call", level_price=499.0, stop_price=498.6, reasoning="test"
    )
    broker = GuardedStub(bars, quote or current_quote(), underlying)
    broker.cursor = len(bars) - 1
    return broker, snapshot, candidate, decision


def test_guarded_entry_sizes_from_ask_plus_reserve():
    broker, snapshot, candidate, decision = support_setup()
    settings = Settings(_env_file=None, entry_budget_reserve_pct=0.05)
    position, action, skip = execute_entry(
        broker, settings, decision, "call", snapshot, candidate
    )
    assert skip is None
    assert action.qty == 2  # $2.00 ask * 1.05 * 2 * 100 = $420
    assert position.qty_remaining == 2
    assert broker.preflights[0]["accepted"] is True


@pytest.mark.parametrize(
    ("quote_kwargs", "reason"),
    [
        ({"bid": None, "ask": 2.0}, "valid positive bid/ask"),
        ({"bid": 2.1, "ask": 2.0}, "valid positive bid/ask"),
        ({"bid": 1.0, "ask": 2.0}, "spread"),
        ({"age_s": 10}, "stale"),
        ({"delayed": True}, "delayed"),
    ],
)
def test_bad_entry_quote_skips_without_placing_order(quote_kwargs, reason):
    quote = current_quote(**quote_kwargs)
    broker, snapshot, candidate, decision = support_setup(quote=quote)
    position, action, skip = execute_entry(
        broker, Settings(_env_file=None), decision, "call", snapshot, candidate
    )
    assert position is action is None
    assert reason in skip
    assert broker.fills == []
    assert broker.preflights[0]["accepted"] is False


@pytest.mark.parametrize(
    ("underlying", "reason"),
    [(498.5, "crossed call stop"), (500.0, "moved away"), (501.0, "moved away")],
)
def test_call_preflight_rejects_crossed_or_degraded_signal(underlying, reason):
    broker, snapshot, candidate, decision = support_setup(underlying=underlying)
    _, _, skip = execute_entry(
        broker, Settings(_env_file=None), decision, "call", snapshot, candidate
    )
    assert reason in skip
    assert broker.fills == []


def test_put_preflight_mirrors_favorable_drift_and_stop():
    snapshot = build_snapshot("SPY", walk(ts(9, 30), [499.0, 499.4, 499.8, 500.0]))
    candidate = SetupCandidate(
        direction="put",
        level=Level(price=500.2, kind="resistance", label="prev_day_high"),
        distance=0.2,
        speed=0.8,
    )
    settings = Settings(_env_file=None)
    quote = current_quote()
    assert "moved away" in validate_entry_preflight(
        quote, 499.0, snapshot, candidate, 500.6, settings
    )
    assert "crossed put stop" in validate_entry_preflight(
        quote, 500.7, snapshot, candidate, 500.6, settings
    )


def test_reserved_ask_can_skip_one_contract_before_market_order():
    broker, snapshot, candidate, decision = support_setup(
        quote=current_quote(bid=4.5, ask=4.8)
    )
    _, _, skip = execute_entry(
        broker, Settings(_env_file=None), decision, "call", snapshot, candidate
    )
    assert "reserves $504" in skip
    assert broker.fills == []


def test_entry_quote_request_failure_is_journaled_and_skipped():
    broker, snapshot, candidate, decision = support_setup()
    broker.get_option_quote = lambda contract: (_ for _ in ()).throw(RuntimeError("feed down"))
    _, _, skip = execute_entry(
        broker, Settings(_env_file=None), decision, "call", snapshot, candidate
    )
    assert "quote request failed" in skip
    assert broker.fills == []
    assert broker.preflights[0]["accepted"] is False


def test_entry_outside_approach_zone_is_skipped_even_with_large_drift_limit():
    broker, snapshot, candidate, decision = support_setup(underlying=500.6)
    settings = Settings(_env_file=None, max_entry_drift_min_cents=1000)
    _, _, skip = execute_entry(broker, settings, decision, "call", snapshot, candidate)
    assert "left support approach zone" in skip
    assert broker.fills == []


def test_old_fill_and_action_payloads_default_execution_quality_to_none():
    fill = Fill.model_validate({"premium": 2.0, "qty": 1, "ts": datetime.now(ET)})
    action = ExecutedAction.model_validate(
        {
            "kind": "entry", "qty": 1, "premium": 2.0,
            "equity_price": 499.2, "ts": datetime.now(ET),
        }
    )
    assert fill.execution_quality is None
    assert action.execution_quality is None
