"""Regression for the 2026-07-08 duplicate-order incident.

A failing entry order used to escape the graph as a plain exception: the
runner swallowed it, trades_today never advanced, and the same setup fired
a fresh order every minute (five 1-lot orders in five minutes). A failed
entry attempt must now consume a trade so the daily limit brakes the loop
even if the kill switch somehow does not engage.
"""

from pathlib import Path

from tajator.broker.base import OrderFailed
from tajator.broker.stub import StubBroker
from tajator.config import Settings
from tajator.graph.nodes import RuntimeContext
from tajator.journal import Journal
from tajator.runner import TradingSession

CSV = Path(__file__).parent / "data" / "spy_sample_day.csv"


class FailingEntryBroker(StubBroker):
    """Every buy reaches IB but ends unfilled; optionally writes the kill
    switch, mimicking IBBroker's halt on failing entry orders."""

    def __init__(self, *args, kill_switch_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kill_switch_file = kill_switch_file
        self.buy_attempts = 0

    def buy_option(self, contract, qty):
        self.buy_attempts += 1
        if self.kill_switch_file is not None:
            self.kill_switch_file.write_text("entry order failed")
        raise OrderFailed(
            f"BUY {qty}x {contract.local_name} ended Cancelled (filled 0/{qty}). No contracts filled.",
            side="BUY", requested=qty, filled=0, suspect=False,
        )


def run_failing_day(tmp_path, with_kill_switch):
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    broker = FailingEntryBroker.from_csv(CSV, prev_day_high=503.5, prev_day_low=497.0)
    broker.kill_switch_file = settings.kill_switch_file if with_kill_switch else None
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY", use_llm=False
    )
    sess = TradingSession(ctx)
    sess.run_replay(broker, verbose=False)
    return settings, broker, sess, tmp_path


def test_failed_entries_consume_trades_and_stop_the_loop(tmp_path):
    """Kill switch disabled on purpose: the trade counter alone must brake."""
    settings, broker, sess, _ = run_failing_day(tmp_path, with_kill_switch=False)
    assert broker.buy_attempts >= 1, "the scripted setup must trigger at least one attempt"
    assert broker.buy_attempts <= settings.max_trades_per_day, (
        f"{broker.buy_attempts} orders sent — failed entries must count toward the daily limit"
    )
    assert sess.trades_today == broker.buy_attempts
    assert sess.position is None
    content = "\n".join(f.read_text() for f in tmp_path.glob("journal-*.jsonl"))
    assert '"entry_order_failed"' in content


def test_kill_switch_stops_after_first_failed_entry(tmp_path):
    settings, broker, sess, _ = run_failing_day(tmp_path, with_kill_switch=True)
    assert broker.buy_attempts == 1, "once the kill switch is written no further order may go out"
    assert sess.position is None


def test_failed_exit_keeps_position_and_keeps_retrying(tmp_path):
    """A SELL that ends unfilled must leave the position intact and be retried
    on later ticks (live, the runner swallows the error and re-ticks) — the
    session must never think it is flat."""

    class FailingExitBroker(StubBroker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sell_attempts = 0

        def sell_option(self, contract, qty):
            self.sell_attempts += 1
            raise OrderFailed(
                f"SELL {qty}x {contract.local_name} ended Cancelled (filled 0/{qty}). No contracts filled.",
                side="SELL", requested=qty, filled=0, suspect=False,
            )

    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    broker = FailingExitBroker.from_csv(CSV, prev_day_high=503.5, prev_day_low=497.0)
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY", use_llm=False
    )
    sess = TradingSession(ctx)

    # Same cadence as the live runner: a failed tick is swallowed and retried.
    from datetime import datetime, time
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    day = broker.bars[0].ts.astimezone(et).date()
    broker.seek(datetime.combine(day, time(9, 30), tzinfo=et))
    for _ in range(10):
        broker.advance()
    failures = 0
    while broker.now().astimezone(et).time() < time(16, 0):
        try:
            sess.tick()
        except OrderFailed:
            failures += 1
        if not broker.advance():
            break

    assert broker.sell_attempts >= 2, "every sell signal must retry the order"
    assert failures == broker.sell_attempts
    assert sess.position is not None
    assert sess.position.qty_remaining == sum(sess.position.plan.pieces)
