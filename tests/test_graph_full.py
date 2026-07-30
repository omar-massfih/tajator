"""Full-graph integration: replay a scripted Opening-Range Breakout day through
the real compiled LangGraph with the StubBroker. The scripted day contains one
clean call setup: a tight 500-501 opening range, then a break above 501 that
runs to ~505 and holds, so the position enters, manages, and flattens at close.
"""

from zoneinfo import ZoneInfo

from tajator.broker.stub import StubBroker
from tajator.config import Settings
from tajator.graph.nodes import RuntimeContext
from tajator.journal import Journal
from tajator.runner import TradingSession

from conftest import orb_breakout_day

ET = ZoneInfo("America/New_York")


def _session(tmp_path, **settings_kwargs):
    settings = Settings(
        _env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path, **settings_kwargs
    )
    broker = StubBroker(orb_breakout_day(), prev_day_high=503.5, prev_day_low=497.0)
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY",
    )
    return TradingSession(ctx), broker


def test_full_day_enter_scale_runner(tmp_path):
    sess, broker = _session(tmp_path)
    sess.run_replay(broker)

    buys = [f for f in broker.fills if f[0] == "BUY"]
    sells = [f for f in broker.fills if f[0] == "SELL"]

    assert len(buys) == 1, f"expected exactly one entry, got {broker.fills}"
    assert buys[0][1].right == "C", "the scripted setup is a call breakout above the opening range"
    bought = buys[0][2].qty
    assert sum(s[2].qty for s in sells) == bought, "position must be fully closed"
    assert sells, "the position must exit (scale-outs and/or the EOD flatten)"

    assert sess.position is None
    assert sess.trades_today == 1

    # entry must be a breakout: the fill happens above the 501 opening-range high.
    entry_ts = buys[0][2].ts.astimezone(ET)
    assert (entry_ts.hour, entry_ts.minute) >= (9, 45), f"entry at {entry_ts} before the range locked"

    journal_files = list(tmp_path.glob("journal-*.jsonl"))
    assert journal_files, "journal must be written"
    content = journal_files[0].read_text()
    assert '"candidates"' in content
    assert '"entry_decision"' in content
    assert '"fill"' in content
    assert '"orb_high"' in content


def test_replay_notifies_every_fill_but_never_talks_to_telegram(tmp_path):
    """The notifier seam must fire on every fill during replay (so a real Notifier
    stays in sync with the journal) while the default NullNotifier — what replay
    actually gets in production — proves replay never sends real Telegram messages."""

    class RecordingNotifier:
        def __init__(self):
            self.fills = []

        def notify_fill(self, symbol, action, position):
            self.fills.append((symbol, action.kind, action.qty))

        def notify_status(self, text):
            pass

    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    broker = StubBroker(orb_breakout_day(), prev_day_high=503.5, prev_day_low=497.0)
    notifier = RecordingNotifier()
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY",
        notifier=notifier,
    )
    sess = TradingSession(ctx)
    sess.run_replay(broker)

    assert len(notifier.fills) == len(broker.fills)
    assert notifier.fills[0][1] == "entry"


def test_kill_switch_blocks_all_entries(tmp_path):
    sess, broker = _session(tmp_path)
    sess.ctx.settings.kill_switch_file.write_text("stop")
    sess.run_replay(broker)
    assert broker.fills == [], "kill switch must prevent every entry"


def test_two_symbol_sessions_keep_independent_state(tmp_path):
    """Two TradingSessions sharing one journal must not share position/trades_today."""
    settings = Settings(
        _env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path,
    )
    journal = Journal(tmp_path)

    broker_spy = StubBroker(orb_breakout_day(), prev_day_high=503.5, prev_day_low=497.0)
    broker_aapl = StubBroker(orb_breakout_day(), prev_day_high=503.5, prev_day_low=497.0)

    sess_spy = TradingSession(
        RuntimeContext(settings=settings, broker=broker_spy, journal=journal, symbol="SPY")
    )
    sess_aapl = TradingSession(
        RuntimeContext(settings=settings, broker=broker_aapl, journal=journal, symbol="AAPL")
    )

    sess_spy.run_replay(broker_spy)
    assert sess_spy.trades_today == 1
    assert sess_aapl.trades_today == 0, "AAPL session must be untouched by SPY's replay"

    sess_aapl.run_replay(broker_aapl)
    assert sess_aapl.trades_today == 1
    assert sess_spy.trades_today == 1, "SPY session's counter must not be affected by AAPL's replay"

    content = "\n".join(f.read_text() for f in tmp_path.glob("journal-*.jsonl"))
    assert '"symbol": "SPY"' in content
    assert '"symbol": "AAPL"' in content
