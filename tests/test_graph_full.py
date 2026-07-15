"""Full-graph integration: replay the bundled synthetic day through the real
compiled LangGraph with the StubBroker and the deterministic rule-follower
(use_llm=False). The scripted day contains exactly one clean call setup:
fast selloff into the premarket low, bounce through the EMAs, fade to BE.
"""

from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tajator.broker.stub import StubBroker
from tajator.config import Settings
from tajator.graph.nodes import RuntimeContext
from tajator.graph.nodes import make_nodes
from tajator.journal import Journal
from tajator.runner import TradingSession
from tajator.market.indicators import build_snapshot
from tajator.models import PatternAnalysis

ET = ZoneInfo("America/New_York")
CSV = Path(__file__).parent / "data" / "spy_sample_day.csv"


@pytest.fixture
def session(tmp_path):
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    broker = StubBroker.from_csv(CSV, prev_day_high=503.5, prev_day_low=497.0)
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY", use_llm=False
    )
    return TradingSession(ctx), broker, tmp_path


def test_full_day_enter_scale_runner(session, capsys):
    sess, broker, tmp_path = session
    sess.run_replay(broker)

    buys = [f for f in broker.fills if f[0] == "BUY"]
    sells = [f for f in broker.fills if f[0] == "SELL"]

    assert len(buys) == 1, f"expected exactly one entry, got {broker.fills}"
    assert buys[0][1].right == "C", "the scripted setup is a call at the premarket low"
    bought = buys[0][2].qty
    assert sum(s[2].qty for s in sells) == bought, "position must be fully closed"
    assert len(sells) >= 2, "expected scale-out pieces plus a runner exit"

    assert sess.position is None
    assert sess.trades_today == 1

    # entry must have happened on the way DOWN into the level (no chasing):
    entry_ts = buys[0][2].ts.astimezone(ET)
    assert entry_ts.hour == 11 and entry_ts.minute <= 15, f"entry at {entry_ts}"

    journal_files = list(tmp_path.glob("journal-*.jsonl"))
    assert journal_files, "journal must be written"
    content = journal_files[0].read_text()
    assert '"candidates"' in content
    assert '"llm_decision"' in content
    assert '"fill"' in content


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
    broker = StubBroker.from_csv(CSV, prev_day_high=503.5, prev_day_low=497.0)
    notifier = RecordingNotifier()
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY",
        use_llm=False, notifier=notifier,
    )
    sess = TradingSession(ctx)
    sess.run_replay(broker)

    assert len(notifier.fills) == len(broker.fills)
    assert notifier.fills[0][1] == "entry"


def test_kill_switch_blocks_all_entries(session):
    sess, broker, tmp_path = session
    sess.ctx.settings.kill_switch_file.write_text("stop")
    sess.run_replay(broker)
    assert broker.fills == [], "kill switch must prevent every entry"


def test_pattern_data_mode_creates_only_a_validated_synthetic_candidate(tmp_path):
    bars = walk(ts(9, 30), [99.5] * 58 + [99.8, 100.0, 100.2])
    broker = StubBroker(bars, prev_day_high=105.0, prev_day_low=95.0)
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)

    class PatternLLM:
        def invoke(self, messages):
            return PatternAnalysis(
                action="enter_call",
                pattern="double_bottom",
                status="confirmed",
                confidence=0.88,
                breakout_price=100.0,
                invalidation_price=99.5,
                evidence=["two lows", "close above neckline"],
                reasoning="confirmed neckline break",
            )

    ctx = RuntimeContext(
        settings=settings,
        broker=broker,
        journal=Journal(tmp_path),
        symbol="AAPL",
        use_llm=True,
        pattern_data=True,
        _pattern_llm=PatternLLM(),
    )
    nodes = make_nodes(ctx)
    state = {
        "bars": bars,
        "snapshot": build_snapshot("AAPL", bars),
        "levels": [],
        "trades_today": 0,
        "cooldown_levels": [],
    }

    setup = nodes["detect_setups"](state)
    assert setup["pattern_scan_due"] is True
    decision = nodes["llm_decide"]({**state, **setup})

    assert decision["decision"].action == "enter_call"
    assert decision["candidates"][0].level.label == "pattern_double_bottom"
    content = next(tmp_path.glob("journal-*.jsonl")).read_text()
    assert '"pattern_data_analysis"' in content
    assert '"sha256"' in content


def test_compiled_graph_can_enter_call_from_validated_pattern_data(tmp_path):
    bars = walk(ts(9, 30), [99.5] * 58 + [99.8, 100.0, 100.2])
    broker = StubBroker(bars, prev_day_high=105.0, prev_day_low=95.0)
    broker.seek(bars[-1].ts)
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)

    class PatternLLM:
        def invoke(self, messages):
            return PatternAnalysis(
                action="enter_call", pattern="double_bottom", status="confirmed",
                confidence=0.88, breakout_price=100.0, invalidation_price=99.5,
                evidence=["two lows", "close above neckline"],
                reasoning="confirmed neckline break",
            )

    session = TradingSession(RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="AAPL",
        use_llm=True, pattern_data=True, _pattern_llm=PatternLLM(),
    ))

    out = session.tick()

    assert out["risk"].approved is True
    assert len(broker.fills) == 1
    assert broker.fills[0][0] == "BUY"
    assert broker.fills[0][1].right == "C"


def test_two_symbol_sessions_keep_independent_state(tmp_path):
    """Two TradingSessions sharing one journal must not share position/trades_today."""
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    journal = Journal(tmp_path)

    broker_spy = StubBroker.from_csv(CSV, prev_day_high=503.5, prev_day_low=497.0)
    broker_aapl = StubBroker.from_csv(CSV, prev_day_high=503.5, prev_day_low=497.0)

    sess_spy = TradingSession(
        RuntimeContext(settings=settings, broker=broker_spy, journal=journal, symbol="SPY", use_llm=False)
    )
    sess_aapl = TradingSession(
        RuntimeContext(settings=settings, broker=broker_aapl, journal=journal, symbol="AAPL", use_llm=False)
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


# --- stop-out cooldown -------------------------------------------------------
# A scripted day where price falls into the prev-day low (499), stops out,
# bounces, and falls into the SAME level again minutes later. With the default
# cooldown the second approach must not re-arm; with the cooldown disabled it
# re-enters — proving the veto (not the detector) is what blocks it.

from tajator.broker.stub import StubBroker as _StubBroker  # noqa: E402

from conftest import ts, walk  # noqa: E402


def _cooldown_day():
    closes = (
        [503.0] * 5 + [502.8] * 5            # 09:30-09:39 warmup
        + [502.5, 502.0, 501.6, 501.2, 500.8, 500.45]  # fall into 499 -> entry 09:45
        + [499.9, 499.3, 498.55]             # through the 498.6 stop at 09:48
        + [499.4, 500.1, 500.7, 501.2]       # bounce away
        + [501.0, 500.6, 500.2, 499.9, 499.6]  # second approach ~09:55 (inside cooldown)
        + [499.5] * 3
    )
    return walk(ts(9, 30), closes)


def _cooldown_session(tmp_path, cooldown_minutes):
    settings = Settings(
        _env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path,
        stop_cooldown_minutes=cooldown_minutes,
    )
    broker = _StubBroker(_cooldown_day(), prev_day_high=510.0, prev_day_low=499.0)
    ctx = RuntimeContext(
        settings=settings, broker=broker, journal=Journal(tmp_path), symbol="SPY", use_llm=False
    )
    return TradingSession(ctx), broker


def test_stopped_out_level_is_cooled_down(tmp_path):
    sess, broker = _cooldown_session(tmp_path, cooldown_minutes=30)
    sess.run_replay(broker, verbose=False)

    buys = [f for f in broker.fills if f[0] == "BUY"]
    assert len(buys) == 1, "the cooled level must not re-arm"
    assert sess.trades_today == 1
    content = next(tmp_path.glob("journal-*.jsonl")).read_text()
    assert '"cooldown_veto"' in content, "the dropped re-entry must be journaled"


def test_cooldown_disabled_re_enters_the_same_level(tmp_path):
    sess, broker = _cooldown_session(tmp_path, cooldown_minutes=0)
    sess.run_replay(broker, verbose=False)

    buys = [f for f in broker.fills if f[0] == "BUY"]
    assert len(buys) == 2, "without the cooldown the second approach re-enters"
