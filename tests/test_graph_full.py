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
from tajator.journal import Journal
from tajator.runner import TradingSession

ET = ZoneInfo("America/New_York")
CSV = Path(__file__).parent / "data" / "spy_sample_day.csv"


@pytest.fixture
def session(tmp_path):
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)
    broker = StubBroker.from_csv(CSV, prev_day_high=503.5, prev_day_low=497.0)
    ctx = RuntimeContext(settings=settings, broker=broker, journal=Journal(tmp_path), use_llm=False)
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


def test_kill_switch_blocks_all_entries(session):
    sess, broker, tmp_path = session
    sess.ctx.settings.kill_switch_file.write_text("stop")
    sess.run_replay(broker)
    assert broker.fills == [], "kill switch must prevent every entry"
