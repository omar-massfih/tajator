import json
from datetime import datetime
from zoneinfo import ZoneInfo

from tajator.journal import Journal
from tajator.models import Decision

ET = ZoneInfo("America/New_York")


def test_journal_appends_jsonl_with_models(tmp_path):
    journal = Journal(tmp_path)
    ts = datetime(2026, 7, 6, 11, 0, tzinfo=ET)
    decision = Decision(action="wait", reasoning="nothing setting up")
    journal.write("llm_decision", ts=ts, decision=decision)
    journal.write("fill", ts=ts, qty=1, premium=2.5)

    path = tmp_path / "journal-2026-07-06.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["type"] == "llm_decision"
    assert lines[0]["decision"]["action"] == "wait"
    assert lines[1]["qty"] == 1
