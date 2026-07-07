from tajator.llm.decide import decide_entry, decide_scale, format_snapshot
from tajator.market.indicators import build_snapshot
from tajator.models import Decision, Level, SetupCandidate

from conftest import ts, walk


class FakeLLM:
    """Stands in for the structured-output chain: .invoke(messages) -> Decision."""

    def __init__(self, decision=None, error=None):
        self.decision, self.error = decision, error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.decision


def test_format_snapshot_contains_key_facts():
    bars = walk(ts(9, 30), [500.0, 499.8, 499.5, 499.2])
    snap = build_snapshot("SPY", bars)
    level = Level(price=499.0, kind="support", label="prev_day_low")
    cand = SetupCandidate(direction="call", level=level, distance=0.2, speed=-0.8,
                          note="falling into prev_day_low @ 499.0")
    text = format_snapshot(bars, snap, [level], [cand], trades_today=1)
    assert "price 499.20" in text
    assert "prev_day_low" in text
    assert "DETECTED SETUP CANDIDATES" in text
    assert "trades taken today: 1" in text


def test_entry_error_falls_back_to_wait():
    d = decide_entry(FakeLLM(error=TimeoutError("llm timeout")), "snapshot")
    assert d.action == "wait"


def test_scale_error_falls_back_to_scale_out():
    d = decide_scale(FakeLLM(error=TimeoutError("llm timeout")), "snapshot")
    assert d.action == "scale_out"


def test_scale_cannot_answer_enter():
    rogue = Decision(action="enter_call", level_price=1.0, stop_price=0.6, reasoning="x")
    d = decide_scale(FakeLLM(decision=rogue), "snapshot")
    assert d.action == "scale_out"


def test_entry_passes_through_valid_decision():
    good = Decision(action="enter_call", level_price=499.0, stop_price=498.6,
                    confidence="high", reasoning="fast drop into prev-day low")
    fake = FakeLLM(decision=good)
    d = decide_entry(fake, "snapshot")
    assert d == good
    system_msg = fake.calls[0][0]
    assert "NEVER chase" in system_msg["content"]
