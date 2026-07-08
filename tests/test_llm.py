from tajator.llm.decide import decide_entry, decide_prep, decide_scale, format_prep_snapshot, format_snapshot
from tajator.market.indicators import build_snapshot
from tajator.models import Decision, Level, MorningBriefing, SetupCandidate

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


def test_entry_none_result_falls_back_to_wait():
    # with_structured_output returns None (not an exception) on a failed parse
    d = decide_entry(FakeLLM(decision=None), "snapshot")
    assert d.action == "wait"


def test_scale_none_result_falls_back_to_scale_out():
    d = decide_scale(FakeLLM(decision=None), "snapshot")
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


def test_format_prep_snapshot_contains_distance():
    bars = walk(ts(9, 0), [500.0, 500.5, 500.2, 499.8])
    snap = build_snapshot("SPY", bars)
    level = Level(price=497.0, kind="support", label="prev_day_low")
    text = format_prep_snapshot("SPY", snap, [level])
    assert "pre-market prep" in text
    assert "prev_day_low" in text
    assert "distance" in text


def test_prep_error_falls_back_to_deterministic_levels():
    level = Level(price=497.0, kind="support", label="prev_day_low")
    briefing = decide_prep(FakeLLM(error=TimeoutError("llm timeout")), "SPY", [level], "snapshot")
    assert isinstance(briefing, MorningBriefing)
    assert briefing.bias == "neutral"
    assert briefing.watch_levels[0].tradable is False
    assert briefing.watch_levels[0].level == level


def test_prep_none_result_falls_back_to_deterministic_levels():
    level = Level(price=497.0, kind="support", label="prev_day_low")
    briefing = decide_prep(FakeLLM(decision=None), "SPY", [level], "snapshot")
    assert isinstance(briefing, MorningBriefing)
    assert briefing.bias == "neutral"
    assert briefing.watch_levels[0].tradable is False


def test_prep_passes_through_valid_briefing():
    level = Level(price=497.0, kind="support", label="prev_day_low")
    good = MorningBriefing(
        symbol="SPY", bias="bullish",
        watch_levels=[{"level": level, "tradable": True, "direction": "call", "note": "clean"}],
        summary="watching prev-day low",
    )
    fake = FakeLLM(decision=good)
    briefing = decide_prep(fake, "SPY", [level], "snapshot")
    assert briefing == good
