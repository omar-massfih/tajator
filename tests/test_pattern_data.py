import pytest

from tajator.llm.decide import build_pattern_llm
from tajator.llm.pattern_data import (
    PATTERN_DATA_BARS,
    PATTERN_DIRECTIONS,
    PATTERN_SYSTEM_PROMPT,
    build_pattern_data,
    decide_pattern,
    pattern_messages,
    validate_pattern_entry,
)
from tajator.models import PatternAnalysis
from tajator.config import Settings
from tajator.market.indicators import build_snapshot

from conftest import ts, walk


class FakePatternLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.result


def test_build_pattern_data_serializes_ohlcv_and_limits_lookback():
    bars = walk(ts(9, 30), [100 + index * 0.02 for index in range(PATTERN_DATA_BARS + 10)])

    data = build_pattern_data("AAPL", bars)

    assert data.bar_count == PATTERN_DATA_BARS
    assert len(data.sha256) == 64
    assert "bars: index,time,open,high,low,close,volume,volume_vs_median" in data.text
    assert bars[-1].ts.strftime("%Y-%m-%dT%H:%M:%S%z") in data.text
    assert bars[0].ts.strftime("%Y-%m-%dT%H:%M:%S%z") not in data.text


def test_pattern_message_contains_numeric_data_pivots_and_fixed_catalog():
    bars = walk(ts(9, 30), [100.0, 99.8, 99.6, 99.8, 100.0, 100.2, 100.0])
    data = build_pattern_data("AAPL", bars, pivot_window=1)

    messages = pattern_messages("AAPL current price 100.00", data)

    assert "double top" in messages[0]["content"]
    assert "pivots:" in messages[1]["content"]
    assert "image" not in messages[1]["content"]
    assert data.pivot_count >= 1


def test_pattern_catalog_has_symmetric_direction_mapping():
    assert PATTERN_DIRECTIONS["double_bottom"] == "call"
    assert PATTERN_DIRECTIONS["inverse_head_and_shoulders"] == "call"
    assert PATTERN_DIRECTIONS["double_top"] == "put"
    assert PATTERN_DIRECTIONS["head_and_shoulders"] == "put"
    assert "not guaranteed edges" in PATTERN_SYSTEM_PROMPT


def test_decide_pattern_returns_structured_analysis():
    result = PatternAnalysis(
        action="enter_call",
        pattern="double_bottom",
        status="confirmed",
        confidence=0.88,
        breakout_price=101.0,
        invalidation_price=99.8,
        evidence=["two lows", "close above neckline"],
        reasoning="confirmed double bottom",
    )
    fake = FakePatternLLM(result=result)
    data = build_pattern_data("AAPL", walk(ts(9, 30), [100.0] * 7))

    assert decide_pattern(fake, "context", data) == result
    assert fake.calls


@pytest.mark.parametrize("result", [None, TimeoutError("timeout")])
def test_decide_pattern_fails_closed(result):
    fake = FakePatternLLM(error=result if isinstance(result, Exception) else None, result=result)
    data = build_pattern_data("AAPL", walk(ts(9, 30), [100.0] * 7))

    analysis = decide_pattern(fake, "context", data)

    assert analysis.action == "wait"
    assert analysis.pattern == "none"


def test_codex_cli_supports_pattern_input():
    assert build_pattern_llm("codex").output_model is PatternAnalysis


def _confirmed_call_analysis(**updates):
    payload = {
        "action": "enter_call",
        "pattern": "double_bottom",
        "status": "confirmed",
        "confidence": 0.88,
        "breakout_price": 100.0,
        "invalidation_price": 99.5,
        "evidence": ["two lows", "latest close broke the neckline"],
        "reasoning": "completed double bottom",
    }
    payload.update(updates)
    return PatternAnalysis(**payload)


def test_validate_pattern_entry_builds_normal_candidate_after_recent_breakout(tmp_path):
    bars = walk(ts(9, 30), [99.5] * 57 + [99.8, 100.0, 100.2])
    snapshot = build_snapshot("AAPL", bars)
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL")

    decision, candidate, violations = validate_pattern_entry(
        _confirmed_call_analysis(), bars, snapshot, settings,
    )

    assert violations == []
    assert decision.action == "enter_call"
    assert decision.level_price == 100.0
    assert candidate is not None
    assert candidate.direction == "call"
    assert candidate.level.label == "pattern_double_bottom"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"status": "forming"}, "not confirmed"),
        ({"confidence": 0.5}, "below 0.80 minimum"),
        ({"action": "enter_put"}, "maps to call"),
        ({"breakout_price": 98.0}, "outside the supplied data range"),
    ],
)
def test_validate_pattern_entry_rejects_unconfirmed_or_inconsistent_reads(
    tmp_path, updates, message,
):
    bars = walk(ts(9, 30), [99.5] * 57 + [99.8, 100.0, 100.2])
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL")

    decision, candidate, violations = validate_pattern_entry(
        _confirmed_call_analysis(**updates), bars, build_snapshot("AAPL", bars), settings,
    )

    assert decision.action == "wait"
    assert candidate is None
    assert any(message in violation for violation in violations)
