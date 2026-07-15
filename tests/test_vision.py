import base64
import io

import pytest
from PIL import Image

from tajator.llm.decide import build_vision_llm
from tajator.llm.vision import (
    CHART_BARS,
    PATTERN_DIRECTIONS,
    VISION_SYSTEM_PROMPT,
    decide_vision,
    render_bar_chart,
    validate_vision_entry,
    vision_messages,
)
from tajator.models import VisionPatternAnalysis
from tajator.config import Settings
from tajator.market.indicators import build_snapshot

from conftest import ts, walk


class FakeVisionLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.result


def test_render_bar_chart_is_valid_png_and_limits_lookback():
    bars = walk(ts(9, 30), [100 + index * 0.02 for index in range(CHART_BARS + 10)])

    chart = render_bar_chart("AAPL", bars)

    assert chart.png.startswith(b"\x89PNG\r\n\x1a\n")
    assert chart.bar_count == CHART_BARS
    assert len(chart.sha256) == 64
    with Image.open(io.BytesIO(chart.png)) as image:
        assert image.size == (chart.width, chart.height)
        assert image.format == "PNG"


def test_vision_message_contains_inline_png_and_fixed_catalog():
    chart = render_bar_chart("AAPL", walk(ts(9, 30), [100.0, 100.2, 100.1]))

    messages = vision_messages("AAPL current price 100.10", chart)

    assert "double top" in messages[0]["content"]
    image_block = messages[1]["content"][1]
    assert image_block["type"] == "image"
    assert image_block["mime_type"] == "image/png"
    assert base64.b64decode(image_block["base64"]) == chart.png


def test_vision_catalog_has_symmetric_direction_mapping():
    assert PATTERN_DIRECTIONS["double_bottom"] == "call"
    assert PATTERN_DIRECTIONS["inverse_head_and_shoulders"] == "call"
    assert PATTERN_DIRECTIONS["double_top"] == "put"
    assert PATTERN_DIRECTIONS["head_and_shoulders"] == "put"
    assert "not guaranteed edges" in VISION_SYSTEM_PROMPT


def test_decide_vision_returns_structured_analysis():
    result = VisionPatternAnalysis(
        action="enter_call",
        pattern="double_bottom",
        status="confirmed",
        confidence=0.88,
        breakout_price=101.0,
        invalidation_price=99.8,
        evidence=["two lows", "close above neckline"],
        reasoning="confirmed double bottom",
    )
    fake = FakeVisionLLM(result=result)
    chart = render_bar_chart("AAPL", walk(ts(9, 30), [100.0, 100.2]))

    assert decide_vision(fake, "context", chart) == result
    assert fake.calls


@pytest.mark.parametrize("result", [None, TimeoutError("timeout")])
def test_decide_vision_fails_closed(result):
    fake = FakeVisionLLM(error=result if isinstance(result, Exception) else None, result=result)
    chart = render_bar_chart("AAPL", walk(ts(9, 30), [100.0, 100.2]))

    analysis = decide_vision(fake, "context", chart)

    assert analysis.action == "wait"
    assert analysis.pattern == "none"


def test_codex_cli_supports_vision_input():
    assert build_vision_llm("codex").output_model is VisionPatternAnalysis


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
    return VisionPatternAnalysis(**payload)


def test_validate_vision_entry_builds_normal_candidate_after_recent_breakout(tmp_path):
    bars = walk(ts(9, 30), [99.5] * 57 + [99.8, 100.0, 100.2])
    snapshot = build_snapshot("AAPL", bars)
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL")

    decision, candidate, violations = validate_vision_entry(
        _confirmed_call_analysis(), bars, snapshot, settings,
    )

    assert violations == []
    assert decision.action == "enter_call"
    assert decision.level_price == 100.0
    assert candidate is not None
    assert candidate.direction == "call"
    assert candidate.level.label == "vision_double_bottom"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"status": "forming"}, "not confirmed"),
        ({"confidence": 0.5}, "below 0.80 minimum"),
        ({"action": "enter_put"}, "maps to call"),
        ({"breakout_price": 98.0}, "outside the visible chart range"),
    ],
)
def test_validate_vision_entry_rejects_unconfirmed_or_inconsistent_reads(
    tmp_path, updates, message,
):
    bars = walk(ts(9, 30), [99.5] * 57 + [99.8, 100.0, 100.2])
    settings = Settings(_env_file=None, kill_switch_file=tmp_path / "KILL")

    decision, candidate, violations = validate_vision_entry(
        _confirmed_call_analysis(**updates), bars, build_snapshot("AAPL", bars), settings,
    )

    assert decision.action == "wait"
    assert candidate is None
    assert any(message in violation for violation in violations)
