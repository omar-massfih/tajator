"""Tests for the LLM daily level-planner and its merge into the level set."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tajator.graph.nodes import _merge_anchor_levels
from tajator.llm.codex import _extract_json
from tajator.llm.level_planner import BRIEFING_SCHEMA, build_prompt, plan_levels
from tajator.models import Bar, Level

ET = ZoneInfo("America/New_York")


class _FakeClient:
    def __init__(self, payload=None, exc=None):
        self.payload, self.exc = payload, exc
        self.prompt = None

    def complete(self, prompt, schema):
        self.prompt = prompt
        if self.exc:
            raise self.exc
        return self.payload


def _bars(prices, start=datetime(2026, 6, 15, 9, 20, tzinfo=ET)):
    return [Bar(ts=start + timedelta(minutes=i), open=p, high=p + 0.1, low=p - 0.1, close=p)
            for i, p in enumerate(prices)]


def test_plan_levels_parses_and_assigns_kind_by_spot():
    payload = {"bias": "neutral", "summary": "range day",
               "levels": [{"price": 95.0, "role": "support", "reason": "prior low"},
                          {"price": 110.0, "role": "resistance", "reason": "prior high"}]}
    client = _FakeClient(payload)
    out = plan_levels(client, "AAPL", daily=[], prev_high=111.0, prev_low=94.0,
                      bars=_bars([100.0] * 15), spot=100.0)
    assert [(l.price, l.kind) for l in out] == [(95.0, "support"), (110.0, "resistance")]
    assert all(l.label == "llm_level" for l in out)


def test_plan_levels_returns_empty_on_failure():
    # any exception from the model must fall back to [] (mechanical detection)
    out = plan_levels(_FakeClient(exc=RuntimeError("codex down")), "AAPL",
                      daily=[], prev_high=None, prev_low=None, bars=_bars([100.0] * 15), spot=100.0)
    assert out == []


def test_plan_levels_skips_nonpositive_prices():
    payload = {"bias": "bullish", "summary": "", "levels": [
        {"price": 0.0, "role": "support", "reason": "bad"},
        {"price": 101.0, "role": "resistance", "reason": "ok"}]}
    out = plan_levels(_FakeClient(payload), "AAPL", daily=[], prev_high=None, prev_low=None,
                      bars=_bars([100.0] * 15), spot=100.0)
    assert [l.price for l in out] == [101.0]


def test_prompt_includes_premarket_and_prior_day():
    # bars span 09:20-09:34 -> 10 premarket bars before 09:30
    client = _FakeClient({"bias": "neutral", "summary": "", "levels": []})
    plan_levels(client, "AAPL", daily=_bars([100.0] * 3), prev_high=111.0, prev_low=94.0,
                bars=_bars([100.0] * 15), spot=100.0)
    assert "premarket high" in client.prompt
    assert "Prior day high: 111.0" in client.prompt
    assert "AAPL" in client.prompt


def test_merge_prefers_anchors_and_dedupes_mechanical():
    anchors = [Level(price=100.0, kind="support", label="llm_level")]
    mech = [Level(price=100.05, kind="support", label="prev_day_low"),   # ~dupe of anchor
            Level(price=108.0, kind="resistance", label="premarket_high")]
    merged = _merge_anchor_levels(anchors, mech, price=104.0)
    prices = [l.price for l in merged]
    assert 100.0 in prices and 108.0 in prices and 100.05 not in prices
    # anchor kind recomputed vs live price (100 < 104 -> support)
    assert merged[0].kind == "support"


def test_briefing_schema_shape():
    assert BRIEFING_SCHEMA["required"] == ["bias", "levels", "summary"]


def test_extract_json_tolerates_prose_and_fences():
    assert _extract_json('here you go ```json\n{"a": 1}\n``` done')["a"] == 1
    assert _extract_json('{"b": 2}')["b"] == 2
