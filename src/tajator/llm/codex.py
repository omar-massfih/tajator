"""Codex CLI backend: use a ChatGPT/Codex subscription as the decision LLM.

Set LLM_MODEL=codex (or codex:<model>, e.g. codex:gpt-5.3-codex) and sign in
once with `codex login` — no OpenAI API key needed. Each decision shells out to

    codex exec --ephemeral --skip-git-repo-check -s read-only \
        --output-schema <Decision schema> [-i <chart.png>] -o <answer file> "<prompt>"

run from an empty scratch directory so repository files/AGENTS.md can't leak
into the trading prompt. Codex is used purely as a model here; the read-only
sandbox means it cannot touch anything even if it tried.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

from ..models import Decision, VisionPatternAnalysis

log = logging.getLogger(__name__)

CODEX_TIMEOUT_S = 45  # ticks are 60s; a late answer is dropped, not queued

# Hand-written strict schemas (OpenAI structured-output style): every field
# required, no extras, optionals expressed as nullable.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["wait", "enter_call", "enter_put", "scale_out", "exit"],
        },
        "level_price": {"type": ["number", "null"]},
        "stop_price": {"type": ["number", "null"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "level_price", "stop_price", "confidence", "reasoning"],
    "additionalProperties": False,
}

_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "price": {"type": "number"},
        "kind": {"type": "string", "enum": ["support", "resistance"]},
        "label": {"type": "string"},
    },
    "required": ["price", "kind", "label"],
    "additionalProperties": False,
}

BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "bias": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "watch_levels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": _LEVEL_SCHEMA,
                    "tradable": {"type": "boolean"},
                    "direction": {"type": ["string", "null"], "enum": ["call", "put", None]},
                    "note": {"type": "string"},
                },
                "required": ["level", "tradable", "direction", "note"],
                "additionalProperties": False,
            },
        },
        "cleanest_level": {"type": ["number", "null"]},
        "summary": {"type": "string"},
    },
    "required": ["symbol", "bias", "watch_levels", "cleanest_level", "summary"],
    "additionalProperties": False,
}

VISION_PATTERN_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["wait", "enter_call", "enter_put"]},
        "pattern": {
            "type": "string",
            "enum": [
                "none", "double_top", "double_bottom", "head_and_shoulders",
                "inverse_head_and_shoulders", "triangle_breakout_up",
                "triangle_breakout_down",
            ],
        },
        "status": {"type": "string", "enum": ["none", "forming", "confirmed"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "breakout_price": {"type": ["number", "null"]},
        "invalidation_price": {"type": ["number", "null"]},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": [
        "action", "pattern", "status", "confidence", "breakout_price",
        "invalidation_price", "evidence", "reasoning",
    ],
    "additionalProperties": False,
}


class CodexDecider:
    """Drop-in for the langchain structured-output chain: .invoke(messages) -> output_model."""

    def __init__(
        self,
        model: str | None = None,
        binary: str = "codex",
        timeout_s: int = CODEX_TIMEOUT_S,
        output_model: type[BaseModel] = Decision,
        schema: dict = DECISION_SCHEMA,
    ):
        self.model = model
        self.binary = binary
        self.timeout_s = timeout_s
        self.output_model = output_model
        self._tmp = tempfile.TemporaryDirectory(prefix="tajator-codex-")  # cleaned on GC/exit
        self._workdir = Path(self._tmp.name)
        self._schema_file = self._workdir / "decision.schema.json"
        self._schema_file.write_text(json.dumps(schema))

    def invoke(self, messages: list[dict]) -> BaseModel:
        prompt_parts: list[str] = []
        image_files: list[Path] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                prompt_parts.append(content)
                continue
            for block in content:
                if block.get("type") == "text":
                    prompt_parts.append(block["text"])
                elif block.get("type") == "image":
                    mime_type = block.get("mime_type")
                    if mime_type != "image/png":
                        raise ValueError(f"Codex vision input requires image/png, got {mime_type}")
                    try:
                        payload = base64.b64decode(block["base64"], validate=True)
                    except (KeyError, ValueError) as exc:
                        raise ValueError("Codex vision input contains invalid base64") from exc
                    image_file = self._workdir / f"input-{len(image_files)}.png"
                    image_file.write_bytes(payload)
                    image_files.append(image_file)
        prompt = "\n\n".join(prompt_parts)
        answer_file = self._workdir / "answer.json"
        answer_file.unlink(missing_ok=True)

        cmd = [
            self.binary, "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--color", "never",
            "--output-schema", str(self._schema_file),
            "--output-last-message", str(answer_file),
        ]
        if self.model:
            cmd += ["--model", self.model]
        for image_file in image_files:
            cmd += ["--image", str(image_file)]
        cmd.append(prompt)

        result = subprocess.run(
            cmd, cwd=self._workdir, capture_output=True, text=True, timeout=self.timeout_s
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"codex exec failed (rc={result.returncode}): {result.stderr.strip()[:300]}"
            )

        raw = answer_file.read_text().strip() if answer_file.exists() else ""
        if not raw:
            raw = result.stdout
        return self.output_model.model_validate(_extract_json(raw))


def _extract_json(text: str) -> dict:
    """Parse the answer, tolerating prose or code fences around the JSON object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in codex output: {text[:200]!r}")
    return json.loads(match.group(0))
