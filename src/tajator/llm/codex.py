"""Codex CLI backend: use a ChatGPT/Codex subscription as the model, no API key.

Sign in once with `codex login`, then set LLM_MODEL=codex (or codex:<model>).
Each call shells out to

    codex exec --ephemeral --skip-git-repo-check --sandbox read-only \
        --output-schema <schema> --output-last-message <answer> "<prompt>"

from an empty scratch dir so repository files can't leak into the prompt, and in a
read-only sandbox so the model can touch nothing. This is used only as a text→JSON
oracle; the caller validates the result.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path


class CodexClient:
    """Minimal `prompt + JSON schema -> dict` oracle over the Codex CLI."""

    def __init__(self, model: str = "codex", *, binary: str = "codex", timeout_s: int = 45):
        # LLM_MODEL is "codex" or "codex:<model>"; split off the optional model.
        self.model = model.split(":", 1)[1] if ":" in model else None
        self.binary = binary
        self.timeout_s = timeout_s

    def complete(self, prompt: str, schema: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="tajator-codex-") as tmp:
            workdir = Path(tmp)
            schema_file = workdir / "schema.json"
            schema_file.write_text(json.dumps(schema))
            answer_file = workdir / "answer.json"
            cmd = [
                self.binary, "exec", "--ephemeral", "--skip-git-repo-check",
                "--sandbox", "read-only", "--color", "never",
                "--output-schema", str(schema_file),
                "--output-last-message", str(answer_file),
            ]
            if self.model:
                cmd += ["--model", self.model]
            cmd.append(prompt)
            result = subprocess.run(
                cmd, cwd=workdir, capture_output=True, text=True, timeout=self.timeout_s
            )
            if result.returncode != 0:
                raise RuntimeError(f"codex exec rc={result.returncode}: {result.stderr.strip()[:300]}")
            raw = answer_file.read_text().strip() if answer_file.exists() else result.stdout
        return _extract_json(raw)


def _extract_json(text: str) -> dict:
    """Parse the answer, tolerating prose or code fences around the JSON object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(match.group(0))
