import json
import stat

import pytest

from tajator.llm.codex import CodexDecider, _extract_json
from tajator.llm.decide import build_llm, decide_entry

GOOD_JSON = json.dumps(
    {"action": "enter_call", "level_price": 499.0, "stop_price": 498.6,
     "confidence": "high", "reasoning": "fast drop into prev-day low"}
)


def fake_codex(tmp_path, *, answer=GOOD_JSON, rc=0, write_file=True):
    """A stand-in `codex` executable that honors --output-last-message."""
    script = tmp_path / "codex"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"answer = {answer!r}\n"
        "args = sys.argv[1:]\n"
        "if '--output-last-message' in args and " + str(write_file) + ":\n"
        "    out = args[args.index('--output-last-message') + 1]\n"
        "    open(out, 'w').write(answer)\n"
        "else:\n"
        "    print('thinking...')\n"
        "    print(answer)\n"
        f"sys.exit({rc})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


MESSAGES = [{"role": "system", "content": "rules"}, {"role": "user", "content": "snapshot"}]


def test_codex_decider_parses_structured_answer(tmp_path):
    decider = CodexDecider(binary=fake_codex(tmp_path))
    d = decider.invoke(MESSAGES)
    assert d.action == "enter_call" and d.stop_price == 498.6


def test_codex_falls_back_to_stdout_parsing(tmp_path):
    decider = CodexDecider(binary=fake_codex(tmp_path, write_file=False))
    d = decider.invoke(MESSAGES)
    assert d.action == "enter_call"


def test_codex_failure_bubbles_and_decide_entry_waits(tmp_path):
    decider = CodexDecider(binary=fake_codex(tmp_path, rc=1))
    with pytest.raises(RuntimeError):
        decider.invoke(MESSAGES)
    assert decide_entry(decider, "snapshot").action == "wait"


def test_extract_json_tolerates_fences_and_prose():
    noisy = "Here you go:\n```json\n" + GOOD_JSON + "\n```\nGood luck!"
    assert _extract_json(noisy)["action"] == "enter_call"
    with pytest.raises(ValueError):
        _extract_json("no json here")


def test_build_llm_routes_codex_strings():
    assert isinstance(build_llm("codex"), CodexDecider)
    decider = build_llm("codex:gpt-5.3-codex")
    assert isinstance(decider, CodexDecider) and decider.model == "gpt-5.3-codex"
