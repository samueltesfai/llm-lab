import json
import sys
from pathlib import Path

import pytest

from long_context import case_seed, generate, get_var, meeting_notes, server_inventory


def test_deterministic():
    assert meeting_notes(5, notes=20).messages == meeting_notes(5, notes=20).messages
    assert meeting_notes(5, notes=20).messages != meeting_notes(6, notes=20).messages


def test_case_seed_is_stable_and_configuration_specific():
    params = {"notes": 40, "position": "early"}
    assert case_seed(2026, "meeting_notes", params) == case_seed(2026, "meeting_notes", dict(reversed(params.items())))
    assert case_seed(2026, "meeting_notes", params) != case_seed(2026, "meeting_notes", {**params, "position": "late"})
    assert case_seed(2026, "meeting_notes", params) != case_seed(7, "meeting_notes", params)


def test_run_seed_comes_from_environment(monkeypatch):
    test_vars = {"generator": "meeting_notes", "params": {"notes": 10}}
    default = generate(test_vars)[0].grader_updates["expected"]
    monkeypatch.setenv("EVAL_SEED", "7")
    assert generate(test_vars)[0].grader_updates["expected"] != default


@pytest.mark.parametrize("position, expected_index", [("early", 5), ("middle", 20), ("late", 35), (0.25, 10)])
def test_positions(position, expected_index):
    assert meeting_notes(1, notes=40, position=position).metadata["needle_note_index"] == expected_index


def test_similar_distractors_are_distinct_codes():
    generated = meeting_notes(3, notes=40, similar_distractors=6)
    codes = generated.metadata["distractor_codes"]
    assert len(set(codes)) == 6 and generated.grader_updates["expected"] not in codes
    prompt = "\n".join(m["content"] for m in generated.messages)
    assert "Project Aurora Mobile (a separate project from Project Aurora)" in prompt
    assert prompt.count(generated.grader_updates["expected"]) == 1


def test_inventory_contains_near_misses_and_one_target():
    generated = server_inventory(4, rows=50)
    table = generated.messages[1]["content"]
    for host in ("| db-prod-17 |", "| db-prod-07 |", "| db-stage-17 |"):
        assert table.count(host) == 1
    assert table.count("\n| ") == 51   # header + 50 rows
    assert table.count(generated.grader_updates["expected"]) == 1


def test_invalid_parameters():
    with pytest.raises(ValueError):
        meeting_notes(1, notes=4)
    with pytest.raises(ValueError):
        meeting_notes(1, position="nowhere")


def test_get_var_returns_json_history_for_promptfoo():
    test_vars = {"generator": "server_inventory", "params": {"rows": 30}}
    output = get_var("messages", "", test_vars)["output"]
    assert isinstance(output, str)
    assert json.loads(output) == generate(test_vars)[0].messages


def test_reproduces_experiment_001_prompts():
    experiment = Path(__file__).resolve().parent.parent / "experiments" / "001_context_recall"
    sys.path.insert(0, str(experiment))
    try:
        from run import build_messages
    finally:
        sys.path.remove(str(experiment))
    messages, code = build_messages(90, "middle", 2026)
    generated = meeting_notes(2026, notes=90, position="middle")
    assert generated.messages == messages and generated.grader_updates["expected"] == code
