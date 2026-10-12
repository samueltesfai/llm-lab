import re
from pathlib import Path

import pytest

from llm_eval import CATEGORIES
from llm_eval.generators import case_seed, meeting_notes, server_inventory
from llm_eval.scenarios import ScenarioError, load_scenarios, select_cases

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"


@pytest.fixture(scope="module")
def cases():
    return load_scenarios(SCENARIOS)


def test_repository_scenarios_load(cases):
    assert len({c.id for c in cases}) == len(cases)
    assert {c.category for c in cases} == set(CATEGORIES)


def test_every_category_has_several_standard_cases(cases):
    for category in CATEGORIES:
        standard = [c for c in cases if c.category == category and c.profile == "standard"]
        assert len(standard) >= 4, category


def test_every_case_builds_with_a_grader_expectation(cases):
    for case in cases:
        generated, _ = case.build(2026)
        assert generated.messages[-1]["role"] == "user"
        spec = {**case.grader, **generated.grader_updates}
        if spec["type"] in ("exact", "regex_extract", "number", "choice"):
            assert "expected" in spec, case.id


def test_generated_expected_answer_is_in_prompt(cases):
    for case in (c for c in cases if c.generator):
        generated, _ = case.build(7)
        prompt = "\n".join(m["content"] for m in generated.messages)
        assert prompt.count(generated.grader_updates["expected"]) == 1, case.id


def test_selection(cases):
    standard = select_cases(cases)
    assert all(c.profile == "standard" for c in standard)
    assert all(c.profile == "long" for c in select_cases(cases, profile="long"))
    assert len(select_cases(cases, profile="all")) == len(cases)
    only = select_cases(cases, categories=["corrections"])
    assert only and all(c.category == "corrections" for c in only)
    # explicit case patterns ignore the profile
    picked = select_cases(cases, patterns=["lc_long_*"])
    assert picked and all(c.id.startswith("lc_long_") for c in picked)


def write(tmp_path, text):
    (tmp_path / "x.yaml").write_text(text)
    return tmp_path


@pytest.mark.parametrize("body, message", [
    ("cases:\n  - {id: a, purpose: p, grader: {type: contains}, messages: [{role: user, content: hi}]}",
     "needs 'category'"),
    ("category: c\ncases:\n  - {id: a, purpose: p, grader: {type: nope}, messages: [{role: user, content: hi}]}",
     "grader.type"),
    ("category: c\ncases:\n  - {id: a, purpose: p, grader: {type: exact}, messages: [{role: user, content: hi}]}",
     "requires 'expected'"),
    ("category: c\ncases:\n  - {id: a, purpose: p, grader: {type: contains}, messages: [{role: assistant, content: hi}]}",
     "last message"),
    ("category: c\ncases:\n  - {id: a, purpose: p, grader: {type: contains}}", "exactly one of"),
    ("category: c\ncases:\n  - {id: a, purpose: p, grader: {type: contains}, generator: {name: nope}}",
     "unknown generator"),
    ("category: c\ncases:\n  - {id: a, purpose: p, grader: {type: contains, all: ['(']}, messages: [{role: user, content: hi}]}",
     "bad regex"),
    ("category: c\ncases:\n  - {id: a, purpose: p, grader: {type: contains}, options: {temperature: 1}, messages: [{role: user, content: hi}]}",
     "run-level"),
    ("category: c\ncases:\n  - {id: a, purpose: p, grader: {type: contains}, typo: 1, messages: [{role: user, content: hi}]}",
     "unknown keys"),
    ("category: c\ncases:\n  - {id: a, purpose: p, grader: {type: contains}, cutoff: maybe, messages: [{role: user, content: hi}]}",
     "cutoff must be"),
    ("category: c\ncases:\n  - {id: a, purpose: p, grader: {type: contains}, messages: [{role: user, content: hi}]}\n"
     "  - {id: a, purpose: p, grader: {type: contains}, messages: [{role: user, content: hi}]}", "duplicate id"),
])
def test_validation_errors(tmp_path, body, message):
    with pytest.raises(ScenarioError, match=re.escape(message)):
        load_scenarios(write(tmp_path, body))


def test_defaults_are_merged(tmp_path):
    body = ("category: c\ndefaults: {options: {num_predict: 9}, grader: {type: contains}}\n"
            "cases:\n  - {id: a, purpose: p, options: {top_k: 3}, messages: [{role: user, content: hi}]}")
    (case,) = load_scenarios(write(tmp_path, body))
    assert case.options == {"num_predict": 9, "top_k": 3}
    assert case.grader == {"type": "contains"}


class TestGenerators:
    def test_deterministic(self):
        assert meeting_notes(5, notes=20).messages == meeting_notes(5, notes=20).messages
        assert meeting_notes(5, notes=20).messages != meeting_notes(6, notes=20).messages

    def test_case_seed_is_stable_and_case_specific(self):
        assert case_seed(2026, "a") == case_seed(2026, "a") != case_seed(2026, "b")

    @pytest.mark.parametrize("position, expected_index", [("early", 5), ("middle", 20), ("late", 35), (0.25, 10)])
    def test_positions(self, position, expected_index):
        assert meeting_notes(1, notes=40, position=position).metadata["needle_note_index"] == expected_index

    def test_similar_distractors_are_distinct_codes(self):
        generated = meeting_notes(3, notes=40, similar_distractors=6)
        codes = generated.metadata["distractor_codes"]
        assert len(set(codes)) == 6 and generated.grader_updates["expected"] not in codes
        prompt = "\n".join(m["content"] for m in generated.messages)
        assert "Project Aurora Mobile (a separate project from Project Aurora)" in prompt

    def test_inventory_contains_near_misses(self):
        generated = server_inventory(4, rows=50)
        table = generated.messages[1]["content"]
        for host in ("| db-prod-17 |", "| db-prod-07 |", "| db-stage-17 |"):
            assert table.count(host) == 1
        assert table.count("\n| ") == 51   # header + 50 rows

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            meeting_notes(1, notes=4)
        with pytest.raises(ValueError):
            meeting_notes(1, position="nowhere")
