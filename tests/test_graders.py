import pytest

from llm_eval.graders import GRADERS, clean_answer, grade, normalize_short, strip_fence


def g(answer, **spec):
    return grade(answer, spec)


class TestNormalization:
    def test_strip_fence_whole_answer_only(self):
        assert strip_fence("```json\n{\"a\": 1}\n```") == '{"a": 1}'
        assert strip_fence("text\n```x```") == "text\n```x```"

    def test_think_block_removed(self):
        assert clean_answer("<think>hmm</think>\n42") == "42"

    def test_normalize_short(self):
        assert normalize_short(" `Thursday.` ") == "thursday"
        assert normalize_short("A  b\n c ", case_sensitive=True) == "A b\nc"


class TestExact:
    def test_pass_with_light_normalization(self):
        assert g("  postgresql://u@h:1/db. ", type="exact", expected="postgresql://u@h:1/db").status == "pass"

    def test_multiline(self):
        assert g("```\nal,bo\n5\n```", type="exact", expected="al,bo\n5").status == "pass"

    def test_case_sensitive(self):
        assert g("status=ok", type="exact", expected="STATUS=OK", case_sensitive=True).status == "fail"
        assert g("status=ok", type="exact", expected="STATUS=OK").status == "pass"

    def test_extra_text_fails(self):
        assert g("The answer is 1 2 1 3", type="exact", expected="1 2 1 3").status == "fail"


class TestRegexExtract:
    spec = {"type": "regex_extract", "pattern": r"MARIGOLD-\d{5}", "expected": "MARIGOLD-12345"}

    def test_pass(self):
        assert grade("The code is MARIGOLD-12345.", self.spec).status == "pass"

    def test_wrong_value(self):
        result = grade("MARIGOLD-99999", self.spec)
        assert result.status == "fail" and "missing" in result.detail

    def test_no_match(self):
        assert grade("UNKNOWN", self.spec).status == "fail"

    def test_extra_candidates_review_by_default_and_fail_when_configured(self):
        answer = "MARIGOLD-12345 or maybe MARIGOLD-54321"
        assert grade(answer, self.spec).status == "review"
        assert grade(answer, {**self.spec, "extra": "fail"}).status == "fail"

    def test_set_expectation(self):
        spec = {"type": "regex_extract", "pattern": r"\b(?:Lena|Omar|Ji-woo)\b", "expected": ["Lena", "Omar"]}
        assert grade("Omar, Lena", spec).status == "pass"
        assert grade("Omar", spec).status == "fail"


class TestNumber:
    def test_pass_with_formatting(self):
        assert g("€2,080", type="number", expected=2080).status == "pass"
        assert g("-18 °C", type="number", expected=-18).status == "pass"

    def test_extra_numbers_review(self):
        assert g("1260 + 600 + 220 = 2080", type="number", expected=2080).status == "review"

    def test_wrong(self):
        assert g("2000", type="number", expected=2080).status == "fail"
        assert g("no idea", type="number", expected=1).status == "fail"

    def test_tolerance(self):
        assert g("3.14", type="number", expected=3.1416, tolerance=0.01).status == "pass"


class TestContains:
    def test_all_and_none(self):
        spec = {"type": "contains", "all": ["wednesday", "14:00"], "none": ["tuesday"]}
        assert grade("Wednesday at 14:00", spec).status == "pass"
        assert grade("Wednesday at 14:00 (moved from Tuesday)", spec).status == "fail"
        assert grade("Wednesday", spec).status == "fail"

    def test_any(self):
        spec = {"type": "contains", "any": ["postgres", "pg"]}
        assert grade("PostgreSQL", spec).status == "pass"
        assert grade("MySQL", spec).status == "fail"


class TestChoice:
    @pytest.mark.parametrize("answer", ["C", "(C)", "C.", "The answer is C", "Answer: C", "C) uses a set", "**C**"])
    def test_extracts(self, answer):
        assert g(answer, type="choice", expected="C").status == "pass"

    def test_wrong_choice(self):
        assert g("B", type="choice", expected="C").status == "fail"

    def test_unclear_is_review(self):
        assert g("Probably the one with a set.", type="choice", expected="C").status == "review"


class TestJson:
    schema = {
        "type": "object", "additionalProperties": False, "required": ["id", "tags"],
        "properties": {"id": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}},
    }

    def test_valid(self):
        result = g('{"id": 1, "tags": ["b", "a"]}', type="json", schema=self.schema,
                   expect={"id": 1, "tags": ["A", "B"]}, unordered_lists=True)
        assert result.status == "pass"

    def test_fenced_allowed_by_default(self):
        assert g('```json\n{"id": 1, "tags": []}\n```', type="json", schema=self.schema).status == "pass"

    def test_fence_rejected_when_disallowed(self):
        assert g('```json\n{"id": 1, "tags": []}\n```', type="json", allow_fences=False).status == "fail"

    def test_invalid_json(self):
        result = g("{id: 1}", type="json")
        assert result.status == "fail" and "invalid JSON" in result.detail

    def test_schema_violation_extra_field(self):
        result = g('{"id": 1, "tags": [], "email": "x"}', type="json", schema=self.schema)
        assert result.status == "fail" and "schema" in result.detail

    def test_wrong_type(self):
        assert g('{"id": "1", "tags": []}', type="json", schema=self.schema).status == "fail"

    def test_expected_value_mismatch(self):
        result = g('{"id": 2, "tags": []}', type="json", expect={"id": 1})
        assert result.status == "fail" and "id" in result.detail

    def test_ordered_list_root(self):
        expect = {"$": [{"name": "a"}, {"name": "b"}]}
        assert g('[{"name": "a"}, {"name": "b"}]', type="json", expect=expect).status == "pass"
        assert g('[{"name": "b"}, {"name": "a"}]', type="json", expect=expect).status == "fail"

    def test_bool_is_not_int(self):
        assert g('{"x": true}', type="json", expect={"x": 1}).status == "fail"

    def test_forbidden_keys_and_text(self):
        assert g('{"a": {"phone": "1"}}', type="json", forbidden_keys=["phone"]).status == "fail"
        assert g('{"a": "me@example.org"}', type="json", forbidden_text=["@"]).status == "fail"


class TestConstraints:
    spec = {"type": "constraints", "checks": [
        {"line_count": {"eq": 3}}, {"each_line_matches": r"^- \S"},
        {"each_line_max_words": 5}, {"not_contains": "cloud"}]}

    def test_pass(self):
        assert grade("- one two\n- three four\n- five six", self.spec).status == "pass"

    def test_reports_every_violation(self):
        result = grade("Intro\n- use the cloud now please always\n- b", self.spec)
        assert result.status == "fail"
        assert "lines=3" not in result.detail  # 3 lines is correct here
        assert "don't match" in result.detail and "exceed" in result.detail and "cloud" in result.detail

    def test_unknown_check_raises(self):
        with pytest.raises(ValueError):
            g("x", type="constraints", checks=[{"bogus": 1}])


class TestAbstain:
    def test_marker(self):
        assert g("NOT IN SOURCE", type="abstain").status == "pass"

    def test_invented_detail_fails_even_with_marker(self):
        spec = {"type": "abstain", "forbidden": [r"\d+\s*liters"]}
        assert grade("NOT IN SOURCE, but probably 500 liters", spec).status == "fail"

    def test_abstention_phrase_without_marker_is_review(self):
        assert g("The document does not specify this.", type="abstain").status == "review"

    def test_answering_fails(self):
        assert g("The manager is Alice Smith.", type="abstain").status == "fail"


def test_registry_required_keys_are_tuples():
    for name, (function, required) in GRADERS.items():
        assert callable(function) and isinstance(required, tuple), name
