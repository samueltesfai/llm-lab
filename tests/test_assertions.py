import pytest

import assertions
from assertions import clean_answer, normalize_short, strip_fence

GRADERS = {"exact": assertions.grade_exact, "regex_extract": assertions.grade_regex_extract,
           "number": assertions.grade_number, "choice": assertions.grade_choice,
           "abstain": assertions.grade_abstain}


def grade(answer, spec):
    return GRADERS[spec["type"]](clean_answer(answer), spec)


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


class TestChoice:
    @pytest.mark.parametrize("answer", ["C", "(C)", "C.", "The answer is C", "Answer: C", "C) uses a set", "**C**"])
    def test_extracts(self, answer):
        assert g(answer, type="choice", expected="C").status == "pass"

    def test_wrong_choice(self):
        assert g("B", type="choice", expected="C").status == "fail"

    def test_unclear_is_review(self):
        assert g("Probably the one with a set.", type="choice", expected="C").status == "review"


class TestAbstain:
    def test_marker(self):
        assert g("NOT IN SOURCE", type="abstain").status == "pass"

    def test_invented_detail_fails_even_with_marker(self):
        spec = {"type": "abstain", "forbidden": [r"\d+\s*liters"]}
        assert grade("NOT IN SOURCE, but probably 500 liters", spec).status == "fail"

    def test_marker_with_normalization(self):
        assert g("not in source.", type="abstain").status == "pass"
        assert g("`NOT IN SOURCE`", type="abstain").status == "pass"

    def test_marker_plus_claim_is_not_a_pass(self):
        result = g("NOT IN SOURCE. The operations manager is Alice.", type="abstain")
        assert result.status == "review" and "unsupported claims" in result.detail

    def test_abstention_phrase_without_marker_is_review(self):
        assert g("The document does not specify this.", type="abstain").status == "review"

    def test_answering_fails(self):
        assert g("The manager is Alice Smith.", type="abstain").status == "fail"


class TestPromptfooEntryPoints:
    def test_result_shape_and_review_prefix(self):
        assert assertions.number("2080", {"config": {"expected": 2080}}) == {"pass": True, "score": 1.0, "reason": "pass"}
        result = assertions.choice("not sure", {"config": {"expected": "C"}})
        assert result["pass"] is False and result["reason"].startswith("REVIEW:")

    def test_cut_off_answer_is_explained(self):
        context = {"config": {"expected": 1},
                   "providerResponse": {"metadata": {"metrics": {"done_reason": "length", "output_tokens": 2048}}}}
        assert assertions.number("no", context)["reason"].startswith("answer cut off at 2048 tokens")
        # a cut-off answer that can't be judged is a failure under cutoff: fail, not review
        reason = assertions.choice("unclear", {**context, "config": {"expected": "C"}})["reason"]
        assert reason.startswith("answer cut off") and "REVIEW" not in reason
        # an assertion config error stays an error even when the answer was cut off
        assert assertions.number("no", {**context, "config": {}})["reason"].startswith("ERROR:")

    def test_long_context_answer_regenerates_expected(self):
        import long_context

        test_vars = {"generator": "meeting_notes", "params": {"notes": 12, "position": "late"}}
        generated, _ = long_context.generate(test_vars)
        code = generated.grader_updates["expected"]
        assert assertions.long_context_answer(f"The code is {code}.", {"vars": test_vars})["pass"]
        assert not assertions.long_context_answer(f"{code} or MARIGOLD-00000", {"vars": test_vars})["pass"]

    def test_broken_assertion_config_is_an_error_not_a_failure(self):
        result = assertions.number("42", {"config": {}})              # missing 'expected'
        assert not result["pass"] and result["reason"].startswith("ERROR: assertion config problem")
        result = assertions.regex_extract("x", {"config": {"pattern": "(", "expected": "x"}})
        assert result["reason"].startswith("ERROR:")
        assert assertions.python_function("x", {"config": {}})["reason"].startswith("ERROR:")
        result = assertions.long_context_answer("x", {"vars": {"generator": "nope"}})
        assert result["reason"].startswith("ERROR: cannot regenerate") and "available" in result["reason"]

    def test_python_function_entry_point(self):
        context = {"config": {"function": "add", "tests": [{"args": [1, 2], "expected": 3}]}}
        assert assertions.python_function("```python\ndef add(a, b):\n    return a + b\n```", context)["pass"]
        assert assertions.python_function("no code", context)["reason"] == "no Python code found in answer"
