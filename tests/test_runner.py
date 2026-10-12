import re

import pytest

from llm_eval.ollama import OllamaError
from llm_eval.report import exit_code, summarize
from llm_eval.runner import RunConfig, run_case, run_suite
from llm_eval.scenarios import Case
from tests.fakes import FakeClient, response, unreachable


def make_case(case_id="c1", category="corrections", **kwargs):
    defaults = dict(purpose="p", source="t.yaml", messages=[{"role": "user", "content": "q"}],
                    grader={"type": "exact", "expected": "42"})
    defaults.update(kwargs)
    return Case(id=case_id, category=category, **defaults)


def run_one(answer, case=None, **config):
    client = FakeClient(answer)
    records = run_case(case or make_case(), client, RunConfig(model="fake", **config), "run")
    return records, client


def test_pass_and_record_contents():
    (record,), client = run_one(lambda m, o: response("42"))
    assert record["status"] == "pass"
    assert record["answer"] == "42" and record["expected"] == "42"
    assert record["metrics"]["prompt_tokens"] == 1000
    assert len(record["prompt_sha256_16"]) == 16
    assert client.calls[0]["truncate"] is False


def test_fail():
    (record,), _ = run_one(lambda m, o: response("41"))
    assert record["status"] == "fail" and "expected" in record["detail"]


def test_option_precedence():
    case = make_case(options={"num_predict": 64, "top_k": 5})
    (record,), client = run_one(lambda m, o: response("42"), case, num_ctx=8192, temperature=0.3,
                                seed=7, num_predict=100, extra_options={"top_k": 9})
    assert client.calls[0]["options"] == {"num_predict": 100, "top_k": 9, "num_ctx": 8192,
                                          "temperature": 0.3, "seed": 7}


def test_trials_vary_sampling_seed():
    records, client = run_one(lambda m, o: response("42"), trials=3, seed=10)
    assert [r["trial"] for r in records] == [0, 1, 2]
    assert [c["options"]["seed"] for c in client.calls] == [10, 11, 12]


def test_context_exceeded_is_invalid_not_fail():
    error = OllamaError("context_exceeded", "request (3016 tokens) exceeds the available context size")
    (record,), _ = run_one(lambda m, o: error)
    assert record["status"] == "invalid" and "num_ctx" in record["detail"]


def test_suspected_silent_truncation_is_invalid():
    case = make_case(messages=[{"role": "user", "content": "word " * 2000}])   # 10,000 chars
    (record,), _ = run_one(lambda m, o: response("41", prompt=900), case)
    assert record["status"] == "invalid" and record["truncation_suspected"]


def test_output_cut_off_by_num_predict_is_invalid_unless_passing():
    (record,), _ = run_one(lambda m, o: response("4", done_reason="length"))
    assert record["status"] == "invalid" and "num_predict" in record["detail"]
    (record,), _ = run_one(lambda m, o: response("42", done_reason="length"))
    assert record["status"] == "pass"


def test_cutoff_fail_policy():
    case = make_case(cutoff="fail")
    (record,), _ = run_one(lambda m, o: response("4", done_reason="length"), case)
    assert record["status"] == "fail" and "counted as failure" in record["detail"]
    (record,), _ = run_one(lambda m, o: response("4", done_reason="length"), case, think=True)
    assert record["status"] == "invalid"


def test_context_full_is_invalid():
    (record,), _ = run_one(lambda m, o: response("4", prompt=2040, output=10), num_ctx=2048)
    assert record["status"] == "invalid" and "context window" in record["detail"]


def test_timeout_and_http_errors_continue():
    (record,), _ = run_one(lambda m, o: OllamaError("timeout", "timed out"))
    assert record["status"] == "error" and record["error_kind"] == "timeout"


def test_min_ctx_skip_does_not_call_model():
    records, client = run_one(lambda m, o: response("42"), make_case(min_ctx=32768), num_ctx=16384)
    assert records[0]["status"] == "skipped" and not client.calls


def test_generator_cases_use_generated_expectation():
    case = make_case(messages=None, generator={"name": "meeting_notes", "params": {"notes": 10}},
                     grader={"type": "regex_extract", "pattern": r"MARIGOLD-\d{5}"})

    def answer(messages, options):
        prompt = "\n".join(m["content"] for m in messages)
        return response(re.search(r"Project Aurora is (MARIGOLD-\d{5})", prompt).group(1))

    (record,), _ = run_one(answer, case)
    assert record["status"] == "pass" and record["generator_seed"] is not None
    assert record["expected"].startswith("MARIGOLD-")


def test_thinking_is_recorded_and_not_graded():
    (record,), _ = run_one(lambda m, o: response("42", thinking="maybe 41?"), think=True)
    assert record["status"] == "pass" and record["thinking"] == "maybe 41?"


def test_unreachable_aborts_suite_and_keeps_records():
    cases = [make_case("a"), make_case("b"), make_case("c")]
    replies = iter([response("42"), unreachable(), response("42")])
    client = FakeClient(lambda m, o: next(replies))
    seen = []
    records, aborted = run_suite(cases, client, RunConfig(model="fake"), "run", seen.append)
    assert aborted and [r["status"] for r in records] == ["pass", "error"]
    assert seen == records


def test_summary_and_exit_codes():
    def rec(case_id, category, status, metrics=None):
        return {"case_id": case_id, "category": category, "status": status, "metrics": metrics}

    metrics = {"prompt_tokens": 100, "prompt_cached_tokens": 40, "output_tokens": 5, "wall_s": 1.0,
               "decode_tokens_per_s": 50.0, "prompt_tokens_per_s": None, "cold_load": False}
    records = [rec("a", "x", "pass", metrics), rec("b", "x", "fail", metrics), rec("c", "y", "review"),
               rec("d", "y", "skipped")]
    summary = summarize(records)
    assert summary["categories"]["x"]["pass_rate"] == 0.5
    assert summary["overall"]["pass_rate"] == pytest.approx(1 / 3, abs=0.001)
    assert summary["performance"]["prompt_cached_fraction"] == 0.4
    assert summary["performance"]["decode_tokens_per_s"]["median"] == 50.0
    assert exit_code(summary, aborted=False) == 1
    assert exit_code(summarize([rec("a", "x", "pass"), rec("c", "y", "review")]), False) == 0
    assert exit_code(summarize([rec("a", "x", "pass"), rec("b", "x", "invalid")]), False) == 2
    assert exit_code(summarize([rec("a", "x", "pass")]), aborted=True) == 2
