import json

import summary


def row(test_id, category, *, success=True, failure_reason=0, error=None, reason="", metrics=None):
    return {"testCase": {"description": test_id, "metadata": {"category": category}},
            "success": success, "failureReason": failure_reason, "error": error,
            "gradingResult": {"reason": reason},
            "response": {"metadata": {"model": "m", "think": False, "options": {"num_ctx": 4096},
                                      "metrics": metrics, "server": {}}} if metrics else None}


METRICS = {"decode_tokens_per_s": 100.0, "prompt_tokens_per_s": None, "wall_s": 1.0, "cold_load": False}


def write(tmp_path, rows, name="run.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"results": {"results": rows}}))
    return path


def test_statuses_and_exit_codes(tmp_path):
    rows = [row("a", "x", metrics=METRICS),
            row("b", "x", success=False, failure_reason=1, error="also set on failures", reason="wrong")]
    s = summary.summarize(write(tmp_path, rows))
    assert s["categories"]["x"]["pass"] == 1 and s["categories"]["x"]["fail"] == 1
    assert s["problems"] == [("fail", "b", "wrong")]
    assert summary.exit_status(s) == 1
    assert summary.exit_status(summary.summarize(write(tmp_path, rows[:1]))) == 0


def test_invalid_and_error_are_separated_and_exit_2(tmp_path):
    rows = [row("a", "y", success=False, failure_reason=2, error="INVALID [context_exceeded]: too long"),
            row("b", "y", success=False, failure_reason=2, error="ERROR [timeout]: timed out")]
    s = summary.summarize(write(tmp_path, rows))
    assert s["categories"]["y"]["invalid"] == 1 and s["categories"]["y"]["error"] == 1
    assert summary.exit_status(s) == 2


def test_assertion_errors_are_errors_not_failures(tmp_path):
    def errored(reason):
        r = row("a", "z", success=False, failure_reason=1)
        r["gradingResult"] = {"reason": reason, "componentResults": [{"pass": False, "reason": reason}]}
        return r
    s = summary.summarize(write(tmp_path, [errored("ERROR: assertion config problem (KeyError)"),
                                           errored("Error running Python script: KeyError")]))
    assert s["categories"]["z"]["error"] == 2 and s["categories"]["z"]["fail"] == 0
    assert summary.exit_status(s) == 2


def test_main_prints_comparison_for_several_files(tmp_path, capsys):
    first = write(tmp_path, [row("a", "x", metrics=METRICS)], "1.json")
    second = write(tmp_path, [row("a", "x", success=False, failure_reason=1, metrics=METRICS)], "2.json")
    assert summary.main([str(first), str(second)]) == 0
    out = capsys.readouterr().out
    assert "1/1" in out and "0/1" in out
