import json

import pytest

from llm_eval import cli
from tests.fakes import FakeClient, response, unreachable


@pytest.fixture
def fake(monkeypatch):
    """Install a FakeClient in place of OllamaClient; returns a setter for its behavior."""
    holder = {}

    def install(answer, **kwargs):
        client = FakeClient(answer, **kwargs)
        monkeypatch.setattr(cli, "OllamaClient", lambda host, timeout: client)
        holder["client"] = client
        return client

    return install


def run(tmp_path, *args):
    return cli.main(["--model", "fake", "--results-dir", str(tmp_path), *args])


def only_run_dir(tmp_path):
    (run_dir,) = list(tmp_path.iterdir())
    return run_dir


def test_list_needs_no_server(capsys):
    assert cli.main(["--list", "--category", "programming"]) == 0
    assert "pg_trace_mutable_default" in capsys.readouterr().out


def test_full_run_writes_results_and_exit_code(tmp_path, fake, capsys):
    client = fake(lambda m, o: response("1 2 1 3"))
    code = run(tmp_path, "--case", "pg_trace_*", "--num-ctx", "4096", "--label", "unit")
    assert code == 1   # pg_trace_sort_key fails with this canned answer
    run_dir = only_run_dir(tmp_path)
    meta = json.loads((run_dir / "run.json").read_text())
    records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert meta["status"] == "completed" and meta["label"] == "unit"
    assert meta["config"]["num_ctx"] == 4096 and meta["environment"]["ollama_version"] == "0.0-test"
    assert meta["warmup"]["prompt_tokens"] == 1000
    assert {r["case_id"]: r["status"] for r in records} == {
        "pg_trace_mutable_default": "pass", "pg_trace_sort_key": "fail"}
    assert meta["summary"]["overall"]["pass"] == 1
    assert len(client.calls) == 3   # warm-up + 2 cases
    out = capsys.readouterr().out
    assert "TOTAL" in out and "Performance" in out


def test_all_pass_exit_zero_and_no_warmup(tmp_path, fake):
    client = fake(lambda m, o: response("1 2 1 3"))
    assert run(tmp_path, "--case", "pg_trace_mutable_default", "--no-warmup") == 0
    assert len(client.calls) == 1


def test_missing_model(tmp_path, fake, capsys):
    fake(lambda m, o: response("x"), models=["other:latest"])
    assert run(tmp_path) == cli.EXIT_SETUP_FAILED
    assert "ollama pull fake" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_server_unreachable(tmp_path, monkeypatch, capsys):
    class Down(FakeClient):
        def version(self):
            raise unreachable()

    monkeypatch.setattr(cli, "OllamaClient", lambda host, timeout: Down(lambda m, o: None))
    assert run(tmp_path) == cli.EXIT_SETUP_FAILED
    assert "Cannot use Ollama" in capsys.readouterr().err


def test_server_dies_mid_run(tmp_path, fake):
    calls = iter([response("OK"), response("1 2 1 3"), unreachable()])
    fake(lambda m, o: next(calls))
    assert run(tmp_path, "--case", "pg_trace_*") == 2
    meta = json.loads((only_run_dir(tmp_path) / "run.json").read_text())
    assert meta["status"] == "aborted"


def test_bad_option_syntax():
    with pytest.raises(SystemExit):
        cli.parse_args(["--option", "nonsense"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--option", "temperature=1"])
    assert cli.parse_args(["--option", "top_k=20", "--option", "stop=[\"x\"]"]).extra_options == {
        "top_k": 20, "stop": ["x"]}


def test_no_matching_cases(tmp_path, fake):
    fake(lambda m, o: response("x"))
    assert run(tmp_path, "--case", "does_not_exist") == cli.EXIT_SETUP_FAILED


def test_compare_reads_stored_runs(tmp_path, fake, capsys):
    from llm_eval import compare

    fake(lambda m, o: response("1 2 1 3"))
    run(tmp_path, "--case", "pg_trace_*", "--label", "first")
    capsys.readouterr()
    assert compare.main([str(tmp_path), "--cases"]) == 0
    out = capsys.readouterr().out
    assert "1/2" in out and "pg_trace_sort_key" in out and "(first)" in out
    assert compare.main([str(tmp_path / "missing")]) == 1
