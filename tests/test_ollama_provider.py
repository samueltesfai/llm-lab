import io
import json
import socket
from urllib.error import HTTPError, URLError

import pytest

import ollama_provider
from ollama_provider import OllamaClient, OllamaError, extract_metrics
from tests.fakes import FakeClient, response, unreachable


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def patch_urlopen(monkeypatch, behavior):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["payload"] = json.loads(request.data) if request.data else None
        captured["url"] = request.full_url
        if isinstance(behavior, Exception):
            raise behavior
        return FakeHTTPResponse(json.dumps(behavior).encode())

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)
    return captured


def http_error(code, body):
    return HTTPError("http://x/api/chat", code, "err", {}, io.BytesIO(json.dumps(body).encode()))


def test_chat_payload_and_result(monkeypatch):
    captured = patch_urlopen(monkeypatch, response("hi"))
    body, _ = OllamaClient("http://h:1/").chat("m", [{"role": "user", "content": "x"}], {"num_ctx": 2048})
    assert body["message"]["content"] == "hi"
    payload = captured["payload"]
    assert captured["url"] == "http://h:1/api/chat"
    assert payload["stream"] is False and payload["truncate"] is False
    assert payload["options"] == {"num_ctx": 2048}


def test_truncate_field_omitted_when_allowed(monkeypatch):
    captured = patch_urlopen(monkeypatch, response("hi"))
    OllamaClient("http://x").chat("m", [], {}, truncate=True)
    assert "truncate" not in captured["payload"]


@pytest.mark.parametrize("error, kind", [
    (http_error(400, {"error": "request (3016 tokens) exceeds the available context size (2048 tokens)"}),
     "context_exceeded"),
    (http_error(404, {"error": "model 'nope' not found"}), "model_missing"),
    (http_error(500, {"error": "boom"}), "http"),
    (URLError(ConnectionRefusedError(111, "refused")), "unreachable"),
    (URLError(socket.timeout("timed out")), "timeout"),
    (TimeoutError("timed out"), "timeout"),
])
def test_error_classification(monkeypatch, error, kind):
    patch_urlopen(monkeypatch, error)
    with pytest.raises(OllamaError) as info:
        OllamaClient("http://x").chat("m", [], {})
    assert info.value.kind == kind


def test_missing_message_is_bad_response(monkeypatch):
    patch_urlopen(monkeypatch, {"done": True})
    with pytest.raises(OllamaError) as info:
        OllamaClient("http://x").chat("m", [], {})
    assert info.value.kind == "bad_response"


def test_metrics_prompt_throughput_only_without_cache():
    metrics = extract_metrics(response(prompt=2000, cached=0, output=20), 0.7)
    assert metrics["prompt_uncached_tokens"] == 2000
    assert metrics["prompt_tokens_per_s"] == 10000.0      # 2000 tokens / 0.2 s
    assert metrics["decode_tokens_per_s"] == 200.0        # 20 tokens / 0.1 s
    assert metrics["load_s"] == 0.001 and metrics["cold_load"] is False
    assert metrics["wall_s"] == 0.7


def test_metrics_partially_cached_prompt_has_no_throughput():
    # Fresh tokens after a long cached prefix process slower than a cold prompt.
    metrics = extract_metrics(response(prompt=4964, cached=4452, output=10), 0.1)
    assert metrics["prompt_uncached_tokens"] == 512
    assert metrics["prompt_tokens_per_s"] is None
    assert metrics["decode_tokens_per_s"] is None         # 10 output tokens: too few to time


def test_metrics_cache_count_unavailable():
    raw = response(prompt=2000)
    del raw["prompt_eval_cached_count"]                   # older Ollama
    metrics = extract_metrics(raw, 0.5)
    assert metrics["prompt_cached_tokens"] is None and metrics["prompt_uncached_tokens"] is None
    assert metrics["prompt_tokens_per_s"] is None


def test_metrics_handle_missing_fields_and_cold_load():
    metrics = extract_metrics({"message": {"content": "x"}, "load_duration": 3_000_000_000}, 3.2)
    assert metrics["cold_load"] is True
    assert metrics["prompt_tokens"] is None and metrics["prompt_tokens_per_s"] is None
    assert metrics["decode_tokens_per_s"] is None


# -- call_api: the promptfoo provider entry point, against a scripted client ----------------

@pytest.fixture
def provider(monkeypatch):
    """Install a FakeClient; returns a function running call_api with given answer/metadata."""
    for variable in ollama_provider.ENV_OVERRIDES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(ollama_provider, "gpu_info", lambda: [{"name": "Test GPU", "memory_mib": 1024, "driver": "1"}])
    monkeypatch.delenv("EVAL_SEED", raising=False)

    def call(answer, *, metadata=None, config=None, messages=None, repeat=0, models=("fake:latest",)):
        monkeypatch.setattr(ollama_provider, "_warmed", {})   # per-process memos
        monkeypatch.setattr(ollama_provider, "_server", {})
        client = FakeClient(answer, models=models)
        monkeypatch.setattr(ollama_provider, "OllamaClient", lambda host, timeout: client)
        prompt = json.dumps(messages or [{"role": "user", "content": "q"}])
        result = ollama_provider.call_api(prompt, {"config": {"model": "fake", **(config or {})}},
                                          {"test": {"metadata": metadata or {}}, "repeatIndex": repeat})
        return result, client

    return call


def test_call_api_success(provider):
    result, client = provider(lambda m, o: response("42", prompt=1000, cached=200))
    assert result["output"] == "42"
    assert result["tokenUsage"]["prompt"] == 1000
    assert result["tokenUsage"]["completionDetails"]["cacheReadInputTokens"] == 200
    meta = result["metadata"]
    assert meta["metrics"]["prompt_uncached_tokens"] == 800 and meta["server"]["model_digest"] == "abc123"
    assert meta["warmup"]["prompt_tokens"] == 1000
    assert len(client.calls) == 2 and client.calls[1]["truncate"] is False   # warm-up + the test


def test_option_precedence(provider, monkeypatch):
    monkeypatch.setenv("EVAL_NUM_CTX", "8192")
    monkeypatch.setenv("EVAL_SEED", "10")
    monkeypatch.setenv("EVAL_OPTIONS", '{"top_k": 9}')
    result, client = provider(lambda m, o: response("x"), metadata={"num_predict": 64},
                              config={"temperature": 0.3, "options": {"top_k": 5}}, repeat=2)
    assert client.calls[-1]["options"] == {"num_predict": 64, "top_k": 9, "num_ctx": 8192,
                                           "temperature": 0.3, "seed": 12}


def test_context_exceeded_is_invalid(provider):
    error = OllamaError("context_exceeded", "request (4952 tokens) exceeds the available context size (2048 tokens)")
    result, _ = provider(lambda m, o: error if o["num_predict"] != 4 else response("OK"))
    assert result["error"].startswith("INVALID [context_exceeded]")


def test_unreachable_and_missing_model_are_errors(provider):
    result, _ = provider(unreachable)
    assert result["error"].startswith("ERROR [unreachable]")
    result, _ = provider(lambda m, o: response("x"), models=["other:latest"])
    assert result["error"].startswith("ERROR [model_missing]") and "ollama pull fake" in result["error"]


def test_suspected_truncation_is_invalid(provider):
    long_history = [{"role": "user", "content": "word " * 2000}]
    result, _ = provider(lambda m, o: response("x", prompt=900), messages=long_history)
    assert result["error"].startswith("INVALID: prompt appears truncated")


def test_context_full_is_invalid(provider):
    result, _ = provider(lambda m, o: response("x", prompt=2040, output=10), config={"num_ctx": 2048})
    assert "context window full" in result["error"]


def test_cutoff_policy(provider):
    cut = lambda m, o: response("partial", done_reason="length")
    result, _ = provider(cut)
    assert result["error"].startswith("INVALID: answer cut off by num_predict")
    result, _ = provider(cut, metadata={"cutoff": "fail"})
    assert result["output"] == "partial"                 # graded, and will fail
    result, _ = provider(cut, metadata={"cutoff": "fail"}, config={"think": True})
    assert "error" in result                              # thinking budget: still invalid


def test_thinking_goes_to_metadata(provider):
    result, _ = provider(lambda m, o: response("42", thinking="maybe 41?"))
    assert result["output"] == "42" and result["metadata"]["thinking"] == "maybe 41?"


def test_unknown_cache_count_is_not_reported_as_zero(provider):
    def answer(m, o):
        raw = response("42")
        del raw["prompt_eval_cached_count"]
        return raw
    result, _ = provider(answer)
    assert "completionDetails" not in result["tokenUsage"]


@pytest.mark.parametrize("env, config, message", [
    ({"EVAL_NUM_CTX": "lots"}, {}, "EVAL_NUM_CTX='lots' is invalid"),
    ({"EVAL_SEED": "x"}, {}, "EVAL_SEED='x' is not an integer"),
    ({"EVAL_OPTIONS": "[1, 2]"}, {}, "must be a JSON object"),
    ({"EVAL_OPTIONS": "{bad"}, {}, "EVAL_OPTIONS='{bad' is invalid"),
    ({}, {"num_cxt": 4096}, "unknown provider config keys ['num_cxt']"),
])
def test_configuration_errors_are_actionable(provider, monkeypatch, env, config, message):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    result, client = provider(lambda m, o: response("x"), config=config)
    assert result["error"].startswith("ERROR [config]") and message in result["error"]
    assert not client.calls


def test_malformed_prompt_is_actionable(provider, monkeypatch):
    client = FakeClient(lambda m, o: response("x"))
    monkeypatch.setattr(ollama_provider, "OllamaClient", lambda host, timeout: client)
    for prompt in ("plain text, not JSON", "[]", '[{"role": "user"}]', '{"role": "user", "content": "x"}'):
        result = ollama_provider.call_api(prompt, {"config": {}}, {})
        assert result["error"].startswith("ERROR [config]: the prompt is not a JSON chat history"), prompt


def test_hardware_and_placement_recorded(provider):
    result, _ = provider(lambda m, o: response("42"))
    meta = result["metadata"]
    assert meta["server"]["gpus"] == [{"name": "Test GPU", "memory_mib": 1024, "driver": "1"}]
    assert meta["warmup"]["placement"] == {"size_bytes": 4_000_000_000, "size_vram_bytes": 3_000_000_000,
                                           "context_length": 8192}


def test_gpu_info(monkeypatch):
    class Done:
        stdout = "NVIDIA GeForce RTX 5070 Ti, 16303, 615.71.09\n"
    monkeypatch.setattr(ollama_provider.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(ollama_provider.subprocess, "run", lambda *a, **k: Done())
    assert ollama_provider.gpu_info() == [{"name": "NVIDIA GeForce RTX 5070 Ti", "memory_mib": 16303,
                                           "driver": "615.71.09"}]

    def fails(*a, **k):
        raise ollama_provider.subprocess.CalledProcessError(9, "nvidia-smi")
    monkeypatch.setattr(ollama_provider.subprocess, "run", fails)
    assert ollama_provider.gpu_info() is None
    monkeypatch.setattr(ollama_provider.shutil, "which", lambda name: None)
    assert ollama_provider.gpu_info() is None
