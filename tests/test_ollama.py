import io
import json
import socket
from urllib.error import HTTPError, URLError

import pytest

from llm_eval import ollama
from llm_eval.ollama import ChatResult, OllamaClient, OllamaError, extract_metrics
from tests.fakes import response


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

    monkeypatch.setattr(ollama, "urlopen", fake_urlopen)
    return captured


def http_error(code, body):
    return HTTPError("http://x/api/chat", code, "err", {}, io.BytesIO(json.dumps(body).encode()))


def test_chat_payload_and_result(monkeypatch):
    captured = patch_urlopen(monkeypatch, response("hi"))
    result = OllamaClient("http://h:1/").chat("m", [{"role": "user", "content": "x"}], {"num_ctx": 2048})
    assert result.content == "hi"
    payload = captured["payload"]
    assert captured["url"] == "http://h:1/api/chat"
    assert payload["stream"] is False and payload["truncate"] is False
    assert payload["options"] == {"num_ctx": 2048}


def test_truncate_field_omitted_when_allowed(monkeypatch):
    captured = patch_urlopen(monkeypatch, response("hi"))
    OllamaClient().chat("m", [], {}, truncate=True)
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
        OllamaClient().chat("m", [], {})
    assert info.value.kind == kind


def test_missing_message_is_bad_response(monkeypatch):
    patch_urlopen(monkeypatch, {"done": True})
    with pytest.raises(OllamaError) as info:
        OllamaClient().chat("m", [], {})
    assert info.value.kind == "bad_response"


def test_metrics_use_uncached_tokens_for_prompt_throughput():
    metrics = extract_metrics(ChatResult(response(prompt=2058, cached=58, output=20), 0.7))
    assert metrics["prompt_uncached_tokens"] == 2000
    assert metrics["prompt_tokens_per_s"] == 10000.0      # 2000 tokens / 0.2 s
    assert metrics["decode_tokens_per_s"] == 200.0        # 20 tokens / 0.1 s
    assert metrics["load_s"] == 0.001 and metrics["cold_load"] is False
    assert metrics["wall_s"] == 0.7


def test_metrics_skip_prompt_throughput_when_mostly_cached():
    metrics = extract_metrics(ChatResult(response(prompt=2058, cached=2054), 0.1))
    assert metrics["prompt_tokens_per_s"] is None
    assert metrics["decode_tokens_per_s"] is None         # 10 output tokens: too few to time


def test_metrics_handle_missing_fields_and_cold_load():
    raw = {"message": {"content": "x"}, "load_duration": 3_000_000_000}
    metrics = extract_metrics(ChatResult(raw, 3.2))
    assert metrics["cold_load"] is True
    assert metrics["prompt_tokens"] is None and metrics["prompt_tokens_per_s"] is None
    assert metrics["decode_tokens_per_s"] is None
