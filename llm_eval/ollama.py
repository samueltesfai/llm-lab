"""Minimal Ollama HTTP client (standard library only) and response-metric extraction.

Only the endpoints the suite needs are wrapped: /api/chat, /api/tags, /api/show,
/api/ps and /api/version. Nothing here pulls, creates, or modifies models.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaError(Exception):
    """A request that did not produce a usable model response.

    kind is one of: "unreachable", "timeout", "http", "context_exceeded",
    "model_missing", "bad_response".
    """

    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclass
class ChatResult:
    response: dict          # raw JSON body returned by Ollama
    wall_seconds: float     # client-observed time for the whole HTTP request

    @property
    def content(self) -> str:
        return (self.response.get("message") or {}).get("content") or ""

    @property
    def thinking(self) -> str:
        return (self.response.get("message") or {}).get("thinking") or ""


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 600.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    # -- transport -----------------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.host + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=timeout or self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise _classify_http_error(exc.code, text) from None
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaError("timeout", f"request to {path} timed out after {timeout or self.timeout}s") from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise OllamaError("timeout", f"request to {path} timed out") from exc
            raise OllamaError("unreachable", f"cannot reach Ollama at {self.host}: {exc.reason}") from exc
        except ConnectionError as exc:
            raise OllamaError("unreachable", f"connection to Ollama at {self.host} failed: {exc}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError("bad_response", f"non-JSON response from {path}: {body[:200]!r}") from exc

    # -- endpoints -----------------------------------------------------------------
    def version(self) -> str | None:
        return self._request("/api/version", timeout=10).get("version")

    def list_models(self) -> list[dict]:
        return self._request("/api/tags", timeout=10).get("models", [])

    def show(self, model: str) -> dict:
        return self._request("/api/show", {"model": model}, timeout=30)

    def running(self) -> list[dict]:
        return self._request("/api/ps", timeout=10).get("models", [])

    def chat(self, model: str, messages: list[dict], options: dict, *, think=False,
             keep_alive: str = "5m", truncate: bool = False) -> ChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": keep_alive,
            "options": options,
        }
        # truncate=False asks Ollama to reject an over-long prompt (HTTP 400) instead of
        # silently dropping earlier messages. Older servers ignore the field.
        if not truncate:
            payload["truncate"] = False
        start = time.monotonic()
        response = self._request("/api/chat", payload)
        elapsed = time.monotonic() - start
        if "message" not in response:
            raise OllamaError("bad_response", f"chat response has no message: {str(response)[:200]}")
        return ChatResult(response=response, wall_seconds=elapsed)


def _classify_http_error(status: int, text: str) -> OllamaError:
    try:
        message = json.loads(text).get("error", text)
    except (json.JSONDecodeError, AttributeError):
        message = text
    message = message if isinstance(message, str) else json.dumps(message)
    lowered = message.lower()
    if "exceed" in lowered and "context" in lowered:
        return OllamaError("context_exceeded", message, status)
    if status == 404 and "not found" in lowered:
        return OllamaError("model_missing", message, status)
    return OllamaError("http", f"HTTP {status}: {message}", status)


# -- metrics -------------------------------------------------------------------------

LOAD_COLD_THRESHOLD_S = 1.0     # load_duration above this means weights were (re)loaded
MIN_UNCACHED_FOR_THROUGHPUT = 256  # too few fresh tokens make prompt tokens/s meaningless
MIN_OUTPUT_FOR_THROUGHPUT = 16     # likewise for decode: a 2-token answer gives a noisy rate


def _seconds(response: dict, field: str) -> float | None:
    value = response.get(field)
    return round(value / 1e9, 4) if isinstance(value, (int, float)) else None


def extract_metrics(result: ChatResult) -> dict:
    """Turn Ollama's nanosecond counters into seconds and throughput figures.

    prompt_eval_count includes tokens served from the prompt (KV) cache, while
    prompt_eval_duration only covers tokens actually processed. Prompt throughput is
    therefore computed from *uncached* tokens, and only when the server reports
    prompt_eval_cached_count and enough fresh tokens were processed. Decode throughput
    is only computed for answers of at least MIN_OUTPUT_FOR_THROUGHPUT tokens.
    """
    r = result.response
    prompt_tokens = r.get("prompt_eval_count")
    cached = r.get("prompt_eval_cached_count")
    output_tokens = r.get("eval_count")
    load_s = _seconds(r, "load_duration")
    prompt_s = _seconds(r, "prompt_eval_duration")
    eval_s = _seconds(r, "eval_duration")

    uncached = None
    if isinstance(prompt_tokens, int) and isinstance(cached, int):
        uncached = max(prompt_tokens - cached, 0)

    prompt_tps = None
    if uncached is not None and uncached >= MIN_UNCACHED_FOR_THROUGHPUT and prompt_s:
        prompt_tps = round(uncached / prompt_s, 1)

    decode_tps = None
    if isinstance(output_tokens, int) and output_tokens >= MIN_OUTPUT_FOR_THROUGHPUT and eval_s:
        decode_tps = round(output_tokens / eval_s, 1)

    return {
        "prompt_tokens": prompt_tokens,
        "prompt_cached_tokens": cached,
        "prompt_uncached_tokens": uncached,
        "output_tokens": output_tokens,
        "load_s": load_s,
        "prompt_eval_s": prompt_s,
        "eval_s": eval_s,
        "total_s": _seconds(r, "total_duration"),
        "wall_s": round(result.wall_seconds, 4),
        "prompt_tokens_per_s": prompt_tps,
        "decode_tokens_per_s": decode_tps,
        "cold_load": load_s is not None and load_s > LOAD_COLD_THRESHOLD_S,
        "done_reason": r.get("done_reason"),
        "thinking_chars": len(result.thinking),
    }
