"""Promptfoo custom provider: one Ollama /api/chat request per test case.

Why custom: promptfoo's built-in Ollama provider discards Ollama's timing fields
(load, prompt-processing and generation durations) and can't tell an untrustworthy
result from a model failure. This provider returns:

  output      the answer text (any thinking trace goes to metadata, not the output)
  tokenUsage  prompt / completion tokens; KV-cache hits under completionDetails
  metadata    timing metrics, effective Ollama options, server and model identity
  error       for results that say nothing about the model, all prefixed "INVALID:"
              (Ollama rejected the prompt as too long, the prompt looks truncated,
              or the answer was cut off) or "ERROR:" (unreachable, timeout, HTTP).
              promptfoo counts these as errors, never as failures.

Settings: the provider `config` in promptfooconfig.yaml, overridden per run by the
EVAL_* environment variables in ENV_OVERRIDES. (promptfoo's own `{{ env.X }}`
templating leaves the placeholder in place when X is unset, and --env-file overrides
the shell, so neither can express "default, unless set for this run".)
"""

import json
import os
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULTS = {
    "model": "qwen3.5:9b",
    "host": "http://127.0.0.1:11434",
    "num_ctx": 32768,
    "temperature": 0.0,
    "think": False,
    "num_predict": None,     # when set, overrides every test's metadata.num_predict
    "keep_alive": "5m",
    "timeout": 600,
    "options": {},           # any other Ollama options, e.g. {top_k: 20}
}
DEFAULT_NUM_PREDICT = 512
DEFAULT_SEED = 2026


def _bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


ENV_OVERRIDES = {
    "EVAL_MODEL": ("model", str),
    "EVAL_HOST": ("host", str),
    "EVAL_NUM_CTX": ("num_ctx", int),
    "EVAL_TEMPERATURE": ("temperature", float),
    "EVAL_THINK": ("think", _bool),
    "EVAL_NUM_PREDICT": ("num_predict", int),
    "EVAL_OPTIONS": ("options", json.loads),     # JSON object, e.g. '{"top_k": 20}'
}


class ConfigError(ValueError):
    """A problem in promptfooconfig.yaml, EVAL_* variables, or a test's messages."""


def run_seed() -> int:
    """Seed for sampling and for generated histories (helpers/long_context.py)."""
    value = os.environ.get("EVAL_SEED") or DEFAULT_SEED
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"EVAL_SEED={value!r} is not an integer") from None

def settings(config: dict | None = None) -> dict:
    """Effective settings: DEFAULTS < provider config < EVAL_* environment variables."""
    config = {k: v for k, v in (config or {}).items() if k != "basePath"}   # basePath: added by promptfoo
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        raise ConfigError(f"unknown provider config keys {unknown}; valid keys: {sorted(DEFAULTS)}")
    merged = {**DEFAULTS, **config}
    for variable, (key, parse) in ENV_OVERRIDES.items():
        if os.environ.get(variable):
            try:
                merged[key] = parse(os.environ[variable])
            except ValueError as exc:
                raise ConfigError(f"{variable}={os.environ[variable]!r} is invalid: {exc}") from None
    if not isinstance(merged["options"], dict):
        raise ConfigError("options / EVAL_OPTIONS must be a JSON object, e.g. '{\"top_k\": 20}'")
    merged["seed"] = run_seed()
    return merged


def parse_messages(prompt: str) -> list[dict]:
    """The rendered prompt must be a JSON chat history: [{role, content}, ...]."""
    try:
        messages = json.loads(prompt)
    except json.JSONDecodeError:
        messages = None
    if (not isinstance(messages, list) or not messages
            or not all(isinstance(m, dict) and isinstance(m.get("role"), str)
                       and isinstance(m.get("content"), str) for m in messages)):
        raise ConfigError("the prompt is not a JSON chat history; give the test vars.messages as a list "
                          f"of {{role, content}} messages (got: {prompt[:80]!r})")
    return messages


# -- Ollama HTTP client (standard library only; never pulls or modifies models) ----------

class OllamaError(Exception):
    """kind: unreachable | timeout | http | context_exceeded | model_missing | bad_response"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _classify_http_error(status: int, text: str) -> OllamaError:
    try:
        message = json.loads(text).get("error", text)
    except (json.JSONDecodeError, AttributeError):
        message = text
    message = message if isinstance(message, str) else json.dumps(message)
    lowered = message.lower()
    if "exceed" in lowered and "context" in lowered:
        return OllamaError("context_exceeded", message)
    if status == 404 and "not found" in lowered:
        return OllamaError("model_missing", message)
    return OllamaError("http", f"HTTP {status}: {message}")


class OllamaClient:
    def __init__(self, host: str, timeout: float = 600.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        request = Request(self.host + path, data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"},
                          method="GET" if payload is None else "POST")
        try:
            with urlopen(request, timeout=timeout or self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            raise _classify_http_error(exc.code, exc.read().decode("utf-8", errors="replace")) from None
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaError("timeout", f"{path} timed out after {timeout or self.timeout}s") from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise OllamaError("timeout", f"{path} timed out") from exc
            raise OllamaError("unreachable", f"cannot reach Ollama at {self.host}: {exc.reason}") from exc
        except ConnectionError as exc:
            raise OllamaError("unreachable", f"connection to {self.host} failed: {exc}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError("bad_response", f"non-JSON response from {path}: {body[:200]!r}") from exc

    def version(self) -> str | None:
        return self._request("/api/version", timeout=10).get("version")

    def list_models(self) -> list[dict]:
        return self._request("/api/tags", timeout=10).get("models", [])

    def chat(self, model: str, messages: list[dict], options: dict, *, think=False,
             keep_alive: str = "5m", truncate: bool = False) -> tuple[dict, float]:
        """Returns (Ollama's response body, client-observed wall seconds)."""
        payload = {"model": model, "messages": messages, "stream": False, "think": think,
                   "keep_alive": keep_alive, "options": options}
        # truncate=False makes Ollama reject an over-long prompt (HTTP 400) instead of
        # silently dropping earlier messages. Older servers ignore the field.
        if not truncate:
            payload["truncate"] = False
        start = time.monotonic()
        response = self._request("/api/chat", payload)
        elapsed = time.monotonic() - start
        if "message" not in response:
            raise OllamaError("bad_response", f"chat response has no message: {str(response)[:200]}")
        return response, elapsed


# -- metrics ---------------------------------------------------------------------------------

LOAD_COLD_THRESHOLD_S = 1.0        # load_duration above this: weights were (re)loaded
MIN_UNCACHED_FOR_THROUGHPUT = 256  # shorter prompts make prompt tokens/s meaningless
MIN_OUTPUT_FOR_THROUGHPUT = 16     # likewise for decode: a 2-token answer gives a noisy rate


def _seconds(response: dict, field: str) -> float | None:
    value = response.get(field)
    return round(value / 1e9, 4) if isinstance(value, (int, float)) else None


def extract_metrics(r: dict, wall_seconds: float) -> dict:
    """Ollama's nanosecond counters as seconds, plus throughput.

    Verified against Ollama 0.40.3: prompt_eval_count includes tokens served from the
    KV (prefix) cache, prompt_eval_cached_count reports how many, and
    prompt_eval_duration covers only the uncached ones. Cache reuse is partial and
    model-dependent (qwen3.5 reuses in ~512-token steps), and a few fresh tokens after a
    long cached prefix process slower per token than a cold prompt. So prompt
    throughput is reported only for requests with nothing cached, and as unavailable
    (None) when the server doesn't report the cached count.
    """
    prompt_tokens, cached, output_tokens = r.get("prompt_eval_count"), r.get("prompt_eval_cached_count"), r.get("eval_count")
    load_s, prompt_s, eval_s = _seconds(r, "load_duration"), _seconds(r, "prompt_eval_duration"), _seconds(r, "eval_duration")
    uncached = max(prompt_tokens - cached, 0) if isinstance(prompt_tokens, int) and isinstance(cached, int) else None
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_cached_tokens": cached,
        "prompt_uncached_tokens": uncached,
        "output_tokens": output_tokens,
        "load_s": load_s,
        "prompt_eval_s": prompt_s,
        "eval_s": eval_s,
        "total_s": _seconds(r, "total_duration"),
        "wall_s": round(wall_seconds, 4),
        "prompt_tokens_per_s": (round(uncached / prompt_s, 1)
                                if cached == 0 and uncached >= MIN_UNCACHED_FOR_THROUGHPUT and prompt_s else None),
        "decode_tokens_per_s": (round(output_tokens / eval_s, 1)
                                if isinstance(output_tokens, int) and output_tokens >= MIN_OUTPUT_FOR_THROUGHPUT and eval_s else None),
        "cold_load": load_s is not None and load_s > LOAD_COLD_THRESHOLD_S,
        "done_reason": r.get("done_reason"),
        "thinking_chars": len((r.get("message") or {}).get("thinking") or ""),
    }


# -- result validity -------------------------------------------------------------------------

# More characters per reported prompt token than this means Ollama processed far less
# text than we sent: assume it dropped part of the prompt (prose measures 3-5).
TRUNCATION_CHARS_PER_TOKEN = 8.0
TRUNCATION_MIN_CHARS = 2000


def context_problem(prompt_chars: int, metrics: dict, num_ctx: int) -> str | None:
    """Why the prompt or answer likely didn't fit, or None.

    The truncation check is a HEURISTIC backstop for servers that ignore
    truncate:false. A token count far below what the text implies is strong evidence
    that something was dropped; a plausible count is NOT proof the prompt was complete.
    """
    prompt_tokens = metrics["prompt_tokens"]
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return None
    if prompt_chars >= TRUNCATION_MIN_CHARS and prompt_chars / prompt_tokens > TRUNCATION_CHARS_PER_TOKEN:
        return (f"prompt appears truncated (heuristic): {prompt_chars} characters sent, "
                f"only {prompt_tokens} tokens processed")
    if prompt_tokens + (metrics["output_tokens"] or 0) >= num_ctx:
        return f"context window full: {prompt_tokens} prompt + {metrics['output_tokens']} output tokens >= num_ctx {num_ctx}"
    return None


def ollama_options(s: dict, test_metadata: dict, repeat_index: int) -> dict:
    """Test's num_predict < EVAL_NUM_PREDICT < extra options < run-level settings.
    With --repeat, trial i samples with seed + i so sampled trials aren't identical."""
    options = {"num_predict": test_metadata.get("num_predict", DEFAULT_NUM_PREDICT)}
    if s["num_predict"]:
        options["num_predict"] = s["num_predict"]
    options.update(s["options"])
    options.update(num_ctx=s["num_ctx"], temperature=s["temperature"], seed=s["seed"] + repeat_index)
    return options


# Per-process memo: promptfoo keeps the Python worker alive between calls, so the
# warm-up and server lookup run once per eval (and are cheap if they do repeat).
_warmed: dict = {}
_server: dict = {}


def _warm_up(client: OllamaClient, s: dict) -> dict:
    """Load the model with the run's num_ctx (a different num_ctx forces a reload) so
    test timings start warm; its own metrics are recorded on every row."""
    key = (client.host, s["model"], s["num_ctx"])
    if key not in _warmed:
        _warmed[key] = extract_metrics(*client.chat(
            s["model"], [{"role": "user", "content": "Reply with OK."}],
            {"num_ctx": s["num_ctx"], "num_predict": 4, "temperature": 0},
            keep_alive=s["keep_alive"], truncate=True))
    return _warmed[key]


def _server_info(client: OllamaClient, model: str) -> dict:
    key = (client.host, model)
    if key not in _server:
        entry = next((m for m in client.list_models() if m.get("name") in (model, model + ":latest")), None)
        if entry is None:
            raise OllamaError("model_missing", f"model {model!r} is not installed; run `ollama pull {model}` "
                                               "(the suite never pulls models)")
        details = entry.get("details") or {}
        _server[key] = {"ollama_version": client.version(), "model_digest": entry.get("digest"),
                        "parameter_size": details.get("parameter_size"),
                        "quantization": details.get("quantization_level")}
    return _server[key]


def call_api(prompt: str, options: dict, context: dict) -> dict:
    test_metadata = (context.get("test") or {}).get("metadata") or {}
    try:
        s = settings(options.get("config"))
        messages = parse_messages(prompt)
        request_options = ollama_options(s, test_metadata, context.get("repeatIndex") or 0)
    except ConfigError as exc:
        return {"error": f"ERROR [config]: {exc}"}
    metadata = {"model": s["model"], "think": s["think"], "options": request_options}

    client = OllamaClient(s["host"], timeout=s["timeout"])
    try:
        metadata["server"] = _server_info(client, s["model"])
        metadata["warmup"] = _warm_up(client, s)
        response, wall_seconds = client.chat(s["model"], messages, request_options, think=s["think"],
                                             keep_alive=s["keep_alive"], truncate=False)
    except OllamaError as exc:
        label = "INVALID" if exc.kind == "context_exceeded" else "ERROR"
        return {"error": f"{label} [{exc.kind}]: {exc}", "metadata": metadata}

    metrics = extract_metrics(response, wall_seconds)
    message = response.get("message") or {}
    answer = message.get("content") or ""
    metadata.update(metrics=metrics, answer=answer)
    if message.get("thinking"):
        metadata["thinking"] = message["thinking"]

    problem = context_problem(sum(len(m["content"]) for m in messages), metrics, s["num_ctx"])
    if problem is None and metrics["done_reason"] == "length":
        # Cut off by num_predict: invalid, unless the test declares `cutoff: fail` because
        # its budget is far beyond any correct answer (never under thinking mode, whose
        # token budget the test author didn't plan for).
        if test_metadata.get("cutoff") != "fail" or s["think"]:
            problem = f"answer cut off by num_predict={request_options['num_predict']}"
    if problem:
        return {"error": f"INVALID: {problem}", "metadata": metadata}

    prompt_tokens, output_tokens = metrics["prompt_tokens"] or 0, metrics["output_tokens"] or 0
    token_usage = {"prompt": prompt_tokens, "completion": output_tokens,
                   "total": prompt_tokens + output_tokens, "numRequests": 1}
    if metrics["prompt_cached_tokens"] is not None:   # unknown stays unknown, not 0
        token_usage["completionDetails"] = {"cacheReadInputTokens": metrics["prompt_cached_tokens"]}
    return {"output": answer, "tokenUsage": token_usage, "metadata": metadata}
