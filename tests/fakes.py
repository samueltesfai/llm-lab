"""Canned Ollama responses and a scripted stand-in for OllamaClient."""

from ollama_provider import OllamaError


def response(content="", *, prompt=1000, cached=0, output=20, done_reason="stop",
             load_ns=1_000_000, thinking=None):
    message = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    return {
        "model": "fake", "message": message, "done": True, "done_reason": done_reason,
        "total_duration": 500_000_000, "load_duration": load_ns,
        "prompt_eval_count": prompt, "prompt_eval_cached_count": cached,
        "prompt_eval_duration": 200_000_000, "eval_count": output, "eval_duration": 100_000_000,
    }


class FakeClient:
    """`answer(messages, options)` returns a response dict or an OllamaError to raise."""

    def __init__(self, answer, models=("fake:latest",)):
        self.answer = answer
        self.host = "http://fake"
        self.models = list(models)
        self.calls = []

    def version(self):
        return "0.0-test"

    def list_models(self):
        return [{"name": m, "digest": "abc123", "details": {"parameter_size": "1B"}} for m in self.models]

    def running(self):
        return [{"name": m, "size": 4_000_000_000, "size_vram": 3_000_000_000, "context_length": 8192}
                for m in self.models]

    def chat(self, model, messages, options, *, think=False, keep_alive="5m", truncate=False):
        self.calls.append({"messages": messages, "options": options, "think": think, "truncate": truncate})
        result = self.answer(messages, options)
        if isinstance(result, Exception):
            raise result
        return result, 0.6


def unreachable(*_):
    return OllamaError("unreachable", "connection refused")
