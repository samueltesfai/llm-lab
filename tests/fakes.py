"""A stand-in for OllamaClient that answers from a script instead of a model."""

from llm_eval.ollama import ChatResult, OllamaError


def response(content="", *, prompt=1000, cached=0, output=10, done_reason="stop",
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
    """`answer` maps (messages, options) to a response dict, or raises OllamaError."""

    def __init__(self, answer, models=("fake:latest",), version="0.0-test"):
        self.answer = answer
        self.models = list(models)
        self.calls = []
        self._version = version

    def version(self):
        return self._version

    def list_models(self):
        return [{"name": m, "model": m, "digest": "abc123", "size": 1} for m in self.models]

    def show(self, model):
        return {"details": {"parameter_size": "1B", "quantization_level": "Q4"},
                "model_info": {"fake.context_length": 4096}, "capabilities": ["completion"]}

    def running(self):
        return []

    def chat(self, model, messages, options, *, think=False, keep_alive="5m", truncate=False):
        self.calls.append({"model": model, "messages": messages, "options": options,
                           "think": think, "truncate": truncate})
        result = self.answer(messages, options)
        if isinstance(result, Exception):
            raise result
        return ChatResult(response=result, wall_seconds=0.6)


def unreachable(*_):
    return OllamaError("unreachable", "connection refused")
