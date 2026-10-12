"""Execute selected cases against Ollama and produce one result record per trial.

Final status of a trial:
  pass / fail / review  - from the grader
  invalid - the run can't be trusted as a measure of the model: the prompt exceeded the
            context window (Ollama rejected it or appears to have truncated it), or the
            answer was cut off by num_predict / the context limit before it passed
            (unless the case sets `cutoff: fail`, see scenarios.py).
  error   - the request failed (HTTP error, timeout, malformed response) or the grader
            could not run.
  skipped - not attempted, e.g. the case requires a larger --num-ctx.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

from llm_eval.graders import grade
from llm_eval.ollama import ChatResult, OllamaError, extract_metrics
from llm_eval.scenarios import Case

DEFAULT_NUM_PREDICT = 512
# Above this many characters per reported prompt token, the prompt that reached the model
# is far shorter than the text we sent: assume Ollama dropped part of it. Typical English
# prose and chat templates measure 3-5 characters per token.
TRUNCATION_CHARS_PER_TOKEN = 8.0
TRUNCATION_MIN_CHARS = 2000


@dataclass
class RunConfig:
    model: str
    num_ctx: int = 16384
    temperature: float = 0.0
    seed: int = 2026
    trials: int = 1
    think: bool = False
    num_predict: int | None = None        # overrides every case when set
    extra_options: dict = field(default_factory=dict)   # --option key=value
    keep_alive: str = "5m"
    allow_truncation: bool = False

    def options_for(self, case: Case, trial: int) -> dict:
        options = {"num_predict": DEFAULT_NUM_PREDICT}
        options.update(case.options)
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        options.update(self.extra_options)
        # Run-level settings always win. With trials > 1 the sampling seed varies per
        # trial (seed + trial index) so sampled runs aren't forced to be identical.
        options.update(num_ctx=self.num_ctx, temperature=self.temperature, seed=self.seed + trial)
        return options


class RunAborted(Exception):
    """Ollama became unreachable; continuing would only produce more errors."""

    def __init__(self, message: str, records: list[dict]):
        super().__init__(message)
        self.records = records   # records produced for the case before aborting


def _messages_digest(messages: list[dict]) -> str:
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _base_record(run_id: str, case: Case, trial: int) -> dict:
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_id": case.id,
        "category": case.category,
        "trial": trial,
        "status": None,
        "detail": "",
    }


def context_check(prompt_chars: int, metrics: dict, num_ctx: int) -> tuple[bool, bool]:
    """Return (truncation_suspected, context_full).

    Ollama's prompt_eval_count reports how many prompt tokens the model processed. It
    cannot prove nothing was dropped, but a count far below what the text implies is a
    strong signal, and prompt + output reaching num_ctx means generation ran out of room.
    """
    prompt_tokens = metrics.get("prompt_tokens")
    output_tokens = metrics.get("output_tokens") or 0
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return False, False
    suspected = (prompt_chars >= TRUNCATION_MIN_CHARS
                 and prompt_chars / prompt_tokens > TRUNCATION_CHARS_PER_TOKEN)
    full = prompt_tokens + output_tokens >= num_ctx
    return suspected, full


def run_case(case: Case, client, config: RunConfig, run_id: str) -> list[dict]:
    """Run all trials of one case. Raises RunAborted if Ollama becomes unreachable."""
    if case.min_ctx and config.num_ctx < case.min_ctx:
        record = _base_record(run_id, case, 0)
        record.update(status="skipped", detail=f"requires --num-ctx >= {case.min_ctx}")
        return [record]

    try:
        generated, generator_seed = case.build(config.seed)
    except Exception as exc:   # a generator bug shouldn't kill the whole run
        record = _base_record(run_id, case, 0)
        record.update(status="error", detail=f"generator failed: {type(exc).__name__}: {exc}")
        return [record]
    spec = {**case.grader, **generated.grader_updates}
    messages = generated.messages
    prompt_chars = sum(len(m["content"]) for m in messages)

    records = []
    for trial in range(config.trials):
        record = _base_record(run_id, case, trial)
        options = config.options_for(case, trial)
        record.update(
            expected=spec.get("expected"),
            grader=spec["type"],
            options=options,
            prompt_sha256_16=_messages_digest(messages),
            message_count=len(messages),
            prompt_chars=prompt_chars,
            generator_seed=generator_seed,
            generator_metadata=generated.metadata or None,
        )
        try:
            result: ChatResult = client.chat(
                config.model, messages, options, think=config.think,
                keep_alive=config.keep_alive, truncate=config.allow_truncation,
            )
        except OllamaError as exc:
            record["error_kind"] = exc.kind
            if exc.kind == "context_exceeded":
                record.update(status="invalid", detail=f"prompt does not fit num_ctx: {exc}")
            else:
                record.update(status="error", detail=str(exc))
            records.append(record)
            if exc.kind == "unreachable":
                raise RunAborted(str(exc), records) from exc
            continue

        metrics = extract_metrics(result)
        truncation_suspected, context_full = context_check(prompt_chars, metrics, config.num_ctx)
        output_cut = metrics["done_reason"] == "length" or context_full
        record.update(answer=result.content, metrics=metrics,
                      truncation_suspected=truncation_suspected, output_cut_off=output_cut)
        if result.thinking:
            record["thinking"] = result.thinking

        try:
            graded = grade(result.content, spec)
        except Exception as exc:
            record.update(status="error", detail=f"grader crashed: {type(exc).__name__}: {exc}")
            records.append(record)
            continue
        record["extracted"] = graded.extracted

        if truncation_suspected:
            record.update(status="invalid", detail=(
                f"prompt appears truncated: {prompt_chars} chars but only "
                f"{metrics['prompt_tokens']} prompt tokens reported; grader said {graded.status}"))
        elif output_cut and graded.status != "pass":
            limit = "context window" if context_full else f"num_predict={options['num_predict']}"
            # A case may declare its output budget generous enough that running out is
            # itself a failure (e.g. rambling code). Not in thinking mode, whose budget
            # the case author did not plan for, and never when the context filled up.
            as_failure = case.cutoff == "fail" and not config.think and not context_full
            record.update(status="fail" if as_failure else "invalid", detail=(
                f"answer cut off by {limit} ({metrics['output_tokens']} output tokens"
                f"{'; counted as failure for this case' if as_failure else ''}); "
                f"grader said {graded.status}: {graded.detail}"))
        else:
            record.update(status=graded.status, detail=graded.detail)
        records.append(record)
    return records


def warmup(client, config: RunConfig) -> dict:
    """Load the model with the run's num_ctx so per-case timings start warm.

    Changing num_ctx forces Ollama to reload the model, so the warm-up must use the
    same value as the cases. The first load's cost is reported separately.
    """
    options = {"num_ctx": config.num_ctx, "temperature": 0, "seed": config.seed, "num_predict": 4}
    result = client.chat(config.model, [{"role": "user", "content": "Reply with OK."}], options,
                         think=False, keep_alive=config.keep_alive, truncate=True)
    return extract_metrics(result)


def run_suite(cases: list[Case], client, config: RunConfig, run_id: str,
              on_record: Callable[[dict], None] = lambda r: None) -> tuple[list[dict], str | None]:
    """Run every case; returns (records, abort_reason)."""
    records: list[dict] = []
    for case in cases:
        try:
            case_records = run_case(case, client, config, run_id)
        except RunAborted as exc:
            for record in exc.records:
                records.append(record)
                on_record(record)
            return records, str(exc)
        for record in case_records:
            records.append(record)
            on_record(record)
    return records, None


def config_dict(config: RunConfig) -> dict:
    return asdict(config)
