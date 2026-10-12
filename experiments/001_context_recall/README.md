# Experiment 001 — Context Recall

## Question
How reliably can Qwen3.5-9B retrieve a specific fact from
a long synthetic conversation?

## Method
Generate a deterministic conversation with a known fact,
submit the history to Ollama, and evaluate the answer.

## Variables
- Context window size
- Conversation length
- Position of the target fact

## Measurements
- Recall correctness
- Prompt token count
- Wall-clock latency
- Output token count

## Run
uv run python experiments/001_context_recall/run.py --ctx 16384 --notes 90

Results are appended to `results/001_context_recall/context_results.jsonl`.
The reusable, multi-position version of this test is the `long_context`
category of the evaluation suite (see the top-level README).

## Findings
Three runs on 2026-10-12, all with `qwen3.5:9b` (Q4_K_M), `--ctx 16384`, 90 notes,
fact position `early`, seed 2026 (`results/001_context_recall/context_results.jsonl`):

| Run | Recall | Prompt tokens | Wall | Load | Prompt eval | Notes |
|---|---|---|---|---|---|---|
| 1 | correct | 10,998 | 43.4 s | 23.8 s | not recorded | cold: model not loaded |
| 2 | correct | 10,998 | 4.5 s | 2.4 s | 2.02 s (~5,440 tok/s) | model reloaded, prompt fully processed |
| 3 | correct | 10,998 | 0.15 s | 0.002 s | 0.027 s | immediate repeat: prompt served from cache |

- At ~11k input tokens in a 16k window, the model recalled the exact code with no
  near-limit warning. That is one position and one seed, so it shows the task is
  feasible, not how reliable recall is.
- Model loading dominated the cold request (55% of wall time). Each later change of
  `num_ctx` also forces a reload.
- An identical repeat reported the same `prompt_eval_count` but took 0.027 s of prompt
  evaluation, an apparent 407,000 tok/s. Ollama counts cached prefix tokens in
  `prompt_eval_count`, so throughput computed from it is meaningless for repeats. Newer
  Ollama versions report `prompt_eval_cached_count`, which separates the two.

## Status: concluded
This experiment is kept as the record of how the results above were produced. New
measurements belong in the evaluation suite's `long_context` category. Its generator is
adapted from `run.py` (identical prompts for the same seed), and it adds:

- look-alike distractor facts and a table-lookup variant
- `truncate: false`, so an over-long prompt is reported as invalid instead of silently cut
  (this script only warns within 256 tokens of the limit)
- cache-aware throughput and comparable run metadata

The script remains handy for ad-hoc sweeps (`--notes`, `--position`) and as a
self-contained example of calling the Ollama chat API.
