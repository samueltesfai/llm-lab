# Local LLM Lab

Experiments exploring private, locally hosted AI assistants: what local models can do,
how context-window limits interact with memory, retrieval, and summarization, and what
inference costs in time and memory on consumer hardware.

## Environment

- OS: Fedora Linux
- GPU: NVIDIA RTX 5070 Ti (16 GB VRAM)
- Runtime: Ollama
- Initial model: Qwen3.5-9B (`qwen3.5:9b`)
- Python managed with `uv` (`pyproject.toml`, `uv.lock`)

## Two evaluation layers

1. **Model-level evaluation** (this repository's evaluation suite): how capable,
   reliable, and fast is a model when given *controlled* inputs directly through the
   Ollama API? Every test sends a complete, fixed prompt in a single request.
2. **Application-level evaluation** (future work, not implemented here): how well does
   an actual chatbot (e.g. Open WebUI) manage history, retrieval, persistent memory,
   and the overall user experience?

The suite establishes a baseline for the first layer, so that differences observed
later at the application layer can be attributed to the application rather than to
the model or its settings.

## Setup

```bash
uv sync            # creates .venv with runtime and dev dependencies
ollama pull qwen3.5:9b   # the suite never pulls or modifies models itself
```

Runtime dependencies are `pyyaml` (scenario files) and `jsonschema` (structured-output
validation); `pytest` is a dev dependency. Ollama is reached over plain HTTP from the
standard library.

## Running the evaluation suite

```bash
# Standard suite (37 cases, ~25 s on an RTX 5070 Ti once the model is loaded)
uv run python -m llm_eval --model qwen3.5:9b

# List cases without contacting Ollama
uv run python -m llm_eval --list
uv run python -m llm_eval --list --profile all

# One category, or specific cases (glob patterns; --case ignores --profile)
uv run python -m llm_eval --category corrections
uv run python -m llm_eval --category grounded_qa,programming
uv run python -m llm_eval --case pg_trace_mutable_default
uv run python -m llm_eval --case 'lc_*'

# Expensive long-context profile (11k-26k token prompts)
uv run python -m llm_eval --profile long --num-ctx 32768

# Another model, other inference settings
uv run python -m llm_eval --model llama3.1:8b --num-ctx 8192
uv run python -m llm_eval --temperature 0.7 --trials 5 --seed 1
uv run python -m llm_eval --think --num-predict 4096      # enable thinking mode
uv run python -m llm_eval --option top_k=20 --option repeat_penalty=1.1

# Remote server, print answers of non-passing cases, tag the run
uv run python -m llm_eval --host http://gpu-box:11434 -v --label "after driver update"
```

`uv run python -m llm_eval --help` lists every option. The most important:

| Option | Default | Meaning |
|---|---|---|
| `--model` | `qwen3.5:9b` | Installed Ollama model tag |
| `--host` | `http://127.0.0.1:11434` | Ollama server URL |
| `--num-ctx` | `16384` | Context window requested from Ollama |
| `--temperature` | `0.0` | Sampling temperature |
| `--seed` | `2026` | Seed for scenario generators and sampling; trial *i* samples with `seed + i` |
| `--num-predict` | per case | Overrides every case's output-token limit |
| `--think` | off | Enable the model's thinking mode; raise `--num-predict` with it |
| `--option KEY=VALUE` | | Any other Ollama option (value parsed as JSON) |
| `--trials` | `1` | Repetitions of each case |
| `--profile` | `standard` | `standard`, `long`, or `all` |
| `--no-warmup` | | Skip the untimed request that loads the model first |
| `--allow-truncation` | | Let Ollama silently truncate over-long prompts (not recommended) |

Exit status: `0` all graded cases passed (`review`/`skipped` allowed), `1` at least one
failure, `2` some results were `invalid`/`error` or the run aborted, `3` the run could
not start (Ollama unreachable, model not installed, invalid scenario files), `130`
interrupted.

## What is tested

| Category | Standard / long cases | Evaluates |
|---|---|---|
| `long_context` | 5 / 4 | Finding one precise fact early, mid-way, or late in a long conversation or table, with look-alike distractor facts. Inputs are generated deterministically; the generator is adapted from [experiment 001](experiments/001_context_recall/). |
| `instruction_following` | 6 | JSON that must satisfy a schema (required/forbidden fields, enums, types), simultaneous format rules, exact line/CSV formats. |
| `multi_turn` | 5 | Combining facts spread across a scripted conversation: sums, constraint intersection, assembling a URL, ordering, aggregation. |
| `corrections` | 7 | Corrected, twice-superseded, partially corrected, rejected, and merely tentative changes; plus asking for the *historical* value. |
| `grounded_qa` | 7 | Answering from a supplied document: extraction, combining facts, applying a rule, and abstaining (`NOT IN SOURCE`) when the document lacks the answer. |
| `programming` | 7 | Tracing program output, locating and fixing a bug, choosing a correct implementation, writing functions checked by hidden tests. |

Cases live in [`scenarios/*.yaml`](scenarios/); every case has a `purpose` line. Run
`--list` to read them.

The history in multi-turn, corrections, and grounded-QA cases, **including the assistant
turns, is synthetic**. It was written by hand and sent in a single request. These cases
measure how a model reasons over supplied history, not whether any application
remembers anything.

## Results

Each run writes a new directory; nothing is ever overwritten:

```
results/runs/<UTC timestamp>_<model>_ctx<num_ctx>/
  run.json        # effective config, environment, selection, warm-up, summary
  results.jsonl   # one record per case trial, appended as the run progresses
```

`run.json` records the command line, all effective options, Ollama version, model
digest/quantization/parameter size, model memory placement from `/api/ps`
(`size_vram`, allocated `context_length`), git commit and dirty flag, SHA-256 of every
scenario file, Python version, and whether the code sandbox had network isolation.
Each record in `results.jsonl` holds the full answer (and thinking text, if any), the
expected value, the grader's verdict and detail, the effective options, a hash of the
exact prompt, generator seed and metadata, and all timing/token metrics.

`results/` is git-ignored. Use `git add -f` for runs you want to keep in history.
Results of the original context-recall experiment are in `results/001_context_recall/`.

Compare stored runs:

```bash
uv run python -m llm_eval.compare             # one row per run, per-category pass counts
uv run python -m llm_eval.compare --cases     # plus a case-by-run status matrix
uv run python -m llm_eval.compare results/runs/2026*_ctx32768
```

## Interpreting scores

Every trial gets one status:

| Status | Meaning |
|---|---|
| `pass` | Meets the case's acceptance criteria. |
| `fail` | Objectively wrong: wrong value, invalid JSON, schema violation, failed tests, invented detail. |
| `review` | Possibly acceptable, but a simple rule can't decide, e.g. the right value appears together with other candidates, or an abstention without the required marker, or code the sandbox refused to run. Inspect the answer. |
| `invalid` | The result says nothing reliable about the model, e.g. the prompt didn't fit `num_ctx`, or the answer was cut off by `num_predict`. |
| `error` | The request failed (HTTP error, timeout, malformed response). |
| `skipped` | Not attempted: the case requires a larger `--num-ctx` (`min_ctx`). |

**Pass rate = pass / (pass + fail + review).** `review` counts against the rate until a
human decides; `invalid`, `error`, and `skipped` are excluded and reported separately.

Graders are deterministic: normalized exact match, regex extraction, numeric comparison,
JSON Schema validation, explicit format checks, abstention checks, and executing hidden
unit tests. There is no LLM-as-judge.

Context limits are handled explicitly:

- Requests are sent with `truncate: false`. Recent Ollama versions then reject an
  over-long prompt (HTTP 400, `exceed_context_size_error`) instead of silently dropping
  earlier messages; the suite records that as `invalid` with the token counts. Without
  this, Ollama 0.40 cut a 2,058-token prompt to 937 tokens at `num_ctx=1024` and the
  model simply answered `UNKNOWN`, which would have looked like a retrieval failure.
- For servers that ignore `truncate`, a heuristic marks a result `invalid` if far fewer
  prompt tokens were processed than the prompt's length implies (> 8 characters per
  token). `prompt_eval_count` can't prove that nothing was dropped, only make it
  implausible.
- If prompt plus output reaches `num_ctx`, or the answer hits `num_predict` without
  passing, the result is `invalid`. Cases whose output budget is ~10× what a correct
  answer needs may declare `cutoff: fail`, so rambling past the budget counts as a
  failure. That policy is disabled under `--think`.
- Ollama may allocate a different context than requested (it allocated 2048 when 1024
  was asked). The allocated value from `/api/ps` is stored in `run.json`.

## Interpreting timing

Per request (all in `results.jsonl` → `metrics`):

| Field | Source | Meaning |
|---|---|---|
| `wall_s` | client | Whole HTTP request as observed by the suite |
| `total_s`, `load_s`, `prompt_eval_s`, `eval_s` | Ollama `*_duration` (ns → s) | Server-side total, model load, prompt processing, generation |
| `prompt_tokens` | `prompt_eval_count` | Prompt tokens in the request, **including** cached ones |
| `prompt_cached_tokens` | `prompt_eval_cached_count` | Tokens reused from the KV (prefix) cache |
| `prompt_tokens_per_s` | derived | Uncached tokens / `prompt_eval_s`; only when ≥ 256 uncached tokens |
| `decode_tokens_per_s` | derived | `eval_count / eval_s`; only when ≥ 16 output tokens |
| `cold_load` | derived | `load_s` > 1 s: weights were (re)loaded during this request |

Warm versus cold:

- **Model loading.** Before the cases, an untimed warm-up request loads the model with the
  run's `num_ctx`. Changing `num_ctx` forces Ollama to reload. Its cost is stored as
  `run.json → warmup.load_s`, so case timings start warm. Any later reload (e.g. the
  model was evicted) is flagged per request and in the summary.
- **Prompt (prefix) caching.** Ollama reuses the KV cache for a prompt prefix it has just
  processed, including across runs within the `keep_alive` window. A repeated request
  can then report ~10k prompt tokens processed in 0.03 s. The suite computes prompt
  throughput only from **uncached** tokens, so cached requests don't inflate it. With
  `--trials > 1` at temperature 0, later trials mostly measure the cache.
- Decode throughput is stable and comparable across runs. Prompt throughput depends on
  prompt length and batch settings. Compare it between runs with the same profile.
- Thinking tokens count as output tokens (`eval_count`), so `--think` raises latency.

## Adding a scenario

Add a case to the relevant file in `scenarios/` (or create a new `*.yaml` file; a new
category name works without code changes). No runner edits are needed:

```yaml
  - id: cr_new_example                 # unique across all files
    purpose: What this case evaluates, in one sentence.
    messages:
      - {role: user, content: "The release is on Friday."}
      - {role: assistant, content: "Noted."}
      - {role: user, content: "Correction: Monday. When is the release? One word."}
    grader:
      type: regex_extract
      pattern: '\b(?:monday|friday)\b'
      expected: monday
    options: {num_predict: 32}         # optional; num_ctx/temperature/seed are run-level
```

Available graders (see `llm_eval/graders.py`):

| `type` | Key fields | Use for |
|---|---|---|
| `exact` | `expected` (str or list), `case_sensitive` | Precisely specified output |
| `regex_extract` | `pattern`, `expected` (str or set), `extra: review/fail` | Codes, names, IDs inside free text |
| `number` | `expected`, `tolerance`, `extra` | Numeric answers |
| `contains` | `all`, `any`, `none` (regex lists) | Required and forbidden content |
| `choice` | `expected` letter, `options` | Multiple choice |
| `json` | `schema`, `expect {path: value}`, `forbidden_keys`, `forbidden_text`, `unordered_lists`, `allow_fences` | Structured output |
| `constraints` | `checks: [line_count, word_count, each_line_matches, each_line_max_words, contains, not_contains]` | Format rules |
| `abstain` | `marker` (default `NOT IN SOURCE`), `forbidden` | Unanswerable questions |
| `python_function` | `function`, `tests: [{args, expected} or {args, raises}]` | Code-writing tasks |

Other case keys: `generator` (instead of `messages`; see `llm_eval/generators.py`),
`profile: long`, `min_ctx`, `cutoff: fail`, `tags`, `notes`. A file may set `defaults`
merged into each case, and `shared` for YAML anchors (see `grounded_qa.yaml`). Scenario
files are validated on load: unknown keys, missing grader fields, bad regexes, and
duplicate IDs are reported before anything runs.

Long inputs are generated rather than written out. A generator is a Python function in
`llm_eval/generators.py`, seeded from `--seed` and the case ID, so prompts are
byte-identical across runs with the same seed. Programmatic generation is used only
where hand-written YAML would be impractically long.

## Running model-generated code

`python_function` cases execute code written by the model, with layered safeguards
(`llm_eval/sandbox.py`):

- The code is parsed first: only pure standard-library modules (`math`, `re`,
  `collections`, ...) may be imported, and `open`, `eval`, `exec`, `__import__`,
  `getattr` and dunder attributes are rejected. Rejected code is graded `review`,
  never run.
- It runs in a separate `python -I` process in a temporary directory, with an empty
  environment and stdin closed.
- Limits apply: CPU time, 1 GiB address space, 1 MiB file size, 64 open files, no new
  processes, and a wall-clock timeout.
- When `unshare -rn` works (stock Fedora), the process has no network interfaces.

This is defense in depth for a personal machine, not a hardened sandbox against
adversarial code.

## Limitations

- **This is not a measure of general model intelligence.** 41 hand-written cases cover
  a few practical capabilities. Use established public benchmarks for broad model
  comparisons.
- **Passing does not mean a chatbot has persistent memory, effective document
  retrieval, or reliable cross-session behavior.** All history and documents are
  supplied in one request; nothing is stored or retrieved.
- Small case counts mean one case changes a category's rate a lot. Look at which cases
  changed (`compare --cases`), not only percentages.
- At temperature 0 with a fixed seed, results are reproducible on the same
  model/server/hardware, but a different Ollama version, GPU, or quantization can
  change outputs.
- Token estimates in case descriptions (e.g. "~5k tokens") are for Qwen-family
  tokenizers; other models tokenize differently. The actual count is in each record.
- Truncation detection relies on Ollama honoring `truncate: false` (verified with Ollama
  0.40.3) plus a heuristic. `prompt_eval_count` alone cannot prove a prompt was complete.
- GPU metrics are limited to Ollama's own `/api/ps` memory report; there is no GPU
  profiling.
- Graders accept limited formatting variation (case, surrounding whitespace, a final
  period, a code fence). A model that answers correctly but ignores an explicit format
  instruction can fail or land in `review`. That is intended, but keep it in mind when
  comparing models with different styles.

## Repository layout

```
llm_eval/            evaluation suite
  cli.py             command line, run metadata, result files
  ollama.py          Ollama HTTP client, error classification, timing metrics
  scenarios.py       scenario loading, validation, selection
  generators.py      deterministic long-input generators
  graders.py         deterministic graders
  sandbox.py         restricted execution of model-written functions
  runner.py          runs cases, decides pass/fail/invalid
  report.py          summaries, console output, exit status
  compare.py         side-by-side comparison of stored runs
scenarios/           test cases, one YAML file per category
tests/               tests of the suite itself (mocked Ollama; no server needed)
experiments/         standalone experiments
results/             run output (git-ignored)
```

Run the suite's own tests with `uv run pytest`.

## Experiments

- [001 — Context Recall](experiments/001_context_recall/)
