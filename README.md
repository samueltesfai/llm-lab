# Local LLM Lab

Experiments exploring private, locally hosted AI assistants: what local models can do,
how context-window limits interact with memory, retrieval, and summarization, and what
inference costs in time and memory on consumer hardware.

## Environment

- OS: Fedora Linux
- GPU: NVIDIA RTX 5070 Ti (16 GB VRAM)
- Runtime: Ollama
- Initial model: Qwen3.5-9B (`qwen3.5:9b`)
- Evaluation framework: [promptfoo](https://www.promptfoo.dev/) (Node.js, pinned in `package.json`)
- Python managed with `uv` (`pyproject.toml`, `uv.lock`), for evaluation helpers only

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

## Layout

```
promptfooconfig.yaml     the suite: provider, prompt, run options, scenario files
scenarios/               test cases, one promptfoo test file per category
helpers/                 Python only where promptfoo has no built-in equivalent
  ollama_provider.py     Ollama client: timings, KV-cache accounting, invalid-run detection
  long_context.py        deterministic long chat histories (seeded generators)
  assertions.py          answer checks promptfoo lacks, e.g. normalized match, abstention
  sandbox.py             restricted execution of model-written code
  summary.py             per-category console summary, timings, exit status
tests/                   tests of the helpers (mocked Ollama; no server needed)
experiments/             standalone experiments
results/                 run reports (git-ignored)
```

promptfoo handles test loading, filtering, repetition, built-in assertions
(substring, regex, JSON Schema, exact match, JavaScript), result storage, and the web
viewer. The helpers exist only for logic specific to these evaluations.

## Setup

```bash
uv sync                  # Python environment for the helpers (pytest is the only dependency)
npm install              # promptfoo, pinned
ollama pull qwen3.5:9b   # the suite never pulls or modifies models itself
```

Node.js 22.22+ is required (`sudo dnf install nodejs24`).

## Running

```bash
npm run eval                                              # all 41 cases, ~1 minute warm
npm run eval -- --filter-metadata category=corrections    # one category
npm run eval -- --filter-pattern '^pg_trace'              # cases by id (regex)
npm run eval -- --filter-pattern '^cr_rejected_proposal$' # one case
npm run eval -- --repeat 5                                # repeated trials

EVAL_MODEL=llama3.1:8b EVAL_NUM_CTX=8192 npm run eval     # another model / context
EVAL_TEMPERATURE=0.7 EVAL_SEED=1 npm run eval -- --repeat 5
EVAL_THINK=1 EVAL_NUM_PREDICT=4096 npm run eval           # thinking mode
EVAL_OPTIONS='{"top_k": 20}' npm run eval                 # any other Ollama option
EVAL_HOST=http://gpu-box:11434 npm run eval               # another Ollama server

npm run view             # promptfoo's web viewer: every run, every answer, side by side
npm test                 # the helpers' own tests
```

Arguments after `--` go to `promptfoo eval`, so any of its options work. Per-run
settings are environment variables, since promptfoo cannot override provider config
from the command line. Defaults are in `promptfooconfig.yaml`.

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_MODEL` | `qwen3.5:9b` | Installed Ollama model tag |
| `EVAL_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `EVAL_NUM_CTX` | `32768` | Context window requested from Ollama |
| `EVAL_TEMPERATURE` | `0` | Sampling temperature |
| `EVAL_SEED` | `2026` | Seed for generated histories and sampling; repeat *i* samples with `seed + i` |
| `EVAL_NUM_PREDICT` | per test | Overrides every test's output-token limit |
| `EVAL_THINK` | off | Thinking mode; raise `EVAL_NUM_PREDICT` with it |
| `EVAL_OPTIONS` | `{}` | JSON object of other Ollama options |

The run scripts disable promptfoo's telemetry and update check, and promptfoo's
response cache is off (`evaluateOptions.cache: false`). With the cache on, a rerun
would replay stored answers for up to 14 days, with no new timings. Requests run one
at a time, because concurrency would distort timings.

**Exit status** (from `helpers/summary.py`): `0` every test passed, `1` at least one
failed, `2` some results were invalid or errored, so the pass rate isn't trustworthy.
Any other code means promptfoo itself failed (e.g. invalid YAML). Malformed settings
(`EVAL_NUM_CTX=lots`, a misspelled provider config key, a test without a valid
`messages` list, a broken assertion config) show up as per-test errors whose message
names the problem.

## What is tested

| Category | Cases | Evaluates |
|---|---|---|
| `long_context` | 9 | Finding one precise fact early, mid-way, or late in a 5k–26k-token conversation or table, with look-alike distractor facts. Histories are generated deterministically; the generator is adapted from [experiment 001](experiments/001_context_recall/) and reproduces its prompts exactly. |
| `instruction_following` | 6 | JSON satisfying a schema (required/forbidden fields, enums, types), simultaneous format rules, exact line/CSV formats. |
| `multi_turn` | 5 | Combining facts spread across a scripted conversation: sums, constraint intersection, assembling a URL, ordering, aggregation. |
| `corrections` | 7 | Corrected, twice-superseded, partially corrected, rejected, and merely tentative changes; plus asking for the *historical* value. |
| `grounded_qa` | 7 | Answering from a supplied document: extraction, combining facts, applying a rule, abstaining (`NOT IN SOURCE`) when the answer isn't there. |
| `programming` | 7 | Tracing program output, locating and fixing a bug, choosing a correct implementation, writing functions checked by hidden tests. |

Every test has a `metadata.purpose`. The history in multi-turn, corrections, and
grounded-QA tests, **including the assistant turns, is synthetic**. It was written by
hand and sent in a single request. These tests measure reasoning over supplied
history, not whether any application remembers anything.

## Results

Each `npm run eval` writes **one JSON report per suite execution**, filtered runs
included: `results/runs/<UTC timestamp>.json`. It contains:

- the full configuration
- one entry per test: id, category and purpose, the prompt, the answer, every
  assertion's verdict and reason, and provider metadata: timings, effective Ollama
  options, Ollama version, model digest, quantization, and any thinking trace
- run-level statistics

promptfoo also keeps an index of runs in `results/promptfoo/` for `npm run view`.
`results/` is git-ignored; `git add -f` the reports you want versioned. Results of
the original context-recall experiment are in `results/001_context_recall/`.

Compare runs:

```bash
npm run summary -- results/runs/20261012-220517.json            # one run in detail
npm run summary -- results/runs/*.json                          # one row per run
```

## Interpreting results

| Status | Meaning |
|---|---|
| pass | Meets the test's acceptance criteria. |
| fail | Objectively wrong: wrong value, invalid JSON, schema violation, failed hidden tests, invented detail. A reason starting `REVIEW:` means "possibly acceptable, but no simple rule can decide" (e.g. the right value alongside other candidates, or an abstention without the agreed marker). Inspect those. |
| invalid | Says nothing about the model: the prompt didn't fit `num_ctx`, looks truncated, or the answer was cut off by `num_predict`. promptfoo shows these as errors; the summary separates them. |
| error | Infrastructure, not the model: the request failed (unreachable server, missing model, timeout, HTTP error), the configuration is malformed, or a check couldn't run (broken assertion config, sandbox unavailable). |

All checks are deterministic: promptfoo's built-in assertions, plus the Python ones
in `helpers/assertions.py` (normalized exact match, number, option-letter, and
name-set extraction, abstention, regenerated long-context answers, sandboxed
hidden tests). There is no LLM-as-judge.

Context limits are handled explicitly:

- The provider sends `truncate: false`. Ollama then rejects an over-long prompt
  (`exceed_context_size_error`) instead of silently dropping earlier messages, and the
  test is reported invalid with the token counts. Without this, Ollama 0.40 cut a
  2,058-token prompt to 937 tokens at `num_ctx=1024`, and the model answered `UNKNOWN`,
  which would have looked like a retrieval failure.
- For servers that ignore `truncate`, a result is invalid if far fewer prompt tokens
  were processed than its length implies (more than 8 characters per token), or if
  prompt plus output filled `num_ctx`. `prompt_eval_count` can't prove a prompt was
  complete, only make truncation implausible.
- An answer cut off by `num_predict` is invalid. Exception: tests whose budget is about
  10× a correct answer declare `cutoff: fail`, so rambling counts as a failure. That
  policy is disabled in thinking mode.

## Interpreting timing

The provider records, per test (`response.metadata.metrics` in the report):

| Field | Source | Meaning |
|---|---|---|
| `wall_s` | client | Whole request as observed by the provider |
| `total_s`, `load_s`, `prompt_eval_s`, `eval_s` | Ollama `*_duration` (ns → s) | Server total, model load, prompt processing, generation |
| `prompt_tokens` | `prompt_eval_count` | Prompt tokens, **including** cached ones |
| `prompt_cached_tokens` | `prompt_eval_cached_count` | Tokens reused from Ollama's KV (prefix) cache |
| `prompt_tokens_per_s` | derived | `prompt_tokens / prompt_eval_s`; only when nothing was cached and ≥ 256 tokens, otherwise unavailable (`null`) |
| `decode_tokens_per_s` | derived | `eval_count / eval_s`; only when ≥ 16 output tokens |
| `cold_load` | derived | `load_s` > 1 s: weights were (re)loaded during this request |

- **Warm vs. cold.** Before the first test, the provider sends a warm-up request with
  the run's `num_ctx`; changing `num_ctx` forces a reload. Its metrics are stored as
  `metadata.warmup`, and later reloads are flagged per test and in the summary.
- **Prefix caching.** Ollama reuses the KV cache for a prompt prefix it just processed,
  including across runs within the `keep_alive` window. Verified against Ollama 0.40.3:
  `prompt_eval_count` includes cached tokens, `prompt_eval_cached_count` reports them,
  and `prompt_eval_duration` covers only the rest. An exact repeat of a 4,964-token
  prompt reported 4,960 cached tokens and 0.03 s. Reuse is partial and model-dependent:
  a shared 4,950-token prefix gave 4,452 cached tokens, since qwen3.5 reuses in ~512-token
  steps. A few fresh tokens after a long cached prefix also process slower per token
  than a cold prompt (~3,800 vs ~5,000 tok/s here). So prompt throughput is reported
  only for requests with nothing cached, and as unavailable when Ollama omits the
  cached count (older versions). With `--repeat` at temperature 0, later repeats are
  mostly cached.
- Decode throughput is the most comparable figure across runs. Thinking tokens count
  as output tokens.

## Adding a test

Add an entry to the relevant file in `scenarios/` (new files go in the config's
`tests` list). Prefer a built-in assertion; use a Python one from
`helpers/assertions.py` only when none fits.

```yaml
- description: cr_new_example            # unique id
  metadata:
    category: corrections
    purpose: What this test evaluates, in one sentence.
    num_predict: 32                      # optional output budget (default 512)
  vars:
    messages:
      - {role: user, content: "The release is on Friday."}
      - {role: assistant, content: "Noted."}
      - {role: user, content: "Correction: Monday. When is the release? One word."}
  assert:
    - {type: icontains, value: monday}
    - {type: not-icontains, value: friday}
```

Python assertions take a `config`, e.g.
`{type: python, value: file://helpers/assertions.py:number, config: {expected: 2080}}`.
Available: `exact`, `number`, `choice`, `regex_extract`, `abstain`,
`python_function`, `long_context_answer`; see their docstrings for config keys.
Long-context tests name a generator in `vars` (see `scenarios/long_context.yaml`).

## Running model-generated code

`python_function` assertions run code written by the model, through layered
safeguards in `helpers/sandbox.py`:

1. **Static policy.** The code is parsed first. Only pure standard-library modules may
   be imported, and file, eval, and introspection builtins and dunder attributes are
   rejected. Rejected code is graded `REVIEW:`, never run.
2. **Separate process.** It runs in a `python -I` process in a temporary directory,
   with an empty environment.
3. **Resource limits.** CPU, 1 GiB memory, file size, open files, no new processes,
   and a wall-clock timeout.
4. **No network.** It runs under `unshare -rn`, which works on stock Fedora, and has no
   network interfaces. If that isn't available, the sandbox **fails closed**: the code
   isn't run, and the test is reported as an error.

Verified here: inside the sandbox, `fork`, network connections and a 2 GiB allocation
all fail.

**Trust boundary.** This protects a personal machine against accidental damage or
casual misuse by the model under test. It is not a hardened sandbox:
- There's no filesystem isolation. The process can read anything your user can, if the
  code gets past the static check.
- The static check is a denylist and can have gaps.
- It depends on the host kernel and Python.

Don't use it to run deliberately adversarial code.

## Limitations

- **This is not a measure of general model intelligence.** 41 hand-written tests cover
  a few practical capabilities. Use established public benchmarks for broad
  comparisons.
- **Passing does not mean a chatbot has persistent memory, effective document
  retrieval, or reliable cross-session behavior.** All history and documents are
  supplied in one request.
- Small case counts mean one test moves a category's rate a lot. Look at which tests
  changed, not only percentages.
- At temperature 0 with a fixed seed, results reproduce on the same
  model/server/hardware. A different Ollama version, GPU, or quantization can change
  outputs.
- Token estimates in test purposes are for Qwen-family tokenizers. The actual count is
  in each result.
- Overflow detection relies on Ollama honoring `truncate: false` (verified with 0.40.3).
  The backup check for servers that ignore it is a heuristic: it catches gross
  truncation, but a plausible token count doesn't prove the prompt was complete.
- promptfoo has only pass/fail/error. "Review" and "invalid" are encoded in reason and
  error text and separated again by `helpers/summary.py`.
- GPU metrics are limited to what Ollama reports; there is no GPU profiling.

## Experiments

- [001 — Context Recall](experiments/001_context_recall/) (concluded; superseded by the `long_context` tests)
