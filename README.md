# Local LLM Lab

Experiments exploring private, locally hosted AI assistants: what local models can do,
how context-window limits interact with memory, retrieval, and summarization, and what
inference costs in time and memory on consumer hardware.

## Environment

- Runtime: [Ollama](https://ollama.com/) on Linux, with a GPU that has enough memory for
  the model under test
- Default model: Qwen3.5-9B (`qwen3.5:9b`)
- Evaluation framework: [promptfoo](https://www.promptfoo.dev/) (Node.js, pinned in `package.json`)
- Python managed with `uv` (`pyproject.toml`, `uv.lock`), for evaluation helpers only

The hardware isn't fixed here, because each run records its own (see [Results](#results)).

## Two evaluation layers

1. **Model-level evaluation** (this repository's evaluation suite): how capable,
   reliable, and fast is a model when given *controlled* inputs directly through the
   Ollama API? Every test sends a complete, fixed prompt in a single request.
2. **Application-level evaluation** (future work, not implemented here): how well does
   an actual chatbot (e.g. Open WebUI) manage history, retrieval, persistent memory,
   and the overall user experience?

The suite establishes a model-level baseline that helps distinguish model limitations
from application-level behavior when investigating failures in the complete chatbot.
It can't attribute differences on its own: an application may use different prompts,
system instructions, inference settings, retrieved documents, or conversation
formatting, and the same model can behave quite differently with those inputs.

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
results/                 run reports (git-ignored, except baseline/ and 001_context_recall/)
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
- the hardware, in each entry's provider metadata:
  - the machine's GPUs (name, memory, driver) via `nvidia-smi`, recorded as
    unavailable without it; this describes the machine running the suite, which is
    the Ollama server only when Ollama runs locally
  - where Ollama placed the model (`warmup.placement`): its size, how much sits in
    GPU memory, and the context actually allocated. A partly CPU-offloaded model is
    much slower, and timings alone wouldn't show it.
- run-level statistics

promptfoo also keeps an index of runs in `results/promptfoo/` for `npm run view`.
`results/` is git-ignored, with two exceptions:

- **`results/baseline/`** holds one versioned reference run, so a fresh checkout has
  an example of the expected output and a fixed point of comparison.
  `2026-10-12_qwen3.5-9b_ctx32768.json` is the full suite on `qwen3.5:9b` (Q4_K_M, digest
  `56671c2ab938`) with Ollama 0.40.3, the default settings, and seed 2026: 35 of 41
  passed. Its hardware is recorded in the report; `npm run summary` shows it. Timings
  from other hardware aren't comparable with it; pass/fail results usually are. All six failures are genuine model errors:
  - two wrong sums (`mt_trip_total`, `gq_combine_capacity`)
  - a tie broken in the wrong order (`if_csv_sorted_exact`)
  - two wrong program traces (`pg_trace_*`)
  - a code answer that rambled past its 2,048-token budget (`pg_write_parse_duration`)

  Replace it deliberately when the suite or the reference model changes, not after
  every run.
- **`results/001_context_recall/`** holds the original context-recall experiment's
  results.

Compare runs:

```bash
npm run summary -- results/baseline/*.json                      # the reference run in detail
npm run summary -- results/baseline/*.json results/runs/<new>.json   # a new run beside it
npm run summary -- results/runs/*.json                          # one row per run
```

## Interpreting results

| Status | Meaning |
|---|---|
| pass | Meets the test's acceptance criteria. |
| fail | Objectively wrong: wrong value, invalid JSON, schema violation, failed hidden tests, invented detail. Some failures are marked **REVIEW** in the summary (reason `REVIEW:` in promptfoo): "possibly acceptable, but no simple rule can decide", e.g. the right value alongside other candidates, or an abstention without the agreed marker. They count as failures until you inspect them. |
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
  than a cold prompt (~3,800 vs ~5,000 tok/s on the baseline run's hardware). So prompt throughput is reported
  only for requests with nothing cached, and as unavailable when Ollama omits the
  cached count (older versions). With `--repeat` at temperature 0, later repeats are
  mostly cached.
- Decode throughput is the most comparable figure across runs. Thinking tokens count
  as output tokens.

## Adding a test

### How a test runs

A test is a YAML entry in one of the `scenarios/*.yaml` files. When you run
`npm run eval`, promptfoo does the following for each test:

1. **Builds the prompt.** It reads the test's `vars.messages`, a chat history, and
   renders it through the single prompt template in `promptfooconfig.yaml`.
2. **Calls the model.** It hands the history to the provider
   (`helpers/ollama_provider.py`), which sends one request to Ollama. The run-wide
   settings (model, `num_ctx`, temperature, seed) come from the config and `EVAL_*`
   variables. The test can only set its own output budget (`metadata.num_predict`).
3. **Checks the answer.** It runs every assertion under `assert` against the model's
   answer, called `output`. **The test passes only if every assertion passes.**
4. **Records the result.** The answer, each assertion's verdict and reason, and the
   provider's timings go into the run report and the web viewer.

If the provider decides the run can't be trusted (prompt too long, answer cut off),
the test becomes *invalid* and no assertion runs. See [Interpreting results](#interpreting-results).

### Anatomy of a test

```yaml
- description: cr_release_moved           # unique id, shown everywhere; prefix with the category
  metadata:
    category: corrections                 # grouping in the summary; filter with --filter-metadata
    purpose: Report the corrected release day, not the original.   # one sentence: what it measures
    num_predict: 32                       # optional: max answer tokens (default 512)
    cutoff: fail                          # optional: see below (default: invalid)
  vars:
    messages:                             # the full conversation sent to the model
      - {role: system, content: "Answer with one word."}
      - {role: user, content: "The release is on Friday."}
      - {role: assistant, content: "Noted: Friday."}
      - {role: user, content: "Correction: it moved to Monday. When is the release?"}
  assert:                                 # every assertion must pass
    - {type: icontains, value: monday}
    - {type: not-icontains, value: friday}
```

| Field | Required | Meaning |
|---|---|---|
| `description` | yes | Unique test ID. It's what `--filter-pattern` matches, and what the summary and viewer display. Start it with the category prefix (`lc_`, `if_`, `mt_`, `cr_`, `gq_`, `pg_`). |
| `metadata.category` | yes | The category the summary groups by. `--filter-metadata category=…` selects it. A new name creates a new category; no code changes needed. |
| `metadata.purpose` | yes | One sentence on what capability the test checks. Documentation for whoever reads the results. |
| `metadata.num_predict` | no | The answer's token budget. Keep it tight for short answers and generous for code. If the answer runs out of budget before passing, the test is **invalid**. |
| `metadata.cutoff` | no | `fail` makes running out of budget count as a **failure** instead. Use it only when the budget is far beyond what a correct answer needs (~10×), so rambling is the model's fault. It's ignored in thinking mode. |
| `vars.messages` | yes | The conversation (see below). |
| `assert` | yes | The checks on the answer (see below). |

### Supplying the conversation

**Write it out.** `messages` is a list of `{role, content}` entries:
- `role` is `system`, `user`, or `assistant`.
- The last message should be the `user` question you're grading.
- Assistant turns are **synthetic**: you write them, so they are history, not
  something the model produced during the test. That's how multi-turn and corrections
  scenarios are built.
- For multi-line text, use a YAML block scalar:

```yaml
      - role: user
        content: |
          First line of a longer message.
          Second line, kept exactly as written.
```

**Reuse a long text across tests in one file.** Define it once with a YAML anchor
(`&name`) and refer to it with an alias (`*name`). `grounded_qa.yaml` does this for
its reference document:

```yaml
      - {role: user, content: &document "The full reference document..."}   # first test: define
      ...
      - {role: user, content: *document}                                     # later tests: reuse
```

**Generate it** (long inputs only). Instead of writing messages, name a generator
from `helpers/long_context.py` and its parameters, and grade with
`long_context_answer`. That assertion regenerates the expected answer from the same
seed, so you never write the answer down:

```yaml
- description: lc_notes_late_60
  metadata: {category: long_context, num_predict: 64, purpose: Recall a fact late in a 60-note log.}
  vars:
    generator: meeting_notes
    params: {notes: 60, position: late, similar_distractors: 3}
    messages: file://helpers/long_context.py
  assert:
    - {type: python, value: file://helpers/assertions.py:long_context_answer}
```

| Generator | Parameters | Builds | Expected answer |
|---|---|---|---|
| `meeting_notes` | `notes` (≥ 8, ~120 tokens each), `position` (`early`/`middle`/`late` or a fraction 0–1), `similar_distractors` (0–6) | A meeting-log conversation with one target fact. Distractors repeat the same sentence pattern for similar project names. | the `MARIGOLD-ddddd` code |
| `server_inventory` | `rows` (≥ 20, ~45 tokens each), `position` | A markdown host table with near-miss hostnames | the target host's IP address |

### Choosing assertions

Use the simplest check that is objective and not brittle. Two things to rule out:
- a correct answer failing over formatting the prompt didn't require
- a wrong answer passing by mentioning the right value among others

**promptfoo built-ins** (deterministic; no code needed):

| `type` | Passes when | Example |
|---|---|---|
| `equals` | The output is exactly `value`, character for character. There's no trimming and no case folding. If `value` is an object or list, the output must parse as JSON equal to it. | `{type: equals, value: "STATUS=OK"}` |
| `contains` / `icontains` | The output contains `value`. `icontains` ignores case. | `{type: icontains, value: postgres}` |
| `contains-all` / `icontains-all` | It contains every string in the list. | `{type: icontains-all, value: [Dana, "40221"]}` |
| `contains-any` / `icontains-any` | It contains at least one string in the list. | `{type: icontains-any, value: [yes, correct]}` |
| `starts-with` | The output begins with `value`. | `{type: starts-with, value: "- "}` |
| `regex` | A JavaScript regex matches somewhere in the output. It is **case-sensitive**; for case-insensitive matching use `javascript` (next rows). | `{type: regex, value: '\b40221\b'}` |
| `is-json` | The **whole** output parses as JSON (no code fences or prose). If `value` is given, it is a JSON Schema the parsed result must satisfy. | see `instruction_following.yaml` |
| `word-count` | The word count is within bounds. | `{type: word-count, value: {min: 3, max: 12}}` |
| `javascript` | A JavaScript expression over `output` returns true. Write one line, or several lines ending in `return`. You can return `{pass, score, reason}` to explain a failure. | `{type: javascript, value: '/^no\b/i.test(output)'}` |
| `python` | A Python function returns pass. Used for the project's own checks below. | `{type: python, value: file://helpers/assertions.py:number, config: {expected: 8}}` |

**Negation.** Prefix almost any type with `not-` to invert it:
`not-icontains`, `not-regex`, `not-contains-any`. This is how you say "must not
mention the old value".

**Multi-line JavaScript example**, which also explains its failure:

```yaml
    - type: javascript
      value: |
        const lines = output.trim().split('\n');
        if (lines.length !== 3) return {pass: false, score: 0, reason: `${lines.length} lines, expected 3`};
        return true;
```

promptfoo also offers model-graded types (`llm-rubric`, `factuality`, `similar`, …).
They need a second "judge" model and aren't deterministic, so this suite deliberately
doesn't use them.

**Project assertions** (`helpers/assertions.py`). Use these when the built-ins would
be brittle. Each is called as
`{type: python, value: file://helpers/assertions.py:<name>, config: {...}}`:

| Name | Use it for | `config` keys |
|---|---|---|
| `exact` | A precisely specified short answer, where harmless formatting shouldn't fail it. Before comparing, it ignores case and extra whitespace, strips quotes and backticks around the answer, a final period, and a code fence around the whole answer. | `expected` (string, or a list of accepted answers), `case_sensitive` (default `false`) |
| `number` | Numeric answers. Every number in the answer must equal `expected`. If the right number appears among others (e.g. a worked calculation), the result is **review**. | `expected`, `tolerance` (default 0), `extra`: `review` (default) or `fail` |
| `choice` | Multiple choice. It extracts one option letter from forms like `C`, `(C)`, `Answer: C`, or `**C**`. If it can't find a single letter, the result is **review**. | `expected` (a letter), `options` (default `ABCD`) |
| `regex_extract` | Answers you identify by pattern, such as names, codes or days. It collects every distinct match, and these must equal `expected`. | `pattern`, `expected` (string, or a list compared as a set), `ignore_case` (default `true`), `extra`: `review` (default) or `fail` when other candidates appear too |
| `abstain` | Questions that can't be answered from the supplied material. It passes on the agreed marker, and fails if invented specifics appear, even next to the marker. A refusal worded differently is **review**. | `marker` (default `NOT IN SOURCE`), `forbidden` (regexes for invented details), `abstention_phrases` |
| `python_function` | Code-writing tasks. It extracts the function from the answer and runs hidden tests in the sandbox. Return values are compared after JSON normalization, so tuples equal lists. | `function` (name), `tests`: a list of `{args: [...], expected: ...}` or `{args: [...], raises: ValueError}`, optionally `kwargs`; `timeout` (default 10 s) |
| `long_context_answer` | Generated long-context tests. | none; it reads `vars.generator` and `vars.params` |

**How a project assertion reports its result:**
- **Review** counts as a failure, marked `REVIEW` in the summary (its promptfoo
  reason starts with `REVIEW:`). It means the answer may be acceptable but a rule
  can't decide; read it yourself. A test with `cutoff: fail` whose answer ran out of
  budget is a plain failure, never review: running out of budget is the failure.
- **Error** means a broken `config` (e.g. a missing `expected`). It shows up with a
  reason starting `ERROR:`, and the summary counts it as an error, never as a model
  failure.

### Workflow for a new test

1. **Pick the file** for the capability, e.g. `scenarios/corrections.yaml`.
2. **Write the test** and **verify the expected answer independently.** For example,
   run a program whose output the model must predict, and do arithmetic twice. A wrong
   expectation is the most common bug in a test suite.
3. **Run only the new test:** `npm run eval -- --filter-pattern '^cr_release_moved$'`.
4. **Read the result,** in the summary or `npm run view`, and ask two questions:
   - Would a correct answer in a slightly different format still pass? If not, the
     prompt should demand the format, or a more tolerant assertion should replace it.
   - Could a wrong answer pass? Add a `not-` assertion for the tempting wrong value.
5. If you changed anything in `helpers/`, run `npm test`.

**A new category** needs a new file. Create `scenarios/<name>.yaml` (a YAML list of
tests) and add `file://scenarios/<name>.yaml` to the `tests` list in
`promptfooconfig.yaml`. The summary picks up the new `metadata.category`
automatically.

**A new generator** is a function in `helpers/long_context.py`. It takes `(seed,
**params)` and returns `Generated(messages, grader_updates={"expected": ...})`.
Register it in `GENERATORS` with a regex that finds candidate answers in the model's
output, and add a test to `tests/test_long_context.py`.

## Running model-generated code

`python_function` assertions run code written by the model, through layered
safeguards in `helpers/sandbox.py`:

1. **Static policy.** The code is parsed first. Only pure standard-library modules may
   be imported, and file, eval, and introspection builtins and dunder attributes are
   rejected. Rejected code is never run and is graded review. Code that doesn't parse
   at all is a failure.
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
