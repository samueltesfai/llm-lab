# Design Specification: Open WebUI Integration Evaluations for `llm-lab`

**Status:** Implementation specification for a coding agent  
**Prepared:** 2026-10-13  
**Repository:** https://github.com/samueltesfai/llm-lab  
**Implementation target:** locally hosted Open WebUI + Ollama (`qwen3.5:9b` initially)  
**Framework:** existing pinned Promptfoo + Python helpers managed with `uv`

> **Execution gate:** First establish and verify a standalone, real Open WebUI API smoke test. **Do not build the Promptfoo integration suite or reorganize its source layout until that smoke test passes.** If the installed Open WebUI version differs from the current online API documentation, discover and validate the compatible behavior; do not silently substitute a stateless API call or mark a blocked capability as a model failure.

## 1. Goal and boundaries

Add reproducible **application-level** evaluations alongside the repository's existing **model-level** evaluations. The new tests determine whether Open WebUI correctly persists and selects information, supports model-directed retrieval and memory, compacts context, and retrieves documents when driven through its actual saved-chat application workflow. Results should distinguish **application configuration, state setup, retrieval/tool choice, model use of retrieved facts, context limits, and infrastructure failures**.

The existing direct-Ollama suite remains the model reference. Do **not** re-run the same 41 questions unchanged via a different endpoint and call that application testing. Instead, seed state *inside Open WebUI*, ask questions that cannot be answered from the evaluation prompt alone, and observe which application mechanism provided the evidence. Maintain a small number of deterministic, synthetic scenarios that test real architectural differences.

**Required deliverables:** working smoke test and verification evidence; standardized repository layout and working commands; minimal Open WebUI client/provider; Promptfoo scenarios and fixtures; unit tests with mocked HTTP; setup/cleanup safeguards; executable documentation; persisted evaluation reports. All new functionality must be implementable using the user's existing local Open WebUI/Ollama installation; do not build a new UI, inference engine, generic agent framework, separate evaluation framework, or custom memory policy.

**Non-goals:** replacing Open WebUI; implementing alternative frontend support; code-interpreter grading or container orchestration; developing custom `request()` filters or context summarizers as a prerequisite; estimating a model's general intelligence; introducing LLM-as-judge grading; implementing a multi-user production test service. Code execution through Open WebUI/Open Terminal is a *feature of the assistant* rather than a replacement for the project's existing `helpers/sandbox.py` grading sandbox; it is not part of this initial integration suite.

### Baseline facts about the repository

Verified against public `main` on 2026-10-13:

- `promptfooconfig.yaml` at root selects `file://helpers/ollama_provider.py` and six scenario files under root `scenarios/`.
- **41 direct-Ollama cases** across `long_context`, `instruction_following`, `multi_turn`, `corrections`, `grounded_qa`, and `programming`.
- `helpers/`: `ollama_provider.py`, `long_context.py`, `assertions.py`, `sandbox.py`, `summary.py`.
- `tests/` contains pytest coverage for those helpers; `package.json` pins Promptfoo **0.123.1**, `uv` manages Python, and `npm run eval`, `npm run view`, `npm test` already work.
- Tracked reference result: `results/baseline/2026-10-12_qwen3.5-9b_ctx32768.json` (35/41 passes; 6 model failures; 0 invalid/error), run on Ollama **0.40.3** with `qwen3.5:9b` Q4_K_M, `num_ctx=32768`, thinking off. This is **historical evidence**, not a new integration baseline.
- `experiments/001_context_recall/` and `results/001_context_recall/` are retained historical research; leave them in place.

Preserve the current model-level semantics, reported metrics, result file, and CLI compatibility. No changes to the known six failure classifications are expected from the structural migration.

## 2. Core conceptual model

Treat the following as **distinct experimental mechanisms**, even when the user question is identical:

| Mechanism | State location | How information reaches Qwen | What is being tested |
|---|---|---|---|
| Active-chat history | Saved current chat | Prior turns supplied in a later completion request | Correct construction/persistence of a real conversation and use of prior turns |
| Past-chat search | *Other* saved chats, same account | Qwen chooses `search_chats`; optionally `view_chat` retrieves one full chat | Tool availability, tool choice, lexical search and follow-up reading |
| Injected persistent memory | User-scoped saved memory records | Open WebUI injects memories into system context | Injection and answer use, *not* tool selection |
| Tool-driven memory | Same memory store | Qwen chooses `search_memories`, `list_memories`, `add_memory`, etc. | Autonomous tool selection and memory CRUD |
| Context compaction | Active chat + saved summary checkpoint | Open WebUI summarizes old messages and keeps recent ones | Preservation of facts despite a shortened inference context |
| Knowledge/document retrieval | Uploaded files/knowledge | Full-context attachment, focused RAG, or model-directed knowledge tools | Extraction/indexing, selection, grounding, attribution |

A saved previous conversation is **not automatically injected** into a new chat. Past-chat search is a separate model-directed operation and `search_chats` excludes the current chat. Stored memories are separate from saved conversations. Memory tools can be available even when automatic memory injection is disabled. Both are separate from the model's ordinary within-context reasoning. See official [History & Search](https://docs.openwebui.com/features/chat-conversations/chat-features/history-search/) and [Memory & Personalization](https://docs.openwebui.com/features/chat-conversations/memory/).

**Experiment invariant:** For a cross-chat question, do not send any of the seeded source conversation's text directly in the *new* chat's `messages` payload. For a tool-selection experiment, do not invoke the tool directly from the test runner as a shortcut. For a memory-creation experiment, do not precreate the memory whose creation is being tested.

## 3. Required execution order and stop conditions

Implement in this order. Every gate must be verified and documented before advancing.

### Gate A — Standalone API compatibility smoke test (first)

Write a minimal standalone Python smoke test (e.g. `scripts/smoke_openwebui.py`) using Python's standard library if practical. It must run **without Promptfoo** and use the same API authentication, message-tree, and tool-capable completion path that the final integration tests will use. It must not depend on the new repository layout. No new Python package merely for HTTP, retrying or polling.

**Preconditions:**

1. Obtain `OPENWEBUI_URL`, `OPENWEBUI_API_KEY`, and `OPENWEBUI_MODEL` from environment variables. The API key must belong to a **dedicated test account**, not the everyday user. Never put the key in source, logs, committed output, or fixture files.
2. Check the installed Open WebUI version (from a documented supported version endpoint if available, otherwise record the operator-supplied container image tag/digest). Record the backing Ollama version/model identifier separately; do not assume that online docs match the installed release.
3. `GET /api/models` (authenticated), confirm the exact intended model ID; verify the API key/permissions. Determine whether API keys are enabled globally and for the test user.
4. Require explicit confirmation that the account/instance is test-only and that the run may create and delete test-owned records. Do **not** change global/admin settings programmatically during smoke testing.

**Smoke sequence (must actually execute):**

1. Generate valid UUIDs for initial user message, assistant placeholder, chat/session IDs as appropriate; create a uniquely named chat using `POST /api/v1/chats/new` with valid `history.messages`, `history.currentId`, `parentId` and `childrenIds` tree relationships. Verify the returned chat ID.
2. Call `POST /api/chat/completions` using the **saved-chat native flow**: `chat_id`, assistant message `id`, `session_id`, `stream: true`, selected model, and `messages` containing the current in-chat history. Disable title/tags/follow-up background tasks for determinism. **Do not send a `tools` field**, even an empty array, in tests that rely on Open WebUI's built-in tools; it bypasses built-in resolution.
3. Recognize the asynchronous acknowledgment (`task_ids`, `chat_id`) as **not the answer**. Poll `GET /api/tasks/chat/{chat_id}` until there are no active task IDs, with bounded elapsed timeout and error handling; then `GET /api/v1/chats/{chat_id}`. Read the *specific* assistant placeholder by ID, require a completed answer, and record whether `output`, `usage`, and `sources` were returned. If the installed version differs, use the documented compatible completion-detection mechanism and verify message `done` rather than assuming any nonempty content implies completion.
4. Append a genuine second user turn **to the same saved chat**, maintaining the message tree and `currentId`, using the supported chat-update endpoint `POST /api/v1/chats/{chat_id}` as needed; trigger a second completion with a new assistant placeholder. Ask about an exact synthetic fact introduced earlier. Verify both turns were persisted, the answer references the fact, and the frontend-style tree remains coherent.
5. Create **another** fresh chat with the same test user. Confirm by reading its persisted record that it contains **no copied fact** from the first chat and is independent. A successful independent chat is sufficient at smoke stage; a native history-search tool-call proof belongs to the additional smoke qualification below.
6. Delete **only the smoke-created chats** using `DELETE /api/v1/chats/{id}`. Verify disappearance/absence with the appropriate documented read behavior. Cleanup must run in a `finally` block even when the response, polling, or assertions fail; report a cleanup failure explicitly.

**Native-tool qualification — required before enabling any tool-driven scenarios:** Using the same saved-chat `stream: true` + `session_id` workflow, configure/verify Native Function Calling and the model's Chat History built-in-tool category, then seed a different chat containing a synthetic fact and ask for that fact from a new chat. Confirm the **server executed** `search_chats` and/or `view_chat` (when exposed in structured output/trace) or obtain comparable installed-version evidence. A final answer alone is not proof that the tool loop ran. If built-in tool execution cannot be verified, the smoke's core chat path can pass, but **tool-driven integration scenarios are blocked**; record that distinction rather than silently dropping `session_id` or changing to a stateless completion.

**Smoke evidence:** Write a short, sanitized machine-readable JSON result under ignored `results/runs/` capturing endpoint/status checks, Open WebUI version/image identity when available, model ID, creation/read/second-turn/delete results, native-tool qualification, timings, and failure stage. Do not store API keys or sensitive headers. Do not claim feature parity without confirming it. Provide a one-command documented invocation and a mocked unit test for success, timeout, wrong message ID, missing model, and cleanup-on-error.

**Gate A acceptance:** A real local run demonstrates authentication, model discovery, one saved-chat two-turn cycle, completion detection, saved-history integrity, and verified deletion; native-tool qualification succeeds separately for tool-dependent scenarios. If the basic smoke fails, **stop before repository refactor or suite construction**, record the exact API incompatibility and the installed version. No mocked or dry-run-only success satisfies the gate.

### Gate B — Standardize the project layout (after successful basic smoke)

Use **symmetrical** suite locations. Avoid one root suite and one nested suite.

```text
llm-lab/
  evals/
    ollama/
      promptfooconfig.yaml
      scenarios/
        long_context.yaml
        instruction_following.yaml
        multi_turn.yaml
        corrections.yaml
        grounded_qa.yaml
        programming.yaml
    openwebui/
      promptfooconfig.yaml
      scenarios/
        conversation.yaml
        history_search.yaml
        memory.yaml
        compaction.yaml
        knowledge.yaml
      fixtures/
        cedar_conversation.json
        cedar_reference.md
        cedar_reference_old.md
  helpers/
    ollama_provider.py
    openwebui_provider.py
    openwebui_client.py              # only if separation clearly simplifies tests
    long_context.py
    assertions.py
    sandbox.py
    summary.py
  scripts/
    smoke_openwebui.py
  tests/
    ...existing pytest tests...
    test_openwebui_provider.py
    test_openwebui_smoke.py
  experiments/
    001_context_recall/              # leave untouched
  results/
    001_context_recall/              # leave untouched
    baseline/                        # preserve tracked historical baseline
    runs/                            # generated; gitignored
    promptfoo/                       # generated; gitignored
  package.json
  package-lock.json
  pyproject.toml
  README.md
```

This is a **target shape**, not an instruction to create empty scenario files or split a tiny module prematurely. `helpers/` is shared; test-specific execution lives in its two providers. Do not move the historical experiment or the historical result. Migrate existing scenario files and config via `git mv`; update paths based on **observed Promptfoo path resolution** (config-relative vs cwd-relative), not assumption. Keep the Python import path, evaluation helpers and existing tests working.

CLI requirements:

- `npm run eval:ollama`: original 41 cases under `evals/ollama/`.
- `npm run eval:openwebui`: new integration cases under `evals/openwebui/` (explicitly requires a configured running instance; must not silently start or mutate one's personal instance).
- `npm run eval`: backward-compatible alias for `eval:ollama` (does not run both).
- `npm run view`, `npm test`, and `npm run summary` (if already supported) retain current behavior.
- Both suites share the pinned Promptfoo version, `uv` environment and `helpers/assertions.py`, and use consistently named ignored run-report outputs, e.g. `results/runs/ollama/` vs `results/runs/openwebui/`. Preserve the tracked baseline at its current location.
- Retain telemetry and update checks disabled, Promptfoo response caching disabled, and evaluation concurrency **1**.

**Gate B acceptance:** The migrated Ollama suite still has **41 discoverable cases**; its existing tests pass; paths resolve from the documented repository root; the historical baseline JSON is byte-for-byte unchanged; shell commands and README examples work. A fresh full-model run is preferred when Ollama is available, but structural checks and pytest must always run. Migration must not alter grader behavior or test data.

### Gate C — Stateful Open WebUI Promptfoo suite

Only begin after Gate A's basic smoke and Gate B. Tool-specific scenarios require the native-tool qualification, independently.

Use a **single narrow Python Promptfoo provider** (`helpers/openwebui_provider.py`), backed by a small HTTP helper only if needed. Promptfoo owns loading test cases, filtering, repeat runs, assertions, reports and viewer. The provider owns **prepare → seed → exercise → observe → return answer → cleanup**. Tests should be declarative rather than Python scripts per scenario. Avoid generic task engines, plugin architectures, fixture DSLs and redundant assertion implementations.

Each test case identifies a versioned fixture (path or simple key), an experimental mode/profile, the final evaluation question and objective assertions. The fixture contents are **seeded into Open WebUI's persistent state**, not sent as hidden context in the evaluated question. Favor a small number of simple fixture types (saved chat, memory, file), backed by straight Python logic over an elaborate schema. A scenario can have a few actual API chat turns where behavior depends on how those turns are generated; a stored conversation fixture may also be seeded through the documented chat APIs for deterministic history-search tests. Label the seed method explicitly.

#### Test lifecycle contract

1. **Prepare:** Check authenticated model availability, expected Open WebUI version/profile, `num_ctx` and feature prerequisites; generate test-run and per-case UUIDs; start from isolated test state. Do not assume that a request flag overrides server/admin permissions. If a required capability is not enabled, return an **infrastructure/configuration error**, not a wrong model answer.
2. **Seed:** Establish one deterministic synthetic project fact in the mechanism under test (saved chat, persistent memory, knowledge collection). Confirm via read API that it was persisted/processed before asking anything. Save all created IDs to a cleanup manifest as soon as each API creation succeeds. **Do not inject the source fact into the evaluated question/chat in ways that bypass the targeted mechanism.**
3. **Exercise:** Run the application-level completion using the smoke-verified saved-chat path; drive native tools in Open WebUI, not inside the provider. Where the experiment depends on autonomous memory creation, introduce the fact conversationally and allow the model to choose whether to store it.
4. **Observe:** Retrieve the exact completed assistant message and available structured `output`, `usage`, `sources`, tool actions, `context_usage`, effective model/settings and elapsed wall time. Annotate unavailable fields as **not observed**; never equate missing trace data with proof a tool did not execute. Distinguish assistant output from async acknowledgment or incomplete generation.
5. **Grade:** Return final answer text as Promptfoo `output`; reuse deterministic assertions in `helpers/assertions.py` or Promptfoo built-ins. Return structured metadata without mixing it into the answer. For tool-selection tests, provide specific machine-checkable metadata (e.g. search invoked yes/no/unknown, returned relevant evidence yes/no/unknown) rather than grading on answer text alone when trace is available.
6. **Clean up:** Delete only test-owned chats, memory records, files and collections, in dependency-safe order; verify deletion; run in `finally`. Never blanket-delete user data. If records remain, classify this as an error and print their IDs for recovery. If an aborted run leaves test-owned resources, provide a documented safe recovery mechanism based on stored IDs, never an unrestricted purge.

**Isolation:** Require an explicitly designated test account, ideally a disposable Open WebUI instance using its own data volume. Memory and history are user-scoped; distinct test-run identifiers alone are not enough to prevent unrelated saved memories from influencing answers. At minimum, assert the test user's starting relevant state is empty/known, or use disposable per-run state. Never execute destructive cleanup against an unverified personal account. Tests should run sequentially; repetitions need fresh, independently seeded state. If automatic memory review/background tasks are on, record them and ensure they cannot contaminate tests whose goal is to isolate other mechanisms.

**Error taxonomy:**

- `pass` — expected observable application behavior and answer.
- `fail` — valid completed experiment, wrong answer or confirmed incorrect tool/state behavior.
- `invalid` — no defensible judgment due to context truncation, cut-off, incomplete answer, unconfirmed readiness, wrong fixture state or ambiguous observability needed for grading.
- `error` — authentication, missing API/model/config, HTTP failures, timeouts, indexing failure, tool transport failure, incompatible version, cleanup failure, provider/fixture bug.

Use the existing provider's `INVALID:` and `ERROR:` conventions where compatible with Promptfoo's failure classification, and make the console/report summary understand application metadata without assuming Ollama-only metrics such as warmup or GPU placement are present. When final-answer correctness can be judged but tool traces are absent, report answer correctness and **tool-choice unknown** rather than inventing proof of the mechanism.

## 4. API contract reference and prerequisites

**Source of truth:** the **installed version's observed behavior** first, official Open WebUI documentation second. These endpoints and fields are published in current docs/source as of 2026-10-13, but individual local releases may differ. Validate on Gate A, record response shapes, and keep adaptation minimal and version-scoped.

| Operation | HTTP call | Notes |
|---|---|---|
| Authenticate/discover models | `GET /api/models` | Bearer API key; verify exact model ID and access |
| Create saved chat | `POST /api/v1/chats/new` | Provide chat metadata and valid message tree; caller-generated UUIDs |
| Update chat/message tree | `POST /api/v1/chats/{id}` | Preserve `history.messages`, `currentId`, `childrenIds`, parent relationships |
| Execute native multi-round turn | `POST /api/chat/completions` | Requires saved `chat_id` + assistant `id` + `stream:true` + `session_id` for built-in tools; asynchronous task acknowledgment |
| Await completion | `GET /api/tasks/chat/{id}` | Bounded polling until no pending task IDs; also inspect message completion state |
| Read final result | `GET /api/v1/chats/{id}` | `chat.history.messages[assistant_id]`; may include `content`, `output`, `usage`, `sources`, `context_usage` |
| Delete test chat | `DELETE /api/v1/chats/{id}` | Verify deletion; requires delete permission |
| List/create/delete saved memories | `GET /api/v1/memories/`, `POST /api/v1/memories/add`, `DELETE /api/v1/memories/{id}` | Confirm per-user permissions and record schema with local version |
| Upload document | `POST /api/v1/files/` | Multipart upload; processing asynchronous |
| Wait for processed file | `GET /api/v1/files/{id}/process/status` | `pending`/`completed`/`failed`; **not** chat attachment's `processed` field |
| Attach to KB | `POST /api/v1/knowledge/{id}/file/add` | Only after processing complete; create collection via installed-version API |
| Retrieve from KB | Chat completion with appropriate `files` attachment or model's knowledge tools | Focused Retrieval and Full Context differ; record selected mode |
| Inspect/trigger compaction | `GET /api/v1/chats/{id}` (`context_usage`); `POST /api/v1/chats/{id}/compact` | Distinguish manual endpoint from automatic threshold-triggered compaction |
| Delete files/KBs | Installed-version file/knowledge delete endpoints | **Validate documented/backend routes before implementation**, check vector cleanup; never guess an endpoint from naming pattern |

**Important distinction for API requests:** The documented *stateless* completion path cannot be substituted for the full native agentic loop. With `chat_id`, assistant `id`, `stream:true` and `session_id`, Open WebUI itself resolves and executes built-in tools; it writes the result to the saved chat. The HTTP response may be only an accepted-task acknowledgment. Omitting `session_id` disables built-in tools, and specifying a custom `tools` array bypasses Open WebUI's built-in tool resolution. `features.memory=true` alone does not override global/per-user/per-model permissions. Keep unrelated features (web search, code interpreter, image generation) disabled in controlled tests; preserve required memory and history categories per profile. Refer to [Server-Side Tool Calling](https://docs.openwebui.com/reference/server-side-tool-calling/) and [Backend-Controlled API Flow](https://docs.openwebui.com/reference/api-flow/).

**Tool invocation semantics:** `search_chats` and `view_chat`, and `search_memories` / `list_memories` / `add_memory` / memory update tools are **model-selected built-ins**, not endpoints the runner should call to simulate Qwen's behavior. Seeding via the memories REST API is allowed for *retrieval* experiments. In history-search tests, seed one old chat and ask in a distinct new chat. The chat tool search is text matching, not necessarily semantic search; include exact-keyword and paraphrase variants, and treat retrieval-tool selection as a separate outcome from reading its returned evidence.

**File and knowledge readiness:** Upload completion does **not** imply indexing readiness. Poll `process/status` to `completed` before adding to a collection or asking grounded questions; fail early on `failed` or timeout. Collection creation and deletion APIs and access permissions must be version-verified. Keep fixture docs small, uniquely named, deterministic and non-sensitive.

## 5. Configuration profiles and experimental controls

Maintain **separate run profiles** from **seed fixtures**. The same fixture/question may run under different known settings; do not encode entire conversations in provider configuration or overwrite server settings per test. Some settings are instance-wide or model-wide, not per request; record their verified values before a run and instruct operators to use explicit isolated instances/profiles when necessary.

| Profile | Saved chat history | History tools | Automatic memory injection | Memory tools | Compaction | Purpose |
|---|---:|---:|---:|---:|---:|---|
| `plain` | Yes, current chat | Off | Off | Off | Off | Control and same-chat continuity |
| `history_tools` | Yes | On | Off | Off | Off | Cross-chat search and tool selection |
| `memory_injected` | Yes | Off | On | Optional Off | Off | Information availability without search choice |
| `memory_tools` | Yes | Off | **Off** | On | Off | Autonomous memory search/create/update |
| `compaction` | Yes | Off | Off | Off | On | Long-session summary preservation |
| `knowledge` | Yes | Off | Off | Off | Off | File/collection retrieval modes |

This table defines intended experimental conditions. It is **not** proof that every toggle is available on every installation. Require preflight checks or explicitly documented manual settings. Avoid a generic profile-switching engine or fragile automated admin UI configuration.

**Comparable model settings:** Match Qwen and Ollama settings where feasible to direct-Ollama reference (`qwen3.5:9b`, `num_ctx=32768`, temperature `0`, thinking disabled, seed `2026`). Open WebUI may use Ollama's much smaller default context unless `num_ctx` is explicitly set at model/request level; verify the actual effective context (e.g. provider settings or Ollama telemetry) before comparing with model-level baselines. Do not assume the Open WebUI API enforces Ollama `truncate:false`. Detect suspected truncation/cutoff and classify outcomes cautiously. Record model digest, Ollama/Open WebUI versions, profile, tool categories, effective options, compaction threshold, and retrieval settings when accessible.

**Compaction settings:** Built-in feature is off by default. Current docs list `ENABLE_CONTEXT_COMPACTION`, `CONTEXT_COMPACTION_TOKEN_THRESHOLD` (default `80000`), `CONTEXT_COMPACTION_TOKEN_CAP`, `CONTEXT_COMPACTION_RETENTION_PERCENTAGE` (default `40`), `CONTEXT_COMPACTION_MODEL`, `CONTEXT_COMPACTION_PROMPT_TEMPLATE`, plus model-level threshold override. Use a **well-below-32K** threshold for compacted tests and a no-compaction comparison. Crossing the threshold does **not guarantee compaction**: a suitable turn boundary may be absent or summary may fail; check observable checkpoint/usage and logs where supported. Keep the full stored transcript distinct from what reached inference; `context_usage` is an estimate, not exact Ollama token accounting. **Do not implement a custom summarizer for this suite.**

**Memory settings:** Current docs list `ENABLE_MEMORIES` (default true), `ENABLE_MEMORY_SYSTEM_CONTEXT` (default true), `MEMORIES_CONTEXT_CHAR_LIMIT` and `MEMORIES_USER_CHAR_LIMIT` (both default 2000), and optional `ENABLE_MEMORY_BACKGROUND_REVIEW` (default off). Memory tools require Native Function Calling and per-model categories; stored-memory injection is independent of those tool calls. In `memory_tools`, disable system injection to attribute outcomes to retrieval. Do not mix background auto-review into autonomous tool-creation tests unless specifically asserted and verified; it can change memory state without a tool call from Qwen.

**Costs:** Record at least completed-turn wall time, observed number of tool invocations/inference rounds, last-call usage and aggregate usage when available. Current API docs distinguish final-call `prompt_tokens`/`completion_tokens` from cumulative `input_tokens`/`output_tokens`/`total_tokens` across tool rounds; do not sum them again as if they were disjoint. Do not claim Ollama's token throughput or KV-cache hits when the application API does not expose them. Tool schemas and retrieved results add context; `view_chat` can return a **full old conversation**, potentially expensive on a 32K model.

## 6. Initial deterministic scenario specifications

Start with **six mechanism families**, using a coherent fictional project (“Project Cedar”) and distinct random/test-run-specific identifiers to prevent accidental hits. Keep explicit known answers and one confounding factor at a time. Scenario IDs must be stable, distinct, discoverable/filterable in Promptfoo; fixture facts can be namespaced by run ID without changing meaning. Do not assume model action equals app action.

### C1. Same-chat continuity (`conversation`)

- **Seed:** Create an actual chat via API. User introduces “Project Cedar stores data in SQLite because it must operate offline”; assistant actually replies. Add one or more unrelated real turns (not synthetically pasted assistant turns).
- **Exercise:** In the *same chat*, ask what database Cedar uses and why.
- **Expected:** SQLite; offline requirement. Reject PostgreSQL and invented rationale.
- **Observables:** Stored messages, parent/child message tree, whether information reached later completion; answer correctness. No history-search call is expected or required.
- **Control:** Direct-Ollama test with same relevant prior turns explicitly supplied, used for diagnosis rather than an extra integration mechanism.

### C2. Cross-chat history search (`history_search`)

- **Seed:** Persist *Chat A* with Cedar decision, confirm retrievable via read API. Start *Chat B* with only the question; saved memories empty and injection off.
- **Exercise:** Ask in B: “What database did we select for Project Cedar?”
- **Variants:** Exact keywords vs natural paraphrase (“Which storage engine did our offline project settle on?”); a negative control with history tools disabled. A third chat with a plausible unrelated database decision can serve as a distractor.
- **Expected:** With relevant history retrieval, SQLite + offline reason. With no retrieval available, answer should **acknowledge lack of knowledge**, not invent a decision.
- **Observables:** Was `search_chats` attempted? Query terms? Search results? Did `view_chat` load A? What content reached Qwen? Final correctness. Trace missing = **unknown**, not proof of no call.
- **Important:** Search returns hits/snippets; `view_chat` can load a complete chat, with higher context cost.

### C3. Injected saved memory (`memory`)

- **Seed:** Via `POST /api/v1/memories/add`, persist a small `context` memory describing the Cedar decision; confirm with `GET /api/v1/memories/`. No old chat containing the fact.
- **Exercise:** Ask in a new chat without restating the fact. Compare `memory_injected` against a known no-memory control.
- **Expected:** Accurate recall when injected; abstention without availability. A memory-tool call is **not required** in injection mode.
- **Observables:** Saved memory ID/type/path and content, relevant injection settings, final answer, inference overhead if measurable.

### C4. Autonomous memory operations (`memory`)

- **Seed:** With memory injection off but memory tools enabled, either (a) manually seed a memory and ask a paraphrased question to test `search_memories`/`list_memories`; or (b) start with empty memory state, tell the model a durable Cedar preference, and **ask it to remember**, to test its use of `add_memory`. These are separate cases and must not be conflated.
- **Exercise:** Search case asks from fresh chat; creation case inspects the memory bank after the assistant response; correction case changes SQLite decision and checks that the current state is correctly updated rather than accumulating contradictory memories.
- **Expected:** Correct answer when found; a durable correct persisted memory after creation; current-state memory after correction.
- **Observables:** Tool call selection and response, persisted memory state (including duplicates), correct final answer; classify transport/config failures separately. Tool-selection reliability for this 9B model is an outcome to measure, not an assumption.

### C5. Built-in context compaction (`compaction`)

- **Seed:** A genuine multi-turn Cedar conversation containing a uniquely identifiable decision early, then enough controlled, realistic distractor turns to exceed a deliberately lowered compaction threshold. Use existing `helpers/long_context.py` logic only where helpful for deterministic text; **establish the messages through Open WebUI's normal saved-chat workflow**.
- **Exercise:** Ask for the early fact after the threshold. Use equivalent un-compacted control with same meaningful facts/turns and comparable model settings.
- **Expected:** Accurate early-fact recall despite reduced inference context; establish whether a compaction checkpoint actually occurred (do not count merely crossing an estimated threshold as proof).
- **Observables:** Full stored transcript intact; usage/threshold/checkpoint evidence if accessible; summaries where safely observable; answer accuracy, summarization overhead and latency. Automatic compaction is distinct from deliberately calling `POST /api/v1/chats/{id}/compact` (manual baseline, if included).
- **Limits:** May fail to split long indivisible exchanges; summary can omit a detail; estimated token usage is not exact; test must not hang generating huge chat histories.

### C6. Grounded document retrieval (`knowledge`)

- **Seed:** Upload concise `cedar_reference.md` containing unique factual decisions and an explicitly **unknown** datum. Wait for processing status `completed`; attach directly or place into a newly created test KB and verify association. For conflict case, upload `cedar_reference_old.md` and a newer clearly dated authority document containing a changed decision.
- **Exercise:** Ask a specific source-grounded question, an out-of-source question, and an old-vs-new authority question. Compare **Full Context** and **Focused Retrieval** as separate, confirmed app modes; model-directed knowledge tools are another separate variant requiring native-tool qualification.
- **Expected:** Accurate grounded fact; explicit abstention for unknown; newer authoritative value for conflict. Never let the final evaluation question quote the answer.
- **Observables:** file processing/index readiness, attached file/KB IDs and mode, retrieved `sources`/passages where available, answer, relevant source version. A wrong answer with correct source supplied differs from no relevant source supplied.
- **Cleanup:** Delete the test KB, files and any associated vectors using locally verified API calls; no deletion of shared collections.

### Grading guidance

Use simple fixed responses where possible, such as “Answer in the form `database | reason`,” but don't require exact phrasing for natural-language answers if it creates false failures. Use independent positive and negative assertions (`SQLite`, `offline`, *not* `PostgreSQL`); custom deterministic assertions only where necessary. Avoid a substring check that passes merely because the wrong option is mentioned among other answers. Unknown tests must distinguish explicit uncertainty from hallucinated claims. Correct answer alone cannot certify that specific retrieval tools ran; report both answer and mechanism evidence.

## 7. Provider and test interface

The Open WebUI provider should accept **one rendered test descriptor**, locate its deterministic fixture, seed required state, execute the question and return `{output, metadata}` or an existing-convention error to Promptfoo. Keep the Promptfoo config simple and consistent with the existing Python `file://` provider. Avoid making test authors write HTTP requests or Python methods per case.

**Illustrative, not an existing Promptfoo-native schema:**

```yaml
- description: ow_history_cedar_exact
  metadata:
    category: history_search
    purpose: Retrieve one project decision from an older chat.
  vars:
    fixture: cedar_conversation
    profile: history_tools
    question: What database did we choose for Project Cedar, and why?
  assert:
    - type: icontains
      value: SQLite
    - type: icontains
      value: offline
    - type: not-icontains
      value: PostgreSQL
```

The provider owns the meaning of `fixture`/`profile`/`question`; implement the smallest necessary schema and document it. Do not mistake a YAML `setup` string for stored application state. Fixture files must be static and contain only synthetic nonsecret content; provider-created UUIDs and ownership state belong in run records, not fixtures.

Provider metadata should, where available, include:

```text
suite = openwebui
scenario_id, profile, fixture_id, run_id
openwebui_version_or_image, ollama_version, model_id, model_digest (if available)
num_ctx_requested, num_ctx_verified (or null), sampling/thinking settings
seed_method; created resource IDs (local ignored log only)
turns; observed tool names + number of rounds + whether relevant evidence returned
last_round_usage; aggregate_usage; context_usage; source references
elapsed_s; indexing_wait_s; cleanup_status; diagnostics
observability flags (observed / unavailable / unknown)
```

Never include API tokens, complete HTTP headers, real personal content, or unredacted unrelated chats. Generated test prompts and answers may appear in ignored Promptfoo outputs; use only fictitious fixtures. Handle polling deadlines per step and entire scenario, limited retries for transient transport failures (not model answers), well-formed JSON checks, and failed background tasks. Avoid silent retries that create duplicate chats or memories.

## 8. Validation and acceptance tests

### Offline/mocked tests (must not need Ollama or Open WebUI)

- Correct auth-header handling without leaking bearer token in logs.
- Chat creation message tree: valid UUIDs, `childrenIds`, `parentId`, `history.currentId`; second-turn links and updates.
- Native-tool request contains `chat_id`, `id`, `session_id`, `stream:true`; no `tools` override; model ID and flags correct.
- Async acknowledgment never mistaken for content; polling success, task failure, task timeout, empty/missing assistant message and incomplete `done`.
- File processing `pending → completed`, `failed`, and timeout.
- Memory seeding/readback and fixture isolation.
- Cleanup for successful run, provider error, timeout, assertion failure, and interrupted partial creation (using persisted manifest).
- HTTP 401/403/404, missing model, unsupported version/endpoint, malformed JSON classified `error`; context/cutoff uncertainty as `invalid`.
- Missing tool trace represented as `unknown` rather than `false`.
- Repeat evaluation creates fresh state; no cross-case bleed.
- Migrated Ollama suite's existing pytest tests remain green.

### Live acceptance matrix

| Deliverable | Required verification |
|---|---|
| Gate A smoke | API path actually exercised, two turns persisted/read correctly, completion verified, cleanup checked; native-tool qualification separately demonstrated or blocking issue recorded |
| Layout migration | `evals/ollama/` has exactly 41 cases; original CLI preserved; existing baseline file untouched; `npm test` passes |
| Open WebUI suite | `npm run eval:openwebui` loads and executes the scenarios against explicit test configuration; saves Promptfoo JSON report; viewable through `npm run view` |
| State safety | Every test fixture created/verified in server state; run cleans up all owned chats/memories/files/KBs; unrelated user records untouched |
| Mechanism fidelity | New-chat questions do not contain seeded answers; memory creation not preseeded; native-tool scenarios use real tool loop; compaction occurrence verified rather than inferred |
| Failure trustworthiness | Setup/indexing/config errors not reported as model failures; incomplete or unsupported observations identified honestly; assertions target objective outcomes |
| Results | Category-level pass/fail/invalid/error, final answer and available structured metadata, version/settings/context/cost info, sanitized run artifacts |

**Required final implementation report from the agent:** Summarize actual Git changes, commands run, smoke evidence (including native-tool qualification), which integration tests were executed *live* versus mocked, final pass/fail/invalid/error counts, installed Open WebUI and Ollama versions, remaining API/version-specific unsupported behavior if any, and cleanup confirmation. Do **not** claim an unexecuted test passed. Do **not** silently relax a gate to finish the project.

## 9. Known risks and mitigations

| Risk | Required handling |
|---|---|
| Current online docs diverge from installed release | Version-pin instance; validate real responses in Gate A; record scoped adapter difference, not broad compatibility machinery |
| Native built-in tools silently absent | Demand saved chat + assistant ID + `stream:true` + `session_id`, verify Native mode/capabilities/permissions, omit user-supplied `tools` |
| WebUI page-level settings differ from API | Use explicit profiles, verify settings and one API-versus-browser parity case for tool behavior; fail preflight rather than assuming parity |
| Test inadvertently supplies target answer | Audit B's request messages and application seed methods; use fixture readback and distinct source/evaluation chats |
| Stateful cross-test contamination | Dedicated/disposable test account, fixed starting state, unique run IDs, sequential execution, cleanup manifest and readback |
| Background memory reviewer changes state | Disable in isolated search/injection tests, or explicitly account for it; distinguish tool-created from review-created memories |
| Ollama default context vs 32K reference | Verify effective context and record it; don't interpret truncation as failed retrieval |
| Compaction threshold crossed but no checkpoint | Verify compaction actually happened; no automatic inference from length/`context_usage` alone |
| File uploaded but not indexed | Poll `process/status=completed`, verify KB association; errors/timeouts not model failures |
| Completion task accepted but answer not finished | Poll task state and assistant ID until completed; timeout robustly |
| `view_chat` inflates context and latency | Log tool calls/usage and differentiate last-call context from aggregate tokens; no invented exact costs |
| Tool/source trace absent | Use `unknown`, grade final answer separately, never fabricate a tool path |
| API auth/deletion permissions | Dedicated key with only necessary permissions, verify deletion, never blanket-wipe personal state |
| Tool interactions are stochastic | Deterministic fixtures/settings; use repeatable trials, not single-run claims of perfect reliability |
| Additional dependencies/custom framework creep | Reuse Python stdlib, `uv`, Promptfoo, existing assertions; keep integration-specific code small |

If the installed version supports it, a lightweight `request()` filter could reveal every outgoing inference payload, including retrieved memory/RAG information on later tool rounds. **Do not install or implement this just to satisfy the first suite.** The current docs say this hook was added in v0.11.2; `request()` runs on each provider call, unlike `inlet()`, which runs once per turn. An observational filter is a diagnostic instrument with its own overhead and data exposure, not a memory policy.

## 10. Official source references and verification caveats

The following are research inputs for the agent, **not substitutes for testing the locally installed version**:

1. [Server-Side Tool Calling over the API](https://docs.openwebui.com/reference/server-side-tool-calling/) — critical saved-chat + async native-tool contract, model/permission preconditions, task polling, completion record.
2. [Backend-Controlled API Flow](https://docs.openwebui.com/reference/api-flow/) — UUIDs, message tree, `currentId` and `childrenIds`, create/update chats and multi-turn completions.
3. [API Endpoints](https://docs.openwebui.com/reference/api-endpoints/) — chat completions, file upload/indexing, cumulative versus last-round usage fields, model inference options.
4. [History & Search](https://docs.openwebui.com/features/chat-conversations/chat-features/history-search/) — chat persistence and model-directed `search_chats` / `view_chat`.
5. [Memory & Personalization](https://docs.openwebui.com/features/chat-conversations/memory/) — tool-directed vs injected memories; settings, user permissions and background review.
6. [Open WebUI memory routes in backend source](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/routers/memories.py) — `GET /`, `POST /add`, `DELETE /{id}` under memory router; review version-specific route mounting and schema.
7. [Context Window / Compaction](https://docs.openwebui.com/troubleshooting/context-window/) — default full-history behavior, built-in summarization, thresholds, optional request filters, `context_usage` caveats.
8. [Environment Configuration](https://docs.openwebui.com/reference/env-configuration/) — flags, thresholds, persistence/config precedence; some documentation pages have differed about compaction-model fallback, so verify effective behavior rather than infer it from one description.
9. [Knowledge Bases and Document Chat](https://docs.openwebui.com/features/workspace/knowledge/) — Full Context vs Focused Retrieval and model-directed knowledge tools.
10. [Filter Function](https://docs.openwebui.com/features/extensibility/plugin/functions/filter/) — `inlet()` vs per-inference `request()` observability, opt-in diagnostic only.

The project’s own checked source remains the authority for existing scenario semantics, reporter behavior, and file layout: [llm-lab](https://github.com/samueltesfai/llm-lab). Current Open WebUI documentation evolves rapidly; claims about endpoint response shape, compaction defaults and feature availability must be validated by the Gate A smoke test and target-version checks.
