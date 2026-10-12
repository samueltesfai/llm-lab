#!/usr/bin/env python3
"""One reproducible, local Ollama context-recall experiment.

No third-party Python packages, Open WebUI, or Internet access required.
Usage: python3 context_recall_test.py [--ctx 16384] [--notes 90]
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "http://127.0.0.1:11434/api/chat"
PROJECTS = ("Cedar", "Atlas", "Nimbus", "Willow", "Keystone", "Harbor", "Summit", "Orchid")
COMPONENTS = ("scheduler", "telemetry", "indexer", "build pipeline", "dashboard", "storage layer")
CONCERNS = ("latency", "observability", "documentation", "integration", "rollout", "testing")
CODE_WORDS = ("PEARL", "FALCON", "CITRUS", "EMBER", "LANTERN", "MESA", "RIVER", "PINE")


def build_messages(note_count: int, position: str, seed: int):
    """Construct one synthetic multi-turn history, then one factual recall question."""
    rng = random.Random(seed)
    needle_code = f"MARIGOLD-{rng.randrange(10000, 99999)}"
    insert_at = {"early": note_count // 8, "middle": note_count // 2, "late": 7 * note_count // 8}[position]
    messages = [{"role": "system", "content": (
        "You are answering questions about a fictional team's project meeting notes. "
        "Use only information in the conversation. For an exact code, return only the code; "
        "if it is not in the notes, answer UNKNOWN."
    )}]
    for i in range(note_count):
        if i == insert_at:
            messages.append({"role": "user", "content": (
                f"Important meeting note: The approved QA reference code for Project Aurora "
                f"is {needle_code}. This is the current reference code for Aurora, "
                "not a password or credential."
            )})
            messages.append({"role": "assistant", "content": "Noted the Aurora QA reference code."})
        project = rng.choice(PROJECTS)
        component = rng.choice(COMPONENTS)
        concern = rng.choice(CONCERNS)
        distractor = f"{rng.choice(CODE_WORDS)}-{rng.randrange(10000, 99999)}"
        messages.append({"role": "user", "content": (
            f"Meeting log {i + 1:03d}: For Project {project}, the {component} team "
            f"reviewed {concern} and assigned reference code {distractor} to that project's "
            "internal checklist. The team discussed whether the pending change could "
            "affect users, how to communicate the decision to stakeholders, which "
            "verification steps would be useful before deployment, and how to "
            "record the outcome in the project tracker. The discussion concluded "
            "that a follow-up review should compare the observed results with the "
            "previous iteration and document anything that remains unresolved."
        )})
        messages.append({"role": "assistant", "content": (
            f"Understood. Project {project}'s {component} review is recorded."
        )})
    messages.append({"role": "user", "content": (
        "What is the exact approved QA reference code for Project Aurora? "
        "Return only the code, or UNKNOWN if it was not provided."
    )})
    return messages, needle_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5:9b", help="An installed Ollama model tag")
    parser.add_argument("--ctx", type=int, default=16384, help="Requested context window in tokens")
    parser.add_argument("--notes", type=int, default=90, help="Number of distracting meeting notes")
    parser.add_argument("--position", choices=("early", "middle", "late"), default="early")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("context_results.jsonl"))
    args = parser.parse_args()
    if args.notes < 8 or args.ctx < 1024:
        parser.error("--notes must be >= 8 and --ctx must be >= 1024")

    messages, expected = build_messages(args.notes, args.position, args.seed)
    prompt_words = sum(len(msg["content"].split()) for msg in messages)
    print(f"Model: {args.model} | requested context: {args.ctx:,} tokens")
    print(f"Synthetic history: {args.notes} distracting notes; ~{prompt_words:,} words")
    print(f"Target location: {args.position} | messages: {len(messages)}")
    print("Sending one request to your local Ollama server; first load may take time...", flush=True)

    payload = {"model": args.model, "messages": messages, "stream": False,
               "think": False, "keep_alive": "5m",
               "options": {"num_ctx": args.ctx, "num_predict": 64,
                           "temperature": 0, "seed": args.seed}}
    request = Request(API_URL, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    start = time.monotonic()
    try:
        with urlopen(request, timeout=900) as response:
            result = json.load(response)
    except HTTPError as exc:
        print(f"Ollama HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except (URLError, TimeoutError) as exc:
        print(f"Couldn't reach Ollama or request timed out: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - start
    answer = result.get("message", {}).get("content", "").strip()
    # Exact-code score; allow a surrounding sentence but not a partial or wrong code.
    found_codes = re.findall(r"\bMARIGOLD-\d{5}\b", answer.upper())
    passed = (found_codes == [expected])
    prompt_tokens = result.get("prompt_eval_count")
    near_limit = (prompt_tokens is not None and prompt_tokens >= args.ctx - 256)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "requested_num_ctx": args.ctx,
        "note_count": args.notes,
        "position": args.position,
        "seed": args.seed,
        "history_word_count_approx": prompt_words,
        "message_count": len(messages),
        "expected": expected,
        "answer": answer,
        "passed_exact_code": passed,
        "prompt_tokens_reported": prompt_tokens,
        "response_tokens_reported": result.get("eval_count"),
        "wall_seconds": round(elapsed, 2),
        "model_total_seconds": round(result.get("total_duration", 0) / 1e9, 2),
        "model_load_seconds": round(result.get("load_duration", 0) / 1e9, 2),
        "possible_context_truncation": near_limit,
        "done_reason": result.get("done_reason"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nExpected: {expected}")
    print(f"Answer:   {answer!r}")
    print(f"Recall:   {'PASS' if passed else 'FAIL'}")
    print(f"Prompt tokens processed (Ollama): {prompt_tokens}")
    print(f"Elapsed wall time: {elapsed:.1f}s | output tokens: {result.get('eval_count')}")
    if near_limit:
        print("WARNING: Prompt token count is near the context cap. Input may have been truncated;")
        print("         don't interpret a recall failure as an attention failure without inspecting this.")
    print(f"Results appended to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
