"""Console summary of promptfoo result files: outcomes per category, and Ollama timings.

promptfoo's own output covers individual tests (`npm run view` for detail). This
adds what it doesn't show: pass / fail / invalid / error per category, the timing
metrics recorded by ollama_provider.py, and an exit status that separates model
failures from untrustworthy runs. Given several files, prints one row per run.

    .venv/bin/python helpers/summary.py results/runs/<run>.json [more.json ...]

Exit status (single file): 0 all passed, 1 some failed, 2 some results were invalid
or errored (their pass rate isn't trustworthy).
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

STATUSES = ("pass", "fail", "invalid", "error")


def _median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.median(values), 1) if values else None


# Assertion reasons that mean the check itself couldn't run (our assertions' "ERROR:"
# prefix, or promptfoo's message for an exception inside a Python assertion).
ASSERTION_ERROR_PREFIXES = ("ERROR:", "Error running Python script")


def _status(row: dict) -> str:
    # promptfoo failureReason: 0 none, 1 an assertion failed, 2 provider/runtime error.
    # (The `error` field is also set on plain assertion failures, so it can't be used.)
    if row.get("failureReason") == 2:
        message = str(row.get("error") or (row.get("response") or {}).get("error") or "")
        return "invalid" if message.startswith("INVALID") else "error"
    if row.get("success"):
        return "pass"
    components = (row.get("gradingResult") or {}).get("componentResults") or []
    if any(str(c.get("reason", "")).startswith(ASSERTION_ERROR_PREFIXES) for c in components if not c.get("pass")):
        return "error"
    return "fail"


def summarize(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))["results"]["results"]
    categories: dict[str, Counter] = defaultdict(Counter)
    problems, metrics = [], []
    for row in rows:
        test = row.get("testCase") or {}
        status = _status(row)
        categories[(test.get("metadata") or {}).get("category", "uncategorized")][status] += 1
        response = row.get("response") or {}
        if (response.get("metadata") or {}).get("metrics"):
            metrics.append(response["metadata"]["metrics"])
        if status != "pass":
            reason = row.get("error") if row.get("failureReason") == 2 else (row.get("gradingResult") or {}).get("reason")
            problems.append((status, test.get("description", "?"), " ".join(str(reason).split())[:160]))
    meta = next(((r.get("response") or {}).get("metadata") for r in rows
                 if (r.get("response") or {}).get("metadata")), None) or {}
    return {
        "file": path.name, "model": meta.get("model"), "think": meta.get("think"),
        "num_ctx": (meta.get("options") or {}).get("num_ctx"), "server": meta.get("server") or {},
        "categories": categories, "problems": problems,
        "decode_tps": _median([m["decode_tokens_per_s"] for m in metrics]),
        "prompt_tps": _median([m["prompt_tokens_per_s"] for m in metrics]),
        "wall_total": round(sum(m["wall_s"] for m in metrics), 1),
        "cold_loads": sum(bool(m["cold_load"]) for m in metrics),
        "warmup_load_s": (meta.get("warmup") or {}).get("load_s"),
    }


def exit_status(summary: dict) -> int:
    total = sum(summary["categories"].values(), Counter())
    if total["invalid"] or total["error"]:
        return 2
    return 1 if total["fail"] else 0


def print_one(s: dict):
    server = s["server"]
    print(f"\n{s['model']} ({server.get('parameter_size')}, {server.get('quantization')}, "
          f"digest {str(server.get('model_digest'))[:12]}), Ollama {server.get('ollama_version')}, "
          f"num_ctx={s['num_ctx']}, think={s['think']}")
    print(f"  {'category':<24}" + "".join(f"{x:>9}" for x in STATUSES))
    total = Counter()
    for category, counts in s["categories"].items():
        total.update(counts)
        print(f"  {category:<24}" + "".join(f"{counts[x]:>9}" for x in STATUSES))
    print(f"  {'TOTAL':<24}" + "".join(f"{total[x]:>9}" for x in STATUSES))
    reviews = [p for p in s["problems"] if p[0] == "fail" and p[2].startswith("REVIEW: ")]
    if reviews:
        print(f"  ({len(reviews)} of the failures may be acceptable and need a human look: marked REVIEW)")
    for status, test_id, reason in s["problems"]:
        label = status.upper()
        if status == "fail" and reason.startswith("REVIEW: "):
            label, reason = "REVIEW", reason[len("REVIEW: "):]
        print(f"  {label:<8}{test_id}: {reason}")
    print(f"  timing: warm-up load {s['warmup_load_s']}s, total wall {s['wall_total']}s, median decode "
          f"{s['decode_tps']} tok/s, median prompt {s['prompt_tps']} tok/s (requests with nothing cached)")
    if s["cold_loads"]:
        print(f"  WARNING: {s['cold_loads']} request(s) included a model reload; their timings aren't comparable")


def print_comparison(summaries: list[dict]):
    categories = sorted({c for s in summaries for c in s["categories"]})
    print(f"{'run':<28}{'model':<16}{'ctx':>6} " + "".join(f"{c[:12]:>13}" for c in categories)
          + f"{'inv/err':>8}{'decode':>8}{'prompt':>8}")
    for s in summaries:
        cells = []
        for c in categories:
            counts = s["categories"].get(c, Counter())
            graded = counts["pass"] + counts["fail"]
            cells.append(f"{counts['pass']}/{graded}" if graded else "-")
        bad = sum(c["invalid"] + c["error"] for c in s["categories"].values())
        print(f"{s['file'][:27]:<28}{str(s['model'])[:15]:<16}{str(s['num_ctx']):>6} "
              + "".join(f"{x:>13}" for x in cells) + f"{bad:>8}{str(s['decode_tps']):>8}{str(s['prompt_tps']):>8}")
    print("category cells: passed/graded (graded = pass + fail; invalid and errors excluded)")


def main(argv=None) -> int:
    paths = [Path(p) for p in (sys.argv[1:] if argv is None else argv)]
    if not paths:
        print(__doc__)
        return 2
    summaries = [summarize(p) for p in paths]
    if len(summaries) > 1:
        print_comparison(summaries)
        return 0
    print_one(summaries[0])
    return exit_status(summaries[0])


if __name__ == "__main__":
    raise SystemExit(main())
