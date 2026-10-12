"""Summaries, console output, and result files."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

STATUSES = ("pass", "fail", "review", "invalid", "error", "skipped")
LABELS = {"pass": "PASS", "fail": "FAIL", "review": "REVIEW", "invalid": "INVALID",
          "error": "ERROR", "skipped": "SKIP"}


def _counts(records: list[dict]) -> dict:
    counter = Counter(r["status"] for r in records)
    counts = {status: counter.get(status, 0) for status in STATUSES}
    graded = counts["pass"] + counts["fail"] + counts["review"]
    counts["pass_rate"] = round(counts["pass"] / graded, 3) if graded else None
    return counts


def _stats(values: list[float]) -> dict | None:
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return None
    return {"median": round(statistics.median(values), 1), "min": round(min(values), 1),
            "max": round(max(values), 1), "n": len(values)}


def summarize(records: list[dict]) -> dict:
    by_category: dict[str, list] = defaultdict(list)
    by_case: dict[str, list] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
        by_case[record["case_id"]].append(record)

    measured = [r["metrics"] for r in records if r.get("metrics")]
    prompt_total = sum(m["prompt_tokens"] or 0 for m in measured)
    cached_total = sum(m["prompt_cached_tokens"] or 0 for m in measured)
    performance = {
        "requests": len(measured),
        "wall_s_total": round(sum(m["wall_s"] for m in measured), 2),
        "wall_s": _stats([m["wall_s"] for m in measured]),
        "decode_tokens_per_s": _stats([m["decode_tokens_per_s"] for m in measured]),
        "prompt_tokens_per_s_uncached": _stats([m["prompt_tokens_per_s"] for m in measured]),
        "prompt_tokens_total": prompt_total,
        "prompt_cached_tokens_total": cached_total,
        "prompt_cached_fraction": round(cached_total / prompt_total, 3) if prompt_total else None,
        "output_tokens_total": sum(m["output_tokens"] or 0 for m in measured),
        "cold_loads_during_cases": sum(bool(m["cold_load"]) for m in measured),
    }
    return {
        "overall": _counts(records),
        "categories": {cat: _counts(recs) for cat, recs in by_category.items()},
        "cases": {
            case_id: {"category": recs[0]["category"], "trials": len(recs),
                      "statuses": dict(Counter(r["status"] for r in recs)),
                      "all_passed": all(r["status"] == "pass" for r in recs)}
            for case_id, recs in by_case.items()
        },
        "performance": performance,
    }


def exit_code(summary: dict, aborted: bool) -> int:
    """0: every graded result passed (review/skipped allowed); 1: some failed;
    2: errors, invalid runs, or an aborted run."""
    overall = summary["overall"]
    if aborted or overall["error"] or overall["invalid"]:
        return 2
    return 1 if overall["fail"] else 0


# -- files ----------------------------------------------------------------------------------

class ResultWriter:
    """results/runs/<run_id>/{run.json, results.jsonl}. Never overwrites an existing run."""

    def __init__(self, results_dir: Path, run_id: str):
        self.directory = Path(results_dir) / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.results_path = self.directory / "results.jsonl"
        self.run_path = self.directory / "run.json"

    def append(self, record: dict):
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_run(self, data: dict):
        temporary = self.run_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(self.run_path)


# -- console --------------------------------------------------------------------------------

def _fmt(value, suffix="", digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value:,}{suffix}"


def print_header(run: dict, case_count: int):
    config, env = run["config"], run["environment"]
    model = env.get("model") or {}
    print(f"Model:    {config['model']}  ({model.get('parameter_size', '?')}, "
          f"{model.get('quantization_level', '?')}, digest {str(model.get('digest', '?'))[:12]})")
    print(f"Ollama:   {env.get('ollama_version', '?')} at {run['host']}")
    print(f"Options:  num_ctx={config['num_ctx']} temperature={config['temperature']} seed={config['seed']} "
          f"think={config['think']} trials={config['trials']}"
          + (f" num_predict={config['num_predict']}" if config["num_predict"] else "")
          + (f" extra={config['extra_options']}" if config["extra_options"] else ""))
    print(f"Cases:    {case_count} selected (profile={run['selection']['profile']})")
    print(f"Results:  {run['results_dir']}")


def print_warmup(metrics: dict | None, loaded_before: bool | None):
    if metrics is None:
        print("Warm-up:  skipped (--no-warmup); the first case may include model load time")
        return
    state = "already loaded" if loaded_before else "not loaded"
    if loaded_before and metrics["cold_load"]:
        state += " but was reloaded (e.g. for a different num_ctx)"
    print(f"Warm-up:  model was {state}; load {_fmt(metrics['load_s'], 's', 2)}, "
          f"request wall {_fmt(metrics['wall_s'], 's', 2)}")


def print_record(record: dict, trials: int = 1, verbose: bool = False):
    label = LABELS.get(record["status"], record["status"].upper())
    trial = f" (trial {record['trial'] + 1}/{trials})" if trials > 1 and record["status"] != "skipped" else ""
    line = f"  [{label:<7}] {record['case_id']}{trial}"
    metrics = record.get("metrics")
    if metrics:
        cached = metrics["prompt_cached_tokens"]
        line += (f"  {_fmt(metrics['wall_s'], 's', 2)}  in {_fmt(metrics['prompt_tokens'])}"
                 + (f" ({_fmt(cached)} cached)" if cached else "")
                 + f"  out {_fmt(metrics['output_tokens'])}")
    print(line)
    if record["status"] != "pass" and record.get("detail"):
        print(f"            {record['detail'][:300]}")
    if verbose and record["status"] in ("fail", "review", "invalid") and record.get("answer") is not None:
        answer = record["answer"].strip().replace("\n", "\n            | ")
        print(f"            | {answer[:1500]}")


def print_summary(summary: dict, aborted: str | None):
    header = f"  {'category':<24}" + "".join(f"{LABELS[s]:>8}" for s in STATUSES) + "   pass rate"
    print("\nResults by category (one count per trial)")
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = list(summary["categories"].items()) + [("TOTAL", summary["overall"])]
    for name, counts in rows:
        rate = counts["pass_rate"]
        print(f"  {name:<24}" + "".join(f"{counts[s]:>8}" for s in STATUSES)
              + f"   {'-' if rate is None else f'{rate:.0%}'}")
    print("  pass rate = pass / (pass + fail + review); invalid, error and skipped are excluded")

    perf = summary["performance"]
    print("\nPerformance")
    if not perf["requests"]:
        print("  no completed requests")
    else:
        decode, prompt, wall = perf["decode_tokens_per_s"], perf["prompt_tokens_per_s_uncached"], perf["wall_s"]
        print(f"  requests: {perf['requests']}  total wall {_fmt(perf['wall_s_total'], 's', 1)}  "
              f"median wall {_fmt(wall['median'], 's', 2) if wall else '-'}")
        if decode:
            print(f"  decode:   median {_fmt(decode['median'])} tok/s (min {_fmt(decode['min'])}, max {_fmt(decode['max'])})")
        if prompt:
            print(f"  prompt:   median {_fmt(prompt['median'])} tok/s over uncached tokens "
                  f"(n={prompt['n']} requests with >=256 fresh tokens)")
        else:
            print("  prompt:   no request processed enough uncached tokens for a throughput figure")
        fraction = perf["prompt_cached_fraction"]
        print(f"  tokens:   {_fmt(perf['prompt_tokens_total'])} prompt "
              f"({'-' if fraction is None else f'{fraction:.0%}'} served from cache), "
              f"{_fmt(perf['output_tokens_total'])} output")
        if perf["cold_loads_during_cases"]:
            print(f"  WARNING: {perf['cold_loads_during_cases']} request(s) included a model (re)load; "
                  "their wall times are not comparable")
    if aborted:
        print(f"\nRUN ABORTED: {aborted}")
