"""Compare stored runs side by side.

    uv run python -m llm_eval.compare                 # every run in results/runs
    uv run python -m llm_eval.compare RUN_DIR ...     # specific runs
    uv run python -m llm_eval.compare --cases         # also show per-case outcomes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_eval import CATEGORIES

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "results" / "runs"
SHORT = {"long_context": "longctx", "instruction_following": "instr", "multi_turn": "multiturn",
         "corrections": "correct", "grounded_qa": "grounded", "programming": "program"}


def load_runs(paths: list[Path]) -> list[dict]:
    run_files = []
    for path in paths:
        if (path / "run.json").exists():
            run_files.append(path / "run.json")
        else:
            run_files.extend(sorted(path.glob("*/run.json")))
    runs = []
    for run_file in run_files:
        try:
            run = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping {run_file}: {exc}", file=sys.stderr)
            continue
        if run.get("summary"):
            runs.append(run)
    return runs


def _rate(counts: dict | None) -> str:
    if not counts:
        return "-"
    graded = counts["pass"] + counts["fail"] + counts["review"]
    return f"{counts['pass']}/{graded}" if graded else "-"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_DIR])
    parser.add_argument("--cases", action="store_true", help="show a per-case status matrix")
    args = parser.parse_args(argv)
    runs = load_runs(args.paths)
    if not runs:
        print("No completed runs found.")
        return 1

    categories = [c for c in CATEGORIES if any(c in r["summary"]["categories"] for r in runs)]
    header = (f"{'run':<44} {'ctx':>6} {'temp':>4} {'think':>5} "
              + " ".join(f"{SHORT.get(c, c)[:9]:>9}" for c in categories)
              + f" {'total':>7} {'inv/err':>7} {'decode':>7} {'prompt':>8}")
    print(header)
    print("-" * len(header))
    for run in runs:
        summary, config = run["summary"], run["config"]
        overall, perf = summary["overall"], summary["performance"]
        decode = (perf.get("decode_tokens_per_s") or {}).get("median")
        prompt = (perf.get("prompt_tokens_per_s_uncached") or {}).get("median")
        print(f"{run['run_id'][:44]:<44} {config['num_ctx']:>6} {config['temperature']:>4} "
              f"{str(config['think'])[0]:>5} "
              + " ".join(f"{_rate(summary['categories'].get(c)):>9}" for c in categories)
              + f" {_rate(overall):>7} {overall['invalid'] + overall['error']:>7}"
              + f" {decode if decode is not None else '-':>7} {prompt if prompt is not None else '-':>8}")
    print("\ncategory columns: passed/graded (graded = pass + fail + review); "
          "decode/prompt: median tokens/s (prompt over uncached tokens)")

    if args.cases:
        case_ids = sorted({cid for r in runs for cid in r["summary"]["cases"]})
        print(f"\n{'case':<34}" + " ".join(f"{i:>8}" for i in range(1, len(runs) + 1)))
        for case_id in case_ids:
            cells = []
            for run in runs:
                case = run["summary"]["cases"].get(case_id)
                if case is None:
                    cells.append("")
                elif case["all_passed"]:
                    cells.append("ok")
                else:
                    cells.append("/".join(s[:7] for s in sorted(case["statuses"])))
            print(f"{case_id:<34}" + " ".join(f"{c:>8}" for c in cells))
        for i, run in enumerate(runs, 1):
            print(f"  {i}: {run['run_id']}" + (f"  ({run['label']})" if run.get("label") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
