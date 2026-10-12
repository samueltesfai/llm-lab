"""Command-line entry point: uv run python -m llm_eval [options]"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from llm_eval import report, sandbox
from llm_eval.ollama import DEFAULT_HOST, OllamaClient, OllamaError
from llm_eval.runner import RunConfig, config_dict, run_suite, warmup
from llm_eval.scenarios import ScenarioError, load_scenarios, select_cases

REPO_ROOT = Path(__file__).resolve().parent.parent
EXIT_SETUP_FAILED = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m llm_eval",
        description="Model-level evaluation suite for Ollama-hosted models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run = parser.add_argument_group("model and inference")
    run.add_argument("--model", default="qwen3.5:9b", help="installed Ollama model tag")
    run.add_argument("--host", default=DEFAULT_HOST, help="Ollama server URL")
    run.add_argument("--num-ctx", type=int, default=16384, help="context window requested from Ollama")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=2026,
                     help="seed for scenario generators and Ollama sampling (trial i uses seed+i)")
    run.add_argument("--num-predict", type=int, default=None,
                     help="max output tokens for every case (default: per-case value, else 512)")
    run.add_argument("--think", action="store_true",
                     help="enable the model's thinking mode (graded on the final answer only); "
                          "raise --num-predict, since thinking tokens count against it")
    run.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                     help="extra Ollama option, e.g. --option top_k=20 (repeatable; value parsed as JSON)")
    run.add_argument("--keep-alive", default="5m", help="how long Ollama keeps the model loaded")
    run.add_argument("--timeout", type=float, default=600, help="per-request timeout in seconds")
    run.add_argument("--allow-truncation", action="store_true",
                     help="let Ollama silently truncate over-long prompts (default: reject them; "
                          "such cases are reported as invalid)")
    run.add_argument("--no-warmup", action="store_true",
                     help="skip the untimed warm-up request that loads the model")

    pick = parser.add_argument_group("case selection")
    pick.add_argument("--profile", choices=("standard", "long", "all"), default="standard",
                      help="standard: quick default suite; long: expensive long-context cases")
    pick.add_argument("--category", action="append", default=[],
                      help="run only this category (repeatable or comma-separated)")
    pick.add_argument("--case", action="append", default=[],
                      help="run only cases whose id matches this glob, e.g. 'lc_*' (ignores --profile)")
    pick.add_argument("--trials", type=int, default=1, help="repetitions of each case")
    pick.add_argument("--list", action="store_true", help="list the selected cases and exit")

    out = parser.add_argument_group("files and output")
    out.add_argument("--scenarios-dir", type=Path, default=REPO_ROOT / "scenarios")
    out.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    out.add_argument("--label", default="", help="free-text note stored with the run, e.g. 'baseline'")
    out.add_argument("-v", "--verbose", action="store_true", help="print answers of non-passing cases")

    args = parser.parse_args(argv)
    args.category = [c for item in args.category for c in item.split(",") if c]
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    if args.num_ctx < 512:
        parser.error("--num-ctx must be >= 512")
    try:
        args.extra_options = _parse_options(args.option)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _parse_options(items: list[str]) -> dict:
    options = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--option expects KEY=VALUE, got {item!r}")
        if key in ("num_ctx", "temperature", "seed", "num_predict"):
            raise ValueError(f"use --{key.replace('_', '-')} instead of --option {key}")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options


def _git_state() -> dict:
    def git(*command):
        try:
            return subprocess.run(["git", *command], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _scenario_hashes(directory: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(directory.glob("*.yaml"))}


def _model_entry(models: list[dict], name: str) -> dict | None:
    candidates = {name, name + ":latest"}
    return next((m for m in models if m.get("name") in candidates or m.get("model") in candidates), None)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        cases = load_scenarios(args.scenarios_dir)
    except ScenarioError as exc:
        print(exc, file=sys.stderr)
        return EXIT_SETUP_FAILED
    selected = select_cases(cases, profile=args.profile, categories=args.category, patterns=args.case)

    if args.list:
        for case in selected:
            extra = f" [{case.profile}" + (f", min_ctx={case.min_ctx}" if case.min_ctx else "") + "]"
            print(f"{case.category:<22} {case.id:<34}{extra if case.profile != 'standard' else ''}")
            print(f"{'':<22} {case.purpose}")
        print(f"\n{len(selected)} case(s)")
        return 0
    if not selected:
        print("No cases match the selection (try --list, --profile all, or check --category/--case).",
              file=sys.stderr)
        return EXIT_SETUP_FAILED

    client = OllamaClient(args.host, timeout=args.timeout)
    try:
        ollama_version = client.version()
        installed = client.list_models()
    except OllamaError as exc:
        print(f"Cannot use Ollama at {args.host}: {exc}\nIs the server running (`ollama serve` / "
              "`systemctl status ollama`)?", file=sys.stderr)
        return EXIT_SETUP_FAILED
    entry = _model_entry(installed, args.model)
    if entry is None:
        names = ", ".join(sorted(m.get("name", "?") for m in installed)) or "none"
        print(f"Model {args.model!r} is not installed on this server (installed: {names}).\n"
              f"Install it yourself with `ollama pull {args.model}`; the suite never pulls models.",
              file=sys.stderr)
        return EXIT_SETUP_FAILED
    try:
        shown = client.show(args.model)
        loaded_before = any(m.get("name") == entry.get("name") for m in client.running())
    except OllamaError as exc:
        print(f"Could not inspect model {args.model!r}: {exc}", file=sys.stderr)
        return EXIT_SETUP_FAILED

    config = RunConfig(
        model=args.model, num_ctx=args.num_ctx, temperature=args.temperature, seed=args.seed,
        trials=args.trials, think=args.think, num_predict=args.num_predict,
        extra_options=args.extra_options, keep_alive=args.keep_alive,
        allow_truncation=args.allow_truncation,
    )
    started = datetime.now(timezone.utc)
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model)
    run_id = f"{started:%Y%m%d-%H%M%S}_{safe_model}_ctx{args.num_ctx}"
    writer = report.ResultWriter(args.results_dir, run_id)
    model_info = {
        "name": entry.get("name"), "digest": entry.get("digest"), "size_bytes": entry.get("size"),
        **(shown.get("details") or {}),
        "native_context_length": next((v for k, v in (shown.get("model_info") or {}).items()
                                       if k.endswith(".context_length")), None),
        "capabilities": shown.get("capabilities"),
    }
    run = {
        "run_id": run_id,
        "label": args.label,
        "status": "running",
        "started_utc": started.isoformat(timespec="seconds"),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "host": args.host,
        "results_dir": str(writer.directory),
        "config": config_dict(config),
        "selection": {"profile": args.profile, "categories": args.category, "case_patterns": args.case,
                      "case_ids": [c.id for c in selected]},
        "environment": {
            "ollama_version": ollama_version,
            "model": model_info,
            "model_loaded_before_run": loaded_before,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": _git_state(),
            "scenario_files_sha256_16": _scenario_hashes(args.scenarios_dir),
            "sandbox_network_isolated": bool(sandbox.network_isolation_prefix()),
        },
    }
    writer.write_run(run)
    report.print_header(run, len(selected))

    warm = None
    if not args.no_warmup:
        try:
            warm = warmup(client, config)
        except OllamaError as exc:
            print(f"Warm-up request failed: {exc}", file=sys.stderr)
            run.update(status="setup_failed", error=str(exc))
            writer.write_run(run)
            return EXIT_SETUP_FAILED
        run["warmup"] = warm
    try:
        # Ollama's view of the loaded model: memory placement and allocated context.
        run["environment"]["ollama_ps"] = next(
            ({k: m.get(k) for k in ("size", "size_vram", "context_length")}
             for m in client.running() if m.get("name") == entry.get("name")), None)
    except OllamaError:
        pass
    report.print_warmup(warm, loaded_before)
    print()

    def on_record(record):
        writer.append(record)
        report.print_record(record, args.trials, args.verbose)

    aborted = None
    try:
        records, aborted = run_suite(selected, client, config, run_id, on_record)
    except KeyboardInterrupt:
        aborted = "interrupted by user"
        records = [json.loads(line) for line in writer.results_path.read_text().splitlines()] \
            if writer.results_path.exists() else []

    summary = report.summarize(records)
    run.update(status="aborted" if aborted else "completed", abort_reason=aborted,
               finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), summary=summary)
    writer.write_run(run)
    report.print_summary(summary, aborted)
    print(f"\nSaved: {writer.run_path}\n       {writer.results_path}")
    return 130 if aborted == "interrupted by user" else report.exit_code(summary, bool(aborted))

