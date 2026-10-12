"""Load, validate and select scenario definitions from scenarios/*.yaml.

File layout:

    category: corrections            # required
    description: ...                 # optional, for humans
    shared: {...}                    # optional, ignored; a place to define YAML anchors
    defaults:                        # optional, merged into every case in the file
      options: {num_predict: 128}
    cases:
      - id: cr_meeting_moved         # unique across all files
        purpose: What capability this case evaluates.
        messages: [...]              # literal chat messages, OR
        generator: {name: ..., params: {...}}   # a deterministic generator
        grader: {type: exact, expected: ...}
        options: {num_predict: 64}   # optional per-case Ollama options
        profile: standard            # standard (default) | long
        min_ctx: 32768               # optional: skip when --num-ctx is smaller
        cutoff: invalid              # answer cut off by num_predict: invalid (default) | fail
        tags: [...]                  # optional
        notes: ...                   # optional free text
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llm_eval import CATEGORIES
from llm_eval.generators import GENERATORS, Generated, case_seed
from llm_eval.graders import GRADERS

CASE_KEYS = {"id", "purpose", "messages", "generator", "grader", "options", "profile",
             "min_ctx", "cutoff", "tags", "notes"}
FILE_KEYS = {"category", "description", "defaults", "shared", "cases"}  # shared: free-form, for YAML anchors
PROFILES = ("standard", "long")
# Options that define the run as a whole; cases may not override them.
RUN_LEVEL_OPTIONS = {"num_ctx", "seed", "temperature"}
ROLES = {"system", "user", "assistant"}


class ScenarioError(Exception):
    pass


@dataclass
class Case:
    id: str
    category: str
    purpose: str
    grader: dict
    source: str
    messages: list[dict] | None = None
    generator: dict | None = None
    options: dict = field(default_factory=dict)
    profile: str = "standard"
    min_ctx: int | None = None
    cutoff: str = "invalid"
    tags: list[str] = field(default_factory=list)
    notes: str | None = None

    def build(self, base_seed: int) -> tuple[Generated, int | None]:
        """Materialize messages and the final grader spec. Returns (generated, seed used)."""
        if self.generator is None:
            return Generated(messages=self.messages, grader_updates={}, metadata={}), None
        seed = case_seed(base_seed, self.id)
        function = GENERATORS[self.generator["name"]]
        return function(seed, **self.generator.get("params", {})), seed


def _check_regexes(spec: dict, where: str, problems: list[str]):
    candidates = []
    for key in ("pattern",):
        if key in spec:
            candidates.append(spec[key])
    for key in ("all", "any", "none", "forbidden", "forbidden_text", "abstention_phrases"):
        candidates.extend(spec.get(key, []))
    for check in spec.get("checks", []):
        if isinstance(check, dict):
            for kind in ("each_line_matches", "contains", "not_contains"):
                if kind in check:
                    candidates.append(check[kind])
    for pattern in candidates:
        try:
            re.compile(pattern)
        except (re.error, TypeError) as exc:
            problems.append(f"{where}: bad regex {pattern!r}: {exc}")


def _validate_case(raw: dict, where: str, problems: list[str]):
    unknown = set(raw) - CASE_KEYS
    if unknown:
        problems.append(f"{where}: unknown keys {sorted(unknown)}")
    for key in ("id", "purpose", "grader"):
        if key not in raw:
            problems.append(f"{where}: missing '{key}'")
    if ("messages" in raw) == ("generator" in raw):
        problems.append(f"{where}: needs exactly one of 'messages' or 'generator'")

    if "messages" in raw:
        messages = raw["messages"]
        if not isinstance(messages, list) or not messages:
            problems.append(f"{where}: 'messages' must be a non-empty list")
        else:
            for i, message in enumerate(messages):
                if (not isinstance(message, dict) or message.get("role") not in ROLES
                        or not isinstance(message.get("content"), str) or set(message) - {"role", "content"}):
                    problems.append(f"{where}: message {i} must be {{role: system|user|assistant, content: str}}")
            if isinstance(messages[-1], dict) and messages[-1].get("role") != "user":
                problems.append(f"{where}: the last message must have role 'user'")

    if "generator" in raw:
        generator = raw["generator"]
        if not isinstance(generator, dict) or generator.get("name") not in GENERATORS:
            problems.append(f"{where}: unknown generator; available: {sorted(GENERATORS)}")

    grader = raw.get("grader")
    if grader is not None:
        if not isinstance(grader, dict) or grader.get("type") not in GRADERS:
            problems.append(f"{where}: grader.type must be one of {sorted(GRADERS)}")
        else:
            _, required = GRADERS[grader["type"]]
            for key in required:
                if key not in grader and not (key == "expected" and "generator" in raw):
                    problems.append(f"{where}: grader '{grader['type']}' requires '{key}'")
            _check_regexes(grader, where, problems)
            for i, test in enumerate(grader.get("tests", []) if grader["type"] == "python_function" else []):
                if not isinstance(test, dict) or ("expected" in test) == ("raises" in test):
                    problems.append(f"{where}: test {i} needs exactly one of 'expected' or 'raises'")

    options = raw.get("options", {})
    if not isinstance(options, dict):
        problems.append(f"{where}: 'options' must be a mapping")
    elif RUN_LEVEL_OPTIONS & set(options):
        problems.append(f"{where}: options {sorted(RUN_LEVEL_OPTIONS & set(options))} are run-level "
                        "settings; set them on the command line")
    if raw.get("cutoff", "invalid") not in ("invalid", "fail"):
        problems.append(f"{where}: cutoff must be 'invalid' or 'fail'")
    if raw.get("profile", "standard") not in PROFILES:
        problems.append(f"{where}: profile must be one of {PROFILES}")


def load_scenarios(directory: Path) -> list[Case]:
    files = sorted(Path(directory).glob("*.yaml"))
    if not files:
        raise ScenarioError(f"no scenario files (*.yaml) in {directory}")
    problems: list[str] = []
    cases: list[Case] = []
    seen: dict[str, str] = {}
    for path in files:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            problems.append(f"{path.name}: YAML error: {exc}")
            continue
        if not isinstance(data, dict) or "category" not in data or not isinstance(data.get("cases"), list):
            problems.append(f"{path.name}: needs 'category' and a 'cases' list")
            continue
        if set(data) - FILE_KEYS:
            problems.append(f"{path.name}: unknown top-level keys {sorted(set(data) - FILE_KEYS)}")
        defaults = data.get("defaults") or {}
        for index, raw in enumerate(data["cases"]):
            if not isinstance(raw, dict):
                problems.append(f"{path.name} case #{index}: must be a mapping")
                continue
            merged = {**defaults, **raw}
            merged["options"] = {**defaults.get("options", {}), **raw.get("options", {})}
            where = f"{path.name}:{raw.get('id', f'#{index}')}"
            before = len(problems)
            _validate_case(merged, where, problems)
            case_id = merged.get("id")
            if case_id in seen:
                problems.append(f"{where}: duplicate id (also in {seen[case_id]})")
            seen[case_id] = path.name
            if len(problems) > before:
                continue
            cases.append(Case(
                id=case_id, category=data["category"], purpose=" ".join(str(merged["purpose"]).split()),
                grader=merged["grader"], source=path.name, messages=merged.get("messages"),
                generator=merged.get("generator"), options=merged["options"],
                profile=merged.get("profile", "standard"), min_ctx=merged.get("min_ctx"),
                cutoff=merged.get("cutoff", "invalid"),
                tags=merged.get("tags", []), notes=merged.get("notes"),
            ))
    if problems:
        raise ScenarioError("invalid scenarios:\n  " + "\n  ".join(problems))
    order = {name: i for i, name in enumerate(CATEGORIES)}
    cases.sort(key=lambda c: order.get(c.category, len(order)))   # stable: file order within a category
    return cases


def select_cases(cases: list[Case], *, profile: str = "standard", categories=None, patterns=None) -> list[Case]:
    """Filter by category and id glob patterns. Explicit --case selection ignores the profile."""
    selected = cases
    if categories:
        selected = [c for c in selected if c.category in categories]
    if patterns:
        return [c for c in selected if any(fnmatch.fnmatchcase(c.id, p) for p in patterns)]
    if profile != "all":
        selected = [c for c in selected if c.profile == profile]
    return selected
