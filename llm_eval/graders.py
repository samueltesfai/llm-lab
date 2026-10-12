"""Deterministic graders. Each takes the model's answer text and the case's grader spec.

Outcomes:
  pass   - the response meets the acceptance criteria.
  fail   - the response objectively violates them (wrong value, invalid JSON, ...).
  review - the response *might* be acceptable but a simple rule cannot decide, e.g. it
           contains the right value alongside other candidate values, or abstains
           without the required marker. Counted separately, never as a pass.
  error  - the grader itself could not run (e.g. the code sandbox failed to start).

There is deliberately no LLM-as-judge and no semantic similarity scoring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

import jsonschema


@dataclass
class GradeResult:
    status: str             # pass | fail | review | error
    detail: str = ""
    extracted: Any = None   # what the grader pulled out of the answer, for the record


# -- shared text handling -------------------------------------------------------------

_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_WHOLE_FENCE = re.compile(r"^```[\w+-]*[ \t]*\n(.*?)\n?```$", re.DOTALL)


def clean_answer(text: str) -> str:
    """Strip an inline <think> block and outer whitespace."""
    return _THINK_BLOCK.sub("", text or "").strip()


def strip_fence(text: str) -> str:
    """If the whole answer is one fenced block, return its contents."""
    match = _WHOLE_FENCE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def normalize_short(text: str, case_sensitive: bool = False) -> str:
    """Normalization for short exact answers: whitespace, wrapping quotes/backticks, final period."""
    lines = [" ".join(line.split()) for line in strip_fence(text).splitlines()]
    value = "\n".join(line for line in lines).strip()
    value = value.strip("`\"'").strip()
    value = value[:-1] if value.endswith(".") and not value.endswith("..") else value
    return value if case_sensitive else value.casefold()


def _flags(spec: dict) -> int:
    return re.IGNORECASE if spec.get("ignore_case", True) else 0


# -- graders ----------------------------------------------------------------------------

def grade_exact(answer: str, spec: dict) -> GradeResult:
    """Whole answer must equal an expected value after light normalization."""
    case_sensitive = spec.get("case_sensitive", False)
    expected = spec["expected"] if isinstance(spec["expected"], list) else [spec["expected"]]
    got = normalize_short(answer, case_sensitive)
    if any(got == normalize_short(str(e), case_sensitive) for e in expected):
        return GradeResult("pass", extracted=got)
    return GradeResult("fail", f"expected {expected[0]!r}, got {got[:200]!r}", got)


def grade_regex_extract(answer: str, spec: dict) -> GradeResult:
    """Find every match of `pattern`; the distinct matches must equal `expected`.

    `expected` may be a string or a list (compared as a set). If all expected values
    are present together with other candidates, the result is `review` by default
    (set `extra: fail` when stating several candidates should count as wrong).
    """
    flags = _flags(spec)
    expected = spec["expected"] if isinstance(spec["expected"], list) else [spec["expected"]]
    fold = (lambda s: s.casefold()) if flags else (lambda s: s)
    want = {fold(str(e)) for e in expected}
    found_raw = [m.group(spec.get("group", 0)) for m in re.finditer(spec["pattern"], answer, flags)]
    found = {fold(f) for f in found_raw}
    if not found:
        return GradeResult("fail", f"no match for {spec['pattern']!r}", [])
    if found == want:
        return GradeResult("pass", extracted=sorted(found))
    if want <= found:
        extra = sorted(found - want)
        status = spec.get("extra", "review")
        return GradeResult(status, f"expected values present but also mentions {extra}", sorted(found))
    missing = sorted(want - found)
    return GradeResult("fail", f"missing {missing}; found {sorted(found)}", sorted(found))


_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def grade_number(answer: str, spec: dict) -> GradeResult:
    """The answer's numbers must be exactly the expected number (within tolerance)."""
    expected = float(spec["expected"])
    tolerance = float(spec.get("tolerance", 0))
    values = []
    for token in _NUMBER.findall(answer):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            continue
    distinct = sorted(set(values))
    if not distinct:
        return GradeResult("fail", "no number in answer", [])
    hits = [v for v in distinct if abs(v - expected) <= tolerance]
    if hits and len(distinct) == 1:
        return GradeResult("pass", extracted=distinct)
    if hits:
        status = spec.get("extra", "review")
        return GradeResult(status, f"expected {spec['expected']} present among {distinct}", distinct)
    return GradeResult("fail", f"expected {spec['expected']}, found {distinct}", distinct)


def grade_contains(answer: str, spec: dict) -> GradeResult:
    """Regex presence checks: every `all` pattern, at least one `any` pattern, no `none` pattern."""
    flags = _flags(spec)
    problems = []
    for pattern in spec.get("all", []):
        if not re.search(pattern, answer, flags):
            problems.append(f"missing /{pattern}/")
    any_patterns = spec.get("any", [])
    if any_patterns and not any(re.search(p, answer, flags) for p in any_patterns):
        problems.append(f"none of {any_patterns} present")
    for pattern in spec.get("none", []):
        if re.search(pattern, answer, flags):
            problems.append(f"forbidden /{pattern}/ present")
    return GradeResult("fail", "; ".join(problems)) if problems else GradeResult("pass")


def grade_choice(answer: str, spec: dict) -> GradeResult:
    """Multiple choice: extract a single option letter."""
    letters = spec.get("options", "ABCD")
    expected = spec["expected"].upper()
    text = strip_fence(answer)
    patterns = [
        rf"^\(?([{letters}])\)?[.:]?$",                                         # "C", "(C)", "C."
        rf"\b(?:answer|option|choice)\s*(?:is)?\s*[:\-]?\s*\(?([{letters}])\)?(?![\w'])",
        rf"^\(?([{letters}])[).:]\s",                                            # "C) because ..."
        rf"^\*\*\(?([{letters}])\)?[.:]?\*\*",                                   # "**C**"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE if "answer" in pattern else 0)
        if match:
            got = match.group(1).upper()
            if got == expected:
                return GradeResult("pass", extracted=got)
            return GradeResult("fail", f"chose {got}, expected {expected}", got)
    return GradeResult("review", "could not identify a single chosen option", None)


def _lookup(document: Any, path: str) -> Any:
    if path in ("$", ""):
        return document
    current = document
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


def _json_equal(a: Any, b: Any, unordered: bool) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k], unordered) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        if not unordered:
            return all(_json_equal(x, y, unordered) for x, y in zip(a, b))
        remaining = list(b)
        for item in a:
            for i, candidate in enumerate(remaining):
                if _json_equal(item, candidate, unordered):
                    del remaining[i]
                    break
            else:
                return False
        return True
    return a == b


def _all_keys(document: Any):
    if isinstance(document, dict):
        for key, value in document.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(document, list):
        for item in document:
            yield from _all_keys(item)


def grade_json(answer: str, spec: dict) -> GradeResult:
    """Parse JSON, validate against a JSON Schema, then check expected values.

    spec keys: schema, expect {path: value}, unordered_lists, forbidden_keys,
    forbidden_text (regexes over the raw answer), allow_fences (default true).
    """
    text = answer.strip()
    if spec.get("allow_fences", True):
        text = strip_fence(text)
    elif text.startswith("```"):
        return GradeResult("fail", "answer wrapped in a code fence, which the instructions forbid")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        return GradeResult("fail", f"invalid JSON: {exc}", text[:200])

    if "schema" in spec:
        validator = jsonschema.Draft202012Validator(spec["schema"])
        errors = sorted(validator.iter_errors(document), key=lambda e: list(e.path))
        if errors:
            messages = [f"{'/'.join(map(str, e.path)) or '$'}: {e.message}" for e in errors[:3]]
            return GradeResult("fail", "schema: " + "; ".join(messages), document)

    present = sorted(set(_all_keys(document)) & set(spec.get("forbidden_keys", [])))
    if present:
        return GradeResult("fail", f"forbidden keys present: {present}", document)
    for pattern in spec.get("forbidden_text", []):
        if re.search(pattern, answer, re.IGNORECASE):
            return GradeResult("fail", f"forbidden content /{pattern}/ present", document)

    unordered = spec.get("unordered_lists", False)
    for path, want in (spec.get("expect") or {}).items():
        try:
            got = _lookup(document, path)
        except (KeyError, IndexError, ValueError, TypeError):
            return GradeResult("fail", f"{path}: missing", document)
        if not _json_equal(got, want, unordered):
            return GradeResult("fail", f"{path}: expected {want!r}, got {got!r}", document)
    return GradeResult("pass", extracted=document)


def grade_constraints(answer: str, spec: dict) -> GradeResult:
    """A list of independent format checks; every check must hold.

    Supported checks (one key per list item):
      line_count: {min, max, eq}     non-blank lines
      word_count: {min, max}
      each_line_matches: regex       every non-blank line must match (re.search)
      each_line_max_words: n
      contains: regex                case-insensitive
      not_contains: regex            case-insensitive
    """
    text = clean_answer(answer)
    lines = [line for line in text.splitlines() if line.strip()]
    words = text.split()
    problems = []

    def bounds(name: str, value: int, rule: dict):
        if "eq" in rule and value != rule["eq"]:
            problems.append(f"{name}={value}, expected {rule['eq']}")
        if "min" in rule and value < rule["min"]:
            problems.append(f"{name}={value} < {rule['min']}")
        if "max" in rule and value > rule["max"]:
            problems.append(f"{name}={value} > {rule['max']}")

    for check in spec["checks"]:
        (kind, rule), = check.items()
        if kind == "line_count":
            bounds("lines", len(lines), rule)
        elif kind == "word_count":
            bounds("words", len(words), rule)
        elif kind == "each_line_matches":
            bad = [line for line in lines if not re.search(rule, line)]
            if bad:
                problems.append(f"{len(bad)} line(s) don't match /{rule}/, e.g. {bad[0][:60]!r}")
        elif kind == "each_line_max_words":
            long_lines = [line for line in lines if len(line.split()) > rule]
            if long_lines:
                problems.append(f"{len(long_lines)} line(s) exceed {rule} words")
        elif kind == "contains":
            if not re.search(rule, text, re.IGNORECASE):
                problems.append(f"missing /{rule}/")
        elif kind == "not_contains":
            if re.search(rule, text, re.IGNORECASE):
                problems.append(f"forbidden /{rule}/ present")
        else:
            raise ValueError(f"unknown constraint check {kind!r}")
    return GradeResult("fail", "; ".join(problems)) if problems else GradeResult("pass")


DEFAULT_ABSTENTION_PHRASES = [
    r"not (?:mentioned|specified|stated|provided|included|given|listed|available)",
    r"does not (?:say|mention|specify|state|include|contain|provide)",
    r"doesn't (?:say|mention|specify|state|include|contain|provide)",
    r"no information", r"cannot be determined", r"can't be determined", r"unknown",
]


def grade_abstain(answer: str, spec: dict) -> GradeResult:
    """For unanswerable questions: the model must give the agreed marker and invent nothing.

      marker present, no forbidden detail      -> pass
      forbidden (invented) detail present      -> fail, with or without the marker
      no marker, but an abstention phrase      -> review (right idea, wrong format)
      otherwise                                -> fail (answered something unsupported)
    """
    marker = spec.get("marker", "NOT IN SOURCE")
    for pattern in spec.get("forbidden", []):
        if re.search(pattern, answer, re.IGNORECASE):
            return GradeResult("fail", f"unsupported detail /{pattern}/ in answer")
    if marker.casefold() in answer.casefold():
        return GradeResult("pass")
    phrases = spec.get("abstention_phrases", DEFAULT_ABSTENTION_PHRASES)
    if any(re.search(p, answer, re.IGNORECASE) for p in phrases):
        return GradeResult("review", f"abstained without the required marker {marker!r}")
    return GradeResult("fail", "answered instead of abstaining")


def grade_python_function(answer: str, spec: dict) -> GradeResult:
    """Extract Python code from the answer and run hidden tests in the sandbox."""
    from llm_eval import sandbox

    code = sandbox.extract_code(answer, spec["function"])
    if code is None:
        return GradeResult("fail", "no Python code found in answer")
    outcome = sandbox.run_function_tests(code, spec["function"], spec["tests"],
                                         timeout=spec.get("timeout", 10))
    return GradeResult(outcome.status, outcome.detail, outcome.results)


GRADERS: dict[str, tuple[Callable[[str, dict], GradeResult], tuple[str, ...]]] = {
    # name: (function, required spec keys)
    "exact": (grade_exact, ("expected",)),
    "regex_extract": (grade_regex_extract, ("pattern", "expected")),
    "number": (grade_number, ("expected",)),
    "contains": (grade_contains, ()),
    "choice": (grade_choice, ("expected",)),
    "json": (grade_json, ()),
    "constraints": (grade_constraints, ("checks",)),
    "abstain": (grade_abstain, ()),
    "python_function": (grade_python_function, ("function", "tests")),
}


def grade(answer: str, spec: dict) -> GradeResult:
    function, _ = GRADERS[spec["type"]]
    # Graders see the answer with any inline <think> block and outer whitespace removed.
    return function(clean_answer(answer), spec)
