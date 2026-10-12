"""Python assertions for checks promptfoo has no built-in assertion for.

Substring, regex, JSON / JSON Schema and format rules use promptfoo's own
assertions in the scenario files. What remains needs custom logic: tolerant answer
extraction (exact-after-normalization, numbers, option letters, sets of names),
scorable abstention, long-context answers regenerated from the seed, and running
model-written code in the sandbox. There is deliberately no LLM-as-judge.

Use from a test:

    - type: python
      value: file://helpers/assertions.py:number
      config: {expected: 2080}

Each grader returns a GradeResult with one of these statuses:
  pass   - meets the acceptance criteria
  fail   - objectively wrong
  review - might be acceptable but no simple rule can decide (e.g. the right value
           alongside other candidates, an abstention without the agreed marker, code
           the sandbox refused to run). promptfoo only has pass/fail, so review is
           reported as a failure whose reason starts with "REVIEW:".
"""

import re
from dataclasses import dataclass

import sandbox   # helpers/ is on PYTHONPATH (promptfooconfig env)


@dataclass
class GradeResult:
    status: str             # pass | fail | review | error
    detail: str = ""


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
        return GradeResult("pass")
    return GradeResult("fail", f"expected {expected[0]!r}, got {got[:200]!r}")


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
        return GradeResult("fail", f"no match for {spec['pattern']!r}")
    if found == want:
        return GradeResult("pass")
    if want <= found:
        extra = sorted(found - want)
        status = spec.get("extra", "review")
        return GradeResult(status, f"expected values present but also mentions {extra}")
    missing = sorted(want - found)
    return GradeResult("fail", f"missing {missing}; found {sorted(found)}")


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
        return GradeResult("fail", "no number in answer")
    hits = [v for v in distinct if abs(v - expected) <= tolerance]
    if hits and len(distinct) == 1:
        return GradeResult("pass")
    if hits:
        status = spec.get("extra", "review")
        return GradeResult(status, f"expected {spec['expected']} present among {distinct}")
    return GradeResult("fail", f"expected {spec['expected']}, found {distinct}")


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
                return GradeResult("pass")
            return GradeResult("fail", f"chose {got}, expected {expected}")
    return GradeResult("review", "could not identify a single chosen option")


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


# -- promptfoo entry points ------------------------------------------------------------------

def _grading_result(result: GradeResult, context: dict) -> dict:
    passed = result.status == "pass"
    reason = result.detail or result.status
    # The provider only lets a cut-off answer through for tests declaring `cutoff: fail`,
    # where running out of budget is itself the failure: lead with that, and don't
    # call it "review" just because the truncated answer couldn't be judged.
    metrics = ((context.get("providerResponse") or {}).get("metadata") or {}).get("metrics") or {}
    cut_off = not passed and metrics.get("done_reason") == "length"
    if cut_off and result.status != "error":
        reason = f"answer cut off at {metrics.get('output_tokens')} tokens before finishing (cutoff: fail); {reason}"
    elif result.status in ("review", "error"):
        reason = f"{result.status.upper()}: {reason}"
    return {"pass": passed, "score": 1.0 if passed else 0.0, "reason": reason}


def _run(grader, output: str, context: dict) -> dict:
    try:
        result = grader(clean_answer(output), context.get("config") or {})
    except (KeyError, TypeError, ValueError, re.error) as exc:
        # A broken assertion config must not count as a model failure; the "ERROR:"
        # reason makes summary.py report it as an error.
        result = GradeResult("error", f"assertion config problem ({type(exc).__name__}: {exc}); "
                                      "check this test's assert config")
    return _grading_result(result, context)


def exact(output, context):
    """Whole answer equals `expected` after light normalization (case, quotes, final period, fence)."""
    return _run(grade_exact, output, context)


def regex_extract(output, context):
    """Distinct matches of `pattern` must equal `expected` (string or set)."""
    return _run(grade_regex_extract, output, context)


def number(output, context):
    """The answer's numbers must be exactly `expected` (within `tolerance`)."""
    return _run(grade_number, output, context)


def choice(output, context):
    """Multiple choice: a single extracted option letter must equal `expected`."""
    return _run(grade_choice, output, context)


def abstain(output, context):
    """Unanswerable question: the agreed marker, and no invented specifics."""
    return _run(grade_abstain, output, context)


def long_context_answer(output, context):
    """The one generated code/IP, regenerated from the test's generator vars. Naming
    several candidates counts as wrong."""
    import long_context

    try:
        generated, pattern = long_context.generate(context.get("vars") or {})
    except (TypeError, ValueError) as exc:   # unknown generator or bad params in the test
        return _grading_result(GradeResult("error", f"cannot regenerate expected answer: {exc}"), context)
    spec = {"pattern": pattern, "expected": generated.grader_updates["expected"], "extra": "fail"}
    return _grading_result(grade_regex_extract(clean_answer(output), spec), context)


def python_function(output, context):
    """Run hidden tests against the function the model wrote, in the sandbox."""
    spec = context.get("config") or {}
    if not isinstance(spec.get("function"), str) or not isinstance(spec.get("tests"), list):
        return _grading_result(GradeResult("error", "assertion config needs 'function' and 'tests'"), context)
    code = sandbox.extract_code(clean_answer(output), spec["function"])
    if code is None:
        return _grading_result(GradeResult("fail", "no Python code found in answer"), context)
    outcome = sandbox.run_function_tests(code, spec["function"], spec["tests"], timeout=spec.get("timeout", 10))
    return _grading_result(GradeResult(outcome.status, outcome.detail), context)
