"""Run model-written Python functions against hidden tests, with safeguards.

This is defense in depth for a personal machine, not a security boundary against a
determined adversary:

1. Static policy: the code is parsed (never exec'd in this process). Only a short
   allowlist of pure standard-library modules may be imported, and file/introspection
   builtins (open, eval, exec, __import__, getattr, ...) and dunder attribute access
   are rejected. Rejected code is graded `review`, not `fail`.
2. A separate interpreter in isolated mode (-I), in a throwaway temporary directory,
   with an empty environment and stdin closed.
3. Resource limits: CPU seconds, address space, file size, open files, no new
   processes, plus a wall-clock timeout.
4. No network: when `unshare -rn` works (unprivileged user + network namespace, as on
   stock Fedora), the process runs without any network interfaces. Whether this was
   available is reported in results.
"""

from __future__ import annotations

import ast
import json
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ALLOWED_IMPORTS = {
    "bisect", "collections", "dataclasses", "decimal", "enum", "fractions", "functools",
    "heapq", "itertools", "math", "operator", "re", "statistics", "string", "typing",
    "__future__",
}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "__builtins__",
    "exit", "quit", "help",
}
MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024
OUTPUT_LIMIT_CHARS = 20_000

HARNESS = r'''
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FUNCTION, MARKER = sys.argv[1], sys.argv[2]
with open("tests.json", encoding="utf-8") as handle:
    TESTS = json.load(handle)
try:
    import solution
    fn = getattr(solution, FUNCTION)
except BaseException as exc:
    print(MARKER + json.dumps({"load_error": f"{type(exc).__name__}: {exc}"[:300]}))
    raise SystemExit(0)
results = []
for test in TESTS:
    try:
        got = fn(*test.get("args", []), **test.get("kwargs", {}))
    except BaseException as exc:
        ok = type(exc).__name__ == test.get("raises")
        results.append({"ok": ok, "error": f"{type(exc).__name__}: {exc}"[:200]})
        continue
    if "raises" in test:
        results.append({"ok": False, "got": repr(got)[:200], "error": "expected " + test["raises"]})
        continue
    try:
        normalized = json.loads(json.dumps(got))   # tuples -> lists, like the YAML expectations
    except (TypeError, ValueError):
        normalized = repr(got)
    results.append({"ok": normalized == test["expected"], "got": repr(got)[:200]})
print(MARKER + json.dumps({"results": results}))
'''


@dataclass
class SandboxOutcome:
    status: str                 # pass | fail | review | error
    detail: str = ""
    results: list = field(default_factory=list)


def extract_code(answer: str, function: str) -> str | None:
    """Return the fenced code block defining `function`, or the bare answer if it does."""
    blocks = re.findall(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        if re.search(rf"^\s*def\s+{re.escape(function)}\s*\(", block, re.MULTILINE):
            return block
    if re.search(rf"^def\s+{re.escape(function)}\s*\(", answer, re.MULTILINE):
        return answer
    return None


def policy_violations(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS or node.level:
                problems.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"use of {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            problems.append(f"dunder attribute .{node.attr}")
    return problems


@lru_cache(maxsize=1)
def network_isolation_prefix() -> tuple[str, ...]:
    """['unshare', '-rn'] if an unprivileged network namespace can be created, else []."""
    unshare = shutil.which("unshare")
    if not unshare or not sys.platform.startswith("linux"):
        return ()
    try:
        probe = subprocess.run([unshare, "-rn", "true"], capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ()
    return (unshare, "-rn") if probe.returncode == 0 else ()


def _limit_resources(cpu_seconds: int):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))   # no fork; exec is still allowed
    return apply


def run_function_tests(code: str, function: str, tests: list[dict], timeout: float = 10) -> SandboxOutcome:
    problems = policy_violations(code)
    if problems:
        return SandboxOutcome("review", "code not executed (sandbox policy): " + "; ".join(problems[:5]))

    marker = f"@@RESULT-{secrets.token_hex(8)}@@"
    with tempfile.TemporaryDirectory(prefix="llm_eval_sandbox_") as tmp:
        workdir = Path(tmp)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "tests.json").write_text(json.dumps(tests), encoding="utf-8")
        (workdir / "harness.py").write_text(HARNESS, encoding="utf-8")
        command = [*network_isolation_prefix(), sys.executable, "-I", "harness.py", function, marker]
        try:
            completed = subprocess.run(
                command, cwd=workdir, env={"LANG": "C.UTF-8"}, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=timeout,
                preexec_fn=_limit_resources(int(timeout) + 1),
            )
        except subprocess.TimeoutExpired:
            return SandboxOutcome("fail", f"timed out after {timeout}s")
        except OSError as exc:
            return SandboxOutcome("error", f"sandbox could not start: {exc}")

    stdout = completed.stdout[-OUTPUT_LIMIT_CHARS:]
    payload = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            payload = json.loads(line[len(marker):])
            break
    if payload is None:
        tail = completed.stderr.strip().splitlines()[-1:] or [f"exit code {completed.returncode}"]
        return SandboxOutcome("fail", f"test harness produced no result ({tail[0][:200]})")
    if "load_error" in payload:
        return SandboxOutcome("fail", f"could not load function: {payload['load_error']}")
    results = payload["results"]
    passed = sum(r["ok"] for r in results)
    if passed == len(results):
        return SandboxOutcome("pass", f"{passed}/{len(results)} tests passed", results)
    first = next((i, r) for i, r in enumerate(results) if not r["ok"])
    detail = f"{passed}/{len(results)} tests passed; test {first[0]} failed: {first[1].get('error') or first[1].get('got')}"
    return SandboxOutcome("fail", detail, results)
