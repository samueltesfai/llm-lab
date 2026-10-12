import sandbox

TESTS = [{"args": [2, 3], "expected": 5}, {"args": ["x", 1], "raises": "TypeError"}]


def test_passing_function():
    outcome = sandbox.run_function_tests("def add(a, b):\n    return a + b\n", "add", TESTS)
    assert outcome.status == "pass", outcome.detail


def test_wrong_function():
    outcome = sandbox.run_function_tests("def add(a, b):\n    return a - b\n", "add", TESTS)
    assert outcome.status == "fail" and "1/2" in outcome.detail


def test_tuples_compare_equal_to_lists():
    outcome = sandbox.run_function_tests("def f():\n    return [(1, 2)]\n", "f", [{"args": [], "expected": [[1, 2]]}])
    assert outcome.status == "pass"


def test_missing_function_fails():
    outcome = sandbox.run_function_tests("def other():\n    pass\n", "add", TESTS)
    assert outcome.status == "fail" and "could not load" in outcome.detail


def test_policy_rejects_dangerous_code_without_running_it():
    for code in ("import os\ndef f(): pass", "def f(): return open('x')", "def f(): return ().__class__",
                 "from subprocess import run\ndef f(): pass", "def f(): return eval('1')"):
        outcome = sandbox.run_function_tests(code, "f", [{"args": [], "expected": 1}])
        assert outcome.status == "review", code


def test_allowed_imports():
    code = "import math\nfrom collections import Counter\ndef f():\n    return math.floor(2.5)\n"
    assert sandbox.run_function_tests(code, "f", [{"args": [], "expected": 2}]).status == "pass"


def test_timeout():
    outcome = sandbox.run_function_tests("def f():\n    while True:\n        pass\n", "f",
                                         [{"args": [], "expected": 1}], timeout=2)
    assert outcome.status == "fail" and "timed out" in outcome.detail


def test_syntax_error_is_reported():
    assert sandbox.run_function_tests("def f(:\n", "f", [{"args": [], "expected": 1}]).status == "review"


def test_extract_code_prefers_block_defining_function():
    answer = "Example:\n```python\nprint(1)\n```\nSolution:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert sandbox.extract_code(answer, "add").startswith("def add")
    assert sandbox.extract_code("def add(a, b):\n    return a + b", "add") is not None
    assert sandbox.extract_code("no code here", "add") is None


def test_fails_closed_without_network_isolation(monkeypatch):
    monkeypatch.setattr(sandbox, "network_isolation_prefix", lambda: ())
    outcome = sandbox.run_function_tests("def add(a, b):\n    return a + b\n", "add", TESTS)
    assert outcome.status == "error" and "not executed" in outcome.detail


def test_isolation_available_here():
    # The suite's code-writing tests need it; if this fails they report errors, not results.
    assert sandbox.network_isolation_prefix(), "unshare -rn unavailable on this machine"
