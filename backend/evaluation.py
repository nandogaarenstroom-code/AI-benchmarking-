"""Objective evaluators used by the benchmark engine."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile


def _number(text):
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(match.group()) if match else None


def evaluate(test, answer):
    method = test.get("evaluation_method")
    answer = (answer or "").strip()
    expected = test.get("expected_answer")
    if method in ("numeric_exact", "numeric"):
        actual, target = _number(answer), _number(str(expected))
        return actual is not None and target is not None and abs(actual - target) < 1e-9
    if method == "choice":
        return answer.lower()[:1] == str(expected).lower()[:1]
    if method == "json_schema":
        try:
            value = json.loads(answer)
            return all(key in value for key in test.get("required_keys", []))
        except (json.JSONDecodeError, TypeError):
            return False
    if method == "contains_all":
        return all(str(item).lower() in answer.lower() for item in test.get("required_terms", []))
    if method == "code":
        return _run_code(answer, test.get("tests", []))
    if method == "rubric":
        return len(answer.split()) >= test.get("minimum_words", 1)
    return answer == str(expected)


def _run_code(source, tests):
    """Run generated code with a timeout and no network access in a temp directory."""
    if not source or any(token in source for token in ("import os", "import socket", "subprocess")):
        return False
    harness = source + "\n" + "\n".join(tests)
    try:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run([sys.executable, "-I", "-c", harness],
                                       cwd=directory, capture_output=True, timeout=2,
                                       check=False)
            return completed.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False
