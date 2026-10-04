"""Objective evaluators used by the benchmark engine."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import os
import shutil


def _number(text):
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(match.group()) if match else None


def evaluate(test, answer):
    method = test.get("evaluation_method")
    answer = (answer or "").strip()
    expected = test.get("expected_answer")
    if method in ("numeric_exact", "numeric"):
        actual, target = _number(answer), _number(str(expected))
        tolerance = float(test.get("tolerance", 1e-9))
        return actual is not None and target is not None and abs(actual - target) <= tolerance
    if method == "choice":
        return answer.lower()[:1] == str(expected).lower()[:1]
    if method == "json_schema":
        try:
            value = json.loads(answer)
            return isinstance(value, dict) and all(key in value for key in test.get("required_keys", []))
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
    """Prefer Docker isolation; use a restricted subprocess only when explicitly allowed."""
    if not source or len(source) > 20_000 or any(token in source for token in ("import os", "import socket", "subprocess", "open(")):
        return False
    harness = source + "\n" + "\n".join(tests)
    try:
        with tempfile.TemporaryDirectory() as directory:
            if shutil.which("docker") and os.environ.get("BENCHMARK_ALLOW_DOCKER", "1") == "1":
                completed = subprocess.run(
                    ["docker", "run", "--rm", "--network=none", "--cpus=0.5", "--memory=128m",
                     "--pids-limit=64", "--read-only", "-i", "python:3-alpine", "python", "-I", "-c", harness],
                    cwd=directory, capture_output=True, timeout=3, check=False)
                return completed.returncode == 0
            completed = subprocess.run([sys.executable, "-I", "-c", harness],
                                       cwd=directory, capture_output=True, timeout=2,
                                       check=False)
            return completed.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False
