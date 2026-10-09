"""Objective evaluators used by the benchmark engine."""
from __future__ import annotations

import json
import re
import subprocess
import shutil

MAX_CODE_SIZE = 20_000
MAX_TEST_COUNT = 20
MAX_TEST_SIZE = 2_000


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
            if not isinstance(value, dict) or not all(
                    key in value for key in test.get("required_keys", [])):
                return False
            return all(
                isinstance(value.get(key), {"string": str, "boolean": bool, "number": (int, float),
                                            "integer": int, "object": dict, "array": list}.get(kind, object))
                and not (kind in ("number", "integer") and isinstance(value.get(key), bool))
                for key, kind in test.get("required_types", {}).items()
            )
        except (json.JSONDecodeError, TypeError):
            return False
    if method == "contains_all":
        return all(str(item).lower() in answer.lower() for item in test.get("required_terms", []))
    if method == "code":
        return _run_code(answer, test.get("tests", []))
    if method == "rubric":
        words = answer.split()
        if "exact_words" in test:
            return len(words) == test["exact_words"]
        return len(words) >= test.get("minimum_words", 1)
    return answer == str(expected)


def _run_code(source, tests):
    """Run generated code only in a resource-limited, network-isolated container."""
    if not isinstance(source, str) or not source or len(source) > MAX_CODE_SIZE:
        return False
    if not isinstance(tests, list) or len(tests) > MAX_TEST_COUNT:
        return False
    if any(not isinstance(test, str) or len(test) > MAX_TEST_SIZE for test in tests):
        return False
    harness = source + "\n" + "\n".join(tests)
    if len(harness) > MAX_CODE_SIZE:
        return False
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        completed = subprocess.run(
            [docker, "run", "--rm", "--pull=never", "--network=none", "--cpus=0.5",
             "--memory=128m", "--pids-limit=64", "--read-only", "--user=65534:65534",
             "--cap-drop=ALL", "--security-opt=no-new-privileges:true", "--ipc=none",
             "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--workdir=/tmp",
             "python:3-alpine", "python", "-I", "-c", harness],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3, check=False)
        return completed.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False
