"""Objective evaluators used by the benchmark engine."""
from __future__ import annotations

import json
import re
import subprocess
import shutil

MAX_CODE_SIZE = 20_000
MAX_TEST_COUNT = 20
MAX_TEST_SIZE = 2_000


def code_evaluator_available():
    return shutil.which("docker") is not None


def _number(text):
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(match.group()) if match else None


def _contains_term(text, term):
    text_tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    term_tokens = re.findall(r"\w+", str(term).casefold(), flags=re.UNICODE)
    return bool(term_tokens) and any(
        text_tokens[index:index + len(term_tokens)] == term_tokens
        for index in range(len(text_tokens) - len(term_tokens) + 1)
    )


def evaluation_criteria(test):
    method = test.get("evaluation_method")
    if method in ("numeric_exact", "numeric"):
        return f"Numeric result must equal {test.get('expected_answer')} within tolerance {test.get('tolerance', 1e-9)}."
    if method == "choice":
        return f"First answer token must equal the expected choice {test.get('expected_answer')}."
    if method == "json_schema":
        types = test.get("required_types", {})
        return "Valid JSON object must contain " + ", ".join(
            f"{key} ({types[key]})" if key in types else key
            for key in test.get("required_keys", [])) + "."
    if method == "contains_all":
        return "Response must contain all required terms: " + ", ".join(
            str(term) for term in test.get("required_terms", [])) + "."
    if method == "rubric":
        count = (f"exactly {test['exact_words']} words"
                 if "exact_words" in test else
                 f"at least {test.get('minimum_words', 1)} words")
        terms = test.get("required_terms", [])
        return f"Response must have {count}" + (
            f" and contain: {', '.join(map(str, terms))}." if terms else ".")
    if method == "code":
        return f"Python answer must pass all {len(test.get('tests', []))} sandboxed assertions."
    return f"Response must exactly match {test.get('expected_answer')}."


def evaluate_with_details(test, answer):
    method = test.get("evaluation_method")
    answer = (answer or "").strip()
    expected = test.get("expected_answer")
    criteria = evaluation_criteria(test)
    passed = False
    explanation = "The response did not satisfy the evaluation criteria."
    if method in ("numeric_exact", "numeric"):
        actual, target = _number(answer), _number(str(expected))
        try:
            tolerance = float(test.get("tolerance", 1e-9))
        except (TypeError, ValueError, OverflowError):
            tolerance = 1e-9
        passed = actual is not None and target is not None and abs(actual - target) <= tolerance
        explanation = (f"Parsed numeric result {actual} matches expected {target}."
                      if passed else f"Parsed numeric result {actual}; expected {target}.")
    elif method == "choice":
        first_word = re.search(r"[a-z0-9]+", answer.lower())
        actual = first_word.group() if first_word else ""
        passed = actual == str(expected).strip().lower()
        explanation = (f"First answer token {actual!r} matches the expected choice."
                      if passed else f"First answer token {actual!r}; expected {expected!r}.")
    elif method == "json_schema":
        try:
            value = json.loads(answer)
            required_types = test.get("required_types", {})
            if not isinstance(required_types, dict):
                explanation = "The required type rules are invalid; the evaluator rejected this test."
            elif not isinstance(value, dict):
                explanation = "The response is valid JSON, but its top-level value is not an object."
            else:
                missing = [key for key in test.get("required_keys", []) if key not in value]
                supported_types = {"string": str, "boolean": bool, "number": (int, float),
                                   "integer": int, "object": dict, "array": list}
                invalid_rule = next((kind for kind in required_types.values()
                                     if kind not in supported_types), None)
                wrong_types = [
                    (key, kind) for key, kind in required_types.items()
                    if kind in supported_types and key in value and
                    (not isinstance(value[key], supported_types[kind]) or
                     (kind in ("number", "integer") and isinstance(value[key], bool)))
                ]
                passed = not missing and invalid_rule is None and not wrong_types
                if missing:
                    explanation = "Missing required JSON key(s): " + ", ".join(missing) + "."
                elif invalid_rule:
                    explanation = f"The evaluator does not support required type {invalid_rule!r}."
                elif wrong_types:
                    explanation = "Wrong JSON value type(s): " + ", ".join(
                        f"{key} should be {kind}" for key, kind in wrong_types) + "."
                else:
                    explanation = "Valid JSON object contains all required keys with the expected types."
        except (json.JSONDecodeError, TypeError):
            explanation = "The response is not valid JSON."
    elif method == "contains_all":
        missing = [str(item) for item in test.get("required_terms", [])
                   if not _contains_term(answer, item)]
        passed = not missing
        explanation = ("All required terms were found." if passed else
                       "Missing required term(s): " + ", ".join(missing) + ".")
    elif method == "code":
        if not code_evaluator_available():
            return {"passed": None, "status": "NOT RUN", "criteria": criteria,
                    "explanation": "Docker is unavailable; generated code was not executed."}
        passed = _run_code(answer, test.get("tests", []))
        explanation = ("All sandboxed assertions passed." if passed else
                       "Code execution failed or one or more sandboxed assertions failed; "
                       "sandbox output is suppressed.")
    elif method == "rubric":
        words = answer.split()
        valid_count = (len(words) == test["exact_words"] if "exact_words" in test
                       else len(words) >= test.get("minimum_words", 1))
        missing = [str(item) for item in test.get("required_terms", [])
                   if not _contains_term(answer, item)]
        passed = valid_count and not missing
        if not valid_count:
            explanation = (f"Response has {len(words)} words; expected exactly "
                           f"{test['exact_words']}." if "exact_words" in test else
                           f"Response has {len(words)} words; expected at least "
                           f"{test.get('minimum_words', 1)}.")
        elif missing:
            explanation = "Missing required term(s): " + ", ".join(missing) + "."
        else:
            explanation = f"Word-count constraint passed ({len(words)} words); all required terms were found."
    elif method != "json_schema":
        passed = answer == str(expected)
        explanation = "Response exactly matches the expected answer." if passed else (
            f"Response did not exactly match expected answer {expected!r}.")
    return {"passed": passed, "status": "PASS" if passed else "FAILED",
            "criteria": criteria, "explanation": explanation}


def evaluate(test, answer):
    return evaluate_with_details(test, answer)["passed"] is True


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
