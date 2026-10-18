"""Objective evaluators used by the benchmark engine."""
from __future__ import annotations

import json
import re
import subprocess
import shutil
import uuid

MAX_CODE_SIZE = 20_000
MAX_TEST_COUNT = 20
MAX_TEST_SIZE = 2_000
DOCKER_PYTHON_IMAGE = (
    "python:3-alpine@sha256:"
    "618e7b661a89d05565fb402a5e1eab8b1b4e5df1f5ea3dc494998d092ffe1213"
)


def code_evaluator_available():
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        result = subprocess.run(
            [docker, "image", "inspect", DOCKER_PYTHON_IMAGE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


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


def _contains_terms_in_order(text, terms):
    text_tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    cursor = 0
    for term in terms:
        term_tokens = re.findall(r"\w+", str(term).casefold(), flags=re.UNICODE)
        if not term_tokens:
            return False
        match = next((index for index in range(cursor, len(text_tokens) - len(term_tokens) + 1)
                      if text_tokens[index:index + len(term_tokens)] == term_tokens), None)
        if match is None:
            return False
        cursor = match + len(term_tokens)
    return True


def _normalized_value(value):
    return " ".join(re.findall(r"\w+", str(value).casefold(), flags=re.UNICODE))


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def _schedule_is_valid(route, items, constraints):
    if (not isinstance(route, list) or len(route) != len(items) or
            any(not isinstance(item, str) for item in route) or
            len(set(route)) != len(items) or set(route) != set(items)):
        return False
    position = {item: index for index, item in enumerate(route)}
    for rule in constraints:
        kind = rule.get("kind")
        if kind == "before" and position[rule["first"]] >= position[rule["second"]]:
            return False
        if kind == "immediately_before" and position[rule["second"]] != position[rule["first"]] + 1:
            return False
        if kind == "not_adjacent" and abs(position[rule["first"]] - position[rule["second"]]) == 1:
            return False
        if kind == "fixed_position" and position[rule["item"]] != rule["position"]:
            return False
    return True


def evaluation_criteria(test):
    method = test.get("evaluation_method")
    if method in ("numeric_exact", "numeric"):
        return (f"Complete response must be one numeric value equal to {test.get('expected_answer')} "
                f"within tolerance {test.get('tolerance', 1e-9)}.")
    if method == "choice":
        return f"Complete response must be the expected choice {test.get('expected_answer')}."
    if method == "exact_any":
        return ("The complete response must match one configured acceptable answer "
                "(ignoring case and punctuation).")
    if method == "json_schema":
        types = test.get("required_types", {})
        return "Valid JSON object must contain " + ", ".join(
            f"{key} ({types[key]})" if key in types else key
            for key in test.get("required_keys", [])) + "."
    if method == "constraint_schedule":
        return ("Return a JSON object with the correct possible flag and a route that visits every "
                "stop exactly once while satisfying all listed constraints.")
    if method == "fact_fields":
        return ("Return a JSON object with exactly the requested factual fields; each field must "
                "match one of its documented acceptable summaries.")
    if method == "contains_all":
        return "Response must contain all required terms: " + ", ".join(
            str(term) for term in test.get("required_terms", [])) + "."
    if method == "ordered_contains":
        return "Response must contain all requested values in this order: " + ", ".join(
            str(term) for term in test.get("required_terms", [])) + "."
    if method == "rubric":
        count = (f"exactly {test['exact_words']} words"
                 if "exact_words" in test else
                 f"at least {test.get('minimum_words', 1)} words")
        terms = test.get("required_terms", [])
        relevance = test.get("minimum_topic_terms", 0)
        return (f"Response must have {count}" +
                (f" and contain: {', '.join(map(str, terms))}." if terms else ".") +
                (f" It must also contain at least {relevance} distinct topical terms; "
                 "a keyword-only list is rejected, but this remains a lexical proxy, not semantic grading."
                 if relevance else ""))
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
        single_value = re.fullmatch(r"-?\d+(?:\.\d+)?", answer) is not None
        passed = (single_value and actual is not None and target is not None and
                  abs(actual - target) <= tolerance)
        explanation = (f"Single numeric result {actual} matches expected {target}."
                      if passed else f"Parsed numeric result {actual}; expected {target}.")
    elif method == "choice":
        actual = _normalized_value(answer)
        passed = actual == _normalized_value(expected)
        explanation = (f"Complete response {actual!r} matches the expected choice."
                      if passed else f"Complete response {actual!r}; expected only {expected!r}.")
    elif method == "exact_any":
        acceptable = test.get("acceptable_answers", [expected])
        if not isinstance(acceptable, list):
            acceptable = []
        normalized = _normalized_value(answer)
        passed = bool(normalized) and normalized in {
            _normalized_value(item) for item in acceptable if isinstance(item, str)
        }
        explanation = ("The complete response matches a reviewed acceptable answer."
                       if passed else "The complete response did not match any reviewed acceptable answer.")
    elif method == "json_schema":
        try:
            value = json.loads(answer, parse_constant=_reject_json_constant)
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
        except (json.JSONDecodeError, TypeError, ValueError):
            explanation = "The response is not valid JSON."
    elif method == "contains_all":
        missing = [str(item) for item in test.get("required_terms", [])
                   if not _contains_term(answer, item)]
        passed = not missing
        explanation = ("All required terms were found." if passed else
                       "Missing required term(s): " + ", ".join(missing) + ".")
    elif method == "ordered_contains":
        required = test.get("required_terms", [])
        found = [_contains_term(answer, term) for term in required]
        in_order = _contains_terms_in_order(answer, required)
        passed = all(found) and in_order
        explanation = ("All requested values were found in order." if passed else
                       "Requested values were missing or appeared in the wrong order.")
    elif method == "constraint_schedule":
        items = test.get("schedule_items", [])
        constraints = test.get("schedule_constraints", [])
        try:
            value = json.loads(answer, parse_constant=_reject_json_constant)
            valid_object = isinstance(value, dict)
        except (json.JSONDecodeError, TypeError, ValueError):
            value, valid_object = {}, False
        possible = value.get("possible") if valid_object else None
        route = value.get("route") if valid_object else None
        checks = [
            {"name": "JSON object", "passed": valid_object},
            {"name": "Possible flag", "passed": isinstance(possible, bool) and
             possible == test.get("schedule_possible"),
             "expected": test.get("schedule_possible"), "actual": possible},
        ]
        if test.get("schedule_possible"):
            route_shape = (isinstance(route, list) and len(route) == len(items) and
                           all(isinstance(item, str) for item in route) and
                           len(set(route)) == len(items) and set(route) == set(items))
            checks.append({"name": "Every stop used exactly once", "passed": route_shape})
            for rule in constraints:
                try:
                    rule_passed = route_shape and _schedule_is_valid(route, items, [rule])
                except (KeyError, TypeError, ValueError):
                    rule_passed = False
                checks.append({"name": f"Constraint: {rule.get('kind')}", "passed": rule_passed})
        else:
            checks.append({"name": "No route returned for impossible case",
                           "passed": route == []})
        successes = sum(check["passed"] for check in checks)
        score = round(10 * successes / len(checks), 2)
        passed = successes == len(checks)
        explanation = (f"{successes} of {len(checks)} route checks passed." if valid_object else
                       "The response is not a valid JSON object.")
    elif method == "fact_fields":
        expected_fields = test.get("expected_fields", {})
        try:
            value = json.loads(answer, parse_constant=_reject_json_constant)
            valid_object = isinstance(value, dict)
        except (json.JSONDecodeError, TypeError, ValueError):
            value, valid_object = {}, False
        checks = [
            {"name": "JSON object", "passed": valid_object},
            {"name": "Exact requested keys",
             "passed": valid_object and set(value) == set(expected_fields),
             "expected": sorted(expected_fields),
             "actual": sorted(value) if valid_object else None},
        ]
        for field, acceptable in expected_fields.items():
            actual = value.get(field) if valid_object else None
            accepted = {_normalized_value(item) for item in acceptable}
            checks.append({"name": field,
                           "passed": valid_object and _normalized_value(actual) in accepted,
                           "expected": acceptable, "actual": actual})
        successes = sum(check["passed"] for check in checks)
        score = round(10 * successes / len(checks), 2)
        passed = successes == len(checks)
        explanation = (f"{successes} of {len(checks)} format and factual checks passed." if valid_object else
                       "The response is not a valid JSON object.")
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
        topic_terms = test.get("topic_terms", [])
        matched_topics = [str(term) for term in topic_terms
                          if isinstance(term, str) and _contains_term(answer, term)]
        minimum_topics = test.get("minimum_topic_terms", 0)
        checks = [{"name": "Word count", "passed": valid_count,
                   "expected": test.get("exact_words", test.get("minimum_words", 1)),
                   "actual": len(words)}]
        checks.extend({"name": f"Required term: {term}",
                       "passed": term not in missing} for term in test.get("required_terms", []))
        if minimum_topics:
            checks.append({"name": "Topical relevance proxy",
                           "passed": len(set(matched_topics)) >= minimum_topics,
                           "expected_distinct_terms": minimum_topics,
                           "matched_terms": sorted(set(matched_topics))})
            topic_vocabulary = {
                token for term in [*test.get("required_terms", []), *topic_terms]
                for token in re.findall(r"\w+", str(term).casefold(), flags=re.UNICODE)
            }
            has_non_keyword_word = any(
                token not in topic_vocabulary
                for token in re.findall(r"\w+", answer.casefold(), flags=re.UNICODE))
            checks.append({"name": "Not a keyword-only list",
                           "passed": has_non_keyword_word})
        passed = all(check["passed"] for check in checks)
        score = round(10 * sum(check["passed"] for check in checks) / len(checks), 2)
        explanation = (f"{sum(check['passed'] for check in checks)} of {len(checks)} lexical "
                       "format/relevance checks passed. Topic-word checks are only a relevance proxy; "
                       "semantic quality and originality are not judged.")
    elif method != "json_schema":
        passed = answer == str(expected)
        explanation = "Response exactly matches the expected answer." if passed else (
            f"Response did not exactly match expected answer {expected!r}.")
    result = {"passed": passed, "status": "PASS" if passed else "FAILED",
              "criteria": criteria, "explanation": explanation}
    if method in ("constraint_schedule", "fact_fields", "rubric"):
        result["score"] = score
        result["checks"] = checks
    return result


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
    container_name = f"ai-benchmark-evaluator-{uuid.uuid4().hex}"
    try:
        completed = subprocess.run(
            [docker, "run", "--rm", "--pull=never", "--name", container_name,
             "--network=none", "--cpus=0.5",
             "--memory=128m", "--pids-limit=64", "--read-only", "--user=65534:65534",
             "--cap-drop=ALL", "--security-opt=no-new-privileges:true", "--ipc=none",
             "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--workdir=/tmp",
             DOCKER_PYTHON_IMAGE, "python", "-I", "-c", harness],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3, check=False)
        return completed.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False
    finally:
        try:
            subprocess.run([docker, "rm", "--force", container_name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=5, check=False)
        except (subprocess.SubprocessError, OSError):
            pass
