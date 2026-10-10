"""Domain benchmark generation, evaluation, performance measurement, and exports."""
from __future__ import annotations

import csv
import copy
import hashlib
import itertools
import json
import math
import random
import statistics
import time
import uuid
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .discovery import hardware_info, model_info, normalize_generation_options, resolve_model
from .evaluation import code_evaluator_available, evaluation_criteria, evaluate_with_details
from .reporting import write_pdf, write_chart

CATEGORIES = ["Reasoning", "Mathematics", "Coding", "Instruction Following",
              "Structured Output", "Knowledge", "Hallucination Resistance",
              "Long Context", "Summarization", "Multilingual", "Creative Generation",
              "Consistency", "Context Retrieval"]
COUNTS = {"QUICK": 25, "STANDARD": 100, "FULL": 252, "PERFORMANCE": 25,
          "QUALITY": 100, "REASONING": 40, "CODING": 40, "LONG CONTEXT": 30}
DIFFICULTIES = ("Easy", "Medium", "Hard", "Uncalibrated")
TASK_SET_VERSION = "pilot-1"
EVALUATOR_VERSION = "pilot-1"
DIFFICULTY_DESCRIPTIONS = {
    "Easy": "A direct, single-step task with one clear requirement.",
    "Medium": "A routine task with a larger input or an additional constraint.",
    "Hard": "A multi-step task or one that combines several constraints.",
    "Uncalibrated": "A pilot task with higher constructed complexity; difficulty has not been calibrated.",
}


def _valid_schedule(route, items, constraints):
    if (not isinstance(route, (list, tuple)) or len(route) != len(items) or
            any(not isinstance(item, str) for item in route) or set(route) != set(items)):
        return False
    position = {item: index for index, item in enumerate(route)}
    for rule in constraints:
        kind = rule["kind"]
        if kind == "before" and position[rule["first"]] >= position[rule["second"]]:
            return False
        if kind == "immediately_before" and position[rule["second"]] != position[rule["first"]] + 1:
            return False
        if kind == "not_adjacent" and abs(position[rule["first"]] - position[rule["second"]]) == 1:
            return False
        if kind == "fixed_position" and position[rule["item"]] != rule["position"]:
            return False
    return True


def _schedule_has_solution(items, constraints):
    return any(_valid_schedule(route, items, constraints)
               for route in itertools.permutations(items))


def _test(category, i, rng, difficulty=None):
    difficulty = difficulty or DIFFICULTIES[i % 4]
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    common = {"id": f"{category.lower().replace(' ', '-')}-{i+1:03d}",
              "category": category, "subcategory": category, "difficulty": difficulty,
              "difficulty_rationale": DIFFICULTY_DESCRIPTIONS[difficulty],
              "difficulty_calibration_status": "not empirically calibrated",
              "task_set_version": TASK_SET_VERSION, "evaluator_version": EVALUATOR_VERSION,
              "max_score": 10}
    if category == "Mathematics":
        if difficulty == "Easy":
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            operation, answer = f"{a} + {b}", a + b
            rationale = "Single-step addition with single-digit operands."
        elif difficulty == "Medium":
            operation, answer = f"{a} × {b}", a * b
            rationale = "Single-step multiplication with two-digit operands."
        elif difficulty == "Hard":
            c = rng.randint(100, 999)
            operation, answer = f"({a} × {b}) + {c}", a * b + c
            rationale = "Two-step arithmetic: multiply, then add a three-digit value."
        else:
            c, d = rng.randint(10, 99), rng.randint(2, 9)
            operation, answer = f"({a} + {b}) × ({c} - {d})", (a + b) * (c - d)
            rationale = "Evaluate two grouped expressions, then multiply their results."
        return {**common, "difficulty_rationale": rationale,
                "prompt": f"Calculate {operation}. Return only the number.",
                "expected_answer": str(answer), "evaluation_method": "numeric", "tolerance": 0}
    if category == "Reasoning":
        items = rng.sample(["Depot", "Clinic", "Market", "Library", "Archive",
                            "Harbor", "Gallery", "Workshop", "Station"], {
            "Easy": 3, "Medium": 4, "Hard": 5, "Uncalibrated": 5}[difficulty])
        constraints = [{"kind": "before", "first": items[0], "second": items[1]}]
        if difficulty in ("Medium", "Hard", "Uncalibrated"):
            constraints.append({"kind": "immediately_before",
                                "first": items[2], "second": items[3]})
        if difficulty in ("Hard", "Uncalibrated"):
            constraints.append({"kind": "not_adjacent",
                                "first": items[0], "second": items[4]})
        if difficulty == "Uncalibrated":
            if rng.choice((True, False)):
                constraints.append({"kind": "fixed_position", "item": items[2],
                                    "position": len(items) - 1})
            else:
                constraints.append({"kind": "fixed_position", "item": items[0],
                                    "position": 0})
        descriptions = {
            "before": lambda rule: f"{rule['first']} must be visited before {rule['second']}",
            "immediately_before": lambda rule: (
                f"{rule['first']} must be immediately followed by {rule['second']}"),
            "not_adjacent": lambda rule: f"{rule['first']} and {rule['second']} cannot be adjacent",
            "fixed_position": lambda rule: (
                f"{rule['item']} must be stop number {rule['position'] + 1}"),
        }
        prompt = (f"A courier must visit each stop exactly once: {', '.join(items)}. "
                  "Follow every route rule:\n- " + "\n- ".join(
                      descriptions[rule["kind"]](rule) for rule in constraints) +
                  '\nReturn only JSON: {"possible": true/false, "route": ["stop", ...]}. '
                  'If no route can satisfy all rules, use "possible": false and an empty route.')
        rationale = {
            "Easy": "Construct a route that satisfies a precedence rule.",
            "Medium": "Combine precedence with an immediate-adjacency constraint.",
            "Hard": "Plan a five-stop route under precedence, adjacency, and exclusion rules.",
            "Uncalibrated": "Resolve interacting route constraints and detect an unsatisfiable case.",
        }[difficulty]
        return {**common, "difficulty_rationale": rationale, "prompt": prompt,
                "expected_answer": "A JSON route satisfying all rules, or impossible if none exists.",
                "evaluation_method": "constraint_schedule", "schedule_items": items,
                "schedule_constraints": constraints,
                "schedule_possible": _schedule_has_solution(items, constraints)}
    if category == "Coding":
        function = "reconcile_stock_events"
        rules = [
            "Each valid record is a dictionary with a non-empty event_id, a non-empty string sku, "
            "and an integer delta.",
            "Positive delta means stock received; negative delta means stock shipped.",
        ]
        checks = [
            "assert reconcile_stock_events([]) == {}",
            f"assert reconcile_stock_events([{{'event_id': 'a', 'sku': 'PEN', 'delta': {a}}}, "
            f"{{'event_id': 'b', 'sku': 'PEN', 'delta': {-b}}}]) == "
            f"{{'PEN': {a - b}}}",
        ]
        rationale = "Aggregate signed inventory changes by SKU."
        if difficulty in ("Medium", "Hard", "Uncalibrated"):
            rules.append("Each record has an event_id; count only the first valid record for each event_id.")
            checks.append(
                f"assert reconcile_stock_events([{{'event_id': 'a', 'sku': 'PEN', 'delta': {a}}}, "
                f"{{'event_id': 'a', 'sku': 'PEN', 'delta': 9}}, "
                f"{{'event_id': 'b', 'sku': 'PEN', 'delta': {-b}}}]) == "
                f"{{'PEN': {a - b}}}")
            rationale += " Ignore duplicate event IDs after the first valid event."
        if difficulty in ("Hard", "Uncalibrated"):
            rules.extend([
                "Ignore records with missing/empty IDs or SKUs, non-integer deltas, or boolean deltas.",
                "A malformed record does not reserve its event_id; a valid zero-net SKU remains in the result.",
            ])
            checks.extend([
                f"assert reconcile_stock_events([{{'event_id': 'bad', 'sku': '', 'delta': 8}}, "
                f"{{'event_id': 'bad', 'sku': 'PEN', 'delta': {a}}}, "
                f"{{'event_id': 'bool', 'sku': 'PEN', 'delta': True}}, "
                f"{{'event_id': 'return', 'sku': 'PEN', 'delta': {-a}}}]) == {{'PEN': 0}}",
                f"assert reconcile_stock_events([{{'event_id': 'x', 'sku': 'PEN', 'delta': {a}}}, "
                f"{{'event_id': 'y', 'sku': 'BOOK', 'delta': {b}}}, "
                f"{{'event_id': 'z', 'sku': 'PEN', 'delta': {-a}}}]) == "
                f"{{'PEN': 0, 'BOOK': {b}}}",
            ])
            rationale += " Handle invalid records, duplicate-ID edge cases, and net-zero totals."
        if difficulty == "Uncalibrated":
            checks.append(
                f"assert reconcile_stock_events([{{'event_id': 'r1', 'sku': 'BOOK', 'delta': {a}}}, "
                f"{{'event_id': 'r2', 'sku': 'PEN', 'delta': {-b}}}, "
                f"{{'event_id': 'r2', 'sku': 'BOOK', 'delta': 7}}, "
                f"{{'event_id': 'r3', 'sku': 'PEN', 'delta': {b}}}, "
                f"{{'event_id': 'r4', 'sku': 'BOOK', 'delta': {-a}}}]) == "
                "{'BOOK': 0, 'PEN': 0}")
            rationale += " Reconcile cross-SKU duplicate IDs without double-counting."
        rules_text = "\n".join(f"- {rule}" for rule in rules)
        source = (
            "def reconcile_stock_events(events):\n"
            "    totals = {}\n"
            "    seen = set()\n"
            "    for event in events:\n"
            "        if not isinstance(event, dict):\n"
            "            continue\n"
            "        event_id = event.get('event_id')\n"
            "        sku = event.get('sku')\n"
            "        delta = event.get('delta')\n"
            "        if not isinstance(event_id, str) or not event_id.strip():\n"
            "            continue\n"
            "        if not isinstance(sku, str) or not sku.strip():\n"
            "            continue\n"
            "        if not isinstance(delta, int) or isinstance(delta, bool):\n"
            "            continue\n"
            "        if event_id in seen:\n"
            "            continue\n"
            "        seen.add(event_id)\n"
            "        totals[sku] = totals.get(sku, 0) + delta\n"
            "    return totals")
        example_events = [
            {"event_id": f"sample-{i}", "sku": "PEN", "delta": a},
            {"event_id": f"sample-{i + 1}", "sku": "PEN", "delta": -b},
        ]
        return {**common, "prompt": f"Implement `{function}(events)` for reconciling a small inventory ledger. "
                                     "Return a dictionary mapping each SKU to its net quantity change. "
                                     "Each record includes `event_id`, `sku`, and `delta`. "
                                     "Follow all rules exactly:\n" + rules_text +
                                     "\nExample records:\n" +
                                     json.dumps(example_events, ensure_ascii=False) +
                                     "\nOutput Python code only.",
                "expected_answer": "Python function that satisfies all inventory reconciliation rules",
                "evaluation_method": "code", "tests": checks, "demo_answer": source,
                "difficulty_rationale": rationale}
    if category == "Structured Output":
        if difficulty in ("Easy", "Medium"):
            keys = ("answer", "valid")
            types = {"answer": "string", "valid": "boolean"}
            rationale = "Return a parseable object with two required fields and exact value types."
        else:
            keys = ("result", "count", "items")
            types = {"result": "string", "count": "integer", "items": "array"}
            rationale = "Return a parseable object with three required fields and exact value types."
        return {**common, "difficulty_rationale": rationale,
                "prompt": f'Return a JSON object with keys {", ".join(repr(key) for key in keys)}; '
                          f'follow these types: {", ".join(f"{key} is {kind}" for key, kind in types.items())}.',
                "expected_answer": "JSON object with the listed keys and types",
                "evaluation_method": "json_schema", "required_keys": list(keys),
                "required_types": types}
    if category == "Knowledge":
        knowledge_by_difficulty = {
            "Easy": (("What is the capital of France?", "Paris"),
                     ("Which planet is known as the Red Planet?", "Mars")),
            "Medium": (("What is the chemical symbol for water?", "H2O"),
                       ("Which ocean lies between Africa and Australia?", "Indian Ocean")),
            "Hard": (("Which gas do plants absorb from the atmosphere during photosynthesis?",
                      "carbon dioxide"),
                     ("Which force keeps planets in orbit around the Sun?", "gravity")),
            "Uncalibrated": (("What is the name of the process by which a liquid changes into a gas?",
                        "evaporation"),
                       ("What is the term for an animal that is active at night?", "nocturnal")),
        }
        question, answer = rng.choice(knowledge_by_difficulty[difficulty])
        return {**common, "prompt": question + " Answer concisely.",
                "difficulty_rationale": "Short factual recall from a seeded question set; this "
                                        "category is not independently calibrated by difficulty.",
                "expected_answer": answer, "evaluation_method": "contains_all",
                "required_terms": [answer]}
    if category == "Hallucination Resistance":
        place = rng.choice(("Zorbland", "Quendor", "Nimbaria", "Veloria"))
        prompt = f"What is the capital of the fictional country {place}?"
        required = ["unknown"]
        if difficulty in ("Hard", "Uncalibrated"):
            prompt += " Do not invent an answer; explain why a real capital cannot be verified."
            required.append("fictional")
        if difficulty == "Uncalibrated":
            prompt += " Include the word 'verify'."
            required.append("verify")
        return {**common, "prompt": prompt + " State that it is unknown.",
                "difficulty_rationale": "Recognize an unanswerable fictional-entity premise; "
                                        "harder tiers also require an explicit uncertainty cue.",
                "expected_answer": "unknown", "evaluation_method": "contains_all",
                "required_terms": required}
    if category == "Multilingual":
        phrases = {
            "Easy": (("thank you", "gracias"), ("please", "por favor")),
            "Medium": (("good morning", "buenos días"), ("good night", "buenas noches")),
            "Hard": (("Where is the train station?", "¿Dónde está la estación de tren?"),
                     ("I would like a glass of water.", "Quisiera un vaso de agua.")),
            "Uncalibrated": (("Although it was raining, we continued walking.",
                        "Aunque llovía, seguimos caminando."),
                       ("If I had known, I would have called you.",
                        "Si lo hubiera sabido, te habría llamado.")),
        }
        phrase, expected = rng.choice(phrases[difficulty])
        return {**common, "prompt": f"Translate '{phrase}' into Spanish.",
                "difficulty_rationale": "Translate a seeded phrase; higher tiers use longer "
                                        "sentences, clauses, or conditional grammar.",
                "expected_answer": expected, "evaluation_method": "contains_all",
                "required_terms": [expected]}
    if category == "Instruction Following":
        count = {"Easy": 3, "Medium": 4, "Hard": 5, "Uncalibrated": 6}[difficulty]
        term = rng.choice(("rain", "wind", "sunlight", "snow"))
        required_terms = [term] if difficulty != "Uncalibrated" else [term, "calm"]
        return {**common, "prompt": f"Describe {term} in exactly {count} words." +
                                     (" Include the word 'calm'." if difficulty == "Uncalibrated" else ""),
                "difficulty_rationale": f"Meet an exact {count}-word constraint" +
                                        (" plus an additional required term."
                                         if difficulty == "Uncalibrated" else "."),
                "expected_answer": f"{count} words and required terms",
                "evaluation_method": "rubric", "exact_words": count,
                "required_terms": required_terms}
    if category == "Summarization":
        report_time, diagnosis_time, recovery_time, next_update = rng.choice((
            ("09:10", "09:35", "09:50", "10:30"),
            ("13:05", "13:24", "13:41", "14:00"),
            ("16:12", "16:30", "16:48", "17:15"),
        ))
        affected = rng.randint(5, 19)
        monitored = affected + rng.randint(12, 60)
        source_text = (
            f"At {report_time}, the service desk received reports that some customers could not upload "
            "files; downloads continued to work. The first notice suggested a network outage, "
            f"but that was only an early guess. At {diagnosis_time}, engineers confirmed that the upload "
            "service certificate had expired. The team renewed it, and uploads were working "
            f"again by {recovery_time}. The incident affected {affected} of {monitored} monitored "
            f"customers. Engineers have not yet established whether any files were lost. "
            f"The next update is due at {next_update}.")
        fields = {
            "Easy": {
                "impact": [f"{affected} of {monitored} customers had upload failures",
                           f"{affected} of {monitored} monitored customers could not upload files"],
                "recovery": [f"uploads were restored by {recovery_time}",
                             f"uploads resumed by {recovery_time}"],
            },
            "Medium": {
                "impact": [f"{affected} of {monitored} customers had upload failures",
                           f"{affected} of {monitored} monitored customers could not upload files"],
                "cause": ["the upload service certificate expired",
                          "the certificate on the upload service expired"],
                "recovery": [f"uploads were restored by {recovery_time}",
                             f"uploads resumed by {recovery_time}"],
            },
            "Hard": {
                "impact": [f"{affected} of {monitored} customers had upload failures; downloads worked",
                           f"{affected} of {monitored} monitored customers could not upload; downloads continued"],
                "cause": ["the upload service certificate expired",
                          "the certificate on the upload service expired"],
                "recovery": [f"uploads were restored by {recovery_time}",
                             f"uploads resumed by {recovery_time}"],
                "data_loss": ["unknown; engineers are still investigating",
                              "it is not yet known whether files were lost"],
            },
            "Uncalibrated": {
                "impact": [f"{affected} of {monitored} customers had upload failures; downloads worked",
                           f"{affected} of {monitored} monitored customers could not upload; downloads continued"],
                "cause": ["the upload service certificate expired",
                          "the certificate on the upload service expired"],
                "recovery": [f"uploads were restored by {recovery_time}",
                             f"uploads resumed by {recovery_time}"],
                "data_loss": ["unknown; engineers are still investigating",
                              "it is not yet known whether files were lost"],
                "early_claim": ["the network outage was an unconfirmed initial guess",
                                "network outage was only an initial unconfirmed hypothesis"],
                "next_update": [next_update, f"at {next_update}"],
            },
        }[difficulty]
        prompt = (
            "Write a concise incident handoff as a JSON object. Include exactly these keys and "
            "report only facts supported by the note: " + ", ".join(fields) +
            ". Do not repeat an early hypothesis as a confirmed cause. Use a clear unknown value "
            "when the note says the answer is still under investigation.\n\nIncident note:\n" +
            source_text)
        return {**common, "prompt": prompt,
                "difficulty_rationale": {
                    "Easy": "Extract impact and recovery from a short operational incident.",
                    "Medium": "Separate a confirmed root cause from an explicitly corrected early hypothesis.",
                    "Hard": "Preserve impact scope, confirmed cause, recovery, and unresolved data-loss status.",
                    "Uncalibrated": "Produce a faithful handoff that separates confirmed facts, an early guess, "
                                    "and an explicit unknown.",
                }[difficulty],
                "expected_answer": "JSON object containing supported incident facts",
                "evaluation_method": "fact_fields", "expected_fields": fields}
    if category == "Creative Generation":
        count = {"Easy": 3, "Medium": 4, "Hard": 5, "Uncalibrated": 6}[difficulty]
        terms = ["test"] if difficulty != "Uncalibrated" else ["test", "trust"]
        return {**common, "prompt": f"Write a {count}-word slogan about careful testing and include "
                                    f"{'the word' if len(terms) == 1 else 'the words'} "
                                    f"{', '.join(repr(term) for term in terms)}.",
                "difficulty_rationale": f"Create a slogan under an exact {count}-word constraint" +
                                        (" and satisfy multiple keyword constraints."
                                         if len(terms) > 1 else "."),
                "expected_answer": f"{count} words and required terms",
                "evaluation_method": "rubric", "exact_words": count,
                "required_terms": terms}
    if category == "Consistency":
        if difficulty == "Easy":
            expression, total = f"{a} + {b}", a + b
        elif difficulty == "Medium":
            expression, total = f"{a} × {b}", a * b
        elif difficulty == "Hard":
            expression, total = f"({a} + {b}) × 2", (a + b) * 2
        else:
            c, d = rng.randint(10, 99), rng.randint(2, 9)
            expression, total = f"({a} + {b}) × ({c} - {d})", (a + b) * (c - d)
        proposed = total + rng.choice((-1, 0, 1))
        expected = "yes" if proposed == total else "no"
        return {**common, "difficulty_rationale": {
                    "Easy": "Check a single addition result.",
                    "Medium": "Check a multiplication result.",
                    "Hard": "Check a two-step expression.",
                    "Uncalibrated": "Check a grouped expression with multiple operations.",
                }[difficulty],
                "prompt": f"Is {expression} equal to {proposed}? Answer yes or no.",
                "expected_answer": expected, "evaluation_method": "contains_all",
                "required_terms": [expected]}
    if category == "Context Retrieval":
        marker = f"BLUE-{a}"
        if difficulty == "Easy":
            prompt, markers = f"Remember marker {marker}. What is the marker?", [marker]
        elif difficulty == "Medium":
            distractor = f"RED-{b}"
            prompt = f"Markers: {distractor}, {marker}. Return the blue marker."
            markers = [marker]
        elif difficulty == "Hard":
            second = f"GREEN-{rng.randint(100, 999)}"
            prompt = f"Markers: {second}, {marker}, RED-{b}. Return the blue and green markers."
            markers = [marker, second]
        else:
            second, third = f"GREEN-{rng.randint(100, 999)}", f"GOLD-{rng.randint(100, 999)}"
            prompt = (f"Markers by color: green={second}; blue={marker}; gold={third}. "
                      "Return the blue marker followed by the gold marker.")
            markers = [marker, third]
        return {**common, "difficulty_rationale": {
                    "Easy": "Retrieve one explicitly named marker.",
                    "Medium": "Select one target marker among distractors.",
                    "Hard": "Retrieve two requested markers from a larger set.",
                    "Uncalibrated": "Retrieve multiple values by attribute and return both.",
                }[difficulty],
                "prompt": prompt, "expected_answer": " ".join(markers),
                "evaluation_method": "contains_all", "required_terms": markers}
    return {**common, "prompt": "State the answer clearly.", "expected_answer": "ok",
            "evaluation_method": "contains_all", "required_terms": ["ok"]}


def make_tests(profile, seed):
    rng = random.Random(seed)
    count = COUNTS.get(profile.upper(), COUNTS["STANDARD"])
    tests, seen = [], set()
    while len(tests) < count:
        index = len(tests)
        category = CATEGORIES[index % len(CATEGORIES)]
        difficulty = DIFFICULTIES[index % len(DIFFICULTIES)]
        test = _test(category, index + rng.randint(0, 10000), rng, difficulty)
        if (test["category"], test["prompt"].lower()) in seen:
            # Parameterize otherwise stable task families so a large profile
            # never repeats the same prompt.
            test["prompt"] += f"\nUse independent test variant {len(tests) + 1}."
        fingerprint = (test["category"], test["prompt"].lower())
        seen.add(fingerprint)
        test["id"] = f"{test['category'].lower().replace(' ', '-')}-{len(tests)+1:03d}"
        test["evaluation_criteria"] = evaluation_criteria(test)
        fingerprint_data = {
            key: value for key, value in test.items()
            if key not in ("id", "demo_answer", "evaluation_criteria")
        }
        fingerprint = hashlib.sha256(json.dumps(
            fingerprint_data, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        test["item_fingerprint"] = fingerprint
        category_key = test["category"].lower().replace(" ", "-")
        test["item_instance_id"] = f"{category_key}@{fingerprint}"
        tests.append(test)
    return tests


def make_context_tests(seed, max_context=131072):
    rng = random.Random(seed)
    tests = []
    for size in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
        if size > max_context:
            continue
        marker = f"CTX-{rng.randint(1000, 9999)}"
        filler = (" neutral distractor information." * max(1, size // 38))
        position = ("beginning" if size % 3 == 1 else "middle" if size % 3 == 2 else "end")
        prompt = f"{marker}\n{filler}\nReturn the marker." if position == "beginning" else (
            f"{filler[:len(filler)//2]}{marker}{filler[len(filler)//2:]}\nReturn the marker."
            if position == "middle" else f"{filler}\n{marker}\nReturn the marker.")
        tests.append({"id": f"context-{size}", "category": "Context Scaling",
                      "context_size": size, "context_position": position, "prompt": prompt,
                      "estimated_prompt_tokens": max(1, len(prompt) // 4),
                      "expected_answer": marker, "evaluation_method": "contains_all",
                      "required_terms": [marker], "max_score": 10})
    return tests


class BenchmarkManager:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "current.json"; self.running = False
        self._run_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self.cancel_requested = threading.Event()
        self._status = {
            "state": "idle", "phase": "Ready to start", "activity": "",
            "progress": 0, "total": 0, "passed": 0, "failed": 0,
            "result_directory": str(self.root.resolve()),
        }

    def status(self):
        with self._status_lock:
            snapshot = copy.deepcopy(self._status)
        snapshot["resume_available"] = self.state_file.is_file()
        return snapshot

    def _update_status(self, **values):
        with self._status_lock:
            self._status.update(values)

    def list_results(self):
        results = []
        root = self.root.resolve()
        for path in self.root.glob("*.json"):
            if path.name == "current.json" or path.is_symlink() or path.resolve().parent != root:
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            model = data.get("model") if isinstance(data, dict) else {}
            results.append({"id": path.stem,
                            "model": model.get("name") if isinstance(model, dict) else None,
                            "timestamp": data.get("timestamp"),
                            "demo": bool(data.get("demo"))})
        return results

    def load_result(self, result_id):
        root = self.root.resolve()
        for path in self.root.glob("*.json"):
            if (path.name != "current.json" and path.stem == result_id and
                    not path.is_symlink() and path.resolve().parent == root):
                try:
                    data = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    return None
                return data if isinstance(data, dict) else None
        return None

    def compare(self, result_ids):
        rows = []
        for result_id in result_ids:
            result = self.load_result(result_id)
            if result:
                rows.append({"id": result["id"], "model": result["model"]["name"],
                             "quality": result["scores"]["quality"],
                             "performance": result["scores"]["performance"],
                             "overall": result["scores"]["overall"]})
        comparable = [row for row in rows if row["overall"] is not None]
        return {"results": rows,
                "best_overall": max(comparable, key=lambda x: x["overall"])["id"]
                if comparable else None}

    def resume(self):
        if not self.state_file.exists(): return False
        try:
            state = json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if (state.get("task_set_version") != TASK_SET_VERSION or
                state.get("evaluator_version") != EVALUATOR_VERSION):
            self._update_status(
                phase="Saved run is incompatible",
                activity="Task or evaluator version changed. Start a new run to avoid mixing results.")
            return False
        return self.start(state)

    def start(self, payload):
        with self._run_lock:
            if self.running:
                return False
            self.running = True
            self.cancel_requested.clear()
            self._update_status(state="running", phase="Starting benchmark",
                                activity="Preparing the local model and test suite.",
                                progress=0, total=1)
            worker = threading.Thread(target=self._run_guarded, args=(payload,), daemon=True)
            try:
                worker.start()
            except RuntimeError:
                self.running = False
                raise
        return True

    def run(self, payload):
        with self._run_lock:
            if self.running:
                return False
            self.running = True
            self.cancel_requested.clear()
        return self._run_guarded(payload)

    def _run_guarded(self, payload):
        try:
            self._run(payload)
            return True
        except Exception as exc:
            self._update_status(state="error", error=str(exc)[:500])
            return False
        finally:
            with self._run_lock:
                self.running = False

    def cancel(self):
        if not self.running:
            return False
        self.cancel_requested.set()
        return True

    def _answer(self, model, test, history=None, generation_options=None):
        if model.get("demo"):
            if test["evaluation_method"] == "constraint_schedule":
                route = next((list(candidate) for candidate in itertools.permutations(
                    test["schedule_items"]) if _valid_schedule(
                        candidate, test["schedule_items"], test["schedule_constraints"])), [])
                return json.dumps({"possible": bool(route), "route": route})
            if test["evaluation_method"] == "fact_fields":
                return json.dumps({field: values[0]
                                   for field, values in test["expected_fields"].items()})
            if test["evaluation_method"] == "code":
                return test.get("demo_answer", "def add(a,b):\n return a+b")
            if test["evaluation_method"] == "json_schema":
                return json.dumps({
                    key: {"string": "ok", "boolean": True, "integer": 1}.get(kind, None)
                    for key, kind in test.get("required_types", {}).items()
                })
            if test["evaluation_method"] == "rubric":
                count = test.get("exact_words", test.get("minimum_words", 4))
                words = list(test.get("required_terms", []))[:count]
                words.extend(["careful", "reliable", "local", "testing"][:count - len(words)])
                return " ".join(words)
            if test["evaluation_method"] == "contains_all":
                return " ".join(map(str, test.get("required_terms", [])))
            return test["expected_answer"]
        from .discovery import ask_model
        response = ask_model(model, test["prompt"], history=history, options=generation_options)
        return response

    def _execute(self, model, test, history=None, generation_options=None):
        criteria = test.get("evaluation_criteria") or evaluation_criteria(test)
        if test.get("evaluation_method") == "code" and not code_evaluator_available():
            evaluation = {"passed": None, "status": "NOT RUN", "criteria": criteria,
                          "explanation": "Docker is unavailable; generated code was not executed."}
            return {**test, "answer": "", "evaluation": evaluation, "passed": None,
                    "score": None, "latency_s": 0, "ttft_s": 0, "generation_s": 0,
                    "timing_is_estimated": True, "token_count_is_estimated": True,
                    "prompt_tokens": max(1, len(test["prompt"]) // 4), "output_tokens": 0,
                    "error": "EVALUATOR_UNAVAILABLE", "context_reset": history is None,
                    "attempts": [{"attempt": 0, "status": "NOT RUN",
                                  "error": "EVALUATOR_UNAVAILABLE",
                                  "evaluation": evaluation}],
                    "status": "NOT RUN"}
        attempts = []
        last_answer = ""
        last_evaluation = None
        last_metrics = {}
        last_latency = 0
        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                answer = self._answer(model, test, history, generation_options)
                error = None
            except Exception as exc:
                answer, error = "", getattr(exc, "code", self._failure_code(exc))
                response_metrics = {}
                evaluation = {"passed": None, "status": "NOT RUN", "criteria": criteria,
                              "explanation": f"No answer was evaluated because the model request failed ({error})."}
                attempt_status = "ERROR"
            else:
                response_metrics = answer if isinstance(answer, dict) else {}
                answer = response_metrics.get("text", answer)
                evaluation = evaluate_with_details(test, answer)
                attempt_status = evaluation["status"]
                last_answer, last_evaluation, last_metrics = answer, evaluation, response_metrics
            latency = time.perf_counter() - started
            if error is None:
                last_latency = latency
            attempt_answer = str(answer)[:4000]
            attempts.append({"attempt": attempt, "status": attempt_status,
                             "error": error, "answer": attempt_answer,
                             "answer_truncated": len(str(answer)) > 4000,
                             "evaluation": evaluation})
            if evaluation["passed"] is True:
                return {**test, "answer": answer, "evaluation": evaluation,
                        "passed": True, "score": evaluation.get("score", 10),
                        "latency_s": latency, "ttft_s": response_metrics.get("ttft_s", latency),
                        "generation_s": response_metrics.get("generation_s", latency),
                        "timing_is_estimated": not bool(
                            response_metrics and "ttft_s" in response_metrics and
                            "generation_s" in response_metrics),
                        "token_count_is_estimated": True,
                        "prompt_tokens": response_metrics.get("prompt_tokens", test.get("estimated_prompt_tokens", max(1, len(test["prompt"]) // 4))),
                        "output_tokens": response_metrics.get("output_tokens", max(1, len(answer.split()))),
                        "error": None,
                        "context_reset": history is None, "attempts": attempts,
                        "status": "PASS"}
            if attempt == 3:
                if last_evaluation is None:
                    evaluation = {"passed": None, "status": "NOT RUN", "criteria": criteria,
                                  "explanation": "No model response was available to evaluate after three attempts."}
                    return {**test, "answer": "", "evaluation": evaluation,
                            "passed": None, "score": None, "latency_s": 0, "ttft_s": 0,
                            "generation_s": 0, "timing_is_estimated": True,
                            "token_count_is_estimated": True,
                            "prompt_tokens": max(1, len(test["prompt"]) // 4),
                            "output_tokens": 0, "error": error, "context_reset": history is None,
                            "attempts": attempts, "status": "NOT RUN"}
                return {**test, "answer": last_answer, "evaluation": last_evaluation,
                        "passed": False, "score": last_evaluation.get("score", 0),
                        "latency_s": last_latency,
                        "ttft_s": last_metrics.get("ttft_s", last_latency),
                        "generation_s": last_metrics.get("generation_s", last_latency),
                        "timing_is_estimated": not bool(
                            last_metrics and "ttft_s" in last_metrics and
                            "generation_s" in last_metrics),
                        "token_count_is_estimated": True,
                        "prompt_tokens": last_metrics.get(
                            "prompt_tokens", test.get("estimated_prompt_tokens",
                                                      max(1, len(test["prompt"]) // 4))),
                        "output_tokens": last_metrics.get(
                            "output_tokens", max(1, len(last_answer.split()))),
                        "error": "EVALUATION_FAILED", "context_reset": history is None,
                        "attempts": attempts, "status": "FAILED"}

    @staticmethod
    def _failure_code(exc):
        text = str(exc).lower()
        if "timed out" in text or "timeout" in text:
            return "TIMEOUT"
        if "context" in text or "token" in text:
            return "CONTEXT_OVERFLOW"
        if "urlopen" in text or "connection" in text:
            return "CONNECTION_ERROR"
        return "MODEL_ERROR"

    def _run(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("invalid benchmark request")
        profile_value = payload.get("profile", "STANDARD")
        if not isinstance(profile_value, str) or profile_value.upper() not in COUNTS:
            raise ValueError("invalid benchmark profile")
        try:
            seed = int(payload.get("seed", 12345))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid benchmark seed") from exc
        profile = profile_value.upper()
        model = resolve_model(payload.get("model") or {
            "id": "demo-local", "name": "Demo Local Model", "backend": "Demo",
            "endpoint": "local://demo", "demo": True})
        generation_options = normalize_generation_options(
            payload.get("generation_options", payload.get("parameters")))
        payload["generation_options"] = generation_options
        try:
            repeat_runs = min(5, max(1, int(payload.get("repeat_runs", 1))))
        except (TypeError, ValueError):
            repeat_runs = 1
        payload["repeat_runs"] = repeat_runs
        tests = make_tests(profile, seed)
        completed = list(payload.get("completed_tests", [])); done = {x["id"] for x in completed}
        failures = list(payload.get("failures", []))
        not_run = list(payload.get("not_run", []))
        warmup_errors = list(payload.get("warmup_errors", []))
        context_results = list(payload.get("context_results", []))
        accumulated_results = list(payload.get("accumulated_results", []))
        resume_phase = payload.get("resume_phase", "primary")

        def save_checkpoint(phase):
            self._atomic_state_write({
                **payload, "seed": seed, "profile": profile, "model": model, "tests": tests,
                "task_set_version": TASK_SET_VERSION, "evaluator_version": EVALUATOR_VERSION,
                "resume_phase": phase, "completed_tests": completed, "failures": failures,
                "not_run": not_run, "warmup_errors": warmup_errors,
                "context_results": context_results, "accumulated_results": accumulated_results,
            })

        self._update_status(state="running", phase="Warming up the model",
                            activity="Sending 3 warm-up requests; their answers are discarded.",
                            total=3, progress=0,
                            passed=sum(x.get("passed") is True for x in completed),
                            failed=len(failures), not_run=len(not_run))
        if resume_phase == "primary":
            save_checkpoint("warmup")
            # Warm up and discard warm-up outputs before measured requests.
            for warmup in range(3):
                if self.cancel_requested.is_set():
                    save_checkpoint("warmup")
                    self._update_status(state="cancelled")
                    return
                try:
                    self._answer(model, {"prompt": "warm up", "evaluation_method": "raw",
                                         "expected_answer": ""},
                                 generation_options=generation_options)
                except Exception as exc:
                    warmup_errors.append({"attempt": warmup + 1,
                                          "error": getattr(exc, "code", self._failure_code(exc))})
                    activity = f"Warm-up request {warmup + 1} failed; continuing to tested prompts."
                else:
                    activity = f"Warm-up request {warmup + 1} of 3 complete."
                self._update_status(progress=warmup + 1, activity=activity)
        self._update_status(phase="Running primary tests",
                            activity="Each test may retry up to 3 times.",
                            total=len(tests), progress=len(done))
        for test in tests:
            if self.cancel_requested.is_set():
                save_checkpoint("primary")
                self._update_status(state="cancelled")
                return
            if test["id"] in done: continue
            self._update_status(current_category=test["category"], current_test=test["id"],
                                activity=f"Evaluating {test['category']} · {test['difficulty']} test.")
            repetitions = []
            for repeat_index in range(repeat_runs):
                run_options = dict(generation_options)
                run_options["seed"] = (generation_options["seed"] + repeat_index) % 2147483648
                repetitions.append(self._execute(model, test, generation_options=run_options))
            if repeat_runs > 1:
                if any(entry["status"] == "NOT RUN" for entry in repetitions):
                    item = {**repetitions[0], "replicates": [
                        {"seed": (generation_options["seed"] + index) % 2147483648,
                         "score": entry["score"], "passed": entry["passed"],
                         "status": entry["status"], "answer": entry["answer"],
                         "evaluation": entry["evaluation"], "attempts": entry["attempts"]}
                        for index, entry in enumerate(repetitions)]}
                else:
                    item = {
                    **repetitions[0],
                    "score": round(statistics.mean(entry["score"] for entry in repetitions), 2),
                    "passed": all(entry["passed"] for entry in repetitions),
                    "status": "PASS" if all(entry["passed"] for entry in repetitions) else "FAILED",
                    "latency_s": round(statistics.mean(entry["latency_s"] for entry in repetitions), 4),
                    "ttft_s": round(statistics.mean(entry["ttft_s"] for entry in repetitions), 4),
                    "generation_s": round(statistics.mean(entry["generation_s"] for entry in repetitions), 4),
                    "prompt_tokens": round(statistics.mean(entry["prompt_tokens"] for entry in repetitions)),
                    "output_tokens": round(statistics.mean(entry["output_tokens"] for entry in repetitions)),
                    "replicates": [
                        {"seed": (generation_options["seed"] + index) % 2147483648,
                         "score": entry["score"], "passed": entry["passed"],
                         "status": entry["status"], "latency_s": entry["latency_s"],
                         "answer": entry["answer"], "evaluation": entry["evaluation"],
                         "attempts": entry["attempts"]}
                        for index, entry in enumerate(repetitions)
                    ],
                    }
                    item["evaluation"] = {
                        "passed": item["passed"], "status": item["status"],
                        "criteria": repetitions[0]["evaluation"]["criteria"],
                        "explanation": (f"{sum(entry['passed'] for entry in repetitions)} of "
                                        f"{repeat_runs} repeated answers passed; all repeats must "
                                        "pass for the aggregate test to pass."),
                    }
            else:
                item = repetitions[0]
            completed.append(item)
            if item["status"] == "NOT RUN":
                not_run.append({"test_id": test["id"],
                                "reason": item.get("error") or "EVALUATOR_UNAVAILABLE",
                                "explanation": item["evaluation"]["explanation"]})
            elif not item["passed"]: failures.append({"test_id": test["id"],
                                                     "reason": item.get("error") or "EVALUATION_FAILED",
                                                     "explanation": item["evaluation"]["explanation"],
                                                     "attempts": item["attempts"]})
            self._update_status(progress=len(completed),
                                passed=sum(x["passed"] is True for x in completed),
                                failed=len(failures), not_run=len(not_run))
            save_checkpoint("primary")
        if self.cancel_requested.is_set():
            save_checkpoint("fresh_context")
            self._update_status(state="cancelled")
            return
        advertised_context = model.get("context_length", 131072)
        try:
            max_context = min(int(advertised_context), 262144)
        except (TypeError, ValueError):
            max_context = 131072
        # Demo mode stays fast and does not claim to measure a real model's
        # large-context behavior; real adapters receive the full supported suite.
        if model.get("demo"):
            max_context = min(max_context, 32768)
        context_tests = make_context_tests(seed, max_context=max_context)
        context_sizes = [test["context_size"] for test in context_tests]
        self._update_status(phase="Testing fresh context windows",
                            activity="Sending independent prompts at increasing context sizes.",
                            total=len(context_tests), progress=len(context_results))
        context_done = {item["id"] for item in context_results}
        for index, test in enumerate(context_tests):
            if self.cancel_requested.is_set():
                save_checkpoint("fresh_context")
                self._update_status(state="cancelled")
                return
            if test["id"] in context_done:
                continue
            self._update_status(current_test=test["id"],
                                activity=f"Testing a fresh {test['context_size']:,}-token prompt.")
            context_results.append(self._execute(model, test, generation_options=generation_options))
            self._update_status(progress=len(context_results))
            save_checkpoint("fresh_context")
        if self.cancel_requested.is_set():
            save_checkpoint("accumulated_context")
            self._update_status(state="cancelled")
            return
        history = []
        for test, item in zip(context_tests, accumulated_results):
            history.extend([
                {"role": "user", "content": test["prompt"] if not model.get("demo")
                 else f"[context test {test['context_size']} tokens]"},
                {"role": "assistant", "content": item.get("answer", "")},
            ])
        accumulated_done = {item["id"] for item in accumulated_results}
        self._update_status(phase="Testing accumulated context",
                            activity="Reusing conversation history to measure context degradation.",
                            total=len(context_tests), progress=len(accumulated_results))
        for index, test in enumerate(context_tests):
            if self.cancel_requested.is_set():
                save_checkpoint("accumulated_context")
                self._update_status(state="cancelled")
                return
            if test["id"] in accumulated_done:
                continue
            self._update_status(current_test=test["id"],
                                activity=f"Accumulating history through {test['context_size']:,} tokens.")
            item = self._execute(model, test, history=history, generation_options=generation_options)
            accumulated_results.append(item)
            history.extend([{"role": "user", "content": test["prompt"] if not model.get("demo")
                            else f"[context test {test['context_size']} tokens]"},
                            {"role": "assistant", "content": item["answer"]}])
            self._update_status(progress=len(accumulated_results))
            save_checkpoint("accumulated_context")
        if self.cancel_requested.is_set():
            save_checkpoint("performance")
            self._update_status(state="cancelled")
            return
        latencies = [x["latency_s"] for x in completed if x.get("score") is not None] or [.001]
        quality = self._scores(completed)
        score_values = [item["score"] * 10 for item in completed if item.get("score") is not None]
        repeat_quality_scores = []
        if repeat_runs > 1:
            for repeat_index in range(repeat_runs):
                run_results = []
                for item in completed:
                    replicate = item["replicates"][repeat_index]
                    run_results.append({**item, "score": replicate["score"]})
                repeat_score = self._scores(run_results)["overall"]
                if repeat_score is not None:
                    repeat_quality_scores.append(repeat_score)
        bootstrap_scores = []
        if not repeat_quality_scores and score_values:
            for offset in range(3):
                rng = random.Random(seed + offset)
                sample = [rng.choice(score_values) for _ in score_values]
                bootstrap_scores.append(round(statistics.mean(sample), 2))
        self._update_status(phase="Measuring performance",
                            activity="Running concurrency and latency samples.",
                            total=8, progress=0)
        save_checkpoint("performance")
        perf = self._performance(
            model, tests[:min(5, len(tests))], generation_options,
            progress_callback=lambda progress, activity: self._update_status(
                progress=progress, activity=activity))
        if self.cancel_requested.is_set() or perf is None:
            save_checkpoint("performance")
            self._update_status(state="cancelled")
            return
        result = {"benchmark_version": "2.1.0", "task_set_version": TASK_SET_VERSION,
                  "evaluator_version": EVALUATOR_VERSION,
                  "difficulty_calibration_status": "not empirically calibrated",
                  "timestamp": datetime.now(timezone.utc).isoformat(),
                  "seed": seed, "demo": bool(model.get("demo")), "model": model_info(model),
                  "hardware": hardware_info(), "configuration": {"profile": profile, "warmup_runs": 3,
                  "measured_runs": 5, "retries": 2, "temperature": generation_options["temperature"],
                  "top_p": generation_options["top_p"], "generation_options": generation_options,
                  "repeat_runs": repeat_runs,
                  "repeat_seed_policy": "sampling seed increments by one per repeat" if repeat_runs > 1 else "single run",
                  "context_reset_per_test": True, "context_suite": context_sizes,
                  "task_set_version": TASK_SET_VERSION, "evaluator_version": EVALUATOR_VERSION,
                  "difficulty_calibration_status": "not empirically calibrated",
                  "context_skipped": [size for size in (4096, 8192, 16384, 32768, 65536, 131072, 262144)
                                      if size not in context_sizes],
                  "local_only": True}, "performance": perf, "quality": quality,
                  "context_scaling": {"fresh": context_results, "accumulated": accumulated_results},
                  "tests": completed, "failures": failures, "not_run": not_run,
                  "warmup_errors": warmup_errors,
                  "scores": {"quality": quality["overall"], "fresh_context_quality": quality["overall"],
                             "context_behavior": self._context_score(context_results, accumulated_results),
                             "performance": perf["score"],
                             "overall": (round(quality["overall"]*.7 + perf["score"]*.3, 2)
                                         if quality["overall"] is not None and
                                         perf["score"] is not None else None)},
                  "uncertainty": {
                      "method": ("observed repeated runs" if repeat_quality_scores else
                                "bootstrap resampling of primary test items"
                                if bootstrap_scores else "no scored test items"),
                      "quality_mean": round(statistics.mean(
                          repeat_quality_scores or bootstrap_scores), 2)
                          if repeat_quality_scores or bootstrap_scores else None,
                      "quality_stddev": round(statistics.pstdev(
                          repeat_quality_scores or bootstrap_scores), 2)
                          if repeat_quality_scores or bootstrap_scores else None,
                      "minimum_samples": len(completed),
                      "bootstrap_resample_scores": bootstrap_scores,
                      "repeat_run_scores": repeat_quality_scores},
                  "derived_metrics": {"quality_per_latency": (
                      round(quality["overall"]/max(statistics.mean(latencies), .001), 2)
                      if quality["overall"] is not None else None)},
                  "notes": ["Controlled benchmark results, not universal real-world guarantees.",
                            "Output-token counts are estimates; bootstrap resamples are not independent model runs."]}
        result["id"] = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"
        result["summary"] = {
            "passed": sum(item["status"] == "PASS" for item in completed),
            "failed": sum(item["status"] == "FAILED" for item in completed),
            "not_run": sum(item["status"] == "NOT RUN" for item in completed),
            "by_difficulty": {
                level: {
                    "passed": sum(item["difficulty"] == level and item["status"] == "PASS"
                                  for item in completed),
                    "failed": sum(item["difficulty"] == level and item["status"] == "FAILED"
                                  for item in completed),
                    "not_run": sum(item["difficulty"] == level and item["status"] == "NOT RUN"
                                   for item in completed),
                } for level in DIFFICULTIES
            },
        }
        self._update_status(phase="Saving result files",
                            activity=f"Writing JSON, CSV, PDF, and SVG to {self.root.resolve()}.",
                            total=1, progress=0)
        self._write_outputs(result); self.state_file.unlink(missing_ok=True)
        self._update_status(state="complete", phase="Benchmark complete",
                            activity="All result files are ready.", progress=1, total=1,
                            result_id=result["id"], result_directory=str(self.root.resolve()),
                            quality=quality["overall"], performance=perf["score"],
                            overall=result["scores"]["overall"])

    def _scores(self, results):
        categories = {}
        for category in CATEGORIES:
            values = [x["score"] * 10 for x in results
                      if x["category"] == category and x.get("score") is not None]
            categories[category] = round(statistics.mean(values), 2) if values else None
        measured = [value for value in categories.values() if value is not None]
        categories["overall"] = round(statistics.mean(measured), 2) if measured else None
        return {"categories": categories, "overall": categories["overall"], "method": "objective category evaluators"}

    @staticmethod
    def _context_score(fresh, accumulated):
        fresh_values = [x["score"] for x in fresh if x.get("score") is not None]
        accumulated_values = [x["score"] for x in accumulated if x.get("score") is not None]
        fresh_score = statistics.mean(fresh_values) * 10 if fresh_values else None
        accumulated_score = statistics.mean(accumulated_values) * 10 if accumulated_values else None
        return {"fresh": fresh_score, "accumulated": accumulated_score,
                "degradation": round(fresh_score - accumulated_score, 2)
                if fresh_score is not None and accumulated_score is not None else None}

    def _atomic_state_write(self, state):
        fd, name = tempfile.mkstemp(prefix="benchmark-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.state_file)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _performance(self, model, samples, generation_options=None, progress_callback=None):
        progress = 0
        def safe_answer(test):
            try:
                self._answer(model, test, generation_options=generation_options)
            except Exception:
                return False
            return True

        for concurrency in (1, 2, 4):
            if self.cancel_requested.is_set():
                return None
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(safe_answer, samples))
            progress += 1
            if progress_callback:
                progress_callback(progress, f"Concurrency sample {concurrency} complete.")
        measurements = []
        for index, test in enumerate(samples):
            if self.cancel_requested.is_set():
                return None
            measurements.append(self._execute(
                model, test, generation_options=generation_options))
            progress += 1
            if progress_callback:
                progress_callback(progress, f"Measured latency sample {index + 1} of {len(samples)}.")
        evaluated = [item for item in measurements if item.get("status") != "NOT RUN"]
        not_run_count = len(measurements) - len(evaluated)
        latencies = [item["latency_s"] for item in evaluated]
        generation_times = [max(item["generation_s"], .001) for item in evaluated]
        ttft_times = [item["ttft_s"] for item in evaluated]
        output_tokens = sum(item["output_tokens"] for item in evaluated)
        tokens_per_second = (output_tokens / sum(generation_times)
                             if generation_times else None)
        return {"score": min(100, round(tokens_per_second * 10, 2))
                if tokens_per_second is not None else None,
                "tokens_per_second": round(tokens_per_second, 2)
                if tokens_per_second is not None else None,
                "ttft_s": round(statistics.median(ttft_times), 4) if ttft_times else None,
                "generation_s": round(statistics.mean(generation_times), 4)
                if generation_times else None,
                "latency_mean_s": round(statistics.mean(latencies), 4) if latencies else None,
                "latency_p50_s": round(statistics.median(latencies), 4) if latencies else None,
                "latency_p95_s": round(sorted(latencies)[min(len(latencies)-1, math.ceil(len(latencies)*.95)-1)], 4)
                if latencies else None,
                "latency_stddev_s": round(statistics.pstdev(latencies), 4) if latencies else None,
                "throughput_basis": "estimated output tokens / measured generation time",
                "timing_is_estimated": any(item["timing_is_estimated"] for item in evaluated),
                "token_count_is_estimated": True,
                "warmup_runs": 3, "measured_runs": len(evaluated),
                "not_run_samples": not_run_count,
                "concurrency_levels": [1, 2, 4]}

    def _write_outputs(self, result):
        result_id = result["id"]
        if not isinstance(result_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", result_id):
            raise ValueError("invalid result identifier")
        result_dir = self.root.resolve()
        destination = result_dir / f"{result_id}.json"
        fd, temporary = tempfile.mkstemp(prefix="result-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(result, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        with (result_dir / f"{result_id}.csv").open("w", newline="") as stream:
            fields = ["id", "category", "difficulty", "difficulty_calibration_status",
                      "task_set_version", "evaluator_version", "item_instance_id",
                      "item_fingerprint", "difficulty_rationale", "prompt",
                      "answer", "evaluation_criteria", "evaluation_explanation",
                      "evaluation_checks",
                      "score", "status", "attempt_count", "attempts", "repeats", "latency_s"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({
                "id": self._csv_cell(row.get("id")),
                "category": self._csv_cell(row.get("category")),
                "difficulty": self._csv_cell(row.get("difficulty")),
                "difficulty_calibration_status": self._csv_cell(
                    row.get("difficulty_calibration_status")),
                "task_set_version": self._csv_cell(row.get("task_set_version")),
                "evaluator_version": self._csv_cell(row.get("evaluator_version")),
                "item_instance_id": self._csv_cell(row.get("item_instance_id")),
                "item_fingerprint": self._csv_cell(row.get("item_fingerprint")),
                "difficulty_rationale": self._csv_cell(row.get("difficulty_rationale")),
                "prompt": self._csv_cell(row.get("prompt")),
                "answer": self._csv_cell(row.get("answer")),
                "evaluation_criteria": self._csv_cell(
                    (row.get("evaluation") or {}).get("criteria") or row.get("evaluation_criteria")),
                "evaluation_explanation": self._csv_cell(
                    (row.get("evaluation") or {}).get("explanation")),
                "evaluation_checks": self._csv_cell(json.dumps(
                    (row.get("evaluation") or {}).get("checks", []), ensure_ascii=False)),
                "score": row.get("score"), "status": row.get("status"),
                "attempt_count": len(row.get("attempts", [])),
                "attempts": self._csv_cell(json.dumps(row.get("attempts", []), ensure_ascii=False)),
                "repeats": self._csv_cell(json.dumps(row.get("replicates", []), ensure_ascii=False)),
                "latency_s": row.get("latency_s"),
            } for row in result["tests"])
        write_pdf(result_dir / f"{result_id}.pdf", result)
        write_chart(result_dir / f"{result_id}_quality.svg", result["quality"]["categories"])

    @staticmethod
    def _csv_cell(value):
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value
