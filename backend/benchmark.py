"""Domain benchmark generation, evaluation, performance measurement, and exports."""
from __future__ import annotations

import csv
import copy
import hashlib
import hmac
import itertools
import json
import math
import random
import secrets
import stat
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
from .reporting import write_pdf, write_chart, write_html

CATEGORIES = ["Reasoning", "Mathematics", "Coding", "Instruction Following",
              "Structured Output", "Knowledge", "Hallucination Resistance",
              "Long Context", "Summarization", "Multilingual", "Creative Generation",
              "Consistency", "Context Retrieval"]
COUNTS = {"QUICK": 25, "STANDARD": 100, "FULL": 252, "PERFORMANCE": 25,
          "QUALITY": 100, "REASONING": 40, "CODING": 40, "LONG CONTEXT": 30}
DIFFICULTIES = ("Easy", "Medium", "Hard", "Uncalibrated")
TASK_SET_VERSION = "pilot-3"
EVALUATOR_VERSION = "pilot-3"
DIFFICULTY_DESCRIPTIONS = {
    "Easy": "A direct, single-step task with one clear requirement.",
    "Medium": "A routine task with a larger input or an additional constraint.",
    "Hard": "A multi-step task or one that combines several constraints.",
    "Uncalibrated": "A pilot task with higher constructed complexity; difficulty has not been calibrated.",
}
MAX_CHECKPOINT_SIZE = 128 * 1024 * 1024


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


def _inventory_adjustment_test(common, difficulty, a, b):
    function = "apply_stock_adjustments"
    rules = [
        "Return a new dictionary; do not mutate the opening-stock dictionary.",
        "Opening stock is a dictionary from SKU strings to integer quantities; ignore invalid entries.",
        "An adjustment is a two-item (sku, delta) pair; add every valid signed integer delta.",
        "Ignore malformed pairs, empty SKUs, non-integer deltas, and boolean deltas.",
        "Trim surrounding whitespace from SKU keys; preserve valid zero balances.",
    ]
    checks = [
        "assert apply_stock_adjustments({}, []) == {}",
        f"assert apply_stock_adjustments({{'PEN': 2}}, [('PEN', {a}), ('PEN', {-b})]) == "
        f"{{'PEN': {2 + a - b}}}",
    ]
    if difficulty in ("Medium", "Hard", "Uncalibrated"):
        checks.append(
            f"assert apply_stock_adjustments({{'PEN': 2, 'BOOK': 4}}, "
            f"[('PEN', {a}), ('BOOK', {-b}), ('PEN', 3)]) == "
            f"{{'PEN': {5 + a}, 'BOOK': {4 - b}}}")
    if difficulty in ("Hard", "Uncalibrated"):
        checks.extend([
            f"assert apply_stock_adjustments({{'PEN': 5, 'BOOK': 3, 'BAD': True, ' ': 7}}, "
            f"[None, [], (' ', 2), ('PEN', True), ('PEN', '4'), ('PEN', {-5})]) == "
            "{'PEN': 0, 'BOOK': 3}",
            f"assert apply_stock_adjustments(None, [('PEN', {a})]) == {{'PEN': {a}}}",
            "stock = {'PEN': 4}; apply_stock_adjustments(stock, [('PEN', 2)]); assert stock == {'PEN': 4}",
        ])
    if difficulty == "Uncalibrated":
        checks.append(
            f"assert apply_stock_adjustments({{' PEN ': 8}}, [('PEN', {-b}), (' PEN ', {b})]) "
            "== {'PEN': 8}")
    source = (
        "def apply_stock_adjustments(opening_stock, adjustments):\n"
        "    result = {}\n"
        "    if isinstance(opening_stock, dict):\n"
        "        for sku, quantity in opening_stock.items():\n"
        "            if (isinstance(sku, str) and sku.strip() and isinstance(quantity, int)\n"
        "                    and not isinstance(quantity, bool)):\n"
        "                result[sku.strip()] = result.get(sku.strip(), 0) + quantity\n"
        "    if isinstance(adjustments, (list, tuple)):\n"
        "        for adjustment in adjustments:\n"
        "            if not isinstance(adjustment, (list, tuple)) or len(adjustment) != 2:\n"
        "                continue\n"
        "            sku, delta = adjustment\n"
        "            if (not isinstance(sku, str) or not sku.strip() or not isinstance(delta, int)\n"
        "                    or isinstance(delta, bool)):\n"
        "                continue\n"
        "            sku = sku.strip()\n"
        "            result[sku] = result.get(sku, 0) + delta\n"
        "    return result")
    return {
        **common,
        "subcategory": "Inventory adjustments",
        "prompt": f"Implement `{function}(opening_stock, adjustments)`. Return a new dictionary of "
                  "current quantities by SKU. Follow the contract exactly:\n" +
                  "\n".join(f"- {rule}" for rule in rules) +
                  f"\nExample: opening stock {{'PEN': 2}} and adjustments [('PEN', {a})] "
                  f"produce {{'PEN': {2 + a}}}.\nReturn Python code only.",
        "expected_answer": "Python function that applies valid signed stock adjustments.",
        "evaluation_method": "code", "tests": checks, "demo_answer": source,
        "difficulty_rationale": {
            "Easy": "Apply signed adjustments to one valid opening quantity.",
            "Medium": "Aggregate repeated updates across multiple SKUs.",
            "Hard": "Reject malformed snapshots and updates without losing valid balances.",
            "Uncalibrated": "Normalize SKU whitespace, preserve zero balances, and avoid input mutation.",
        }[difficulty],
    }


def _long_context_test(common: dict, difficulty: str, rng: random.Random) -> dict:
    noise_count = {"Easy": 24, "Medium": 40, "Hard": 56, "Uncalibrated": 72}[difficulty]
    noise = [
        f"Archive note {index + 1}: batch {rng.randint(1000, 9999)} was checked by desk "
        f"{rng.choice(('north', 'east', 'south', 'west'))}; status {rng.choice(('closed', 'queued', 'held'))}."
        for index in range(noise_count)
    ]
    if difficulty == "Easy":
        asset_ids = rng.sample(range(100, 1000), 8)
        codes = rng.sample(range(1000, 10000), 8)
        target, code = f"AS-{asset_ids[0]}", f"PIN-{codes[0]}"
        records = [f"Asset AS-{asset_id} has access code PIN-{pin}."
                   for asset_id, pin in zip(asset_ids, codes)]
        prompt_question = f"What is the access code for asset {target}?"
        answer = code
        rationale = "Locate one requested value among unrelated archive notes."
    elif difficulty == "Medium":
        person = rng.choice(("Ari", "Bo", "Cy", "Dee", "Eli", "Flo", "Gia", "Hal"))
        staff_ids = [f"ST-{staff_id}" for staff_id in rng.sample(range(100, 1000), 8)]
        people = list(zip(("Ari", "Bo", "Cy", "Dee", "Eli", "Flo", "Gia", "Hal"), staff_ids))
        asset_ids = rng.sample(range(100, 1000), 8)
        codes = rng.sample(range(1000, 10000), 8)
        assets = [(staff_id, f"AS-{asset_id}", f"PIN-{code}")
                  for staff_id, asset_id, code in zip(staff_ids, asset_ids, codes)]
        selected = assets[("Ari", "Bo", "Cy", "Dee", "Eli", "Flo", "Gia", "Hal").index(person)]
        records = [f"Staff directory: {name} has staff ID {staff_id}."
                   for name, staff_id in people]
        records.extend(f"Asset register: {asset} is assigned to {staff_id}; access code {code}."
                       for staff_id, asset, code in assets)
        prompt_question = f"Find the access code for the asset assigned to {person}."
        answer = selected[2]
        rationale = "Cross-reference a named person to a staff ID and then an assigned asset."
    elif difficulty == "Hard":
        asset_ids = rng.sample(range(100, 1000), 19)
        asset = f"AS-{asset_ids[0]}"
        logs = [(f"20{year:02d}-04-{day:02d}", rng.choice(("ready", "held", "repaired")))
                for year, day in ((24, 2), (25, 11), (26, 7))]
        records = [f"Inspection log: asset {asset} was marked {state} on {date}."
                   for date, state in logs]
        records.extend(
            f"Inspection log: asset AS-{asset_id} was marked "
            f"{rng.choice(('ready', 'held', 'repaired'))} on 2025-04-{rng.randint(1, 28):02d}."
            for asset_id in asset_ids[1:])
        _, latest_state = max(logs)
        prompt_question = f"What is the latest recorded status of asset {asset}?"
        answer = latest_state
        rationale = "Compare dated records for the same asset and ignore older entries and distractors."
    else:
        staff_ids = [f"ST-{value}" for value in rng.sample(range(100, 1000), 8)]
        people = [(f"Person-{index}", staff_ids[index]) for index in range(8)]
        unit_ids = [f"UNIT-{value}" for value in rng.sample(range(10, 100), 8)]
        units = list(zip(staff_ids, unit_ids))
        regions = [(unit, f"REGION-{chr(65 + index)}")
                   for index, (_, unit) in enumerate(units)]
        asset_ids = rng.sample(range(100, 1000), 8)
        codes = rng.sample(range(1000, 10000), 8)
        assets = [(unit, f"AS-{asset_id}", f"PIN-{code}")
                  for (_, unit), asset_id, code in zip(units, asset_ids, codes)]
        target_region = rng.choice([region for _, region in regions])
        target_units = [unit for unit, region in regions if region == target_region]
        target_assets = [asset for unit, asset, _ in assets if unit in target_units]
        target_asset = rng.choice(target_assets)
        answer = next(code for _, asset, code in assets if asset == target_asset)
        records = [f"People index: {person} belongs to staff ID {staff_id}."
                   for person, staff_id in people]
        records.extend(f"Unit directory: staff ID {staff_id} belongs to {unit}."
                       for staff_id, unit in units)
        records.extend(f"Regional directory: {unit} is located in {region}."
                       for unit, region in regions)
        records.extend(f"Asset register: {asset} belongs to {unit}; access code {code}."
                       for unit, asset, code in assets)
        prompt_question = (f"Using all four indexes, return the access code for an asset whose "
                           f"unit is in {target_region}.")
        rationale = "Join people, staff, unit, region, and asset records while ignoring distractors."
    rng.shuffle(records)
    rng.shuffle(noise)
    prompt = ("Use the archive records below. Treat unrelated notes as distractors and answer only "
              "with the requested value.\n\nRecords:\n" +
              "\n".join(records + noise) + f"\n\nQuestion: {prompt_question}")
    return {**common, "difficulty_rationale": rationale, "prompt": prompt,
            "expected_answer": answer, "evaluation_method": "exact_any",
            "acceptable_answers": [answer]}


def _test(category: str, i: int, rng: random.Random, difficulty: str | None = None) -> dict:
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
            "Easy": 3, "Medium": 4, "Hard": 5, "Uncalibrated": 6}[difficulty])
        constraints = [{"kind": "before", "first": items[0], "second": items[1]}]
        if difficulty in ("Medium", "Hard", "Uncalibrated"):
            medium_variant = rng.choice(("adjacency", "separation"))
            if medium_variant == "adjacency":
                constraints.append({"kind": "immediately_before",
                                    "first": items[2], "second": items[3]})
            else:
                constraints.append({"kind": "not_adjacent",
                                    "first": items[0], "second": items[3]})
        if difficulty == "Hard":
            if rng.choice((True, False)):
                constraints.extend([
                    {"kind": "immediately_before", "first": items[2], "second": items[3]},
                    {"kind": "not_adjacent", "first": items[0], "second": items[4]},
                ])
            else:
                constraints.extend([
                    {"kind": "immediately_before", "first": items[1], "second": items[2]},
                    {"kind": "fixed_position", "item": items[4], "position": len(items) - 1},
                    {"kind": "not_adjacent", "first": items[0], "second": items[4]},
                ])
        if difficulty == "Uncalibrated":
            variant = rng.choice(("contradictory_position", "satisfiable_position", "precedence_cycle"))
            if variant == "contradictory_position":
                constraints.append({"kind": "immediately_before",
                                    "first": items[2], "second": items[3]})
                constraints.append({"kind": "fixed_position", "item": items[2],
                                    "position": len(items) - 1})
            elif variant == "satisfiable_position":
                constraints.append({"kind": "fixed_position", "item": items[0],
                                    "position": 0})
            else:
                constraints.extend([
                    {"kind": "before", "first": items[1], "second": items[2]},
                    {"kind": "before", "first": items[2], "second": items[0]},
                ])
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
        if rng.choice((True, False)):
            test = _inventory_adjustment_test(common, difficulty, a, b)
            test["subcategory"] = "Inventory adjustments"
            return test
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
                "Trim surrounding whitespace from event IDs and SKUs; ignore IDs or SKUs that are empty after trimming.",
                "Ignore records with missing IDs or SKUs, non-integer deltas, or boolean deltas.",
                "A malformed record does not reserve its event_id; a valid zero-net SKU remains in the result.",
            ])
            checks.extend([
                "assert reconcile_stock_events([None, [], {}, {'event_id': '', 'sku': 'PEN', 'delta': 2}, {'event_id': 'x', 'sku': ' ', 'delta': 2}]) == {}",
                f"assert reconcile_stock_events([{{'event_id': ' same ', 'sku': '', 'delta': 9}}, "
                f"{{'event_id': 'same', 'sku': ' PEN ', 'delta': {a}}}, "
                f"{{'event_id': ' same ', 'sku': 'BOOK', 'delta': 99}}]) == "
                f"{{'PEN': {a}}}",
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
            "        event_id = event_id.strip()\n"
            "        sku = sku.strip()\n"
            "        if event_id in seen:\n"
            "            continue\n"
            "        seen.add(event_id)\n"
            "        totals[sku] = totals.get(sku, 0) + delta\n"
            "    return totals")
        example_events = [
            {"event_id": f"sample-{i}", "sku": "PEN", "delta": a},
            {"event_id": f"sample-{i + 1}", "sku": "PEN", "delta": -b},
        ]
        return {**common, "subcategory": "Inventory event reconciliation",
                "prompt": f"Implement `{function}(events)` for reconciling a small inventory ledger. "
                                     "Return a dictionary mapping each SKU to its net quantity change. "
                                     "Each record includes `event_id`, `sku`, and `delta`. "
                                     "Follow all rules exactly:\n" + rules_text +
                                     "\nExample records:\n" +
                                     json.dumps(example_events, ensure_ascii=False) +
                                     "\nOutput Python code only.",
                "expected_answer": "Python function that satisfies all inventory reconciliation rules",
                "evaluation_method": "code", "tests": checks, "demo_answer": source,
                "difficulty_rationale": rationale}
    if category == "Long Context":
        return _long_context_test(common, difficulty, rng)
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
            "Easy": (
                ("geography", "What is the capital of France?",
                 ("Paris", "The capital of France is Paris.")),
                ("astronomy", "Which planet is known as the Red Planet?",
                 ("Mars", "The Red Planet is Mars.")),
                ("biology", "What is the basic unit of life?",
                 ("cell", "The cell is the basic unit of life.")),
                ("mathematics", "How many sides does a triangle have?",
                 ("three", "A triangle has three sides.")),
            ),
            "Medium": (
                ("chemistry", "What is the chemical symbol for water?",
                 ("H2O", "Water's chemical formula is H2O.")),
                ("geography", "Which ocean lies between Africa and Australia?",
                 ("Indian Ocean", "The Indian Ocean lies between Africa and Australia.")),
                ("physics", "What is the SI unit of electric current?",
                 ("ampere", "Electric current is measured in amperes.")),
                ("biology", "Which part of a plant primarily absorbs water from soil?",
                 ("roots", "A plant's roots primarily absorb water from soil.")),
            ),
            "Hard": (
                ("biology", "Which gas do plants absorb from the atmosphere during photosynthesis?",
                 ("carbon dioxide", "Plants absorb carbon dioxide during photosynthesis.")),
                ("physics", "Which force keeps planets in orbit around the Sun?",
                 ("gravity", "Gravity keeps planets in orbit around the Sun.")),
                ("astronomy", "What is the name of the galaxy that contains our Solar System?",
                 ("Milky Way", "Our Solar System is in the Milky Way galaxy.")),
                ("chemistry", "What is the pH of a neutral aqueous solution at room temperature?",
                 ("7", "A neutral aqueous solution has a pH of 7 at room temperature.")),
            ),
            "Uncalibrated": (
                ("chemistry", "What is the name of the process by which a liquid changes into a gas?",
                 ("evaporation", "The process is called evaporation.")),
                ("biology", "What is the term for an animal that is active at night?",
                 ("nocturnal", "A nocturnal animal is active at night.")),
                ("geography", "What imaginary line divides Earth into the Northern and Southern Hemispheres?",
                 ("equator", "The Equator divides Earth into the Northern and Southern Hemispheres.")),
                ("physics", "What is the approximate speed of light in a vacuum, in kilometers per second?",
                 ("299792", "Light travels at approximately 299,792 kilometers per second in a vacuum.")),
            ),
        }
        domain, question, acceptable = rng.choice(knowledge_by_difficulty[difficulty])
        return {**common, "subcategory": f"Knowledge: {domain}",
                "prompt": question + " Answer concisely.",
                "difficulty_rationale": "Short factual recall from a seeded question set; this "
                                        "category is not independently calibrated by difficulty.",
                "expected_answer": acceptable[0], "evaluation_method": "exact_any",
                "acceptable_answers": list(acceptable)}
    if category == "Hallucination Resistance":
        place = rng.choice(("Zorbland", "Quendor", "Nimbaria", "Veloria"))
        prompt = (f"What is the capital of the fictional country {place}? State that it is unknown "
                  "and do not suggest a real capital.")
        acceptable = [
            f"{place} is fictional, so its capital is unknown.",
            f"The capital of {place} is unknown because {place} is fictional.",
        ]
        if difficulty in ("Hard", "Uncalibrated"):
            prompt += " Explicitly say that a real capital cannot be verified."
            acceptable = [
                f"I cannot verify a real capital for {place} because it is fictional.",
                f"{place} is fictional, so no real capital can be verified.",
            ]
        if difficulty == "Uncalibrated":
            prompt += " Include the word 'verify'."
            acceptable = [
                f"I cannot verify a real capital for {place} because it is fictional.",
                f"{place} is fictional, so I cannot verify a real capital.",
            ]
        return {**common, "prompt": prompt,
                "difficulty_rationale": "Recognize an unanswerable fictional-entity premise; "
                                        "harder tiers also require explicit verification language. "
                                        "Only listed concise, non-hallucinated answers are accepted.",
                "expected_answer": acceptable[0], "evaluation_method": "exact_any",
                "acceptable_answers": acceptable}
    if category == "Multilingual":
        phrases = {
            "Easy": (("thank you", ("gracias", "muchas gracias")),
                     ("please", ("por favor", "si es tan amable"))),
            "Medium": (("good morning", ("buenos días", "muy buenos días")),
                       ("good night", ("buenas noches", "que pases buenas noches"))),
            "Hard": (("Where is the train station?",
                      ("¿Dónde está la estación de tren?", "¿Dónde queda la estación de tren?")),
                     ("I would like a glass of water.",
                      ("Quisiera un vaso de agua.", "Me gustaría un vaso de agua."))),
            "Uncalibrated": (("Although it was raining, we continued walking.",
                              ("Aunque llovía, seguimos caminando.",
                               "Aunque estaba lloviendo, continuamos caminando.")),
                             ("If I had known, I would have called you.",
                              ("Si lo hubiera sabido, te habría llamado.",
                               "De haberlo sabido, te habría llamado."))),
        }
        phrase, acceptable = rng.choice(phrases[difficulty])
        return {**common, "prompt": f"Translate '{phrase}' into Spanish. Return only the translation.",
                "difficulty_rationale": "Translate a seeded phrase; higher tiers use longer "
                                        "sentences, clauses, or conditional grammar. Multiple "
                                        "listed target-language variants are accepted.",
                "expected_answer": acceptable[0], "evaluation_method": "exact_any",
                "acceptable_answers": list(acceptable)}
    if category == "Instruction Following":
        count = {"Easy": 3, "Medium": 4, "Hard": 5, "Uncalibrated": 6}[difficulty]
        term = rng.choice(("rain", "wind", "sunlight", "snow"))
        required_terms = [term] if difficulty != "Uncalibrated" else [term, "calm"]
        topic_terms = {
            "rain": ("rain", "wet", "drops", "weather", "water"),
            "wind": ("wind", "breeze", "air", "gust", "weather"),
            "sunlight": ("sunlight", "sun", "bright", "light", "warm"),
            "snow": ("snow", "cold", "flakes", "winter", "white"),
        }[term]
        return {**common, "prompt": f"Describe {term} in exactly {count} words." +
                                     (" Include the word 'calm'." if difficulty == "Uncalibrated" else "") +
                                     " Make the description relevant, not just a list of repeated words.",
                "difficulty_rationale": f"Meet an exact {count}-word constraint" +
                                        (" plus an additional required term and topic relevance."
                                         if difficulty == "Uncalibrated" else "."),
                "expected_answer": f"{count} words and required terms",
                "evaluation_method": "rubric", "exact_words": count,
                "required_terms": required_terms, "topic_terms": list(topic_terms),
                "minimum_topic_terms": 2}
    if category == "Summarization":
        report_time, diagnosis_time, recovery_time, next_update = rng.choice((
            ("09:10", "09:35", "09:50", "10:30"),
            ("13:05", "13:24", "13:41", "14:00"),
            ("16:12", "16:30", "16:48", "17:15"),
        ))
        affected = rng.randint(5, 19)
        monitored = affected + rng.randint(12, 60)
        scenario = rng.choice((
            {
                "service": "file uploads", "working": "downloads", "affected_unit": "customers",
                "impact": "some customers could not upload files; downloads continued to work",
                "impact_summary": "had upload failures",
                "hypothesis": "a network outage", "cause": "the upload service certificate had expired",
                "fix": "renewed the certificate", "recovered": "uploads",
                "unknown": "whether any files were lost",
                "cause_variants": ("the upload service certificate expired",
                                   "the certificate on the upload service expired",
                                   "an expired certificate disrupted the upload service"),
                "unknown_variants": ("unknown; engineers are still investigating",
                                     "it is not yet known whether files were lost",
                                     "whether any files were lost remains under investigation"),
            },
            {
                "service": "checkout", "working": "product browsing",
                "affected_unit": "checkout sessions",
                "impact": "some customers could not complete checkout; product browsing continued to work",
                "impact_summary": "failed",
                "hypothesis": "database overload", "cause": "the checkout signing key had expired",
                "fix": "rotated the signing key", "recovered": "checkout",
                "unknown": "whether any orders were duplicated",
                "cause_variants": ("the checkout signing key expired",
                                   "the signing key used by checkout had expired",
                                   "an expired signing key disrupted checkout"),
                "unknown_variants": ("unknown; engineers are still investigating",
                                     "it is not yet known whether orders were duplicated",
                                     "whether any duplicate orders occurred remains under investigation"),
            },
            {
                "service": "notifications", "working": "the core application",
                "affected_unit": "notification deliveries",
                "impact": "some notifications were delayed; the core application remained available",
                "impact_summary": "were delayed",
                "hypothesis": "an external provider outage",
                "cause": "the notification retry queue had been misconfigured",
                "fix": "corrected the retry queue configuration", "recovered": "notifications",
                "unknown": "whether any notifications were permanently lost",
                "cause_variants": ("the notification retry queue was misconfigured",
                                   "a misconfiguration affected the notification retry queue",
                                   "incorrect retry queue settings delayed notifications"),
                "unknown_variants": ("unknown; engineers are still investigating",
                                     "it is not yet known whether notifications were lost",
                                     "whether any notifications were permanently lost remains under investigation"),
            },
        ))
        impact_variants = (
            f"{affected} of {monitored} {scenario['affected_unit']} {scenario['impact_summary']}",
            f"{scenario['impact_summary'].capitalize()} for {affected} of {monitored} "
            f"monitored {scenario['affected_unit']}",
        )
        source_text = (
            f"At {report_time}, the service desk received reports that {scenario['impact']}. "
            f"The first notice suggested {scenario['hypothesis']}, but that was only an early guess. "
            f"At {diagnosis_time}, engineers confirmed that {scenario['cause']}. The team "
            f"{scenario['fix']}, and {scenario['recovered']} were working again by {recovery_time}. "
            f"The incident affected {affected} of {monitored} monitored {scenario['affected_unit']}. "
            f"Engineers have not yet established {scenario['unknown']}. "
            f"The next update is due at {next_update}.")
        fields = {
            "Easy": {
                "impact": list(impact_variants),
                "recovery": [f"{scenario['recovered']} were restored by {recovery_time}",
                             f"{scenario['recovered']} resumed by {recovery_time}",
                             f"{scenario['recovered']} were back by {recovery_time}"],
            },
            "Medium": {
                "impact": list(impact_variants),
                "cause": list(scenario["cause_variants"]),
                "recovery": [f"{scenario['recovered']} were restored by {recovery_time}",
                             f"{scenario['recovered']} resumed by {recovery_time}",
                             f"{scenario['recovered']} were back by {recovery_time}"],
            },
            "Hard": {
                "impact": [f"{impact_variants[0]}; {scenario['working']} remained available",
                           f"{impact_variants[1]}; {scenario['working']} continued working"],
                "cause": list(scenario["cause_variants"]),
                "recovery": [f"{scenario['recovered']} were restored by {recovery_time}",
                             f"{scenario['recovered']} resumed by {recovery_time}",
                             f"{scenario['recovered']} were back by {recovery_time}"],
                "unresolved": list(scenario["unknown_variants"]),
            },
            "Uncalibrated": {
                "impact": [f"{impact_variants[0]}; {scenario['working']} remained available",
                           f"{impact_variants[1]}; {scenario['working']} continued working"],
                "cause": list(scenario["cause_variants"]),
                "recovery": [f"{scenario['recovered']} were restored by {recovery_time}",
                             f"{scenario['recovered']} resumed by {recovery_time}",
                             f"{scenario['recovered']} were back by {recovery_time}"],
                "unresolved": list(scenario["unknown_variants"]),
                "early_claim": [f"{scenario['hypothesis']} was an unconfirmed initial guess",
                                f"{scenario['hypothesis']} was only an initial unconfirmed hypothesis",
                                f"the initial {scenario['hypothesis']} theory was not confirmed"],
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
        topic_terms = ("test", "testing", "careful", "checks", "quality", "reliable",
                       "bugs", "trust", "results")
        return {**common, "prompt": f"Write a {count}-word slogan about careful testing and include "
                                    f"{'the word' if len(terms) == 1 else 'the words'} "
                                    f"{', '.join(repr(term) for term in terms)}. Make it relevant "
                                    "and avoid repeating words just to meet the count.",
                "difficulty_rationale": f"Create a slogan under an exact {count}-word constraint" +
                                        (" and satisfy multiple keyword constraints."
                                         if len(terms) > 1 else "."),
                "expected_answer": f"{count} words and required terms",
                "evaluation_method": "rubric", "exact_words": count,
                "required_terms": terms, "topic_terms": list(topic_terms),
                "minimum_topic_terms": 2}
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
                "prompt": f"Is {expression} equal to {proposed}? Return only yes or no.",
                "expected_answer": expected, "evaluation_method": "choice"}
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
                "evaluation_method": "ordered_contains", "required_terms": markers}
    return {**common, "prompt": "State the answer clearly.", "expected_answer": "ok",
            "evaluation_method": "contains_all", "required_terms": ["ok"]}


def make_tests(profile: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    count = COUNTS.get(profile.upper(), COUNTS["STANDARD"])
    tests, seen, fingerprints = [], set(), set()
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
        if fingerprint in fingerprints:
            raise ValueError("duplicate generated test fingerprint")
        fingerprints.add(fingerprint)
        test["item_fingerprint"] = fingerprint
        category_key = test["category"].lower().replace(" ", "-")
        test["item_instance_id"] = f"{category_key}@{fingerprint}"
        tests.append(test)
    return tests


def make_context_tests(seed: int, max_context: int = 131072,
                       output_budget: int = 0) -> list[dict]:
    rng = random.Random(seed)
    tests = []
    marker_positions = ("beginning", "middle", "end")
    filler_unit = " neutral distractor information."
    for index, size in enumerate((4096, 8192, 16384, 32768, 65536, 131072, 262144)):
        if size > max_context or size <= max(0, output_budget) + 32:
            continue
        marker = f"CTX-{rng.randint(1000, 9999)}"
        position = marker_positions[index % len(marker_positions)]
        fixed_text = f"{marker}\n\nReturn the marker."
        prompt_budget = max(1, size - max(0, output_budget))
        filler_count = max(0, (prompt_budget * 4 - len(fixed_text)) // len(filler_unit))
        filler = filler_unit * filler_count
        prompt = f"{marker}\n{filler}\nReturn the marker." if position == "beginning" else (
            f"{filler[:len(filler)//2]}{marker}{filler[len(filler)//2:]}\nReturn the marker."
            if position == "middle" else f"{filler}\n{marker}\nReturn the marker.")
        tests.append({"id": f"context-{size}-{position}", "category": "Context Scaling",
                      "context_size": size, "context_position": position, "prompt": prompt,
                      "estimated_prompt_tokens": max(1, len(prompt) // 4),
                      "expected_answer": marker, "evaluation_method": "contains_all",
                      "required_terms": [marker], "max_score": 10})
    return tests


class BenchmarkManager:
    _state_key_lock = threading.Lock()

    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "current.json"; self.running = False
        self._run_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._live_requests = {}
        self.cancel_requested = threading.Event()
        self._status = {
            "state": "idle", "phase": "Ready to start", "activity": "",
            "progress": 0, "total": 0, "passed": 0, "failed": 0,
            "result_directory": self.root.name,
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
        for filename in self._result_names():
            try:
                raw = self._read_result_file(filename)
                if raw is None:
                    continue
                data = json.loads(raw)
            except (OSError, json.JSONDecodeError):
                continue
            model = data.get("model") if isinstance(data, dict) else {}
            if not isinstance(data, dict):
                continue
            results.append({"id": Path(filename).stem,
                            "model": model.get("name") if isinstance(model, dict) else None,
                            "timestamp": data.get("timestamp"),
                            "demo": bool(data.get("demo"))})
        return results

    def load_result(self, result_id):
        if (not isinstance(result_id, str) or
                re.fullmatch(r"[A-Za-z0-9_-]{1,100}", result_id) is None or
                result_id == "current"):
            return None
        for filename in self._result_names():
            if Path(filename).name == f"{result_id}.json":
                try:
                    raw = self._read_result_file(filename)
                    if raw is None:
                        return None
                    data = json.loads(raw)
                except (OSError, json.JSONDecodeError):
                    return None
                return data if isinstance(data, dict) else None
        return None

    def _result_names(self):
        with os.scandir(self.root) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False) and entry.name.endswith(".json"):
                    yield entry.name
                elif (entry.is_dir(follow_symlinks=False) and
                      re.fullmatch(r"[A-Za-z0-9_-]{1,100}", entry.name)):
                    try:
                        with os.scandir(entry.path) as children:
                            for child in children:
                                if child.is_file(follow_symlinks=False) and child.name.endswith(".json"):
                                    yield f"{entry.name}/{child.name}"
                    except OSError:
                        continue

    @staticmethod
    def _relative_parts(filename):
        if not isinstance(filename, str):
            return None
        parts = filename.split("/")
        if (len(parts) not in (1, 2) or
                (len(parts) == 2 and not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", parts[0]))):
            return None
        return parts

    def _read_result_file(self, filename):
        parts = self._relative_parts(filename)
        if (not parts or re.fullmatch(r"[A-Za-z0-9_-]{1,100}\.json", parts[-1]) is None
                or parts[-1] == "current.json"):
            return None
        try:
            raw = self._read_safe_file(parts)
            return raw.decode("utf-8") if raw is not None else None
        except (OSError, UnicodeDecodeError):
            return None

    def _read_safe_file(self, parts):
        root = self.root.resolve()
        if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
            directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                if len(parts) == 2:
                    child_fd = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                       dir_fd=directory_fd)
                    os.close(directory_fd)
                    directory_fd = child_fd
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW |
                             getattr(os, "O_NONBLOCK", 0), dir_fd=directory_fd)
            finally:
                os.close(directory_fd)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    return None
                with os.fdopen(fd, "rb") as stream:
                    fd = -1
                    return stream.read()
            finally:
                if fd >= 0:
                    os.close(fd)
        path = root.joinpath(*parts)
        if (any(root.joinpath(*parts[:index]).is_symlink() for index in range(1, len(parts)+1))
                or not path.resolve().is_relative_to(root) or not path.is_file()):
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def read_artifact(self, filename):
        parts = self._relative_parts(filename)
        if (not parts or
                re.fullmatch(
                    r"benchmark_[0-9]{8}_[0-9]{6}_[a-f0-9]{32}"
                    r"(?:\.json|\.csv|\.pdf|\.html|_quality\.svg)", parts[-1]) is None):
            return None
        try:
            return self._read_safe_file(parts)
        except OSError:
            return None

    def compare(self, result_ids):
        rows = []
        for result_id in result_ids:
            result = self.load_result(result_id)
            if result:
                rows.append({"id": result["id"], "model": result["model"]["name"],
                             "quality": result["scores"]["quality"],
                             "performance": result["scores"]["performance"],
                             "overall": result["scores"]["overall"],
                             "demo": bool(result.get("demo")),
                             "task_set_version": result.get("task_set_version"),
                             "evaluator_version": result.get("evaluator_version"),
                             "profile": (result.get("configuration") or {}).get("profile"),
                             "generation_options": (result.get("configuration") or {}).get(
                                 "generation_options"),
                             "repeat_runs": (result.get("configuration") or {}).get("repeat_runs", 1)})
        comparable = [row for row in rows if row["overall"] is not None]
        warnings = []
        for field, label in (
                ("task_set_version", "task set"), ("evaluator_version", "evaluator"),
                ("profile", "profile"), ("generation_options", "generation settings"),
                ("repeat_runs", "repeat count"), ("demo", "demo/real run type")):
            values = {json.dumps(row.get(field), sort_keys=True) for row in rows}
            if len(values) > 1:
                warnings.append(f"Selected results differ in {label}; scores may not be directly comparable.")
        if any(row["quality"] is None or row["performance"] is None or row["overall"] is None
               for row in rows):
            warnings.append("One or more selected results have unavailable score measurements.")
        return {"results": rows,
                "comparison_warnings": warnings,
                "best_overall": max(comparable, key=lambda x: x["overall"])["id"]
                if comparable else None}

    def _state_key(self, create=False):
        key_path = self.root / ".checkpoint.key"
        with self._state_key_lock:
            if key_path.is_symlink():
                return None
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(key_path, flags)
            except FileNotFoundError:
                if not create:
                    return None
                try:
                    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    fd = os.open(key_path, flags)
                else:
                    key = secrets.token_bytes(32)
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(key)
                        stream.flush()
                        os.fsync(stream.fileno())
                    return key
            try:
                key_stat = os.fstat(fd)
                if not stat.S_ISREG(key_stat.st_mode) or key_stat.st_size != 32:
                    return None
                with os.fdopen(fd, "rb") as stream:
                    fd = -1
                    key = stream.read(33)
                    return key if len(key) == 32 else None
            finally:
                if fd >= 0:
                    os.close(fd)

    def _checkpoint_signature(self, state, key):
        encoded = json.dumps(state, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hmac.new(key, encoded, hashlib.sha256).hexdigest()

    def _read_checkpoint_file(self) -> object:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.state_file, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_size > MAX_CHECKPOINT_SIZE:
                raise ValueError("checkpoint file is invalid or too large")
            with os.fdopen(fd, "rb") as stream:
                fd = -1
                content = stream.read(MAX_CHECKPOINT_SIZE + 1)
            if len(content) > MAX_CHECKPOINT_SIZE:
                raise ValueError("checkpoint file is too large")
            return json.loads(content.decode("utf-8"))
        finally:
            if fd >= 0:
                os.close(fd)

    @staticmethod
    def _valid_checkpoint_outcome(item: dict) -> bool:
        if not isinstance(item, dict):
            return False
        status, passed, score = item.get("status"), item.get("passed"), item.get("score")
        if status == "NOT RUN":
            return passed is None and score is None
        if status not in ("PASS", "FAILED") or passed is not (status == "PASS"):
            return False
        return (isinstance(score, (int, float)) and not isinstance(score, bool) and
                math.isfinite(score) and 0 <= score <= 10)

    def _validate_checkpoint(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("checkpoint must be an object")
        if (state.get("task_set_version") != TASK_SET_VERSION or
                state.get("evaluator_version") != EVALUATOR_VERSION):
            raise ValueError("checkpoint task or evaluator version changed")
        profile, seed = state.get("profile"), state.get("seed")
        if not isinstance(profile, str) or profile.upper() not in COUNTS:
            raise ValueError("checkpoint profile is invalid")
        if (not isinstance(seed, int) or isinstance(seed, bool) or
                not 0 <= seed <= 2_147_483_647):
            raise ValueError("checkpoint seed is invalid")
        if state.get("resume_phase") not in (
                "warmup", "primary", "fresh_context", "accumulated_context", "performance"):
            raise ValueError("checkpoint phase is invalid")
        if not isinstance(state.get("model"), dict):
            raise ValueError("checkpoint model is invalid")
        model = resolve_model(state["model"])
        options = state.get("generation_options")
        if (not isinstance(options, dict) or
                normalize_generation_options(options) != options):
            raise ValueError("checkpoint generation settings are invalid")
        repeats = state.get("repeat_runs")
        if not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= 5:
            raise ValueError("checkpoint repeat count is invalid")

        tests = make_tests(profile, seed)
        if len({test["item_fingerprint"] for test in tests}) != len(tests):
            raise ValueError("generated test fingerprints are not unique")
        saved_tests = state.get("tests")
        if (not isinstance(saved_tests, list) or
                [(test.get("id"), test.get("item_fingerprint")) for test in saved_tests
                 if isinstance(test, dict)] !=
                [(test["id"], test["item_fingerprint"]) for test in tests] or
                len(saved_tests) != len(tests)):
            raise ValueError("checkpoint tests do not match the generated suite")

        generated = {test["id"]: test for test in tests}
        completed = state.get("completed_tests")
        if not isinstance(completed, list) or len(completed) > len(tests):
            raise ValueError("checkpoint completed tests are invalid")
        completed_ids = set()
        for item in completed:
            if (not isinstance(item, dict) or item.get("id") not in generated or
                    item.get("id") in completed_ids or
                    item.get("item_fingerprint") != generated[item["id"]]["item_fingerprint"] or
                    not self._valid_checkpoint_outcome(item)):
                raise ValueError("checkpoint contains an invalid completed test")
            completed_ids.add(item["id"])
            replicates = item.get("replicates")
            if repeats == 1:
                if replicates:
                    raise ValueError("checkpoint has unexpected repeats")
            elif (not isinstance(replicates, list) or len(replicates) != repeats or
                  any(not self._valid_checkpoint_outcome(entry) for entry in replicates)):
                raise ValueError("checkpoint repeats are invalid")

        advertised = model.get("context_length", 131072)
        try:
            max_context = min(int(advertised), 262144)
        except (TypeError, ValueError, OverflowError):
            max_context = 131072
        if model.get("demo"):
            max_context = min(max_context, 32768)
        context_tests = {test["id"]: test for test in make_context_tests(
            seed, max_context, output_budget=options["max_tokens"] + 128)}
        for field in ("context_results", "accumulated_results"):
            records = state.get(field)
            if not isinstance(records, list) or len(records) > len(context_tests):
                raise ValueError(f"checkpoint {field} are invalid")
            seen = set()
            for item in records:
                expected = context_tests.get(item.get("id")) if isinstance(item, dict) else None
                if (expected is None or item["id"] in seen or
                        item.get("prompt") != expected["prompt"] or
                        item.get("expected_answer") != expected["expected_answer"] or
                        item.get("context_size") != expected["context_size"] or
                        not self._valid_checkpoint_outcome(item)):
                    raise ValueError(f"checkpoint {field} contain an invalid result")
                seen.add(item["id"])
        for field, limit in (("failures", len(tests)), ("not_run", len(tests))):
            if not isinstance(state.get(field), list) or len(state[field]) > limit:
                raise ValueError(f"checkpoint {field} are invalid")
        if not isinstance(state.get("warmup_errors"), list):
            raise ValueError("checkpoint warmup_errors are invalid")

    def resume(self):
        with self._run_lock:
            if self.running or not self.state_file.exists():
                return False
            try:
                state = self._read_checkpoint_file()
                if not isinstance(state, dict):
                    raise ValueError("checkpoint must be an object")
                signature = state.pop("_integrity", None)
                key = self._state_key()
                if (not isinstance(signature, str) or key is None or
                        not hmac.compare_digest(signature, self._checkpoint_signature(state, key))):
                    raise ValueError("checkpoint integrity check failed")
                self._validate_checkpoint(state)
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                self._update_status(
                    phase="Saved run rejected",
                    activity="The saved checkpoint is invalid or has been modified; start a new run.")
                return False
            return self._start_locked(state)

    def start(self, payload):
        with self._run_lock:
            return self._start_locked(payload)

    def _start_locked(self, payload):
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
        except Exception:
            self._update_status(state="error",
                                error="The benchmark failed; internal details were not exposed.")
            return False
        finally:
            with self._run_lock:
                self.running = False

    def cancel(self):
        with self._run_lock:
            if (not self.running or self._status.get("state") != "running" or
                    self._status.get("phase") == "Saving result files"):
                return False
            self.cancel_requested.set()
            return True

    def _answer(self, model: dict, test: dict, history: list | None = None,
                generation_options: dict | None = None) -> str | dict:
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
                target_topic_words = max(0, count - 1) if test.get("minimum_topic_terms") else count
                for term in test.get("topic_terms", []):
                    if len(words) >= target_topic_words:
                        break
                    if term not in words:
                        words.append(term)
                if test.get("minimum_topic_terms") and len(words) < count:
                    words.append("builds")
                words.extend(["careful", "reliable", "local", "testing"][:count - len(words)])
                return " ".join(words)
            if test["evaluation_method"] == "contains_all":
                return " ".join(map(str, test.get("required_terms", [])))
            return test["expected_answer"]
        from .discovery import ask_model
        request_id = uuid.uuid4().hex

        def progress(metrics):
            self._publish_live_metrics(request_id, metrics)

        progress({"phase": "prefill", "token_count_is_estimated": True,
                  "timing_source": "awaiting backend"})
        try:
            return ask_model(model, test["prompt"], history=history,
                             options=generation_options, progress_callback=progress)
        finally:
            with self._status_lock:
                self._live_requests.pop(request_id, None)
                if "live_metrics" in self._status:
                    self._status["live_metrics"]["active_requests"] = len(self._live_requests)
                    if not self._live_requests:
                        self._status["live_metrics"]["phase"] = "complete"

    def _publish_live_metrics(self, request_id, metrics):
        if not isinstance(metrics, dict):
            return
        normalized = {
            "phase": metrics.get("phase", "prefill"),
            "token_count_is_estimated": metrics.get("token_count_is_estimated", True) is not False,
            "timing_source": metrics.get("timing_source", "unknown"),
        }
        for key in ("prefill_tokens_per_second", "decode_tokens_per_second"):
            value = metrics.get(key)
            normalized[key] = (value if isinstance(value, (int, float)) and
                               not isinstance(value, bool) and math.isfinite(value) and value >= 0
                               else None)
        with self._status_lock:
            self._live_requests[request_id] = normalized
            active = list(self._live_requests.values())
            aggregate = dict(normalized)
            aggregate["active_requests"] = len(active)
            aggregate["phase"] = ("decode" if any(item["phase"] == "decode" for item in active)
                                  else normalized["phase"])
            for key in ("prefill_tokens_per_second", "decode_tokens_per_second"):
                rates = [item[key] for item in active if item[key] is not None]
                aggregate[key] = sum(rates) if rates else None
            aggregate["token_count_is_estimated"] = any(
                item["token_count_is_estimated"] for item in active)
            sources = {item["timing_source"] for item in active}
            aggregate["timing_source"] = next(iter(sources)) if len(sources) == 1 else "mixed"
            self._status["live_metrics"] = aggregate

    @staticmethod
    def _measurement_flags(metrics):
        return {
            "timing_is_estimated": metrics.get("timing_is_estimated", True) is not False,
            "token_count_is_estimated": metrics.get("token_count_is_estimated", True) is not False,
            "timing_source": metrics.get("timing_source", "estimated"),
            **{key: metrics[key] for key in (
                "prefill_tokens_per_second", "decode_tokens_per_second",
                "prompt_token_count_is_estimated", "output_token_count_is_estimated",
                "finish_reason") if key in metrics},
        }

    def _execute(self, model: dict, test: dict, history: list | None = None,
                 generation_options: dict | None = None) -> dict:
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
        last_diagnostic = None
        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                answer = self._answer(model, test, history, generation_options)
                error = None
            except Exception as exc:
                answer, error = "", getattr(exc, "code", self._failure_code(exc))
                diagnostic = getattr(
                    exc, "diagnostic", "The model request failed before a response could be evaluated.")
                response_metrics = {}
                evaluation = {"passed": None, "status": "NOT RUN", "criteria": criteria,
                              "explanation": f"No answer was evaluated because the model request failed: {diagnostic}"}
                attempt_status = "ERROR"
            else:
                response_metrics = answer if isinstance(answer, dict) else {}
                answer = response_metrics.get("text", answer)
                evaluation = evaluate_with_details(test, answer)
                attempt_status = evaluation["status"]
                last_answer, last_evaluation, last_metrics = answer, evaluation, response_metrics
                truncated = response_metrics.get("finish_reason") in ("length", "max_tokens")
                diagnostic = (
                    "The backend stopped at the output-token budget; the answer may be incomplete. "
                    "Increase max_tokens and rerun before interpreting the quality score."
                    if truncated else None)
                if diagnostic:
                    evaluation = {**evaluation, "explanation":
                                  f"{evaluation.get('explanation', '')} {diagnostic}"}
                    last_evaluation = evaluation
            latency = time.perf_counter() - started
            if error is None:
                last_latency = latency
            else:
                last_diagnostic = diagnostic
            attempt_answer = str(answer)[:4000]
            attempts.append({"attempt": attempt, "status": attempt_status,
                             "error": error, "answer": attempt_answer,
                             "diagnostic": diagnostic,
                             "answer_truncated": len(str(answer)) > 4000,
                             "evaluation": evaluation})
            if evaluation["passed"] is True:
                return {**test, "answer": answer, "evaluation": evaluation,
                        "passed": True, "score": evaluation.get("score", 10),
                        "latency_s": latency, "ttft_s": response_metrics.get("ttft_s", latency),
                        "generation_s": response_metrics.get("generation_s", latency),
                        **self._measurement_flags(response_metrics),
                        "prompt_tokens": response_metrics.get("prompt_tokens", test.get("estimated_prompt_tokens", max(1, len(test["prompt"]) // 4))),
                        "output_tokens": response_metrics.get("output_tokens", max(1, len(answer.split()))),
                        "error": None, "diagnostic": diagnostic,
                        "output_truncated": truncated,
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
                            "diagnostic": last_diagnostic,
                            "attempts": attempts, "status": "NOT RUN"}
                return {**test, "answer": last_answer, "evaluation": last_evaluation,
                        "passed": False, "score": last_evaluation.get("score", 0),
                        "latency_s": last_latency,
                        "ttft_s": last_metrics.get("ttft_s", last_latency),
                        "generation_s": last_metrics.get("generation_s", last_latency),
                        **self._measurement_flags(last_metrics),
                        "prompt_tokens": last_metrics.get(
                            "prompt_tokens", test.get("estimated_prompt_tokens",
                                                      max(1, len(test["prompt"]) // 4))),
                        "output_tokens": last_metrics.get(
                            "output_tokens", max(1, len(last_answer.split()))),
                        "error": "EVALUATION_FAILED", "context_reset": history is None,
                        "diagnostic": next((entry["diagnostic"] for entry in reversed(attempts)
                                            if entry["error"] is None), None),
                        "output_truncated": last_metrics.get("finish_reason") in ("length", "max_tokens"),
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

    @staticmethod
    def _aggregate_repeats(
            test: dict, repetitions: list[dict], generation_options: dict,
            repeat_runs: int) -> dict:
        measured = [
            entry for entry in repetitions
            if entry["status"] != "NOT RUN" and entry.get("score") is not None
        ]
        representative = measured[0] if measured else repetitions[0]
        replicates = [
            {"seed": (generation_options["seed"] + index) % 2147483648,
             "score": entry["score"], "passed": entry["passed"],
             "status": entry["status"], "latency_s": entry["latency_s"],
             "answer": entry["answer"], "evaluation": entry["evaluation"],
             "error": entry.get("error"), "diagnostic": entry.get("diagnostic"),
             "attempts": entry["attempts"]}
            for index, entry in enumerate(repetitions)
        ]
        missing = repeat_runs - len(measured)
        passed = all(entry["passed"] for entry in measured) if measured else None
        item = {
            **representative,
            "score": round(statistics.mean(entry["score"] for entry in measured), 2)
            if measured else None,
            "passed": passed,
            "status": ("PASS" if passed is True else
                       "FAILED" if passed is False else "NOT RUN"),
            "latency_s": round(statistics.mean(entry["latency_s"] for entry in measured), 4)
            if measured else 0,
            "ttft_s": round(statistics.mean(entry["ttft_s"] for entry in measured), 4)
            if measured else 0,
            "generation_s": round(statistics.mean(
                entry["generation_s"] for entry in measured), 4)
            if measured else 0,
            "prompt_tokens": round(statistics.mean(
                entry["prompt_tokens"] for entry in measured))
            if measured else max(1, len(test["prompt"]) // 4),
            "output_tokens": round(statistics.mean(
                entry["output_tokens"] for entry in measured))
            if measured else 0,
            "replicates": replicates,
            "missing_repeat_count": missing,
        }
        if measured:
            item["evaluation"] = {
                "passed": passed, "status": item["status"],
                "criteria": representative["evaluation"]["criteria"],
                "explanation": (
                    f"{sum(entry['passed'] is True for entry in measured)} of "
                    f"{len(measured)} measured repeats passed; {missing} repeat(s) were NOT RUN."),
            }
        return item

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
                                          "error": getattr(exc, "code", self._failure_code(exc)),
                                          "diagnostic": getattr(
                                              exc, "diagnostic",
                                              "The warm-up request failed before completion.")})
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
            item = (self._aggregate_repeats(
                test, repetitions, generation_options, repeat_runs)
                    if repeat_runs > 1 else repetitions[0])
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
        context_tests = make_context_tests(
            seed, max_context=max_context, output_budget=generation_options["max_tokens"] + 128)
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
        result = {"benchmark_version": "2.2.0", "task_set_version": TASK_SET_VERSION,
                  "evaluator_version": EVALUATOR_VERSION,
                  "quality_aggregation": "equal-weighted mean of available category scores",
                  "difficulty_calibration_status": "not empirically calibrated",
                  "timestamp": datetime.now(timezone.utc).isoformat(),
                  "seed": seed, "demo": bool(model.get("demo")),
                  "model": {**model_info(model), "id": model.get("id")},
                  "hardware": hardware_info(), "configuration": {"profile": profile, "warmup_runs": 3,
                  "measured_runs": 5, "retries": 2, "temperature": generation_options["temperature"],
                  "top_p": generation_options["top_p"], "generation_options": generation_options,
                  "repeat_runs": repeat_runs,
                  "repeat_seed_policy": "sampling seed increments by one per repeat" if repeat_runs > 1 else "single run",
                  "context_reset_per_test": True, "context_suite": context_sizes,
                  "context_output_token_reserve": generation_options["max_tokens"],
                  "context_chat_overhead_token_reserve": 128,
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
                            "Token counts and timings use backend measurements where available; "
                            "consult estimation flags and timing_source. Bootstrap resamples are not independent model runs.",
                            "Context prompt sizing uses a character-based token estimate with output and chat overhead reserved; "
                            "the backend tokenizer may differ."]}
        result["id"] = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"
        not_run_by_code = {}
        for item in completed:
            for outcome in item.get("replicates") or [item]:
                if outcome["status"] == "NOT RUN" and outcome.get("error"):
                    code = outcome["error"]
                    not_run_by_code[code] = not_run_by_code.get(code, 0) + 1
        result["summary"] = {
            "passed": sum(item["status"] == "PASS" for item in completed),
            "failed": sum(item["status"] == "FAILED" for item in completed),
            "not_run": sum(item["status"] == "NOT RUN" for item in completed),
            "repeat_outcomes": {
                "requested_per_test": repeat_runs,
                "scored": sum(
                    replicate.get("score") is not None
                    for item in completed for replicate in
                    (item.get("replicates") or [item])),
                "not_run": sum(
                    replicate.get("status") == "NOT RUN"
                    for item in completed for replicate in
                    (item.get("replicates") or [item])),
            },
            "not_run_by_code": dict(sorted(not_run_by_code.items())),
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
        with self._run_lock:
            if self.cancel_requested.is_set():
                save_checkpoint("performance")
                self._update_status(state="cancelled")
                return
            self._update_status(phase="Saving result files",
                                activity="Writing JSON, CSV, PDF, HTML, and SVG to the model results directory.",
                                total=1, progress=0)
        self._write_outputs(result); self.state_file.unlink(missing_ok=True)
        self._update_status(state="complete", phase="Benchmark complete",
                            activity="All result files are ready.", progress=1, total=1,
                            result_id=result["id"], result_directory=self.root.name,
                            quality=quality["overall"], performance=perf["score"],
                            overall=result["scores"]["overall"])

    def _scores(self, results: list[dict]) -> dict:
        categories = {}
        for category in CATEGORIES:
            values = [x["score"] * 10 for x in results
                      if x["category"] == category and x.get("score") is not None]
            categories[category] = round(statistics.mean(values), 2) if values else None
        measured = [value for value in categories.values() if value is not None]
        categories["overall"] = round(statistics.mean(measured), 2) if measured else None
        return {"categories": categories, "overall": categories["overall"],
                "method": "equal-weighted mean of available category scores"}

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
        signed_state = dict(state)
        signed_state.pop("_integrity", None)
        key = self._state_key(create=True)
        if key is None:
            raise OSError("could not load checkpoint signing key")
        signed_state["_integrity"] = self._checkpoint_signature(signed_state, key)
        encoded = json.dumps(signed_state, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_CHECKPOINT_SIZE:
            raise ValueError("checkpoint exceeds the configured size limit")
        fd, name = tempfile.mkstemp(prefix="benchmark-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(5):
                try:
                    os.replace(name, self.state_file)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.1 * (2 ** attempt))
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _performance(self, model, samples, generation_options=None, progress_callback=None):
        progress = 0
        def safe_answer(test):
            started = time.perf_counter()
            try:
                answer = self._answer(model, test, generation_options=generation_options)
            except Exception:
                return {"success": False, "latency_s": time.perf_counter() - started,
                        "output_tokens": 0}
            text = answer.get("text", "") if isinstance(answer, dict) else answer
            return {"success": True, "latency_s": time.perf_counter() - started,
                    "output_tokens": answer.get("output_tokens", max(1, len(str(text).split())))
                    if isinstance(answer, dict) else max(1, len(str(text).split())),
                    "token_count_is_estimated": (
                        answer.get("token_count_is_estimated", True) is not False
                        if isinstance(answer, dict) else True)}

        concurrency_results = []
        for concurrency in (1, 2, 4):
            if self.cancel_requested.is_set():
                return None
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                outcomes = list(pool.map(safe_answer, samples))
            elapsed = time.perf_counter() - started
            successful = [item for item in outcomes if item["success"]]
            estimated = not successful or any(item["token_count_is_estimated"] for item in successful)
            throughput = (round(sum(item["output_tokens"] for item in successful) / elapsed, 2)
                          if elapsed else None)
            concurrency_results.append({
                "workers": concurrency, "requests": len(outcomes),
                "successful_requests": len(successful),
                "failed_requests": len(outcomes) - len(successful),
                "elapsed_s": round(elapsed, 4),
                "requests_per_second": round(len(successful) / elapsed, 2) if elapsed else None,
                "output_tokens_per_second": throughput,
                "estimated_output_tokens_per_second": throughput if estimated else None,
                "token_count_is_estimated": estimated,
                "latency_mean_s": round(statistics.mean(
                    item["latency_s"] for item in successful), 4) if successful else None,
            })
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
        estimated = not evaluated or any(
            item.get("token_count_is_estimated", True) for item in evaluated)
        timing_sources = {item.get("timing_source", "unknown") for item in evaluated}
        prefill_rates = [item["prefill_tokens_per_second"] for item in evaluated
                         if isinstance(item.get("prefill_tokens_per_second"), (int, float))]
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
                "prefill_tokens_per_second": round(statistics.median(prefill_rates), 2)
                if prefill_rates else None,
                "decode_tokens_per_second": round(tokens_per_second, 2)
                if tokens_per_second is not None else None,
                "throughput_basis": ("estimated" if estimated else "backend-reported") +
                " output tokens / generation time (see timing provenance)",
                "timing_is_estimated": not evaluated or any(
                    item["timing_is_estimated"] for item in evaluated),
                "token_count_is_estimated": estimated,
                "timing_source": next(iter(timing_sources)) if len(timing_sources) == 1 else "mixed"
                if timing_sources else "unavailable",
                "warmup_runs": 3, "measured_runs": len(evaluated),
                "not_run_samples": not_run_count,
                "concurrency_levels": [1, 2, 4],
                "concurrency_results": concurrency_results}

    def _write_outputs(self, result):
        result_id = result["id"]
        if not isinstance(result_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", result_id):
            raise ValueError("invalid result identifier")
        model = result["model"]
        identity = json.dumps(
            [model.get("backend"), model.get("endpoint"), model.get("id", model.get("name"))],
            ensure_ascii=False, separators=(",", ":"))
        slug = re.sub(r"[^a-z0-9]+", "-", str(model.get("name") or model.get("id") or "model").lower())
        folder = f"{slug.strip('-')[:60] or 'model'}-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
        names = {kind: f"{result_id}.{kind}" for kind in ("json", "csv", "pdf", "html")}
        names["quality_chart"] = f"{result_id}_quality.svg"
        result["files"] = {kind: f"{folder}/{name}" for kind, name in names.items()}
        result["model_folder"] = folder
        root = self.root.resolve()
        if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                try:
                    os.mkdir(folder, dir_fd=root_fd)
                except FileExistsError:
                    pass
                directory_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                       dir_fd=root_fd)
            finally:
                os.close(root_fd)
        else:
            destination = root / folder
            destination.mkdir(exist_ok=True)
            if destination.is_symlink() or destination.resolve().parent != root:
                raise ValueError("unsafe model results directory")
            directory_fd = None
        try:
            with tempfile.TemporaryDirectory(prefix="export-", dir=root) as staging:
                self._write_output_files(result, Path(staging))
                # Publish JSON last so history only sees runs with all exports present.
                for kind in ("csv", "pdf", "html", "quality_chart", "json"):
                    if directory_fd is not None:
                        os.replace(Path(staging) / names[kind], names[kind],
                                   dst_dir_fd=directory_fd)
                    else:
                        os.replace(Path(staging) / names[kind], root / folder / names[kind])
        finally:
            if directory_fd is not None:
                os.close(directory_fd)

    def _write_output_files(self, result, result_dir):
        result_id = result["id"]
        destination = result_dir / f"{result_id}.json"
        fd, temporary = tempfile.mkstemp(prefix="result-", suffix=".tmp", dir=result_dir)
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
            fields = ["run_type", "id", "category", "subcategory", "difficulty", "difficulty_calibration_status",
                      "task_set_version", "evaluator_version", "item_instance_id",
                      "item_fingerprint", "difficulty_rationale", "prompt",
                      "answer", "evaluation_criteria", "evaluation_explanation",
                      "evaluation_checks",
                      "score", "status", "failure_code", "failure_diagnostic",
                      "attempt_count", "attempts", "repeats", "latency_s"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({
                "run_type": "DEMO DATA" if result.get("demo") else "MODEL DATA",
                "id": self._csv_cell(row.get("id")),
                "category": self._csv_cell(row.get("category")),
                "subcategory": self._csv_cell(row.get("subcategory")),
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
                "failure_code": row.get("error"),
                "failure_diagnostic": self._csv_cell(row.get("diagnostic")),
                "attempt_count": len(row.get("attempts", [])),
                "attempts": self._csv_cell(json.dumps(row.get("attempts", []), ensure_ascii=False)),
                "repeats": self._csv_cell(json.dumps(row.get("replicates", []), ensure_ascii=False)),
                "latency_s": row.get("latency_s"),
            } for row in result["tests"])
        write_pdf(result_dir / f"{result_id}.pdf", result)
        write_chart(result_dir / f"{result_id}_quality.svg", result["quality"]["categories"])
        write_html(result_dir / f"{result_id}.html", result)

    @staticmethod
    def _csv_cell(value):
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value
