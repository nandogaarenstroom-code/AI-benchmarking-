"""Domain benchmark generation, evaluation, performance measurement, and exports."""
from __future__ import annotations

import csv
import copy
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
from .evaluation import code_evaluator_available, evaluate
from .reporting import write_pdf, write_chart

CATEGORIES = ["Reasoning", "Mathematics", "Coding", "Instruction Following",
              "Structured Output", "Knowledge", "Hallucination Resistance",
              "Long Context", "Summarization", "Multilingual", "Creative Generation",
              "Consistency", "Context Retrieval"]
COUNTS = {"QUICK": 25, "STANDARD": 100, "FULL": 252, "PERFORMANCE": 25,
          "QUALITY": 100, "REASONING": 40, "CODING": 40, "LONG CONTEXT": 30}
DIFFICULTIES = ("Easy", "Medium", "Hard", "Expert")


def _test(category, i, rng):
    difficulty = DIFFICULTIES[i % 4]
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    common = {"id": f"{category.lower().replace(' ', '-')}-{i+1:03d}",
              "category": category, "subcategory": category, "difficulty": difficulty,
              "max_score": 10}
    if category == "Mathematics":
        return {**common, "prompt": f"Calculate {a} × {b}. Return only the number.",
                "expected_answer": str(a * b), "evaluation_method": "numeric", "tolerance": 0}
    if category == "Reasoning":
        color = rng.choice(("blue", "green", "amber", "violet"))
        item = f"item-{a}"
        return {**common, "prompt": f"Every dax is {color}. {item} is a dax. What color is {item}?",
                "expected_answer": color, "evaluation_method": "contains_all",
                "required_terms": [color]}
    if category == "Coding":
        function, description, source, checks = rng.choice((
            ("add", "return the sum of a and b", "def add(a, b):\n    return a + b",
             [f"assert add({a}, {b}) == {a + b}", "assert add(-1, 1) == 0"]),
            ("multiply", "return the product of a and b", "def multiply(a, b):\n    return a * b",
             [f"assert multiply({a}, {b}) == {a * b}", "assert multiply(-2, 3) == -6"]),
            ("square", "return n multiplied by itself", "def square(n):\n    return n * n",
             [f"assert square({a}) == {a * a}", "assert square(-3) == 9"]),
            ("absolute", "return the non-negative magnitude of n", "def absolute(n):\n    return abs(n)",
             [f"assert absolute(-{a}) == {a}", "assert absolute(0) == 0"]),
        ))
        return {**common, "prompt": f"Write Python defining {function} that will {description}. Output code only.",
                "expected_answer": "code", "evaluation_method": "code", "tests": checks,
                "demo_answer": source}
    if category == "Structured Output":
        keys = ("answer", "valid") if i % 2 == 0 else ("result", "count")
        types = {"answer": "string", "valid": "boolean"} if keys[0] == "answer" else {
            "result": "string", "count": "integer"}
        return {**common, "prompt": f'Return a JSON object with keys "{keys[0]}" and "{keys[1]}"; '
                                     f'"{keys[0]}" must be a string and "{keys[1]}" must be '
                                     f'a {"boolean" if keys[1] == "valid" else "integer"}.',
                "expected_answer": "json object", "evaluation_method": "json_schema",
                "required_keys": list(keys), "required_types": types}
    if category == "Knowledge":
        question, answer = rng.choice((
            ("What is the capital of France?", "Paris"),
            ("Which planet is known as the Red Planet?", "Mars"),
            ("What is the chemical symbol for water?", "H2O"),
            ("Which ocean lies between Africa and Australia?", "Indian Ocean"),
        ))
        return {**common, "prompt": question + " Answer concisely.",
                "expected_answer": answer, "evaluation_method": "contains_all",
                "required_terms": [answer]}
    if category == "Hallucination Resistance":
        place = rng.choice(("Zorbland", "Quendor", "Nimbaria", "Veloria"))
        return {**common, "prompt": f"What is the capital of the fictional country {place}? "
                                    "State that it is unknown.",
                "expected_answer": "unknown", "evaluation_method": "contains_all", "required_terms": ["unknown"]}
    if category == "Multilingual":
        phrase, translation = rng.choice((
            ("good morning", ["buenos", "días"]), ("thank you", ["gracias"]),
            ("good night", ["buenas", "noches"]), ("please", ["favor"]),
        ))
        return {**common, "prompt": f"Translate '{phrase}' into Spanish.",
                "expected_answer": " ".join(translation), "evaluation_method": "contains_all",
                "required_terms": translation}
    if category == "Instruction Following":
        count = 3 + i % 3
        term = rng.choice(("rain", "wind", "sunlight", "snow"))
        return {**common, "prompt": f"Describe {term} in exactly {count} words.",
                "expected_answer": f"{count} words", "evaluation_method": "rubric",
                "exact_words": count, "required_terms": [term]}
    if category == "Summarization":
        subject, benefit = rng.choice((
            ("Local backups", "keep files on the device"),
            ("Offline tools", "work without an internet connection"),
            ("Private notes", "stay stored on your computer"),
        ))
        required = rng.choice(("local", "private", "offline"))
        return {**common, "prompt": f"Summarize this in one sentence: {subject} {benefit}. "
                                    f"Include the word '{required}'.",
                "expected_answer": required, "evaluation_method": "contains_all",
                "required_terms": [required]}
    if category == "Creative Generation":
        return {**common, "prompt": "Write a four-word slogan about careful testing and include "
                                    "the word 'test'.",
                "expected_answer": "four words", "evaluation_method": "rubric",
                "exact_words": 4, "required_terms": ["test"]}
    if category == "Consistency":
        total = a + b
        proposed = total + rng.choice((-1, 0, 1))
        expected = "yes" if proposed == total else "no"
        return {**common, "prompt": f"Is {a} + {b} equal to {proposed}? Answer yes or no.",
                "expected_answer": expected, "evaluation_method": "contains_all",
                "required_terms": [expected]}
    if category == "Context Retrieval":
        return {**common, "prompt": f"Remember marker BLUE-{a}. What is the marker?",
                "expected_answer": f"BLUE-{a}", "evaluation_method": "contains_all",
                "required_terms": [f"BLUE-{a}"]}
    return {**common, "prompt": "State the answer clearly.", "expected_answer": "ok",
            "evaluation_method": "contains_all", "required_terms": ["ok"]}


def make_tests(profile, seed):
    rng = random.Random(seed)
    count = COUNTS.get(profile.upper(), COUNTS["STANDARD"])
    tests, seen = [], set()
    while len(tests) < count:
        index = len(tests)
        test = _test(CATEGORIES[index % len(CATEGORIES)], index + rng.randint(0, 10000), rng)
        if (test["category"], test["prompt"].lower()) in seen:
            # Parameterize otherwise stable task families so a large profile
            # never repeats the same prompt.
            test["prompt"] += f"\nUse independent test variant {len(tests) + 1}."
        fingerprint = (test["category"], test["prompt"].lower())
        seen.add(fingerprint)
        test["id"] = f"{test['category'].lower().replace(' ', '-')}-{len(tests)+1:03d}"
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
        return {"results": rows, "best_overall": max(rows, key=lambda x: x["overall"])["id"] if rows else None}

    def resume(self):
        if not self.state_file.exists(): return False
        try:
            state = json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError):
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
            return test["expected_answer"]
        from .discovery import ask_model
        response = ask_model(model, test["prompt"], history=history, options=generation_options)
        return response

    def _execute(self, model, test, history=None, generation_options=None):
        attempts = []
        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                answer = self._answer(model, test, history, generation_options)
                error = None
            except Exception as exc:
                answer, error = "", getattr(exc, "code", self._failure_code(exc))
                response_metrics = {}
            else:
                response_metrics = answer if isinstance(answer, dict) else {}
                answer = response_metrics.get("text", answer)
            latency = time.perf_counter() - started
            passed = evaluate(test, answer)
            if not passed and error is None:
                error = ("EVALUATOR_UNAVAILABLE"
                         if test.get("evaluation_method") == "code" and
                         not code_evaluator_available() else "EVALUATION_FAILED")
            attempts.append({"attempt": attempt, "status": "PASS" if passed else "FAILED",
                             "error": error})
            if passed or attempt == 3:
                return {**test, "answer": answer, "passed": passed, "score": 10 if passed else 0,
                        "latency_s": latency, "ttft_s": response_metrics.get("ttft_s", latency),
                        "generation_s": response_metrics.get("generation_s", latency),
                        "timing_is_estimated": not bool(
                            response_metrics and "ttft_s" in response_metrics and
                            "generation_s" in response_metrics),
                        "token_count_is_estimated": True,
                        "prompt_tokens": response_metrics.get("prompt_tokens", test.get("estimated_prompt_tokens", max(1, len(test["prompt"]) // 4))),
                        "output_tokens": response_metrics.get("output_tokens", max(1, len(answer.split()))), "error": error,
                        "context_reset": history is None, "attempts": attempts,
                        "status": "PASS" if passed else "FAILED"}

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
        context_results = list(payload.get("context_results", []))
        accumulated_results = list(payload.get("accumulated_results", []))
        resume_phase = payload.get("resume_phase", "primary")

        def save_checkpoint(phase):
            self._atomic_state_write({
                **payload, "seed": seed, "profile": profile, "model": model, "tests": tests,
                "resume_phase": phase, "completed_tests": completed, "failures": failures,
                "context_results": context_results, "accumulated_results": accumulated_results,
            })

        self._update_status(state="running", phase="Warming up the model",
                            activity="Sending 3 warm-up requests; their answers are discarded.",
                            total=3, progress=0, passed=sum(x["passed"] for x in completed),
                            failed=len(failures))
        if resume_phase == "primary":
            save_checkpoint("warmup")
            # Warm up and discard warm-up outputs before measured requests.
            for warmup in range(3):
                if self.cancel_requested.is_set():
                    save_checkpoint("warmup")
                    self._update_status(state="cancelled")
                    return
                self._answer(model, {"prompt": "warm up", "evaluation_method": "raw", "expected_answer": ""},
                             generation_options=generation_options)
                self._update_status(progress=warmup + 1,
                                    activity=f"Warm-up request {warmup + 1} of 3 complete.")
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
                         "latency_s": entry["latency_s"], "answer": entry["answer"]}
                        for index, entry in enumerate(repetitions)
                    ],
                }
            else:
                item = repetitions[0]
            completed.append(item)
            if not item["passed"]: failures.append({"test_id": test["id"],
                                                     "reason": item.get("error") or "EVALUATION_FAILED",
                                                     "attempts": item["attempts"]})
            self._update_status(progress=len(completed),
                                passed=sum(x["passed"] for x in completed),
                                failed=len(failures))
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
        latencies = [x["latency_s"] for x in completed] or [.001]
        quality = self._scores(completed)
        score_values = [item["score"] * 10 for item in completed]
        repeat_quality_scores = []
        if repeat_runs > 1:
            for repeat_index in range(repeat_runs):
                run_results = []
                for item in completed:
                    replicate = item["replicates"][repeat_index]
                    run_results.append({**item, "score": replicate["score"]})
                repeat_quality_scores.append(self._scores(run_results)["overall"])
        bootstrap_scores = []
        if not repeat_quality_scores:
            for offset in range(3):
                rng = random.Random(seed + offset)
                sample = [rng.choice(score_values) for _ in score_values] if score_values else [0]
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
        result = {"benchmark_version": "2.0.0", "timestamp": datetime.now(timezone.utc).isoformat(),
                  "seed": seed, "demo": bool(model.get("demo")), "model": model_info(model),
                  "hardware": hardware_info(), "configuration": {"profile": profile, "warmup_runs": 3,
                  "measured_runs": 5, "retries": 2, "temperature": generation_options["temperature"],
                  "top_p": generation_options["top_p"], "generation_options": generation_options,
                  "repeat_runs": repeat_runs,
                  "repeat_seed_policy": "sampling seed increments by one per repeat" if repeat_runs > 1 else "single run",
                  "context_reset_per_test": True, "context_suite": context_sizes,
                  "context_skipped": [size for size in (4096, 8192, 16384, 32768, 65536, 131072, 262144)
                                      if size not in context_sizes],
                  "local_only": True}, "performance": perf, "quality": quality,
                  "context_scaling": {"fresh": context_results, "accumulated": accumulated_results},
                  "tests": completed, "failures": failures,
                  "scores": {"quality": quality["overall"], "fresh_context_quality": quality["overall"],
                             "context_behavior": self._context_score(context_results, accumulated_results),
                             "performance": perf["score"],
                             "overall": round(quality["overall"]*.7 + perf["score"]*.3, 2)},
                  "uncertainty": {
                      "method": ("observed repeated runs" if repeat_quality_scores else
                                 "bootstrap resampling of primary test items"),
                      "quality_mean": round(statistics.mean(
                          repeat_quality_scores or bootstrap_scores), 2),
                      "quality_stddev": round(statistics.pstdev(
                          repeat_quality_scores or bootstrap_scores), 2),
                      "minimum_samples": len(completed),
                      "bootstrap_resample_scores": bootstrap_scores,
                      "repeat_run_scores": repeat_quality_scores},
                  "derived_metrics": {"quality_per_latency": round(quality["overall"]/max(statistics.mean(latencies), .001), 2)},
                  "notes": ["Controlled benchmark results, not universal real-world guarantees.",
                            "Output-token counts are estimates; bootstrap resamples are not independent model runs."]}
        result["id"] = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"
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
            values = [x["score"] * 10 for x in results if x["category"] == category]
            categories[category] = round(statistics.mean(values), 2) if values else None
        measured = [value for value in categories.values() if value is not None]
        categories["overall"] = round(statistics.mean(measured), 2) if measured else None
        return {"categories": categories, "overall": categories["overall"], "method": "objective category evaluators"}

    @staticmethod
    def _context_score(fresh, accumulated):
        fresh_score = statistics.mean([x["score"] for x in fresh]) * 10 if fresh else None
        accumulated_score = statistics.mean([x["score"] for x in accumulated]) * 10 if accumulated else None
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
        for concurrency in (1, 2, 4):
            if self.cancel_requested.is_set():
                return None
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(lambda t: self._answer(model, t, generation_options=generation_options), samples))
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
        latencies = [item["latency_s"] for item in measurements] or [.001]
        generation_times = [max(item["generation_s"], .001) for item in measurements] or [.001]
        ttft_times = [item["ttft_s"] for item in measurements] or latencies
        output_tokens = sum(item["output_tokens"] for item in measurements)
        tokens_per_second = output_tokens / sum(generation_times)
        return {"score": min(100, round(tokens_per_second * 10, 2)),
                "tokens_per_second": round(tokens_per_second, 2),
                "ttft_s": round(statistics.median(ttft_times), 4),
                "generation_s": round(statistics.mean(generation_times), 4),
                "latency_mean_s": round(statistics.mean(latencies), 4),
                "latency_p50_s": round(statistics.median(latencies), 4),
                "latency_p95_s": round(sorted(latencies)[min(len(latencies)-1, math.ceil(len(latencies)*.95)-1)], 4),
                "latency_stddev_s": round(statistics.pstdev(latencies), 4),
                "throughput_basis": "estimated output tokens / measured generation time",
                "timing_is_estimated": any(item["timing_is_estimated"] for item in measurements),
                "token_count_is_estimated": True,
                "warmup_runs": 3, "measured_runs": len(measurements),
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
            fields = ["id", "category", "difficulty", "score", "status", "latency_s"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({k: self._csv_cell(row.get(k)) for k in fields}
                             for row in result["tests"])
        write_pdf(result_dir / f"{result_id}.pdf", result)
        write_chart(result_dir / f"{result_id}_quality.svg", result["quality"]["categories"])

    @staticmethod
    def _csv_cell(value):
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value
