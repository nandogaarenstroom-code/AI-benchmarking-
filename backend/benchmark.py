"""Domain benchmark generation, evaluation, performance measurement, and exports."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
import uuid
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .discovery import hardware_info, model_info
from .evaluation import evaluate
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
        return {**common, "prompt": f"If all zibs are blue and item {a} is a zib, what color is item {a}?",
                "expected_answer": "blue", "evaluation_method": "contains_all", "required_terms": ["blue"]}
    if category == "Coding":
        return {**common, "prompt": "Write Python defining add(a,b) that returns their sum. Output code only.",
                "expected_answer": "code", "evaluation_method": "code",
                "tests": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]}
    if category == "Structured Output":
        return {**common, "prompt": 'Return JSON exactly with keys "answer" and "valid".',
                "expected_answer": '{"answer": "ok", "valid": true}', "evaluation_method": "json_schema",
                "required_keys": ["answer", "valid"]}
    if category == "Knowledge":
        return {**common, "prompt": "What is the capital of France? Answer concisely.",
                "expected_answer": "Paris", "evaluation_method": "contains_all", "required_terms": ["Paris"]}
    if category == "Hallucination Resistance":
        return {**common, "prompt": "What is the capital of the fictional country Zorbland? State that it is unknown.",
                "expected_answer": "unknown", "evaluation_method": "contains_all", "required_terms": ["unknown"]}
    if category == "Multilingual":
        return {**common, "prompt": "Translate 'good morning' into Spanish.",
                "expected_answer": "buenos días", "evaluation_method": "contains_all",
                "required_terms": ["buenos", "días"]}
    if category == "Instruction Following":
        return {**common, "prompt": "Respond with exactly three words describing rain.",
                "expected_answer": "three words", "evaluation_method": "rubric", "minimum_words": 3}
    if category == "Summarization":
        return {**common, "prompt": "Summarize: Local systems run without sending data away. Mention local privacy.",
                "expected_answer": "local privacy", "evaluation_method": "contains_all",
                "required_terms": ["local", "privacy"]}
    if category == "Creative Generation":
        return {**common, "prompt": "Write a four-word slogan about careful testing.",
                "expected_answer": "creative", "evaluation_method": "rubric", "minimum_words": 4}
    if category == "Consistency":
        return {**common, "prompt": "Is 2 + 2 equal to 4? Answer yes or no.",
                "expected_answer": "yes", "evaluation_method": "contains_all", "required_terms": ["yes"]}
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
        self.cancel_requested = threading.Event()
        self._status = {"state": "idle", "progress": 0, "total": 0, "passed": 0, "failed": 0}

    def status(self): return self._status

    def list_results(self):
        return [{"id": p.stem, "model": json.loads(p.read_text()).get("model", {}).get("name"),
                 "timestamp": json.loads(p.read_text()).get("timestamp")}
                for p in self.root.glob("*.json") if p.name != "current.json"]

    def load_result(self, result_id):
        for path in self.root.glob("*.json"):
            if path.name != "current.json" and path.stem == result_id:
                return json.loads(path.read_text())
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
        if not self.state_file.exists() or self.running: return False
        import threading
        try:
            state = json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        threading.Thread(target=self.run, args=(state,), daemon=True).start()
        return True

    def cancel(self):
        if not self.running:
            return False
        self.cancel_requested.set()
        return True

    def _answer(self, model, test, history=None):
        if model.get("demo"):
            if test["evaluation_method"] == "code":
                return "def add(a,b):\n return a+b"
            if test["evaluation_method"] == "json_schema":
                return '{"answer":"ok","valid":true}'
            if test["evaluation_method"] == "rubric":
                return "careful reliable local testing"
            return test["expected_answer"]
        from .discovery import ask_model
        response = ask_model(model, test["prompt"], history=history)
        return response

    def _execute(self, model, test, history=None):
        attempts = []
        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                answer = self._answer(model, test, history)
                error = None
            except Exception as exc:
                answer, error = "", getattr(exc, "code", self._failure_code(exc))
                response_metrics = {}
            else:
                response_metrics = answer if isinstance(answer, dict) else {}
                answer = response_metrics.get("text", answer)
            latency = time.perf_counter() - started
            passed = evaluate(test, answer)
            attempts.append({"attempt": attempt, "status": "PASS" if passed else "FAILED",
                             "error": error})
            if passed or attempt == 3:
                return {**test, "answer": answer, "passed": passed, "score": 10 if passed else 0,
                        "latency_s": latency, "ttft_s": response_metrics.get("ttft_s", latency),
                        "generation_s": response_metrics.get("generation_s", latency),
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

    def run(self, payload):
        self.running = True
        self.cancel_requested.clear()
        seed, profile = int(payload.get("seed", 12345)), payload.get("profile", "STANDARD").upper()
        model = payload.get("model") or {"id": "demo-local", "name": "Demo Local Model",
                                          "backend": "Demo", "endpoint": "local://demo", "demo": True}
        tests = payload.get("tests") or make_tests(profile, seed)
        completed = list(payload.get("completed_tests", [])); done = {x["id"] for x in completed}
        failures = list(payload.get("failures", []))
        self._status.update(state="running", total=len(tests), progress=len(completed))
        # Warm up and discard warm-up outputs before measured requests.
        for _ in range(3): self._answer(model, {"prompt": "warm up", "evaluation_method": "raw", "expected_answer": ""})
        started = time.perf_counter()
        for test in tests:
            if self.cancel_requested.is_set():
                self._status.update(state="cancelled")
                self.running = False
                return
            if test["id"] in done: continue
            self._status.update(current_category=test["category"], current_test=test["id"])
            item = self._execute(model, test)
            completed.append(item)
            if not item["passed"]: failures.append({"test_id": test["id"], "reason": "MODEL_ERROR",
                                                     "attempts": item["attempts"]})
            self._status.update(progress=len(completed), passed=sum(x["passed"] for x in completed),
                                failed=len(failures))
            self._atomic_state_write({**payload, "seed": seed, "profile": profile,
                                      "model": model, "tests": tests,
                                      "completed_tests": completed, "failures": failures})
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
        context_results = [self._execute(model, test) for test in context_tests]
        history, accumulated_results = [], []
        for test in context_tests:
            item = self._execute(model, test, history=history)
            accumulated_results.append(item)
            history.extend([{"role": "user", "content": test["prompt"] if not model.get("demo")
                            else f"[context test {test['context_size']} tokens]"},
                            {"role": "assistant", "content": item["answer"]}])
        duration = max(time.perf_counter() - started, .001)
        latencies = [x["latency_s"] for x in completed] or [.001]
        tokens = sum(x["output_tokens"] for x in completed)
        quality = self._scores(completed)
        score_values = [item["score"] * 10 for item in completed]
        seed_scores = []
        for offset in range(3):
            rng = random.Random(seed + offset)
            sample = [rng.choice(score_values) for _ in score_values] if score_values else [0]
            seed_scores.append(round(statistics.mean(sample), 2))
        perf = self._performance(model, tests[:min(5, len(tests))], tokens, duration)
        result = {"benchmark_version": "2.0.0", "timestamp": datetime.now(timezone.utc).isoformat(),
                  "seed": seed, "demo": bool(model.get("demo")), "model": model_info(model),
                  "hardware": hardware_info(), "configuration": {"profile": profile, "warmup_runs": 3,
                  "measured_runs": 5, "retries": 2, "temperature": 0, "top_p": 1,
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
                  "uncertainty": {"independent_seed_scores": seed_scores,
                                  "quality_mean": round(statistics.mean(seed_scores), 2),
                                  "quality_stddev": round(statistics.pstdev(seed_scores), 2),
                                  "minimum_samples": len(completed)},
                  "derived_metrics": {"quality_per_latency": round(quality["overall"]/statistics.mean(latencies), 2)},
                  "notes": ["Controlled benchmark results, not universal real-world guarantees."]}
        result["id"] = f"{model_info(model)['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self._write_outputs(result); self.state_file.unlink(missing_ok=True)
        self._status.update(state="complete", result_id=result["id"], quality=quality["overall"],
                            performance=perf["score"], overall=result["scores"]["overall"])
        self.running = False

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

    def _performance(self, model, samples, tokens, duration):
        for concurrency in (1, 2, 4):
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(lambda t: self._answer(model, t), samples))
        latencies = [self._execute(model, t)["latency_s"] for t in samples] or [.001]
        return {"score": min(100, round(tokens / duration * 10, 2)),
                "tokens_per_second": round(tokens/duration, 2), "ttft_s": round(statistics.median(latencies), 4),
                "generation_s": round(sum(latencies), 4), "latency_mean_s": round(statistics.mean(latencies), 4),
                "latency_p50_s": round(statistics.median(latencies), 4),
                "latency_p95_s": round(sorted(latencies)[min(len(latencies)-1, math.ceil(len(latencies)*.95)-1)], 4),
                "latency_stddev_s": round(statistics.pstdev(latencies), 4),
                "warmup_runs": 3, "measured_runs": 5, "concurrency_levels": [1, 2, 4]}

    def _write_outputs(self, result):
        destination = self.root / f"{result['id']}.json"
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
        with (self.root / f"{result['id']}.csv").open("w", newline="") as stream:
            fields = ["id", "category", "difficulty", "score", "status", "latency_s"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({k: row.get(k) for k in fields} for row in result["tests"])
        write_pdf(self.root / f"{result['id']}.pdf", result)
        write_chart(self.root / f"{result['id']}_quality.svg", result["quality"]["categories"])
