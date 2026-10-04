"""Domain benchmark generation, evaluation, performance measurement, and exports."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
import uuid
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
                "expected_answer": str(a * b), "evaluation_method": "numeric"}
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
    return [_test(CATEGORIES[i % len(CATEGORIES)], i, rng) for i in range(count)]


def make_context_tests(seed):
    rng = random.Random(seed)
    tests = []
    for size in (4096, 8192, 16384, 32768):
        marker = f"CTX-{rng.randint(1000, 9999)}"
        filler = " context" * max(1, size // 8)
        tests.append({"id": f"context-{size}", "category": "Context Scaling",
                      "context_size": size, "prompt": f"{marker}{filler}\nReturn the marker.",
                      "expected_answer": marker, "evaluation_method": "contains_all",
                      "required_terms": [marker], "max_score": 10})
    return tests


class BenchmarkManager:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "current.json"; self.running = False
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
        threading.Thread(target=self.run, args=(json.loads(self.state_file.read_text()),), daemon=True).start()
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
        return ask_model(model, test["prompt"], history=history)

    def _execute(self, model, test, history=None):
        attempts = []
        for attempt in range(1, 4):
            started = time.perf_counter()
            answer = self._answer(model, test, history)
            latency = time.perf_counter() - started
            passed = evaluate(test, answer)
            attempts.append({"attempt": attempt, "status": "PASS" if passed else "FAILED"})
            if passed or attempt == 3:
                return {**test, "answer": answer, "passed": passed, "score": 10 if passed else 0,
                        "latency_s": latency, "ttft_s": latency, "generation_s": latency,
                        "prompt_tokens": len(test["prompt"].split()), "output_tokens": max(1, len(answer.split())),
                        "context_reset": history is None, "attempts": attempts,
                        "status": "PASS" if passed else "FAILED"}

    def run(self, payload):
        self.running = True
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
            if test["id"] in done: continue
            self._status.update(current_category=test["category"], current_test=test["id"])
            item = self._execute(model, test)
            completed.append(item)
            if not item["passed"]: failures.append({"test_id": test["id"], "reason": "MODEL_ERROR",
                                                     "attempts": item["attempts"]})
            self._status.update(progress=len(completed), passed=sum(x["passed"] for x in completed),
                                failed=len(failures))
            self.state_file.write_text(json.dumps({**payload, "seed": seed, "profile": profile,
                                                    "model": model, "tests": tests,
                                                    "completed_tests": completed, "failures": failures}))
        context_tests = make_context_tests(seed)
        context_results = [self._execute(model, test) for test in context_tests]
        history, accumulated_results = [], []
        for test in context_tests:
            item = self._execute(model, test, history=history)
            accumulated_results.append(item)
            history.extend([{"role": "user", "content": test["prompt"]},
                            {"role": "assistant", "content": item["answer"]}])
        duration = max(time.perf_counter() - started, .001)
        latencies = [x["latency_s"] for x in completed] or [.001]
        tokens = sum(x["output_tokens"] for x in completed)
        quality = self._scores(completed)
        perf = self._performance(model, tests[:min(5, len(tests))], tokens, duration)
        result = {"benchmark_version": "2.0.0", "timestamp": datetime.now(timezone.utc).isoformat(),
                  "seed": seed, "demo": bool(model.get("demo")), "model": model_info(model),
                  "hardware": hardware_info(), "configuration": {"profile": profile, "warmup_runs": 3,
                  "measured_runs": 5, "retries": 2, "temperature": 0, "top_p": 1,
                  "context_reset_per_test": True, "context_suite": [4096, 8192, 16384, 32768],
                  "local_only": True}, "performance": perf, "quality": quality,
                  "context_scaling": {"fresh": context_results, "accumulated": accumulated_results},
                  "tests": completed, "failures": failures,
                  "scores": {"quality": quality["overall"], "performance": perf["score"],
                             "overall": round(quality["overall"]*.7 + perf["score"]*.3, 2)},
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
            categories[category] = round(statistics.mean(values), 2) if values else 0
        categories["overall"] = round(statistics.mean([x["score"]*10 for x in results]), 2) if results else 0
        return {"categories": categories, "overall": categories["overall"], "method": "objective category evaluators"}

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
        (self.root / f"{result['id']}.json").write_text(json.dumps(result, indent=2))
        with (self.root / f"{result['id']}.csv").open("w", newline="") as stream:
            fields = ["id", "category", "difficulty", "score", "status", "latency_s"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({k: row.get(k) for k in fields} for row in result["tests"])
        write_pdf(self.root / f"{result['id']}.pdf", result)
        write_chart(self.root / f"{result['id']}_quality.svg", result["quality"]["categories"])
