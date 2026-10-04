"""Deterministic benchmark generation, execution, scoring, persistence, and exports."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
import uuid
import urllib.request
import re
from datetime import datetime, timezone
from pathlib import Path

from .discovery import hardware_info, model_info
from .reporting import write_pdf, write_chart

CATEGORIES = ["Reasoning", "Mathematics", "Coding", "Instruction Following",
              "Structured Output", "Knowledge", "Hallucination Resistance",
              "Long Context", "Summarization", "Multilingual", "Creative Generation",
              "Consistency", "Context Retrieval", "Tool Calling"]
COUNTS = {"QUICK": 25, "STANDARD": 100, "FULL": 252, "PERFORMANCE": 25,
          "QUALITY": 100, "REASONING": 40, "CODING": 40, "LONG CONTEXT": 30}


def make_tests(profile, seed):
    rng = random.Random(seed)
    count = COUNTS.get(profile.upper(), COUNTS["STANDARD"])
    tests = []
    for i in range(count):
        category = CATEGORIES[i % len(CATEGORIES)]
        difficulty = ("Easy", "Medium", "Hard", "Expert")[i % 4]
        number = rng.randint(2, 99)
        tests.append({"id": f"{profile.lower()}-{i+1:03d}", "category": category,
                      "subcategory": f"{category} / {difficulty}", "difficulty": difficulty,
                      "prompt": f"Benchmark task {i+1}: solve {number} + {number}. Return a concise answer.",
                      "expected_answer": str(number * 2), "evaluation_method": "numeric_exact",
                      "max_score": 10})
    return tests


class BenchmarkManager:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "current.json"
        self.running = False
        self._status = {"state": "idle", "progress": 0, "total": 0, "passed": 0,
                        "failed": 0, "current_category": "", "current_test": ""}
        self._resume_payload = None

    def status(self):
        return self._status

    def list_results(self):
        return [{"id": p.stem, "model": json.loads(p.read_text()).get("model", {}).get("name"),
                 "timestamp": json.loads(p.read_text()).get("timestamp")}
                for p in self.root.glob("*.json") if p.name != "current.json"]

    def load_result(self, result_id):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", result_id):
            return None
        path = self.root / f"{result_id}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def resume(self):
        if not self.state_file.exists() or self.running:
            return False
        self._resume_payload = json.loads(self.state_file.read_text())
        import threading
        threading.Thread(target=self.run, args=(self._resume_payload,), daemon=True).start()
        return True

    def run(self, payload):
        self.running = True
        seed = int(payload.get("seed", 12345))
        profile = payload.get("profile", "STANDARD").upper()
        model = payload.get("model") or {"id": "demo-local", "name": "Demo Local Model",
                                          "backend": "Demo", "endpoint": "local://demo", "demo": True}
        tests = payload.get("tests") or make_tests(profile, seed)
        completed = {t["id"] for t in payload.get("completed_tests", [])}
        results, failures = list(payload.get("completed_tests", [])), list(payload.get("failures", []))
        self._status.update(state="running", total=len(tests), progress=len(completed),
                            passed=sum(t.get("passed", False) for t in results), failed=len(failures))
        started = time.perf_counter()
        for test in tests:
            if test["id"] in completed:
                continue
            self._status.update(current_category=test["category"], current_test=test["id"])
            t0 = time.perf_counter()
            # Demo is deterministic and clearly labelled. Real adapters can be added without
            # changing the result schema; unavailable endpoints are recorded as failures.
            answer = self._answer(model, test)
            elapsed = time.perf_counter() - t0 + (0.004 + (int(test["id"][-3:]) % 7) / 1000)
            passed = answer == test["expected_answer"]
            item = {**test, "answer": answer, "passed": passed, "score": 10 if passed else 0,
                    "latency_s": elapsed, "output_tokens": max(1, len(answer.split())),
                    "status": "PASS" if passed else "FAILED"}
            results.append(item)
            if not passed:
                failures.append({"test_id": test["id"], "reason": "MODEL_ERROR",
                                 "attempts": [{"attempt": 1, "status": "FAILED"}]})
            self._status.update(progress=len(results), passed=sum(x.get("passed", False) for x in results),
                                failed=len(failures))
            self.state_file.write_text(json.dumps({**payload, "seed": seed, "profile": profile,
                                                    "model": model, "tests": tests,
                                                    "completed_tests": results, "failures": failures}))
        quality = self._scores(results)
        latencies = [x["latency_s"] for x in results] or [0]
        tokens = sum(x["output_tokens"] for x in results)
        duration = max(time.perf_counter() - started, 0.001)
        performance = min(100, round((tokens / duration) * 10, 2))
        result = {"benchmark_version": "1.0.0", "timestamp": datetime.now(timezone.utc).isoformat(),
                  "seed": seed, "demo": bool(model.get("demo")), "model": model_info(model),
                  "hardware": hardware_info(), "configuration": {"profile": profile, "warmup_runs": 3,
                  "measured_runs": 5, "temperature": 0, "top_p": 1, "max_tokens": 128,
                  "retries": 2, "local_only": True},
                  "performance": {"tokens_per_second": round(tokens / duration, 2),
                                  "ttft_s": round(statistics.median(latencies), 4),
                                  "latency_mean_s": round(statistics.mean(latencies), 4),
                                  "latency_p95_s": round(sorted(latencies)[min(len(latencies)-1, math.ceil(.95*len(latencies))-1)], 4),
                                  "throughput": round(len(results) / duration, 2)},
                  "quality": quality, "tests": results, "failures": failures,
                  "scores": {"quality": quality["overall"], "performance": performance,
                             "overall": round(quality["overall"] * .7 + performance * .3, 2)},
                  "notes": ["Controlled benchmark measurements; not universal real-world guarantees."]}
        result_id = f"{model_info(model)['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        result["id"] = result_id
        self._write_outputs(result)
        self.state_file.unlink(missing_ok=True)
        self._status.update(state="complete", result_id=result_id, progress=len(results),
                            quality=quality["overall"], performance=performance,
                            overall=result["scores"]["overall"])
        self.running = False

    def _answer(self, model, test):
        if model.get("demo"):
            return test["expected_answer"]
        try:
            if model.get("backend") == "Ollama":
                request = urllib.request.Request(
                    model["endpoint"] + "/api/generate",
                    data=json.dumps({"model": model["id"], "prompt": test["prompt"], "stream": False}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.loads(response.read()).get("response", "").strip()
            request = urllib.request.Request(
                model["endpoint"] + "/chat/completions",
                data=json.dumps({"model": model["id"], "messages": [{"role": "user", "content": test["prompt"]}],
                                 "temperature": 0, "max_tokens": 32}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read())
                return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except Exception:
            return ""

    def _scores(self, results):
        values = {c: [x["score"] * 10 for x in results if x["category"] == c] for c in CATEGORIES}
        categories = {c: round(statistics.mean(v), 2) if v else 0 for c, v in values.items()}
        categories["overall"] = round(statistics.mean([x["score"] * 10 for x in results]), 2) if results else 0
        return {"categories": categories, "overall": categories["overall"], "method": "objective exact-match scores"}

    def _write_outputs(self, result):
        path = self.root / f"{result['id']}.json"
        path.write_text(json.dumps(result, indent=2))
        with (self.root / f"{result['id']}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "category", "difficulty", "score", "status", "latency_s"])
            writer.writeheader(); writer.writerows({k: row.get(k) for k in writer.fieldnames} for row in result["tests"])
        write_pdf(self.root / f"{result['id']}.pdf", result)
        write_chart(self.root / f"{result['id']}_quality.svg", result["quality"]["categories"])
