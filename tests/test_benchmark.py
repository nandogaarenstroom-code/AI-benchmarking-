import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from backend.benchmark import BenchmarkManager, make_context_tests, make_tests
from backend.discovery import (
    GENERATION_DEFAULTS, _llama_runtime_details, ask_model, normalize_generation_options,
    resolve_model,
)
from backend.evaluation import evaluate
from backend.reporting import write_pdf


class BenchmarkTests(unittest.TestCase):
    def test_llama_runtime_details_only_include_reported_values(self):
        with patch("backend.discovery._get", return_value={
                "version": "b1234", "default_generation_settings": {
                    "n_ctx": 8192, "temperature": 0.7, "top_p": 0.8, "top_k": 30,
                    "min_p": 0.1, "repeat_penalty": 1.15, "n_predict": 256, "seed": 19,
                },
                "n_gpu_layers": 0}):
            self.assertEqual(_llama_runtime_details("http://localhost:8080/v1"), {
                "llama_cpp": True, "backend_version": "b1234",
                "context_length": 8192, "gpu_offload": 0,
                "generation_defaults": {
                    "temperature": 0.7, "top_p": 0.8, "top_k": 30, "min_p": 0.1,
                    "repeat_penalty": 1.15, "max_tokens": 256, "seed": 19,
                },
            })
        with patch("backend.discovery._get", return_value=None):
            self.assertEqual(_llama_runtime_details("http://localhost:8080/v1"), {})
        with patch("backend.discovery._get", return_value={
                "default_generation_settings": {
                    "temperature": 0.7, "n_predict": -1, "seed": 4294967295,
                }}):
            self.assertEqual(
                _llama_runtime_details("http://localhost:8080/v1")["generation_defaults"],
                {"temperature": 0.7},
            )

    def test_generation_options_are_normalized(self):
        self.assertEqual(GENERATION_DEFAULTS["max_tokens"], 1024)
        self.assertEqual(normalize_generation_options()["max_tokens"], 1024)
        options = normalize_generation_options({
            "temperature": 3, "top_p": "invalid", "top_k": -1,
            "min_p": 0.2, "repeat_penalty": float("nan"), "max_tokens": 5000, "seed": 7,
        })
        self.assertEqual(options, {
            "temperature": 2.0, "top_p": 1.0, "top_k": 0, "min_p": 0.2,
            "repeat_penalty": 1.1, "max_tokens": 4096, "seed": 7,
        })

    def test_generation_options_reach_ollama_adapter(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self):
                return iter([b'{"response":"ok","done":true}\n'])

        captured = {}
        def fake_urlopen(request, timeout):
            captured.update(json.loads(request.data))
            return Response()

        options = {"temperature": 0.4, "top_p": 0.9, "top_k": 20, "min_p": 0.1,
                   "repeat_penalty": 1.2, "max_tokens": 64, "seed": 23}
        with patch("backend.discovery._open", side_effect=fake_urlopen):
            ask_model({"backend": "Ollama", "endpoint": "http://localhost:11434", "id": "model"},
                      "prompt", options=options)
        self.assertEqual(captured["options"], {
            "temperature": 0.4, "top_p": 0.9, "top_k": 20, "min_p": 0.1,
            "repeat_penalty": 1.2, "num_predict": 64, "seed": 23,
        })

    def test_generation_options_reach_openai_compatible_adapter(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self):
                return iter([b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
                             b'data: [DONE]\n'])

        captured = {}
        def fake_urlopen(request, timeout):
            captured.update(json.loads(request.data))
            return Response()

        options = {"temperature": 0.4, "top_p": 0.9, "top_k": 20, "min_p": 0.1,
                   "repeat_penalty": 1.2, "max_tokens": 64, "seed": 23}
        with patch("backend.discovery._open", side_effect=fake_urlopen):
            ask_model({"backend": "OpenAI-compatible", "endpoint": "http://localhost:8080/v1",
                       "id": "model", "llama_cpp": True}, "prompt", options=options)
        self.assertEqual(captured["temperature"], 0.4)
        self.assertEqual(captured["top_p"], 0.9)
        self.assertEqual(captured["top_k"], 20)
        self.assertEqual(captured["min_p"], 0.1)
        self.assertEqual(captured["repeat_penalty"], 1.2)
        self.assertEqual(captured["max_tokens"], 64)
        self.assertEqual(captured["seed"], 23)
        self.assertTrue(captured["stream"])

    def test_openai_adapter_omits_llama_cpp_only_sampler_options(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self): return iter([b'data: [DONE]\n'])

        captured = {}
        def fake_urlopen(request, timeout):
            captured.update(json.loads(request.data))
            return Response()

        with patch("backend.discovery._open", side_effect=fake_urlopen):
            ask_model({"backend": "OpenAI-compatible", "endpoint": "http://localhost:8080/v1",
                       "id": "model"}, "prompt")
        self.assertNotIn("top_k", captured)
        self.assertNotIn("min_p", captured)
        self.assertNotIn("repeat_penalty", captured)

    def test_domain_evaluators(self):
        self.assertTrue(evaluate({"evaluation_method": "numeric", "expected_answer": "42"}, "42"))
        self.assertTrue(evaluate({"evaluation_method": "json_schema", "required_keys": ["ok"]}, '{"ok": true}'))
        with patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                patch("backend.evaluation.subprocess.run",
                      return_value=CompletedProcess(args=[], returncode=0)) as run:
            self.assertTrue(evaluate(
                {"evaluation_method": "code", "tests": ["assert add(1, 2) == 3"]},
                "def add(a,b): return a+b"))
        command = run.call_args.args[0]
        self.assertIn("--network=none", command)
        self.assertIn("--cap-drop=ALL", command)
        self.assertIn("--read-only", command)
        self.assertFalse(evaluate({"evaluation_method": "numeric", "expected_answer": "42"}, "41"))

    def test_code_evaluation_fails_closed_without_docker(self):
        with patch("backend.evaluation.shutil.which", return_value=None), \
                patch("backend.evaluation.subprocess.run") as run:
            self.assertFalse(evaluate(
                {"evaluation_method": "code", "tests": ["assert True"]}, "print('not run')"))
        run.assert_not_called()

    def test_code_evaluation_rejects_invalid_or_oversized_tests(self):
        with patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                patch("backend.evaluation.subprocess.run") as run:
            self.assertFalse(evaluate({"evaluation_method": "code", "tests": ["x" * 2001]}, "pass"))
            self.assertFalse(evaluate({"evaluation_method": "code", "tests": "assert True"}, "pass"))
        run.assert_not_called()

    def test_deterministic_test_generation(self):
        self.assertEqual(make_tests("QUICK", 7), make_tests("QUICK", 7))
        self.assertEqual(len(make_tests("FULL", 7)), 252)
        prompts = [test["prompt"] for test in make_tests("FULL", 7)]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertEqual([x["context_size"] for x in make_context_tests(7)],
                         [4096, 8192, 16384, 32768, 65536, 131072])
        self.assertEqual([x["context_size"] for x in make_context_tests(7, 32768)],
                         [4096, 8192, 16384, 32768])

    def test_demo_run_exports_json_csv_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            with patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                    patch("backend.evaluation.subprocess.run",
                          return_value=CompletedProcess(args=[], returncode=0)):
                manager.run({"profile": "QUICK", "generation_options": {"temperature": 0.6},
                             "model": {"id": "demo-local", "name": "Demo", "backend": "Demo",
                                       "endpoint": "local://demo", "demo": True}})
            files = list(Path(directory).glob("benchmark_*"))
            self.assertTrue(any(p.suffix == ".json" for p in files))
            self.assertTrue(any(p.suffix == ".csv" for p in files))
            self.assertTrue(any(p.suffix == ".pdf" for p in files))
            data = json.loads(next(p for p in files if p.suffix == ".json").read_text())
            self.assertEqual(data["scores"]["quality"], 100)
            self.assertTrue(data["demo"])
            self.assertEqual(data["configuration"]["generation_options"]["temperature"], 0.6)
            self.assertTrue(data["configuration"]["context_reset_per_test"])
            self.assertTrue(all(item["context_reset"] for item in data["tests"]))
            self.assertIn("fresh", data["context_scaling"])
            self.assertIn("accumulated", data["context_scaling"])
            self.assertEqual(manager.compare([data["id"]])["best_overall"], data["id"])
            self.assertEqual(manager.status()["state"], "complete")
            self.assertEqual(manager.status()["phase"], "Benchmark complete")
            self.assertEqual(manager.status()["result_directory"], str(Path(directory).resolve()))

    def test_pdf_includes_dashboard_measurements_and_escapes_model_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            write_pdf(path, {
                "model": {
                    "name": r"Local (test) \ model", "backend": "OpenAI-compatible",
                    "backend_version": "b1234", "context_length": 8192, "gpu_offload": 20,
                    "generation_defaults": {"max_tokens": 2048},
                },
                "timestamp": "2026-10-08T11:33:52Z",
                "scores": {
                    "quality": 75, "performance": 42.5, "overall": 65.25,
                    "context_behavior": {"fresh": 90, "accumulated": 70, "degradation": 20},
                },
                "quality": {"categories": {"Reasoning": 100, "Math": None, "overall": 100}},
                "performance": {
                    "score": 42.5, "tokens_per_second": 12.3, "ttft_s": 0.25,
                    "generation_s": 2.5, "latency_mean_s": 0.5, "latency_p50_s": 0.4,
                    "latency_p95_s": 0.8, "latency_stddev_s": 0.1, "warmup_runs": 3,
                    "measured_runs": 5, "concurrency_levels": [1, 2, 4],
                },
                "configuration": {
                    "profile": "STANDARD", "generation_options": {"max_tokens": 1024},
                    "repeat_runs": 2,
                },
                "uncertainty": {
                    "quality_mean": 75, "quality_stddev": 5, "repeat_run_scores": [70, 80],
                },
                "context_scaling": {
                    "fresh": [{"context_size": 4096, "score": 10}],
                    "accumulated": [{"context_size": 4096, "score": 8}],
                },
                "tests": [{"passed": True}, {"passed": False}],
                "failures": [{"test_id": "test"}],
            })
            report = path.read_bytes()
            self.assertIn(b"Reasoning: 100/100", report)
            self.assertIn(b"Math: Not tested", report)
            self.assertIn(b"Throughput: 12.3 tokens/second", report)
            self.assertIn(b"Fresh context score: 90/100", report)
            self.assertIn(b"4,096 tokens | 100/100 | 80/100", report)
            self.assertIn(b"max_tokens=1024", report)
            self.assertIn(b"Local \\(test\\) \\\\ model", report)

    def test_repeated_demo_run_records_quality_variability(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            with patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                    patch("backend.evaluation.subprocess.run",
                          return_value=CompletedProcess(args=[], returncode=0)):
                manager.run({"profile": "QUICK", "repeat_runs": 2,
                             "generation_options": {"seed": 31},
                             "model": {"id": "demo-local", "name": "Repeat Demo", "backend": "Demo",
                                       "endpoint": "local://demo", "demo": True}})
            result_path = next(Path(directory).glob("benchmark_*.json"))
            result = json.loads(result_path.read_text())
            self.assertEqual(result["configuration"]["repeat_runs"], 2)
            self.assertEqual(result["uncertainty"]["repeat_run_scores"], [100.0, 100.0])
            self.assertEqual(result["uncertainty"]["quality_stddev"], 0.0)
            self.assertEqual([rep["seed"] for rep in result["tests"][0]["replicates"]], [31, 32])

    def test_custom_tests_are_ignored_and_run_state_recovers_after_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            self.assertTrue(manager.run({
                "profile": "QUICK", "tests": [{"id": "attack", "evaluation_method": "code",
                                               "tests": ["raise RuntimeError()"]}],
                "model": {"id": "demo-local", "name": "demo", "backend": "Demo",
                          "endpoint": "local://demo", "demo": True},
            }))
            result_path = next(Path(directory).glob("benchmark_*.json"))
            result = json.loads(result_path.read_text())
            self.assertEqual(len(result["tests"]), 25)
            self.assertNotEqual(result["tests"][0]["id"], "attack")
            self.assertFalse(manager.running)
            self.assertFalse(manager.run({"seed": "invalid"}))
            self.assertFalse(manager.running)
            self.assertEqual(manager.status()["state"], "error")

    def test_model_resolution_rejects_caller_chosen_endpoint(self):
        with patch("backend.discovery.discover_models", return_value=[]):
            with self.assertRaises(ValueError):
                resolve_model({"id": "model", "backend": "OpenAI-compatible",
                               "endpoint": "http://127.0.0.1:9"})

    def test_artifact_id_is_path_safe_and_csv_cells_are_neutralized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            manager = BenchmarkManager(root)
            result = {
                "id": "benchmark_20261008_000000_abcdef",
                "model": {"name": "test"}, "timestamp": "now",
                "scores": {"quality": 0, "performance": 0, "overall": 0},
                "tests": [{"id": "test", "category": "=1+1", "difficulty": "Easy",
                           "score": 0, "status": "PASS", "latency_s": 0}],
                "failures": [], "quality": {"categories": {}},
            }
            manager._write_outputs(result)
            self.assertTrue((root / f"{result['id']}.json").is_file())
            csv_row = (root / f"{result['id']}.csv").read_text().splitlines()[1]
            self.assertIn(",'=1+1,", csv_row)
            result["id"] = "../escaped"
            with self.assertRaises(ValueError):
                manager._write_outputs(result)

    def test_result_listing_and_loading_ignore_symlinks_outside_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            root.mkdir()
            external = Path(directory) / "private.json"
            external.write_text('{"model":{"name":"private"}}')
            (root / "linked.json").symlink_to(external)
            manager = BenchmarkManager(root)
            self.assertEqual(manager.list_results(), [])
            self.assertIsNone(manager.load_result("linked"))


if __name__ == "__main__":
    unittest.main()
