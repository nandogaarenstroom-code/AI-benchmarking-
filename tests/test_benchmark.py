import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.benchmark import BenchmarkManager, make_context_tests, make_tests
from backend.discovery import ask_model, normalize_generation_options
from backend.evaluation import evaluate


class BenchmarkTests(unittest.TestCase):
    def test_generation_options_are_normalized(self):
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
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
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
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            ask_model({"backend": "OpenAI-compatible", "endpoint": "http://localhost:8080/v1",
                       "id": "model"}, "prompt", options=options)
        self.assertEqual(captured["temperature"], 0.4)
        self.assertEqual(captured["top_p"], 0.9)
        self.assertEqual(captured["top_k"], 20)
        self.assertEqual(captured["min_p"], 0.1)
        self.assertEqual(captured["repeat_penalty"], 1.2)
        self.assertEqual(captured["max_tokens"], 64)
        self.assertEqual(captured["seed"], 23)
        self.assertTrue(captured["stream"])

    def test_domain_evaluators(self):
        self.assertTrue(evaluate({"evaluation_method": "numeric", "expected_answer": "42"}, "42"))
        self.assertTrue(evaluate({"evaluation_method": "json_schema", "required_keys": ["ok"]}, '{"ok": true}'))
        self.assertTrue(evaluate({"evaluation_method": "code", "tests": ["assert add(1, 2) == 3"]},
                                 "def add(a,b): return a+b"))
        self.assertFalse(evaluate({"evaluation_method": "numeric", "expected_answer": "42"}, "41"))

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
            manager.run({"profile": "QUICK", "generation_options": {"temperature": 0.6},
                         "model": {"name": "Demo", "backend": "Demo",
                                   "endpoint": "local://demo", "demo": True}})
            files = list(Path(directory).glob("Demo_*"))
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


if __name__ == "__main__":
    unittest.main()
