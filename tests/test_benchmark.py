import json
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import socket
from pathlib import Path

from backend.benchmark import BenchmarkManager, make_context_tests, make_tests
from backend.discovery import AdapterError, ask_model
from backend.runtime import check_environment
from backend.evaluation import evaluate


class BenchmarkTests(unittest.TestCase):
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
        for profile in ("STANDARD", "PERFORMANCE", "QUALITY", "REASONING", "CODING", "LONG CONTEXT"):
            self.assertGreater(len(make_tests(profile, 7)), 0)

    def test_adapter_stream_handles_split_sse_chunks(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return None
            def __iter__(self):
                return iter([b'data: {"choices":[{"delta":{"content":"hel', b'lo"}}]}\n\n',
                             b'data: [DONE]\n\n'])

        model = {"id": "model", "backend": "OpenAI-compatible",
                 "endpoint": "http://localhost:8080/v1"}
        with patch("backend.discovery.urllib.request.urlopen", return_value=Response()):
            self.assertEqual(ask_model(model, "hello")["text"], "hello")

    def test_adapter_errors_are_typed(self):
        model = {"id": "model", "backend": "OpenAI-compatible",
                 "endpoint": "http://localhost:8080/v1"}
        with patch("backend.discovery.urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError("url", 500, "bad", {}, None)):
            with self.assertRaisesRegex(AdapterError, "HTTP 500"):
                ask_model(model, "hello")
        with patch("backend.discovery.urllib.request.urlopen", side_effect=socket.timeout("slow")):
            with self.assertRaises(AdapterError) as caught:
                ask_model(model, "hello")
            self.assertEqual(caught.exception.code, "TIMEOUT")

    def test_demo_run_exports_json_csv_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            manager.run({"profile": "QUICK", "model": {"name": "Demo", "backend": "Demo",
                                                       "endpoint": "local://demo", "demo": True}})
            files = list(Path(directory).glob("Demo_*"))
            self.assertTrue(any(p.suffix == ".json" for p in files))
            self.assertTrue(any(p.suffix == ".csv" for p in files))
            self.assertTrue(any(p.suffix == ".pdf" for p in files))
            data = json.loads(next(p for p in files if p.suffix == ".json").read_text())
            self.assertEqual(data["scores"]["quality"], 100)
            self.assertTrue(data["demo"])
            self.assertTrue(data["configuration"]["context_reset_per_test"])
            self.assertTrue(all(item["context_reset"] for item in data["tests"]))
            self.assertIn("fresh", data["context_scaling"])
            self.assertIn("accumulated", data["context_scaling"])
            self.assertEqual(manager.compare([data["id"]])["best_overall"], data["id"])

    def test_cancel_and_corrupt_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            manager.state_file.write_text("{not json")
            self.assertFalse(manager.resume())
            self.assertFalse(manager.cancel())

    def test_environment_check_reports_unavailable_docker(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"BENCHMARK_USE_DOCKER": "1"}):
            with patch("backend.runtime.shutil.which", return_value=None, create=True):
                self.assertTrue(any("Docker" in warning for warning in check_environment(directory)))


if __name__ == "__main__":
    unittest.main()
