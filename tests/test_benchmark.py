import json
import tempfile
import unittest
from pathlib import Path

from backend.benchmark import BenchmarkManager, make_context_tests, make_tests
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
        self.assertEqual([x["context_size"] for x in make_context_tests(7)],
                         [4096, 8192, 16384, 32768, 65536, 131072])
        self.assertEqual([x["context_size"] for x in make_context_tests(7, 32768)],
                         [4096, 8192, 16384, 32768])

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


if __name__ == "__main__":
    unittest.main()
