import json
import tempfile
import unittest
from pathlib import Path

from backend.benchmark import BenchmarkManager, make_tests


class BenchmarkTests(unittest.TestCase):
    def test_deterministic_test_generation(self):
        self.assertEqual(make_tests("QUICK", 7), make_tests("QUICK", 7))
        self.assertEqual(len(make_tests("FULL", 7)), 252)

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


if __name__ == "__main__":
    unittest.main()
