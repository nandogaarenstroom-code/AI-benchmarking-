import json
import threading
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from backend.benchmark import (
    EVALUATOR_VERSION, TASK_SET_VERSION, BenchmarkManager, make_context_tests, make_tests,
)
from backend.discovery import (
    AdapterError, _engine_probe, _get, _llama_runtime_details, ask_model, discover_models,
    model_info, normalize_generation_options, resolve_model,
)
from backend.evaluation import evaluate, evaluate_with_details
from backend.reporting import write_chart, write_pdf


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

    def test_discovered_model_includes_only_managed_server_startup_settings(self):
        launch = {
            "source": "path", "model_file": "model.gguf",
            "context_size_requested": 65536, "gpu_layers_requested": 24,
            "host": "127.0.0.1", "port": 8080,
        }

        def fake_get(url, timeout=0.7):
            if url == "http://127.0.0.1:8080/v1/models":
                return {"data": [{"id": "local-model"}]}
            return None

        with patch("backend.discovery._get", side_effect=fake_get), \
                patch("backend.discovery._llama_runtime_details", return_value={
                    "llama_cpp": True, "context_length": 65536}), \
                patch("backend.discovery.llama_server_manager.managed_launch",
                      return_value=launch):
            models = discover_models()
        model = next(item for item in models if item.get("id") == "local-model")
        self.assertEqual(model["managed_launch"], launch)
        self.assertEqual(model_info(model)["managed_launch"], launch)

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

    def test_openai_streaming_metrics_use_first_content_chunk(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self):
                return iter([b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
                             b'data: [DONE]\n'])

        with patch("backend.discovery._open", return_value=Response()), \
                patch("backend.discovery.time.perf_counter", side_effect=[10, 10.2, 10.8, 11]):
            result = ask_model({"backend": "OpenAI-compatible", "endpoint": "http://localhost:8080/v1",
                                "id": "model"}, "prompt")
        self.assertAlmostEqual(result["ttft_s"], 0.2)
        self.assertAlmostEqual(result["generation_s"], 0.6)

    def test_adapter_timeout_and_malformed_stream_are_reported(self):
        model = {"backend": "OpenAI-compatible", "endpoint": "http://localhost:8080/v1",
                 "id": "model"}
        with patch("backend.discovery._open", side_effect=TimeoutError("slow")):
            with self.assertRaises(AdapterError) as caught:
                ask_model(model, "prompt")
        self.assertEqual(caught.exception.code, "TIMEOUT")

        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self): return iter([b"data: not-json\n"])

        with patch("backend.discovery._open", return_value=Response()):
            with self.assertRaises(AdapterError) as caught:
                ask_model(model, "prompt")
        self.assertEqual(caught.exception.code, "SERVER_ERROR")

    def test_discovery_rejects_oversized_and_non_object_responses(self):
        class Response:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, size): return self.body

        with patch("backend.discovery._open", return_value=Response(b"x" * 1_048_577)):
            self.assertIsNone(_get("http://localhost/model", timeout=0.1))
        with patch("backend.discovery._open", return_value=Response(b"[]")):
            self.assertIsNone(_get("http://localhost/model", timeout=0.1))

    def test_engine_probe_reports_detected_models_and_malformed_responses(self):
        class Response:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, size): return self.body

        with patch("backend.discovery._open", return_value=Response(
                b'{"data":[{"id":"local-model"},{}]}')):
            status = _engine_probe("LM Studio", "http://localhost:1234/v1",
                                   "http://localhost:1234/v1/models", "data")
        self.assertEqual(status["status"], "detected")
        self.assertEqual(status["models"], ["local-model"])
        with patch("backend.discovery._open", return_value=Response(b"not-json")):
            status = _engine_probe("LM Studio", "http://localhost:1234/v1",
                                   "http://localhost:1234/v1/models", "data")
        self.assertEqual(status["status"], "error")

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

    def test_openai_compatible_adapter_integrates_with_loopback_server(self):
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                body = json.dumps({"data": [{"id": "stub-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = (b'data: {"choices":[{"delta":{"content":"Paris"}}]}\n\n'
                        b'data: [DONE]\n\n')
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/v1"
            self.assertEqual(_get(endpoint + "/models"), {"data": [{"id": "stub-model"}]})
            model = {"id": "stub-model", "backend": "OpenAI-compatible",
                     "endpoint": endpoint, "llama_cpp": False}
            result = ask_model(model, "What is the capital of France?")
            self.assertEqual(result["text"], "Paris")
            self.assertGreaterEqual(result["output_tokens"], 1)
            self.assertTrue(captured[0]["stream"])
            self.assertEqual(captured[0]["messages"], [
                {"role": "user", "content": "What is the capital of France?"}])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_json_types_and_exact_word_count_are_enforced(self):
        schema = {"evaluation_method": "json_schema", "required_keys": ["answer", "valid"],
                  "required_types": {"answer": "string", "valid": "boolean"}}
        self.assertTrue(evaluate(schema, '{"answer":"ok","valid":true}'))
        self.assertFalse(evaluate(schema, '{"answer":false,"valid":"true"}'))
        self.assertFalse(evaluate({**schema, "required_types": ["invalid"]},
                                  '{"answer":"ok","valid":true}'))
        rubric = {"evaluation_method": "rubric", "exact_words": 3,
                  "required_terms": ["rain"]}
        self.assertTrue(evaluate(rubric, "soft rain falls"))
        self.assertFalse(evaluate(rubric, "soft rain falls gently"))
        self.assertFalse(evaluate(rubric, "soft snow falls"))

    def test_reviewed_exact_match_requires_the_complete_answer(self):
        translation = {
            "evaluation_method": "exact_any",
            "expected_answer": "Gracias",
            "acceptable_answers": ["Gracias", "Muchas gracias."],
        }
        self.assertTrue(evaluate(translation, "MUCHAS GRACIAS!"))
        self.assertFalse(evaluate(translation, "I cannot translate this, but gracias is one option."))
        self.assertFalse(evaluate(translation, "Gracias for your help."))
        numeric = {"evaluation_method": "numeric", "expected_answer": "42", "tolerance": 0}
        self.assertTrue(evaluate(numeric, "42"))
        self.assertFalse(evaluate(numeric, "The answer is 42, not 41."))
        self.assertFalse(evaluate(
            {"evaluation_method": "choice", "expected_answer": "yes"}, "yes, but actually no"))
        self.assertTrue(evaluate(
            {"evaluation_method": "ordered_contains", "required_terms": ["BLUE-2", "GOLD-8"]},
            "BLUE-2 then GOLD-8"))
        self.assertFalse(evaluate(
            {"evaluation_method": "ordered_contains", "required_terms": ["BLUE-2", "GOLD-8"]},
            "GOLD-8 then BLUE-2"))

    def test_rubric_reports_format_required_term_and_relevance_checks(self):
        rubric = {"evaluation_method": "rubric", "exact_words": 3,
                  "required_terms": ["rain"], "topic_terms": ["rain", "wet", "drops"],
                  "minimum_topic_terms": 2}
        result = evaluate_with_details(rubric, "rain rain rain")
        self.assertFalse(result["passed"])
        self.assertEqual(result["score"], 6.67)
        self.assertFalse(next(check for check in result["checks"]
                              if check["name"] == "Topical relevance proxy")["passed"])
        relevant = evaluate_with_details(rubric, "rain wet drops")
        self.assertTrue(relevant["passed"])
        self.assertEqual(relevant["score"], 10)

    def test_evaluators_explain_observable_pass_and_fail_evidence(self):
        test = {"evaluation_method": "contains_all", "required_terms": ["amber", "wug"]}
        passed = evaluate_with_details(test, "The amber wug is here.")
        failed = evaluate_with_details(test, "The amber item is here.")
        self.assertTrue(passed["passed"])
        self.assertIn("All required terms", passed["explanation"])
        self.assertFalse(failed["passed"])
        self.assertIn("wug", failed["explanation"])
        self.assertIn("amber", failed["criteria"])
        self.assertFalse(evaluate(test, "The amber wuglet is here."))
        self.assertTrue(evaluate(
            {"evaluation_method": "contains_all", "required_terms": ["¿Dónde está la estación?"]},
            "Dónde está la estación"))

    def test_attempt_records_retain_answers_and_explain_each_check(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            test = {"id": "math-1", "category": "Mathematics", "prompt": "Answer 42",
                    "evaluation_method": "numeric", "expected_answer": "42"}
            with patch.object(manager, "_answer", side_effect=["41", "42"]):
                result = manager._execute({"demo": False}, test)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual([item["answer"] for item in result["attempts"]], ["41", "42"])
        self.assertIn("expected 42", result["attempts"][0]["evaluation"]["explanation"])
        self.assertTrue(result["attempts"][1]["evaluation"]["passed"])

    def test_all_model_request_errors_are_not_run_not_model_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            test = {"id": "math-1", "category": "Mathematics", "prompt": "Answer 42",
                    "evaluation_method": "numeric", "expected_answer": "42"}
            with patch.object(manager, "_answer", side_effect=AdapterError("TIMEOUT", "slow")):
                result = manager._execute({"demo": False}, test)
        self.assertEqual(result["status"], "NOT RUN")
        self.assertIsNone(result["score"])
        self.assertEqual(result["error"], "TIMEOUT")
        self.assertTrue(all(item["status"] == "ERROR" for item in result["attempts"]))
        self.assertIn("no model response was available", result["evaluation"]["explanation"].lower())

    def test_code_evaluation_fails_closed_without_docker(self):
        with patch("backend.evaluation.shutil.which", return_value=None), \
                patch("backend.evaluation.subprocess.run") as run:
            self.assertFalse(evaluate(
                {"evaluation_method": "code", "tests": ["assert True"]}, "print('not run')"))
        run.assert_not_called()

    def test_unavailable_code_evaluator_is_reported_in_test_results(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            test = {"id": "coding-1", "category": "Coding", "prompt": "define f",
                    "evaluation_method": "code", "tests": ["assert f() == 1"]}
            with patch.object(manager, "_answer", return_value="def f(): return 1"), \
                    patch("backend.evaluation.shutil.which", return_value=None):
                result = manager._execute({"demo": False}, test)
        self.assertIsNone(result["passed"])
        self.assertEqual(result["status"], "NOT RUN")
        self.assertIsNone(result["score"])
        self.assertEqual(result["error"], "EVALUATOR_UNAVAILABLE")
        self.assertIn("not executed", result["evaluation"]["explanation"])

    def test_not_run_checks_do_not_lower_quality_score(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            results = [
                {"category": "Coding", "score": None, "status": "NOT RUN"},
                {"category": "Mathematics", "score": 1, "status": "PASS"},
            ]
            scores = manager._scores(results)
        self.assertIsNone(scores["categories"]["Coding"])
        self.assertEqual(scores["categories"]["Mathematics"], 10)
        self.assertEqual(scores["overall"], 10)

    def test_execution_reports_attempt_progress_and_live_feed_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            events = []
            test = {"id": "math-001", "category": "Mathematics", "difficulty": "Easy",
                    "prompt": "What is 2 + 2?", "expected_answer": "4",
                    "evaluation_method": "numeric", "max_score": 10}
            result = manager._execute({"demo": True}, test, progress_callback=events.append)
        self.assertEqual([event["event"] for event in events],
                         ["attempt_started", "attempt_completed"])
        self.assertEqual(events[-1]["status"], "PASS")
        self.assertIn("latency_s", events[-1])
        live = BenchmarkManager._live_test_record(result)
        self.assertEqual(live["id"], "math-001")
        self.assertEqual(live["status"], "PASS")
        self.assertEqual(len(live["attempts"]), 1)
        self.assertEqual(live["evaluation"]["explanation"],
                         result["evaluation"]["explanation"])

    def test_code_evaluation_rejects_invalid_or_oversized_tests(self):
        with patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                patch("backend.evaluation.subprocess.run") as run:
            self.assertFalse(evaluate({"evaluation_method": "code", "tests": ["x" * 2001]}, "pass"))
            self.assertFalse(evaluate({"evaluation_method": "code", "tests": "assert True"}, "pass"))
        run.assert_not_called()

    def test_deterministic_test_generation(self):
        self.assertEqual(make_tests("QUICK", 7), make_tests("QUICK", 7))
        selected = make_tests("STANDARD", 7, categories=["Coding", "Knowledge"], count=5)
        self.assertEqual(len(selected), 5)
        self.assertEqual({item["category"] for item in selected}, {"Coding", "Knowledge"})
        self.assertEqual(selected, make_tests(
            "STANDARD", 7, categories=["Coding", "Knowledge"], count=5))
        quick = make_tests("QUICK", 7)
        self.assertEqual({item["difficulty"] for item in quick},
                         {"Easy", "Medium", "Hard", "Uncalibrated"})
        self.assertEqual(len(make_tests("FULL", 7)), 252)
        full = make_tests("FULL", 7)
        prompts = [test["prompt"] for test in full]
        self.assertEqual(len(prompts), len(set(prompts)))
        for category in {test["category"] for test in full}:
            self.assertEqual(
                {test["difficulty"] for test in full if test["category"] == category},
                {"Easy", "Medium", "Hard", "Uncalibrated"})
        math_tests = [test for test in full if test["category"] == "Mathematics"]
        self.assertIn("+", next(test["prompt"] for test in math_tests if test["difficulty"] == "Easy"))
        self.assertIn("×", next(test["prompt"] for test in math_tests if test["difficulty"] == "Medium"))
        self.assertIn("(", next(test["prompt"] for test in math_tests
                                if test["difficulty"] == "Uncalibrated"))
        self.assertTrue(all(test["difficulty_calibration_status"] ==
                            "not empirically calibrated" for test in quick))
        self.assertTrue(all(test["task_set_version"] == TASK_SET_VERSION and
                            test["evaluator_version"] == EVALUATOR_VERSION for test in quick))
        self.assertTrue(all(test["item_fingerprint"] and test["item_instance_id"]
                            for test in quick))
        self.assertNotEqual(make_tests("QUICK", 7)[0]["item_fingerprint"],
                            make_tests("QUICK", 8)[0]["item_fingerprint"])
        coding_prompts = [test["prompt"] for test in make_tests("FULL", 7)
                          if test["category"] == "Coding"]
        self.assertGreater(len(set(coding_prompts)), 1)
        pilot_tests = [test for test in full if test["category"] in
                       ("Reasoning", "Coding", "Summarization")]
        self.assertFalse(any("Use independent test variant" in test["prompt"]
                             for test in pilot_tests))
        self.assertEqual([x["context_size"] for x in make_context_tests(7)],
                         [4096, 8192, 16384, 32768, 65536, 131072])
        self.assertEqual([x["context_size"] for x in make_context_tests(7, 32768)],
                         [4096, 8192, 16384, 32768])
        context_tests = make_context_tests(7, 32768)
        self.assertEqual([item["context_position"] for item in context_tests],
                         ["beginning", "middle", "end", "beginning"])
        self.assertTrue(all(item["context_size"] * 0.99 <= item["estimated_prompt_tokens"]
                            <= item["context_size"] for item in context_tests))
        self.assertTrue(all(item["id"].endswith(item["context_position"])
                            for item in context_tests))
        knowledge = [item for seed in range(40) for item in make_tests("FULL", seed)
                     if item["category"] == "Knowledge"]
        self.assertGreaterEqual(len({item["subcategory"] for item in knowledge}), 4)
        self.assertGreaterEqual(len({item["prompt"] for item in knowledge}), 8)

    def test_pilot_generators_vary_constraints_scenarios_and_edge_cases(self):
        reasoning_items = [item for seed in range(40) for item in make_tests("FULL", seed)
                           if item["category"] == "Reasoning" and
                           item["difficulty"] == "Uncalibrated"]
        self.assertGreaterEqual(
            sum(not item["schedule_possible"] for item in reasoning_items), 5)
        self.assertGreaterEqual(
            len({tuple(rule["kind"] for rule in item["schedule_constraints"])
                 for item in reasoning_items}), 3)

        summaries = [item for seed in range(40) for item in make_tests("FULL", seed)
                     if item["category"] == "Summarization"]
        self.assertGreaterEqual(
            len({item["prompt"].split("Incident note:\n", 1)[1].split(". ", 1)[1]
                 for item in summaries}), 2)
        manager = BenchmarkManager(Path("."))
        for item in summaries:
            correct = json.loads(manager._answer({"demo": True}, item))
            self.assertTrue(evaluate(item, json.dumps(correct)))

        coding_items = [item for seed in range(40) for item in make_tests("FULL", seed)
                        if item["category"] == "Coding"]
        self.assertEqual({item["subcategory"] for item in coding_items},
                         {"Inventory adjustments", "Inventory event reconciliation"})
        ledger = next(item for item in coding_items
                      if item["subcategory"] == "Inventory event reconciliation" and
                      item["difficulty"] == "Uncalibrated")
        self.assertTrue(any("None, [], {}" in check for check in ledger["tests"]))
        self.assertTrue(any("' same '" in check and "' PEN '" in check
                            for check in ledger["tests"]))
        self.assertIn("event_id = event_id.strip()", ledger["demo_answer"])
        self.assertIn("sku = sku.strip()", ledger["demo_answer"])
        adjustment = next(item for item in coding_items
                          if item["subcategory"] == "Inventory adjustments" and
                          item["difficulty"] == "Uncalibrated")
        self.assertTrue(any("stock == {'PEN': 4}" in check for check in adjustment["tests"]))
        self.assertGreaterEqual(len(adjustment["tests"]), 6)

        long_context = [item for item in make_tests("FULL", 37)
                        if item["category"] == "Long Context"]
        self.assertEqual({item["difficulty"] for item in long_context},
                         {"Easy", "Medium", "Hard", "Uncalibrated"})
        self.assertTrue(all("archive records" in item["prompt"].lower() for item in long_context))
        self.assertTrue(all(evaluate(item, item["expected_answer"]) for item in long_context))

    def test_original_pilot_evaluators_explain_partial_and_full_results(self):
        tests = make_tests("FULL", 123)
        reasoning = next(test for test in tests
                         if test["category"] == "Reasoning" and
                         test["difficulty"] == "Uncalibrated")
        summarization = next(test for test in tests
                             if test["category"] == "Summarization" and
                             test["difficulty"] == "Uncalibrated")
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            route_answer = manager._answer({"demo": True}, reasoning)
            summary_answer = manager._answer({"demo": True}, summarization)
        route_result = evaluate_with_details(reasoning, route_answer)
        self.assertTrue(route_result["passed"])
        self.assertEqual(route_result["score"], 10)
        self.assertGreater(len(route_result["checks"]), 2)
        wrong_route_decision = json.loads(route_answer)
        wrong_route_decision["possible"] = not wrong_route_decision["possible"]
        partial_route = evaluate_with_details(reasoning, json.dumps(wrong_route_decision))
        self.assertFalse(partial_route["passed"])
        self.assertGreater(partial_route["score"], 0)
        self.assertLess(partial_route["score"], 10)

        summary_result = evaluate_with_details(summarization, summary_answer)
        self.assertTrue(summary_result["passed"])
        self.assertEqual(summary_result["score"], 10)
        incorrect_summary = json.loads(summary_answer)
        incorrect_summary["cause"] = "the network failed"
        partial_summary = evaluate_with_details(summarization, json.dumps(incorrect_summary))
        self.assertFalse(partial_summary["passed"])
        self.assertGreater(partial_summary["score"], 0)
        self.assertLess(partial_summary["score"], 10)
        self.assertTrue(any(not check["passed"] and check["name"] == "cause"
                            for check in partial_summary["checks"]))
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            with patch.object(manager, "_answer", return_value=json.dumps(incorrect_summary)):
                recorded = manager._execute({"demo": False}, summarization)
        self.assertEqual(recorded["status"], "FAILED")
        self.assertEqual(recorded["score"], partial_summary["score"])
        self.assertEqual(recorded["evaluation"]["checks"], partial_summary["checks"])

        coding = next(test for test in tests
                      if test["category"] == "Coding" and
                      test["difficulty"] == "Uncalibrated")
        self.assertIn("first valid record", coding["prompt"].lower())
        self.assertGreaterEqual(len(coding["tests"]), 5)

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
            self.assertEqual(data["summary"]["passed"], 25)
            self.assertEqual(data["summary"]["failed"], 0)
            self.assertEqual(data["summary"]["not_run"], 0)
            self.assertTrue(data["demo"])
            self.assertEqual(data["configuration"]["generation_options"]["temperature"], 0.6)
            self.assertEqual(data["task_set_version"], TASK_SET_VERSION)
            self.assertEqual(data["evaluator_version"], EVALUATOR_VERSION)
            self.assertEqual(data["difficulty_calibration_status"], "not empirically calibrated")
            self.assertTrue(data["configuration"]["context_reset_per_test"])
            self.assertTrue(all(item["context_reset"] for item in data["tests"]))
            self.assertIn("fresh", data["context_scaling"])
            self.assertIn("accumulated", data["context_scaling"])
            self.assertEqual(data["performance"]["measured_runs"], 5)
            self.assertTrue(data["performance"]["token_count_is_estimated"])
            self.assertIn("output tokens", data["performance"]["throughput_basis"])
            self.assertEqual(data["uncertainty"]["method"],
                             "bootstrap resampling of primary test items")
            self.assertNotIn("independent_seed_scores", data["uncertainty"])
            self.assertEqual(len(data["uncertainty"]["bootstrap_resample_scores"]), 3)
            self.assertEqual(manager.compare([data["id"]])["best_overall"], data["id"])
            self.assertEqual(manager.status()["state"], "complete")
            self.assertEqual(manager.status()["phase"], "Benchmark complete")
            self.assertEqual(manager.status()["result_directory"], str(Path(directory).resolve()))
            self.assertTrue(all(item["evaluation"]["criteria"] for item in data["tests"]))
            self.assertTrue(all(item["evaluation"]["explanation"] for item in data["tests"]))
            self.assertTrue(all(item["difficulty_rationale"] for item in data["tests"]))
            csv_header = (Path(directory) / f"{data['id']}.csv").read_text().splitlines()[0]
            self.assertIn("prompt", csv_header)
            self.assertIn("subcategory", csv_header)
            self.assertIn("item_fingerprint", csv_header)

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
            self.assertEqual(result["uncertainty"]["method"], "observed repeated runs")
            self.assertEqual(result["uncertainty"]["bootstrap_resample_scores"], [])

    def test_performance_metrics_use_sampled_generation_time_and_ttft(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            sample = {"prompt": "x", "id": "test", "evaluation_method": "raw",
                      "expected_answer": "x", "category": "Mathematics"}
            measurement = lambda latency, ttft: {
                "latency_s": latency, "ttft_s": ttft, "generation_s": 1.0,
                "output_tokens": 5, "timing_is_estimated": False,
            }
            with patch.object(manager, "_answer", return_value={"text": "x"}), \
                    patch.object(manager, "_execute", side_effect=[
                        measurement(0.5, 0.2), measurement(0.7, 0.3)]):
                result = manager._performance({"demo": False}, [sample, sample])
        self.assertEqual(result["tokens_per_second"], 5.0)
        self.assertEqual(result["score"], 50.0)
        self.assertEqual(result["ttft_s"], 0.25)
        self.assertEqual(result["generation_s"], 1.0)
        self.assertEqual(result["latency_mean_s"], 0.6)
        self.assertFalse(result["timing_is_estimated"])
        self.assertEqual([item["workers"] for item in result["concurrency_results"]], [1, 2, 4])
        self.assertTrue(all(item["successful_requests"] == 2
                            for item in result["concurrency_results"]))

    def test_unavailable_performance_samples_produce_no_score(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            sample = {"prompt": "x", "id": "test", "evaluation_method": "raw",
                      "expected_answer": "x", "category": "Mathematics"}
            with patch.object(manager, "_answer", return_value={"text": "x"}), \
                    patch.object(manager, "_execute", return_value={
                        "status": "NOT RUN", "latency_s": 0, "generation_s": 0,
                        "ttft_s": 0, "output_tokens": 0, "timing_is_estimated": True}):
                result = manager._performance({"demo": False}, [sample])
        self.assertIsNone(result["score"])
        self.assertIsNone(result["tokens_per_second"])
        self.assertEqual(result["not_run_samples"], 1)

    def test_status_returns_a_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            snapshot = manager.status()
            snapshot["state"] = "corrupted"
            self.assertEqual(manager.status()["state"], "idle")

    def test_resume_refuses_a_checkpoint_from_another_task_or_evaluator_version(self):
            with tempfile.TemporaryDirectory() as directory:
                manager = BenchmarkManager(Path(directory))
                manager.state_file.write_text(json.dumps({
                    "task_set_version": "older-task-set",
                    "evaluator_version": EVALUATOR_VERSION,
                    "profile": "QUICK",
                }))
                self.assertFalse(manager.resume())
                self.assertIn("version changed", manager.status()["activity"])

    def test_cancelled_context_phase_checkpoints_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            payload = {"profile": "QUICK", "model": {
                "id": "demo-local", "name": "Demo", "backend": "Demo",
                "endpoint": "local://demo", "demo": True}}
            execute = manager._execute
            cancelled = False

            def cancel_after_first_fresh(model, test, history=None, generation_options=None):
                nonlocal cancelled
                result = execute(model, test, history, generation_options)
                if "context_size" in test and history is None and not cancelled:
                    cancelled = True
                    manager.cancel_requested.set()
                return result

            with patch.object(manager, "_execute", side_effect=cancel_after_first_fresh), \
                    patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                    patch("backend.evaluation.subprocess.run",
                          return_value=CompletedProcess(args=[], returncode=0)), \
                    patch("backend.benchmark.hardware_info", return_value={}):
                self.assertTrue(manager.run(payload))
                self.assertEqual(manager.status()["state"], "cancelled")
                state = json.loads(manager.state_file.read_text())
                self.assertEqual(state["resume_phase"], "fresh_context")
                self.assertEqual(len(state["context_results"]), 1)
                self.assertTrue(manager.run(state))
            self.assertEqual(manager.status()["state"], "complete")

    def test_cancelled_performance_phase_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            payload = {"profile": "QUICK", "model": {
                "id": "demo-local", "name": "Demo", "backend": "Demo",
                "endpoint": "local://demo", "demo": True}}

            performance = manager._performance
            stopped = False

            def stop_first_performance(*args, **kwargs):
                nonlocal stopped
                if not stopped:
                    stopped = True
                    manager.cancel_requested.set()
                    return None
                return performance(*args, **kwargs)

            with patch.object(manager, "_performance", side_effect=stop_first_performance), \
                    patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                    patch("backend.evaluation.subprocess.run",
                          return_value=CompletedProcess(args=[], returncode=0)), \
                    patch("backend.benchmark.hardware_info", return_value={}):
                self.assertTrue(manager.run(payload))
                self.assertEqual(manager.status()["state"], "cancelled")
                state = json.loads(manager.state_file.read_text())
                self.assertEqual(state["resume_phase"], "performance")
                self.assertEqual(len(state["context_results"]), len(state["accumulated_results"]))
                self.assertTrue(manager.run(state))
            self.assertEqual(manager.status()["state"], "complete")

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

    def test_reports_label_unscored_tests_and_render_null_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chart = root / "chart.svg"
            write_chart(chart, {"Coding": None, "Mathematics": 100, "overall": None})
            self.assertIn("N/T", chart.read_text())
            pdf = root / "result.pdf"
            write_pdf(pdf, {
                "model": {"name": "test"}, "timestamp": "now",
                "scores": {"quality": None, "performance": None, "overall": None},
                "tests": [{"status": "NOT RUN"}],
                "failures": [],
            })
            self.assertIn(b"0 passed, 0 failed, 1 not run", pdf.read_bytes())
            self.assertIn(b"Not available", pdf.read_bytes())

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

    def test_result_listing_skips_corrupt_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.json").write_text("{")
            manager = BenchmarkManager(root)
            self.assertEqual(manager.list_results(), [])
            self.assertIsNone(manager.load_result("broken"))


if __name__ == "__main__":
    unittest.main()
