import csv
import copy
import json
import os
import urllib.error
import threading
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from unittest.mock import patch

from backend.benchmark import (
    EVALUATOR_VERSION, TASK_SET_VERSION, BenchmarkManager, make_context_tests, make_tests,
)
from backend.discovery import (
    AdapterError, _get, _llama_runtime_details, ask_model, discover_models,
    model_info, normalize_generation_options, resolve_model,
)
from backend.evaluation import DOCKER_PYTHON_IMAGE, evaluate, evaluate_with_details
from backend.hardware import GIB, recommend_settings
from backend.reporting import write_chart, write_pdf, write_html


class BenchmarkTests(unittest.TestCase):
    def test_checkpoint_replace_retries_transient_access_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            manager.state_file.write_text('{"previous": true}')
            real_replace = os.replace
            attempts = 0

            def replace_with_transient_denial(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError(13, "Access is denied", str(destination))
                return real_replace(source, destination)

            with patch("backend.benchmark.os.replace", side_effect=replace_with_transient_denial), \
                    patch("backend.benchmark.time.sleep"):
                manager._atomic_state_write({"saved": True})

            self.assertEqual(attempts, 2)
            saved = json.loads(manager.state_file.read_text())
            self.assertTrue(saved["saved"])
            self.assertRegex(saved["_integrity"], r"^[a-f0-9]{64}$")
            self.assertEqual(list(Path(directory).glob("benchmark-*.tmp")), [])

    def test_checkpoint_replace_preserves_previous_state_after_access_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            manager.state_file.write_text('{"previous": true}')
            with patch("backend.benchmark.os.replace",
                       side_effect=PermissionError(13, "Access is denied")), \
                    patch("backend.benchmark.time.sleep"):
                with self.assertRaises(PermissionError):
                    manager._atomic_state_write({"saved": True})

            self.assertEqual(json.loads(manager.state_file.read_text()), {"previous": True})
            self.assertEqual(list(Path(directory).glob("benchmark-*.tmp")), [])

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
            "repeat_penalty": 1.1, "max_tokens": 5000, "seed": 7,
        })

    def test_generation_options_reach_ollama_adapter(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self):
                return iter([b'{"message":{"content":"ok"},"done":true}\n'])

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
        self.assertEqual(captured["messages"], [{"role": "user", "content": "prompt"}])

    def test_ollama_and_openai_adapters_send_accumulated_history(self):
        history = [
            {"role": "user", "content": "remember this marker"},
            {"role": "assistant", "content": "BLUE-42"},
        ]
        for backend, endpoint, response_body in (
                ("Ollama", "http://localhost:11434",
                 b'{"message":{"content":"BLUE-42"},"done":true}\n'),
                ("OpenAI-compatible", "http://localhost:8080/v1",
                 b'data: {"choices":[{"delta":{"content":"BLUE-42"}}]}\n'
                 b'data: [DONE]\n')):
            with self.subTest(backend=backend):
                class Response:
                    def __enter__(self): return self
                    def __exit__(self, *_): return False
                    def __iter__(self): return iter(response_body.splitlines(keepends=True))

                captured = {}

                def capture(request, timeout):
                    captured.update(json.loads(request.data))
                    return Response()

                with patch("backend.discovery._open", side_effect=capture):
                    answer = ask_model({"backend": backend, "endpoint": endpoint, "id": "model"},
                                       "What marker?", history=history)
                self.assertEqual(answer["text"], "BLUE-42")
                self.assertEqual(captured["messages"], history + [
                    {"role": "user", "content": "What marker?"}])

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

    def test_openai_stream_handles_fragmented_events_and_finish_reason(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self):
                payload = (
                    'data: {"choices":[{"delta":{"content":"café"}}]}\n\n'
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                ).encode()
                return iter([payload[:43], payload[43:46], payload[46:]])

        with patch("backend.discovery._open", return_value=Response()):
            result = ask_model({"backend": "OpenAI-compatible", "endpoint": "http://localhost:8080/v1",
                                "id": "model"}, "prompt")
        self.assertEqual(result["text"], "café")
        self.assertEqual(result["finish_reason"], "stop")

    def test_empty_and_incomplete_streams_are_not_successful_answers(self):
        model = {"backend": "OpenAI-compatible", "endpoint": "http://localhost:8080/v1",
                 "id": "model"}

        class Response:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self): return iter(self.body)

        for body, expected_code in (
                ([b"data: [DONE]\n\n"], "EMPTY_RESPONSE"),
                ([b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'],
                 "INCOMPLETE_STREAM")):
            with self.subTest(expected_code=expected_code), \
                    patch("backend.discovery._open", return_value=Response(body)):
                with self.assertRaises(AdapterError) as caught:
                    ask_model(model, "prompt")
                self.assertEqual(caught.exception.code, expected_code)

        with patch("backend.discovery._open", return_value=Response([
                b'{"response":"","done":true}\n'])):
            with self.assertRaises(AdapterError) as caught:
                ask_model({"backend": "Ollama", "endpoint": "http://localhost:11434",
                           "id": "model"}, "prompt")
        self.assertEqual(caught.exception.code, "EMPTY_RESPONSE")

    def test_adapter_classifies_stream_disconnect_and_http_status(self):
        model = {"backend": "OpenAI-compatible", "endpoint": "http://localhost:8080/v1",
                 "id": "model"}

        class DisconnectedResponse:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"part"}}]}\n\n'
                raise ConnectionResetError("socket reset")

        with patch("backend.discovery._open", return_value=DisconnectedResponse()):
            with self.assertRaises(AdapterError) as caught:
                ask_model(model, "prompt")
        self.assertEqual(caught.exception.code, "CONNECTION_ERROR")
        self.assertNotIn("socket reset", caught.exception.diagnostic)

        with patch("backend.discovery._open", side_effect=urllib.error.HTTPError(
                "http://localhost:8080/v1/chat/completions", 503, "unavailable", {}, None)):
            with self.assertRaises(AdapterError) as caught:
                ask_model(model, "prompt")
        self.assertEqual(caught.exception.code, "HTTP_ERROR")
        self.assertIn("503", caught.exception.diagnostic)

    def test_adapter_errors_are_unscored_not_model_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            model = {"backend": "OpenAI-compatible", "endpoint": "http://localhost:8080/v1",
                     "id": "model"}
            test = {"id": "sample", "prompt": "Return ok", "evaluation_method": "exact_any",
                    "acceptable_answers": ["ok"], "expected_answer": "ok"}
            with patch("backend.discovery.ask_model", side_effect=AdapterError(
                    "EMPTY_RESPONSE", "The model server completed the request without generated text.")):
                result = manager._execute(model, test)

        self.assertEqual(result["status"], "NOT RUN")
        self.assertIsNone(result["score"])
        self.assertEqual(result["error"], "EMPTY_RESPONSE")
        self.assertEqual(len(result["attempts"]), 3)
        self.assertEqual(result["attempts"][0]["status"], "ERROR")
        self.assertIn("without generated text", result["diagnostic"])

    def test_result_comparison_warns_about_incompatible_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            base = {
                "id": "first", "model": {"name": "model-a"},
                "scores": {"quality": 70, "performance": 50, "overall": 64},
                "task_set_version": "pilot-1", "evaluator_version": "pilot-1",
                "demo": False,
                "configuration": {"profile": "QUICK", "repeat_runs": 1,
                                  "generation_options": {"temperature": 0}},
            }
            other = {
                **base, "id": "second", "model": {"name": "model-b"},
                "scores": {"quality": None, "performance": 42, "overall": None},
                "task_set_version": "pilot-2",
                "configuration": {"profile": "STANDARD", "repeat_runs": 2,
                                  "generation_options": {"temperature": 0.5}},
            }
            for result in (base, other):
                (Path(directory) / f"{result['id']}.json").write_text(json.dumps(result))

            comparison = manager.compare(["first", "second"])

        self.assertEqual(comparison["best_overall"], "first")
        self.assertTrue(any("task set" in warning for warning in comparison["comparison_warnings"]))
        self.assertTrue(any("unavailable score" in warning
                            for warning in comparison["comparison_warnings"]))

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

    def test_openai_adapter_omits_llama_cpp_only_sampler_options(self):
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
        command = run.call_args_list[1].args[0]
        self.assertIn("--network=none", command)
        self.assertIn("--cap-drop=ALL", command)
        self.assertIn("--read-only", command)
        self.assertIn(DOCKER_PYTHON_IMAGE, command)
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
        self.assertFalse(evaluate(schema, '{"answer":"ok","valid":true,"extra":NaN}'))
        self.assertFalse(evaluate(schema, '{"answer":"ok","valid":true,"extra":Infinity}'))
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
        self.assertEqual(result["score"], 5)
        self.assertFalse(next(check for check in result["checks"]
                              if check["name"] == "Topical relevance proxy")["passed"])
        self.assertFalse(next(check for check in result["checks"]
                              if check["name"] == "Not a keyword-only list")["passed"])
        self.assertFalse(evaluate(rubric, "rain wet drops"))
        relevant = evaluate_with_details(rubric, "rain and wet")
        self.assertTrue(relevant["passed"])
        self.assertEqual(relevant["score"], 10)

    def test_fingerprints_are_unique_and_duplicate_digests_fail_closed(self):
        tests = make_tests("QUICK", 19)
        self.assertEqual(len({test["item_fingerprint"] for test in tests}), len(tests))

        class DuplicateDigest:
            def hexdigest(self):
                return "same"

        with patch("backend.benchmark.hashlib.sha256", return_value=DuplicateDigest()):
            with self.assertRaisesRegex(ValueError, "duplicate generated test fingerprint"):
                make_tests("QUICK", 19)

    def test_quality_is_equal_weighted_across_available_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            result = manager._scores([
                {"category": "Mathematics", "score": 10},
                {"category": "Mathematics", "score": 10},
                {"category": "Coding", "score": 0},
            ])
        self.assertEqual(result["categories"]["Mathematics"], 100)
        self.assertEqual(result["categories"]["Coding"], 0)
        self.assertEqual(result["overall"], 50)
        self.assertEqual(result["method"], "equal-weighted mean of available category scores")

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

    def test_code_evaluation_requires_pinned_image_and_available_daemon(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            test = {"id": "coding-1", "category": "Coding", "prompt": "define f",
                    "evaluation_method": "code", "tests": ["assert f() == 1"]}
            with patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                    patch("backend.evaluation.subprocess.run",
                          return_value=CompletedProcess(args=[], returncode=1)) as run:
                result = manager._execute({"demo": False}, test)
        self.assertEqual(result["status"], "NOT RUN")
        self.assertIsNone(result["score"])
        self.assertEqual(result["error"], "EVALUATOR_UNAVAILABLE")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0],
                         ["/usr/bin/docker", "image", "inspect", DOCKER_PYTHON_IMAGE])

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

    def test_code_evaluation_rejects_invalid_or_oversized_tests(self):
        with patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                patch("backend.evaluation.subprocess.run") as run:
            self.assertFalse(evaluate({"evaluation_method": "code", "tests": ["x" * 2001]}, "pass"))
            self.assertFalse(evaluate({"evaluation_method": "code", "tests": "assert True"}, "pass"))
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(call.args[0][1:3] == ["image", "inspect"]
                            for call in run.call_args_list))

    def test_docker_timeout_and_failure_force_remove_named_container(self):
        with patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                patch("backend.evaluation.subprocess.run", side_effect=[
                    CompletedProcess(args=[], returncode=0),
                    TimeoutExpired("docker", 3),
                    CompletedProcess(args=[], returncode=0),
                ]) as run:
            self.assertFalse(evaluate(
                {"evaluation_method": "code", "tests": ["assert True"]}, "pass"))
        cleanup = run.call_args_list[2].args[0]
        self.assertEqual(cleanup[:3], ["/usr/bin/docker", "rm", "--force"])
        container_name = cleanup[3]
        self.assertTrue(container_name.startswith("ai-benchmark-evaluator-"))
        launch = run.call_args_list[1].args[0]
        self.assertIn("--name", launch)
        self.assertEqual(launch[launch.index("--name") + 1], container_name)

        with patch("backend.evaluation.shutil.which", return_value="/usr/bin/docker"), \
                patch("backend.evaluation.subprocess.run", side_effect=[
                    CompletedProcess(args=[], returncode=0),
                    CompletedProcess(args=[], returncode=1),
                    CompletedProcess(args=[], returncode=0),
                ]) as run:
            self.assertFalse(evaluate(
                {"evaluation_method": "code", "tests": ["assert True"]}, "pass"))
        self.assertEqual(run.call_count, 3)

    def test_deterministic_test_generation(self):
        self.assertEqual(make_tests("QUICK", 7), make_tests("QUICK", 7))
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

    def test_hardware_context_recommendations_leave_a_probe_window(self):
        for memory in (16, 32, 64):
            with self.subTest(memory_gib=memory):
                recommendation = recommend_settings({"ram_bytes": memory * GIB})
                tokens = recommendation["generation_options"]["max_tokens"]
                context = recommendation["context_size"]
                probes = make_context_tests(7, context, output_budget=tokens + 128)
                self.assertTrue(probes)
                self.assertLessEqual(probes[0]["context_size"], context)

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
            files = list(Path(directory).glob("*/benchmark_*"))
            self.assertTrue(any(p.suffix == ".json" for p in files))
            self.assertTrue(any(p.suffix == ".csv" for p in files))
            self.assertTrue(any(p.suffix == ".pdf" for p in files))
            self.assertTrue(any(p.suffix == ".html" for p in files))
            data = json.loads(next(p for p in files if p.suffix == ".json").read_text())
            self.assertEqual(data["scores"]["quality"], 100)
            self.assertEqual(data["summary"]["passed"], 25)
            self.assertEqual(data["summary"]["failed"], 0)
            self.assertEqual(data["summary"]["not_run"], 0)
            self.assertEqual(data["summary"]["not_run_by_code"], {})
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
            self.assertIsNone(data["scores"]["performance"])
            self.assertIsNone(data["scores"]["overall"])
            self.assertIsNone(manager.compare([data["id"]])["best_overall"])
            self.assertEqual(manager.status()["state"], "complete")
            self.assertEqual(manager.status()["phase"], "Benchmark complete")
            self.assertEqual(manager.status()["result_directory"], Path(directory).name)
            self.assertTrue(all(item["evaluation"]["criteria"] for item in data["tests"]))
            self.assertTrue(all(item["evaluation"]["explanation"] for item in data["tests"]))
            self.assertTrue(all(item["difficulty_rationale"] for item in data["tests"]))
            csv_header = (Path(directory) / data["files"]["csv"]).read_text().splitlines()[0]
            self.assertIn("prompt", csv_header)
            self.assertIn("run_type", csv_header)
            self.assertIn("subcategory", csv_header)
            self.assertIn("item_fingerprint", csv_header)
            self.assertIn("failure_code", csv_header)
            pdf_text = (Path(directory) / data["files"]["pdf"]).read_bytes()
            self.assertIn(b"Not-run reasons: none recorded", pdf_text)
            self.assertIn(b"Run type: DEMO DATA", pdf_text)
            with (Path(directory) / data["files"]["csv"]).open(newline="") as stream:
                self.assertTrue(all(row["run_type"] == "DEMO DATA"
                                    for row in csv.DictReader(stream)))
            chart = (Path(directory) / data["files"]["quality_chart"]).read_text()
            self.assertIn("<svg", chart)

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
            result_path = next(Path(directory).glob("*/benchmark_*.json"))
            result = json.loads(result_path.read_text())
            self.assertEqual(result["configuration"]["repeat_runs"], 2)
            self.assertEqual(result["uncertainty"]["repeat_run_scores"], [100.0, 100.0])
            self.assertEqual(result["uncertainty"]["quality_stddev"], 0.0)
            self.assertEqual([rep["seed"] for rep in result["tests"][0]["replicates"]], [31, 32])
            self.assertEqual(result["uncertainty"]["method"], "observed repeated runs")
            self.assertEqual(result["uncertainty"]["bootstrap_resample_scores"], [])

    def test_later_repeat_adapter_failure_is_summarized_and_retained(self):
        test = {"id": "sample", "category": "Mathematics", "difficulty": "Easy",
                "prompt": "Return 42", "evaluation_method": "exact_any",
                "acceptable_answers": ["42"], "expected_answer": "42"}
        success = {
            **test, "answer": "42", "evaluation": {"criteria": "exact", "explanation": "passed"},
            "passed": True, "score": 10, "status": "PASS", "error": None, "diagnostic": None,
            "latency_s": 0.1, "ttft_s": 0.1, "generation_s": 0.1,
            "prompt_tokens": 2, "output_tokens": 1, "attempts": [],
        }
        failure = {
            **test, "answer": "", "evaluation": {"criteria": "exact", "explanation": "not run"},
            "passed": None, "score": None, "status": "NOT RUN", "error": "TIMEOUT",
            "diagnostic": "The local model server timed out.",
            "latency_s": 0, "ttft_s": 0, "generation_s": 0,
            "prompt_tokens": 2, "output_tokens": 0,
            "attempts": [{"attempt": 1, "status": "ERROR", "error": "TIMEOUT",
                          "diagnostic": "The local model server timed out."}],
        }
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            with patch("backend.benchmark.make_tests", return_value=[test]), \
                    patch("backend.benchmark.make_context_tests", return_value=[]), \
                    patch.object(manager, "_execute", side_effect=[success, failure]), \
                    patch.object(manager, "_performance", return_value={"score": 0}), \
                    patch.object(manager, "_write_outputs") as write_outputs:
                manager.run({"profile": "QUICK", "repeat_runs": 2,
                             "model": {"id": "demo-local", "name": "Demo", "backend": "Demo",
                                       "endpoint": "local://demo", "demo": True}})

        result = write_outputs.call_args.args[0]
        replicate = result["tests"][0]["replicates"][1]
        self.assertEqual(result["summary"]["not_run_by_code"], {"TIMEOUT": 1})
        self.assertEqual(result["tests"][0]["status"], "PASS")
        self.assertEqual(result["tests"][0]["score"], 10)
        self.assertEqual(result["tests"][0]["missing_repeat_count"], 1)
        self.assertEqual(result["summary"]["repeat_outcomes"]["scored"], 1)
        self.assertEqual(result["summary"]["repeat_outcomes"]["not_run"], 1)
        self.assertEqual(replicate["error"], "TIMEOUT")
        self.assertEqual(replicate["diagnostic"], "The local model server timed out.")
        self.assertEqual(replicate["attempts"][0]["error"], "TIMEOUT")

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

    def test_status_suppresses_exception_paths_and_uses_relative_result_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            with patch.object(manager, "_run", side_effect=ValueError(
                    f"failed to read {Path(directory) / 'private.txt'}")):
                self.assertFalse(manager.run({}))
            serialized = json.dumps(manager.status())
            self.assertNotIn(str(Path(directory).resolve()), serialized)
            self.assertNotIn("private.txt", serialized)
            self.assertNotIn("ValueError", serialized)
            self.assertEqual(manager.status()["result_directory"], Path(directory).name)

    def test_resume_refuses_a_checkpoint_from_another_task_or_evaluator_version(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            manager._atomic_state_write({
                "task_set_version": "older-task-set",
                "evaluator_version": EVALUATOR_VERSION,
            })
            self.assertFalse(manager.resume())
            self.assertIn("invalid or has been modified", manager.status()["activity"])

    @staticmethod
    def _write_valid_checkpoint(manager):
        profile, seed = "QUICK", 12345
        manager._atomic_state_write({
            "model": {"id": "demo-local", "name": "Demo Local Model", "backend": "Demo",
                      "endpoint": "local://demo", "demo": True},
            "profile": profile, "seed": seed,
            "generation_options": normalize_generation_options(),
            "repeat_runs": 1, "tests": make_tests(profile, seed),
            "task_set_version": TASK_SET_VERSION, "evaluator_version": EVALUATOR_VERSION,
            "resume_phase": "primary", "completed_tests": [], "failures": [],
            "not_run": [], "warmup_errors": [], "context_results": [],
            "accumulated_results": [],
        })

    def test_resume_rejects_tampered_and_non_object_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            self._write_valid_checkpoint(manager)
            state = json.loads(manager.state_file.read_text())
            state["completed_tests"] = [{"id": "forged", "score": 10, "status": "PASS"}]
            manager.state_file.write_text(json.dumps(state))
            with patch.object(manager, "_start_locked") as start:
                self.assertFalse(manager.resume())
            start.assert_not_called()
            self.assertIn("invalid or has been modified", manager.status()["activity"])

            manager.state_file.write_text("[]")
            self.assertFalse(manager.resume())
            self.assertIn("invalid or has been modified", manager.status()["activity"])

    def test_resume_validates_checkpoint_before_starting(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            self._write_valid_checkpoint(manager)
            with patch.object(manager, "_start_locked", return_value=True) as start:
                self.assertTrue(manager.resume())
            payload = start.call_args.args[0]
            self.assertNotIn("_integrity", payload)
            self.assertEqual(payload["tests"], make_tests("QUICK", 12345))

    def test_resume_accepts_accumulated_warmup_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            self._write_valid_checkpoint(manager)
            state = json.loads(manager.state_file.read_text())
            state.pop("_integrity")
            state["warmup_errors"] = [{"attempt": attempt} for attempt in range(6)]
            manager._validate_checkpoint(state)

    def test_resume_rejects_signed_checkpoint_with_unknown_completed_test(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            tests = make_tests("QUICK", 12345)
            forged = {**tests[0], "id": "forged", "score": 10,
                      "passed": True, "status": "PASS"}
            manager._atomic_state_write({
                "model": {"id": "demo-local", "backend": "Demo", "endpoint": "local://demo"},
                "profile": "QUICK", "seed": 12345,
                "generation_options": normalize_generation_options(), "repeat_runs": 1,
                "tests": tests, "task_set_version": TASK_SET_VERSION,
                "evaluator_version": EVALUATOR_VERSION, "resume_phase": "primary",
                "completed_tests": [forged], "failures": [], "not_run": [],
                "warmup_errors": [], "context_results": [], "accumulated_results": [],
            })
            self.assertFalse(manager.resume())
            self.assertIn("invalid or has been modified", manager.status()["activity"])

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

    def test_cancellation_checkpoints_warmup_primary_and_accumulated_phases(self):
        demo = {"id": "demo-local", "name": "Demo", "backend": "Demo",
                "endpoint": "local://demo", "demo": True}
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))

            def cancel_warmup(model, test, history=None, generation_options=None):
                if test["prompt"] == "warm up":
                    manager.cancel_requested.set()
                    return ""
                return ""

            with patch.object(manager, "_answer", side_effect=cancel_warmup):
                self.assertTrue(manager.run({"profile": "QUICK", "model": demo}))
            self.assertEqual(json.loads(manager.state_file.read_text())["resume_phase"], "warmup")

        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            tests = make_tests("QUICK", 12345)[:2]

            def cancel_primary(model, test, history=None, generation_options=None):
                result = {**test, "answer": "ok", "passed": True, "score": 10,
                          "status": "PASS", "attempts": [], "error": None,
                          "latency_s": 0.1, "ttft_s": 0.1, "generation_s": 0.1,
                          "prompt_tokens": 1, "output_tokens": 1,
                          "timing_is_estimated": True, "evaluation": {
                              "passed": True, "status": "PASS", "criteria": "test",
                              "explanation": "passed"}}
                manager.cancel_requested.set()
                return result

            with patch("backend.benchmark.make_tests", return_value=tests), \
                    patch.object(manager, "_execute", side_effect=cancel_primary):
                self.assertTrue(manager.run({"profile": "QUICK", "model": demo}))
            self.assertEqual(json.loads(manager.state_file.read_text())["resume_phase"], "primary")

        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            context_tests = [
                {"id": "context-4096-beginning", "category": "Context Scaling",
                 "context_size": 4096, "context_position": "beginning", "prompt": "first",
                 "expected_answer": "CTX-1", "estimated_prompt_tokens": 4096,
                 "evaluation_method": "contains_all", "required_terms": ["CTX-1"], "max_score": 10},
                {"id": "context-8192-middle", "category": "Context Scaling",
                 "context_size": 8192, "context_position": "middle", "prompt": "second",
                 "expected_answer": "CTX-2", "estimated_prompt_tokens": 8192,
                 "evaluation_method": "contains_all", "required_terms": ["CTX-2"], "max_score": 10},
            ]

            def cancel_accumulated(model, test, history=None, generation_options=None):
                result = {**test, "answer": test["expected_answer"], "passed": True, "score": 10,
                          "status": "PASS", "attempts": [], "error": None,
                          "latency_s": 0.1, "ttft_s": 0.1, "generation_s": 0.1,
                          "prompt_tokens": 1, "output_tokens": 1,
                          "timing_is_estimated": True, "evaluation": {
                              "passed": True, "status": "PASS", "criteria": "test",
                              "explanation": "passed"}}
                if history is not None:
                    manager.cancel_requested.set()
                return result

            with patch("backend.benchmark.make_tests", return_value=[]), \
                    patch("backend.benchmark.make_context_tests", return_value=context_tests), \
                    patch.object(manager, "_execute", side_effect=cancel_accumulated):
                self.assertTrue(manager.run({"profile": "QUICK", "model": demo}))
            self.assertEqual(
                json.loads(manager.state_file.read_text())["resume_phase"], "accumulated_context")

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

    def test_start_resume_and_cancel_are_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            self._write_valid_checkpoint(manager)
            entered, release, finished = threading.Event(), threading.Event(), threading.Event()

            def block_run(payload):
                entered.set()
                release.wait(timeout=3)
                with manager._run_lock:
                    manager.running = False
                    manager._update_status(state="complete")
                finished.set()

            with patch.object(manager, "_run_guarded", side_effect=block_run):
                self.assertTrue(manager.start({"profile": "QUICK"}))
                self.assertTrue(entered.wait(timeout=2))
                self.assertFalse(manager.start({"profile": "QUICK"}))
                self.assertFalse(manager.resume())
                self.assertTrue(manager.cancel())
                release.set()
                self.assertTrue(finished.wait(timeout=3))
            self.assertFalse(manager.running)

    def test_custom_tests_are_ignored_and_run_state_recovers_after_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            self.assertTrue(manager.run({
                "profile": "QUICK", "tests": [{"id": "attack", "evaluation_method": "code",
                                               "tests": ["raise RuntimeError()"]}],
                "model": {"id": "demo-local", "name": "demo", "backend": "Demo",
                          "endpoint": "local://demo", "demo": True},
            }))
            result_path = next(Path(directory).glob("*/benchmark_*.json"))
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
                "tests": [{"id": "test", "category": "=1+1", "answer": "=2+2",
                           "difficulty": "Easy",
                           "score": 0, "status": "PASS", "latency_s": 0}],
                "failures": [], "quality": {"categories": {}},
            }
            manager._write_outputs(result)
            self.assertTrue((root / result["files"]["json"]).is_file())
            with (root / result["files"]["csv"]).open(newline="") as stream:
                csv_row = next(csv.DictReader(stream))
            self.assertEqual(csv_row["category"], "'=1+1")
            self.assertEqual(csv_row["answer"], "'=2+2")
            result["id"] = "../escaped"
            with self.assertRaises(ValueError):
                manager._write_outputs(result)

    def test_reports_label_unscored_tests_and_render_null_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chart = root / "chart.svg"
            write_chart(chart, {"<script>&": 100, "Coding": None, "overall": None})
            self.assertIn("N/T", chart.read_text())
            self.assertIn("&lt;script&gt;&amp;", chart.read_text())
            pdf = root / "result.pdf"
            write_pdf(pdf, {
                "model": {"name": "evil) Tj (\\\nInjected"}, "timestamp": "now", "demo": True,
                "scores": {"quality": None, "performance": None, "overall": None},
                "tests": [{"status": "NOT RUN"}],
                "failures": [],
            })
            self.assertIn(b"0 passed, 0 failed, 1 not run", pdf.read_bytes())
            self.assertIn(b"Not available", pdf.read_bytes())
            self.assertIn(b"DEMO DATA", pdf.read_bytes())
            self.assertIn(b"evil\\) Tj \\(\\\\ Injected", pdf.read_bytes())
            self.assertNotIn(b"evil) Tj (", pdf.read_bytes())

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

    def test_result_reads_use_safe_files_and_downloads_hide_checkpoint_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            root.mkdir()
            manager = BenchmarkManager(root)
            result_id = "benchmark_20261016_091700_" + "a" * 32
            (root / f"{result_id}.json").write_text('{"id":"safe"}')
            (root / f"{result_id}.csv").write_text("run_type\nMODEL DATA\n")
            manager._atomic_state_write({"resume_phase": "warmup"})
            self.assertEqual(manager.read_artifact(f"{result_id}.csv"), b"run_type\nMODEL DATA\n")
            self.assertIsNone(manager.read_artifact("current.json"))
            self.assertIsNone(manager.read_artifact(".checkpoint.key"))
            self.assertIsNone(manager.load_result("../current"))

            external = Path(directory) / "outside.csv"
            external.write_text("private")
            (root / f"{result_id}.pdf").symlink_to(external)
            self.assertIsNone(manager.read_artifact(f"{result_id}.pdf"))

    def test_result_listing_skips_corrupt_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.json").write_text("{")
            (root / "invalid-utf8.json").write_bytes(b"\xff")
            manager = BenchmarkManager(root)
            self.assertEqual(manager.list_results(), [])
            self.assertIsNone(manager.load_result("broken"))
            self.assertIsNone(manager.load_result("invalid-utf8"))

    @staticmethod
    def _export_fixture():
        return {
            "id": "benchmark_20261018_120000_" + "d" * 32,
            "model": {"name": "../../unsafe model <x>", "backend": "Ollama",
                      "id": "model/a", "endpoint": "http://localhost:11434"},
            "timestamp": "now", "scores": {"quality": 50, "performance": None, "overall": None},
            "tests": [], "quality": {"categories": {"Reasoning": 50, "Coding": None}},
        }

    def test_model_exports_are_grouped_collision_resistant_and_legacy_results_still_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = BenchmarkManager(root)
            result = self._export_fixture()
            manager._write_outputs(result)
            folder = result["model_folder"]
            self.assertEqual(result["artifact_directory"], folder)
            self.assertRegex(folder, r"^unsafe-model-x-[a-f0-9]{20}$")
            self.assertEqual(set(result["files"]), {"json", "csv", "pdf", "html", "quality_chart"})
            self.assertFalse((root / f"{result['id']}.json").exists())
            for filename in result["files"].values():
                self.assertTrue((root / filename).is_file())
                self.assertEqual(manager.read_artifact(filename), (root / filename).read_bytes())
            self.assertEqual(manager.load_result(result["id"])["files"], result["files"])
            other = copy.deepcopy(result)
            other["id"] = "benchmark_20261018_120001_" + "e" * 32
            other["model"]["id"] = "model/b"
            manager._write_outputs(other)
            self.assertNotEqual(folder, other["model_folder"])
            backend = copy.deepcopy(result)
            backend["model"]["backend"] = "OpenAI-compatible"
            manager._write_outputs(backend)
            self.assertNotEqual(folder, backend["model_folder"])
            (root / "legacy.json").write_text('{"id":"legacy","model":{"name":"Legacy"}}')
            self.assertEqual(manager.load_result("legacy")["model"]["name"], "Legacy")
            self.assertEqual({row["id"] for row in manager.list_results()},
                             {result["id"], other["id"], "legacy"})

    def test_nested_results_reject_traversal_and_directory_or_file_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            manager = BenchmarkManager(root)
            result = self._export_fixture()
            manager._write_outputs(result)
            filename = result["files"]["json"]
            external = Path(directory) / "external"
            external.mkdir()
            secret = external / Path(filename).name
            secret.write_text('{"id":"private","model":{"name":"private"}}')
            linked_folder = root / "linked-folder"
            linked_folder.symlink_to(external, target_is_directory=True)
            self.assertIsNone(manager.read_artifact(f"linked-folder/{secret.name}"))
            self.assertIsNone(manager._read_result_file(f"linked-folder/{secret.name}"))
            for bad in (f"../{secret.name}", f"{result['model_folder']}/../{secret.name}",
                        f"/{filename}", filename.replace("/", "\\"), f"./{filename}"):
                self.assertIsNone(manager.read_artifact(bad))
            (root / result["files"]["pdf"]).unlink()
            (root / result["files"]["pdf"]).symlink_to(secret)
            self.assertIsNone(manager.read_artifact(result["files"]["pdf"]))
            (root / filename).unlink()
            (root / filename).symlink_to(secret)
            self.assertIsNone(manager.load_result(result["id"]))
            self.assertEqual(manager.list_results(), [])
            for path in (root / result["model_folder"]).iterdir():
                path.unlink()
            (root / result["model_folder"]).rmdir()
            (root / result["model_folder"]).symlink_to(external, target_is_directory=True)
            with self.assertRaises((OSError, ValueError)):
                manager._write_outputs(result)
            self.assertEqual(secret.read_text(), '{"id":"private","model":{"name":"private"}}')
            self.assertEqual(len(list(external.iterdir())), 1)

    def test_shared_result_reader_rejects_unvalidated_components_and_checkpoint_names(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            manager.state_file.write_text('{"private":"checkpoint"}')
            for parts in (None, [], ["current.json"], ["..", "outside.json"],
                          ["nested/../outside.json"], ["/outside.json"], ["folder", "../outside.json"],
                          ["folder", "file.json", "extra"], ["folder", 1]):
                with self.subTest(parts=parts):
                    self.assertIsNone(manager._read_safe_file(parts))
            filename = "benchmark_20261018_120000_" + "a" * 32 + ".html"
            (Path(directory) / filename).symlink_to(filename)
            self.assertIsNone(manager.read_artifact(filename))

    def test_html_report_is_offline_and_model_output_is_not_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._export_fixture()
            attack = '</script><script>alert("model output")</script><img src=x onerror=alert(1)>'
            result["model"]["name"] = attack
            result["quality"]["categories"][attack] = 12
            result["tests"] = [{"id": attack, "prompt": attack, "answer": attack,
                                "status": "FAILED", "score": 0,
                                "evaluation": {"explanation": attack}}]
            path = Path(directory) / "report.html"
            write_html(path, result)
            document = path.read_text()
            self.assertIn('<canvas id="voxel-lab"', document)
            self.assertIn('Content-Security-Policy', document)
            self.assertIn("script-src 'sha256-", document)
            self.assertIn("&lt;script&gt;", document)
            self.assertIn("\\u003c/script\\u003e", document)
            self.assertNotIn(attack, document)
            self.assertNotIn('<script src=', document)
            self.assertNotIn('<link ', document)
            self.assertIn("Not tested", document)
            self.assertIn("Evaluation evidence", document)

    def test_answer_stream_callback_updates_status_and_preserves_backend_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            metrics = {"phase": "decode", "prefill_tokens_per_second": 250.0,
                       "decode_tokens_per_second": 15.0, "token_count_is_estimated": False,
                       "timing_source": "ollama"}
            response = {"text": "42", "prompt_tokens": 8, "output_tokens": 2,
                        "ttft_s": 0.1, "generation_s": 0.2, "timing_is_estimated": False,
                        **metrics}
            observed = []

            def adapter(*args, progress_callback, **kwargs):
                progress_callback(metrics)
                observed.append(manager.status()["live_metrics"])
                return response

            test = {"id": "math", "prompt": "Answer 42", "category": "Mathematics",
                    "evaluation_method": "numeric", "expected_answer": "42"}
            with patch("backend.discovery.ask_model", side_effect=adapter):
                outcome = manager._execute({"backend": "Ollama"}, test)
            self.assertEqual(observed[0]["decode_tokens_per_second"], 15.0)
            self.assertEqual(observed[0]["active_requests"], 1)
            self.assertFalse(outcome["token_count_is_estimated"])
            self.assertFalse(outcome["timing_is_estimated"])
            self.assertEqual(outcome["timing_source"], "ollama")
            self.assertEqual(outcome["output_tokens"], 2)
            self.assertEqual(manager.status()["live_metrics"]["active_requests"], 0)

    def test_concurrent_live_metrics_aggregate_safely_and_label_partial_estimates(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            barrier = threading.Barrier(4)
            snapshots = []

            def adapter(*args, progress_callback, **kwargs):
                progress_callback({
                    "phase": "decode", "prefill_tokens_per_second": 100,
                    "decode_tokens_per_second": 10, "token_count_is_estimated": True,
                    "timing_source": "client_estimate"})
                barrier.wait(timeout=3)
                snapshots.append(manager.status()["live_metrics"])
                barrier.wait(timeout=3)
                return {"text": "ok"}

            errors = []
            def worker():
                try:
                    manager._answer({}, {"prompt": "x"})
                except Exception as exc:
                    errors.append(exc)

            with patch("backend.discovery.ask_model", side_effect=adapter):
                threads = [threading.Thread(target=worker) for _ in range(4)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)
            self.assertEqual(errors, [])
            self.assertEqual(len(snapshots), 4)
            self.assertTrue(all(item["active_requests"] == 4 for item in snapshots))
            self.assertTrue(all(item["decode_tokens_per_second"] == 40 for item in snapshots))
            self.assertTrue(all(item["token_count_is_estimated"] for item in snapshots))
            self.assertEqual(manager.status()["live_metrics"]["active_requests"], 0)

    def test_completed_worker_rates_are_removed_while_remaining_worker_prefills(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            ready, release = threading.Event(), threading.Event()
            errors = []

            def adapter(model, prompt, progress_callback, **kwargs):
                if prompt == "waiting":
                    progress_callback({"phase": "prefill", "token_count_is_estimated": True,
                                       "timing_source": "client_estimate"})
                    ready.set()
                    if not release.wait(timeout=3):
                        raise TimeoutError("test worker was not released")
                    progress_callback({"phase": "decode", "decode_tokens_per_second": 5,
                                       "token_count_is_estimated": True,
                                       "timing_source": "client_estimate"})
                else:
                    progress_callback({
                        "phase": "decode", "prefill_tokens_per_second": 100,
                        "decode_tokens_per_second": 20, "token_count_is_estimated": False,
                        "timing_is_estimated": False, "timing_source": "ollama"})
                return {"text": "ok"}

            def waiting_worker():
                try:
                    manager._answer({}, {"prompt": "waiting"})
                except Exception as exc:
                    errors.append(exc)

            with patch("backend.discovery.ask_model", side_effect=adapter):
                thread = threading.Thread(target=waiting_worker)
                thread.start()
                try:
                    self.assertTrue(ready.wait(timeout=3))
                    manager._answer({}, {"prompt": "finishing"})
                    metrics = manager.status()["live_metrics"]
                    self.assertEqual(metrics["active_requests"], 1)
                    self.assertEqual(metrics["phase"], "prefill")
                    self.assertIsNone(metrics["decode_tokens_per_second"])
                    self.assertIsNone(metrics["prefill_tokens_per_second"])
                    self.assertEqual(metrics["timing_source"], "client_estimate")
                    self.assertTrue(metrics["token_count_is_estimated"])
                finally:
                    release.set()
                    thread.join(timeout=5)
            self.assertEqual(errors, [])
            self.assertFalse(thread.is_alive())
            completed = manager.status()["live_metrics"]
            self.assertEqual(completed["active_requests"], 0)
            self.assertEqual(completed["phase"], "complete")
            self.assertEqual(completed["decode_tokens_per_second"], 5)

    def test_output_budget_truncation_is_explained_even_if_partial_answer_passes(self):
        test = {"id": "math", "prompt": "42", "category": "Mathematics",
                "evaluation_method": "numeric", "expected_answer": "42"}
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            for answer, status in (("42", "PASS"), ("41", "FAILED")):
                with self.subTest(status=status), patch.object(manager, "_answer", return_value={
                        "text": answer, "finish_reason": "length", "output_tokens": 256}):
                    outcome = manager._execute({}, test)
                self.assertEqual(outcome["status"], status)
                self.assertTrue(outcome["output_truncated"])
                self.assertIn("output-token budget", outcome["diagnostic"])
                self.assertIn("max_tokens", outcome["evaluation"]["explanation"])
                self.assertIn("output-token budget", outcome["attempts"][0]["diagnostic"])

    def test_empty_response_after_reasoning_exhausts_budget_retains_truncation_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            error = AdapterError("EMPTY_RESPONSE", "No generated text.")
            error.truncated = True
            error.finish_reason = "length"
            test = {"id": "math", "prompt": "42", "category": "Mathematics",
                    "evaluation_method": "numeric", "expected_answer": "42"}
            with patch.object(manager, "_answer", side_effect=error):
                outcome = manager._execute({}, test)
            self.assertEqual(outcome["status"], "NOT RUN")
            self.assertTrue(outcome["output_truncated"])
            self.assertEqual(outcome["finish_reason"], "length")
            self.assertIn("max_tokens", outcome["diagnostic"])

    def test_performance_uses_reported_counters_without_calling_them_estimates(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            response = {"text": "42", "output_tokens": 24, "prompt_tokens": 10,
                        "ttft_s": 0.2, "generation_s": 99, "decode_s": 2,
                        "timing_is_estimated": False,
                        "token_count_is_estimated": False, "timing_source": "llama.cpp",
                        "prefill_tokens_per_second": 50}
            test = {"id": "math", "prompt": "42", "category": "Mathematics",
                    "evaluation_method": "numeric", "expected_answer": "42"}
            with patch.object(manager, "_answer", return_value=response):
                performance = manager._performance({}, [test])
            self.assertFalse(performance["token_count_is_estimated"])
            self.assertFalse(performance["timing_is_estimated"])
            self.assertEqual(performance["timing_source"], "llama.cpp")
            self.assertEqual(performance["prefill_tokens_per_second"], 50)
            self.assertEqual(performance["decode_tokens_per_second"], 12)
            self.assertIn("backend-reported", performance["throughput_basis"])
            for sample in performance["concurrency_results"]:
                self.assertFalse(sample["token_count_is_estimated"])
                self.assertIsNone(sample["estimated_output_tokens_per_second"])
                self.assertGreater(sample["output_tokens_per_second"], 0)

    def test_repeat_aggregation_does_not_hide_estimation_or_truncation_in_later_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            test = {"id": "math", "prompt": "42", "category": "Mathematics",
                    "evaluation_method": "numeric", "expected_answer": "42"}
            measured = {"text": "42", "prompt_tokens": 2, "output_tokens": 1,
                        "ttft_s": 0.1, "generation_s": 0.2, "token_count_is_estimated": False,
                        "timing_is_estimated": False, "timing_source": "ollama"}
            estimated = {**measured, "token_count_is_estimated": True,
                         "timing_is_estimated": True, "timing_source": "client_estimate",
                         "finish_reason": "length"}
            with patch.object(manager, "_answer", side_effect=[measured, estimated]):
                repetitions = [manager._execute({}, test) for _ in range(2)]
            result = manager._aggregate_repeats(test, repetitions, {"seed": 1}, 2)
            self.assertTrue(result["token_count_is_estimated"])
            self.assertTrue(result["timing_is_estimated"])
            self.assertEqual(result["timing_source"], "mixed")
            self.assertTrue(result["output_truncated"])
            self.assertIn("max_tokens", result["diagnostic"])
            self.assertFalse(result["replicates"][0]["token_count_is_estimated"])
            self.assertTrue(result["replicates"][1]["token_count_is_estimated"])

    def test_context_prompts_reserve_output_budget_and_chat_overhead(self):
        tests = make_context_tests(7, max_context=8192, output_budget=512 + 128)
        self.assertEqual([test["context_size"] for test in tests], [4096, 8192])
        for test in tests:
            self.assertLessEqual(test["estimated_prompt_tokens"] + 512 + 128, test["context_size"])
            self.assertIn(test["expected_answer"], test["prompt"])
        self.assertEqual(make_context_tests(7, max_context=4096, output_budget=4224), [])

    def test_accumulated_context_preserves_history_without_exceeding_total_window(self):
        options = normalize_generation_options({"max_tokens": 512, "num_ctx": 16384})
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            history = []
            tests = make_context_tests(7, 16384, output_budget=640)
            for original in tests:
                test = manager._accumulated_test(original, history, options)
                required = (manager._history_tokens(history) +
                            test["estimated_prompt_tokens"] + 512 + 128)
                self.assertLessEqual(required, test["context_size"])
                self.assertLessEqual(required, options["num_ctx"])
                self.assertIn(test["expected_answer"], test["prompt"])
                if history:
                    self.assertLess(len(test["prompt"]), len(original["prompt"]))
                if test["context_position"] == "middle":
                    self.assertAlmostEqual(
                        test["prompt"].index(test["expected_answer"]) / len(test["prompt"]),
                        0.5, delta=0.02)
                with patch.object(manager, "_answer", return_value=test["expected_answer"]) as answer:
                    result = manager._execute({}, test, history, options)
                self.assertEqual(result["status"], "PASS")
                answer.assert_called_once()
                history.extend([{"role": "user", "content": test["prompt"]},
                                {"role": "assistant", "content": result["answer"]}])
            self.assertEqual(manager._context_limit({"context_length": 131072}, options), 16384)
            self.assertEqual(manager._context_limit({}, options), 16384)
            self.assertEqual(manager._context_limit({"context_length": "Unknown"}, options), 16384)

    def test_over_budget_context_is_not_sent_to_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            options = normalize_generation_options({"max_tokens": 512, "num_ctx": 4096})
            test = make_context_tests(7, 4096, output_budget=640)[0]
            history = [{"role": "user", "content": "old context" * 3000}]
            with patch.object(manager, "_answer") as answer:
                result = manager._execute({}, test, history, options)
            answer.assert_not_called()
            self.assertEqual(result["status"], "NOT RUN")
            self.assertEqual(result["error"], "CONTEXT_BUDGET_EXCEEDED")
            self.assertIsNone(result["score"])
            self.assertIn("no model request", result["evaluation"]["explanation"])
            self.assertGreater(result["estimated_total_context_tokens"],
                               result["estimated_context_token_limit"])
            self.assertTrue(result["context_budget_is_estimated"])

    def test_context_checkpoint_validation_replays_resized_accumulated_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            model = {"id": "demo-local", "backend": "Demo", "endpoint": "local://demo",
                     "demo": True}
            options = normalize_generation_options({"max_tokens": 512, "num_ctx": 8192})
            context = make_context_tests(7, 8192, output_budget=640)
            history, accumulated = [], []
            for test in context:
                bounded = manager._accumulated_test(test, history, options)
                outcome = manager._execute(model, bounded, history, options)
                accumulated.append(outcome)
                history.extend([{"role": "user", "content": bounded["prompt"]},
                                {"role": "assistant", "content": outcome["answer"]}])
            state = {
                "profile": "QUICK", "seed": 7, "model": model,
                "generation_options": options, "repeat_runs": 1,
                "tests": make_tests("QUICK", 7), "task_set_version": TASK_SET_VERSION,
                "evaluator_version": EVALUATOR_VERSION, "resume_phase": "accumulated_context",
                "completed_tests": [], "failures": [], "not_run": [], "warmup_errors": [],
                "context_results": [], "accumulated_results": accumulated,
            }
            manager._validate_checkpoint(state)
            tampered = copy.deepcopy(state)
            tampered["accumulated_results"][1]["prompt"] = context[1]["prompt"]
            with self.assertRaisesRegex(ValueError, "accumulated_results"):
                manager._validate_checkpoint(tampered)
            reordered = copy.deepcopy(state)
            reordered["accumulated_results"].reverse()
            with self.assertRaisesRegex(ValueError, "accumulated_results"):
                manager._validate_checkpoint(reordered)

    def test_evaluator_becoming_unavailable_after_precheck_remains_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            test = {"id": "code", "category": "Coding", "prompt": "define f",
                    "evaluation_method": "code", "tests": ["assert f() == 1"]}
            evaluation = {"passed": None, "score": None, "status": "NOT RUN",
                          "criteria": "execute tests", "explanation": "Docker became unavailable."}
            with patch("backend.benchmark.code_evaluator_available", return_value=True), \
                    patch("backend.benchmark.evaluate_with_details", return_value=evaluation), \
                    patch.object(manager, "_answer", return_value="def f(): return 1") as answer:
                result = manager._execute({}, test)
            answer.assert_called_once()
            self.assertEqual(result["status"], "NOT RUN")
            self.assertIsNone(result["passed"])
            self.assertIsNone(result["score"])
            self.assertEqual(result["error"], "EVALUATOR_UNAVAILABLE")
            self.assertEqual(result["attempts"][0]["status"], "NOT RUN")

    def test_context_degradation_uses_matched_scored_windows_only(self):
        fresh = [{"id": "small", "score": 10}, {"id": "large", "score": 0}]
        accumulated = [{"id": "small", "score": 8}, {"id": "large", "score": None}]
        scores = BenchmarkManager._context_score(fresh, accumulated)
        self.assertEqual(scores["fresh"], 50)
        self.assertEqual(scores["accumulated"], 80)
        self.assertEqual(scores["degradation"], 20)
        self.assertEqual(scores["matched_count"], 1)
        self.assertEqual(scores["matched_test_ids"], ["small"])
        self.assertIsNotNone(scores["coverage_warning"])
        unmatched = BenchmarkManager._context_score(fresh, [{"id": "other", "score": 10}])
        self.assertIsNone(unmatched["degradation"])

    def test_manager_defaults_do_not_implicitly_apply_hardware_runtime_recommendations(self):
        hardware = {"ram_bytes": 64 * 1024**3, "gpu_vram_bytes": None}
        for supplied, expected in (({}, 2048), ({"max_tokens": 17}, 17),
                                   ({"temperature": 0.5, "num_ctx": 4096}, 2048)):
            with self.subTest(supplied=supplied), tempfile.TemporaryDirectory() as directory:
                manager = BenchmarkManager(Path(directory))
                with patch("backend.benchmark.hardware_info", return_value=hardware) as probe, \
                        patch("backend.benchmark.make_tests", return_value=[]), \
                        patch("backend.benchmark.make_context_tests", return_value=[]), \
                        patch.object(manager, "_performance", return_value={"score": None}), \
                        patch.object(manager, "_write_outputs") as outputs:
                    self.assertTrue(manager.run({"profile": "QUICK", "generation_options": supplied}))
                result = outputs.call_args.args[0]
                options = result["configuration"]["generation_options"]
                self.assertEqual(options["max_tokens"], expected)
                self.assertNotIn("num_gpu", options)
                self.assertNotIn("num_thread", options)
                self.assertEqual(options.get("num_ctx"), supplied.get("num_ctx"))
                self.assertEqual(result["hardware"], hardware)
                probe.assert_called_once_with()

    def test_single_chunk_without_decode_measurement_cannot_inflate_performance_score(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            sample = {"id": "short", "prompt": "x", "category": "Mathematics",
                      "evaluation_method": "raw", "expected_answer": "x"}
            measurement = {
                "status": "PASS", "latency_s": 0.5, "ttft_s": 0.5, "generation_s": 1e-9,
                "output_tokens": 500, "decode_tokens_per_second": None,
                "timing_is_estimated": True, "token_count_is_estimated": False,
                "timing_source": "backend_usage",
            }
            with patch.object(manager, "_answer", return_value={"text": "x"}), \
                    patch.object(manager, "_execute", return_value=measurement):
                performance = manager._performance({}, [sample])
            self.assertIsNone(performance["score"])
            self.assertIsNone(performance["tokens_per_second"])
            self.assertIsNone(performance["decode_tokens_per_second"])
            self.assertEqual(performance["decode_measured_runs"], 0)
            self.assertEqual(performance["decode_unavailable_samples"], 1)
            self.assertEqual(performance["timing_source"], "unavailable")
            self.assertEqual(performance["ttft_s"], 0.5)
            self.assertIsNone(manager._decode_measurement({
                **measurement, "decode_s": 1e-9, "decode_token_count": 500,
                "decode_measurement_s": 1e-9,
            }))

    def test_decode_aggregate_excludes_missing_intervals_and_weights_valid_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(Path(directory))
            sample = {"id": "sample", "prompt": "x", "category": "Mathematics",
                      "evaluation_method": "raw", "expected_answer": "x"}
            base = {"status": "PASS", "latency_s": 8, "ttft_s": 0.2,
                    "generation_s": 2, "output_tokens": 100, "token_count_is_estimated": False}
            client = {**base, "decode_tokens_per_second": 5,
                      "timing_is_estimated": True, "decode_timing_is_estimated": True,
                      "timing_source": "backend_usage"}
            reported = {**base, "output_tokens": 90, "decode_s": 6,
                        "decode_tokens_per_second": 15, "timing_is_estimated": False,
                        "timing_source": "llama.cpp"}
            missing = {**base, "generation_s": 1e-12, "output_tokens": 10000,
                       "decode_tokens_per_second": None, "timing_is_estimated": True}
            with patch.object(manager, "_answer", return_value={"text": "x"}), \
                    patch.object(manager, "_execute", side_effect=[client, reported, missing]):
                performance = manager._performance({}, [sample] * 3)
            self.assertEqual(performance["decode_tokens_per_second"], 12.5)
            self.assertEqual(performance["decode_measured_runs"], 2)
            self.assertEqual(performance["decode_unavailable_samples"], 1)
            self.assertTrue(performance["token_count_is_estimated"])
            self.assertTrue(performance["timing_is_estimated"])
            self.assertEqual(performance["timing_source"], "mixed")
            rate_only = {**reported, "decode_s": None, "generation_s": 1e-12}
            tokens, seconds, estimated = manager._decode_measurement(rate_only)
            self.assertEqual(tokens / seconds, 15)
            self.assertFalse(estimated)


if __name__ == "__main__":
    unittest.main()
