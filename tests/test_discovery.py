import json
import unittest
from unittest.mock import patch

from backend.discovery import (
    AdapterError, MAX_MODEL_RESPONSE_BYTES, ask_model, normalize_generation_options,
)


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        return iter(self.body)


def sse(*items):
    return Response([("data: " + json.dumps(item) + "\n\n").encode() for item in items])


OPENAI = {"id": "model", "backend": "OpenAI-compatible", "endpoint": "http://localhost/v1"}
OLLAMA = {"id": "model", "backend": "Ollama", "endpoint": "http://localhost"}


class DiscoveryTests(unittest.TestCase):
    def test_generous_defaults_and_optional_runtime_normalization(self):
        defaults = normalize_generation_options()
        self.assertEqual(defaults["max_tokens"], 2048)
        self.assertNotIn("num_ctx", defaults)
        self.assertNotIn("num_gpu", defaults)
        self.assertNotIn("num_thread", defaults)
        options = normalize_generation_options({"max_tokens": 999999, "num_ctx": 999999,
                                                "num_gpu": -5, "num_thread": 1000})
        self.assertEqual(options["max_tokens"], 32768)
        self.assertEqual(options["num_ctx"], 131072)
        self.assertEqual(options["num_gpu"], 0)
        self.assertEqual(options["num_thread"], 256)
        self.assertEqual(normalize_generation_options({"max_tokens": 5000})["max_tokens"], 5000)

    def test_runtime_settings_only_reach_ollama(self):
        for model in (OPENAI, {**OPENAI, "llama_cpp": True}, OLLAMA):
            captured = {}

            def opened(request, timeout):
                captured.update(json.loads(request.data))
                if model["backend"] == "Ollama":
                    return Response([b'{"message":{"content":"ok"},"done":true}\n'])
                return sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})

            with self.subTest(model=model), patch("backend.discovery._open", side_effect=opened):
                ask_model(model, "prompt", options={"num_ctx": 8192, "num_gpu": 12, "num_thread": 4})
                if model["backend"] == "Ollama":
                    self.assertEqual(captured["options"]["num_ctx"], 8192)
                    self.assertEqual(captured["options"]["num_gpu"], 12)
                    self.assertEqual(captured["options"]["num_thread"], 4)
                else:
                    self.assertFalse({"num_ctx", "num_gpu", "num_thread"} & captured.keys())

    def test_ollama_authoritative_metrics_reasoning_and_truncation(self):
        events = [
            {"message": {"thinking": "private reasoning"}},
            {"message": {"content": "answer"}},
            {"done": True, "done_reason": "length", "prompt_eval_count": 20,
             "prompt_eval_duration": 2_000_000_000, "eval_count": 12, "eval_duration": 3_000_000_000},
        ]
        progress = []
        with patch("backend.discovery._open", return_value=Response([
                (json.dumps(item) + "\n").encode() for item in events])):
            result = ask_model(OLLAMA, "prompt", progress_callback=progress.append)
        self.assertEqual(result["text"], "answer")
        self.assertEqual(result["prompt_tokens"], 20)
        self.assertEqual(result["output_tokens"], 12)
        self.assertEqual(result["prefill_tokens_per_second"], 10)
        self.assertEqual(result["decode_tokens_per_second"], 4)
        self.assertEqual(result["timing_source"], "ollama")
        self.assertFalse(result["token_count_is_estimated"])
        self.assertTrue(result["truncated"])
        self.assertEqual(progress[0]["phase"], "prefill")
        self.assertIsNone(progress[0]["prefill_tokens_per_second"])
        self.assertEqual(progress[1]["phase"], "generating")
        self.assertEqual(progress[-1]["phase"], "completed")

    def test_llama_timings_and_trailing_usage(self):
        with patch("backend.discovery._open", return_value=sse(
                {"choices": [{"delta": {"reasoning_content": "hidden"}}]},
                {"choices": [{"delta": {"content": "visible"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                 "timings": {"prompt_n": 100, "prompt_ms": 2000,
                             "predicted_n": 20, "predicted_ms": 4000}})):
            result = ask_model({**OPENAI, "llama_cpp": True}, "prompt")
        self.assertEqual(result["text"], "visible")
        self.assertEqual(result["prefill_tokens_per_second"], 50)
        self.assertEqual(result["decode_tokens_per_second"], 5)
        self.assertEqual(result["timing_source"], "llama.cpp")
        self.assertFalse(result["token_count_is_estimated"])
        self.assertFalse(result["truncated"])

    def test_client_estimates_never_treat_ttft_as_prefill(self):
        progress = []
        with patch("backend.discovery._open", return_value=sse(
                {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]})):
            result = ask_model(OPENAI, "prompt", progress_callback=progress.append)
        self.assertTrue(result["token_count_is_estimated"])
        self.assertEqual(result["timing_source"], "client_estimate")
        self.assertIsNone(result["prefill_tokens_per_second"])
        self.assertIsNone(result["prefill_s"])
        self.assertIsNotNone(result["ttft_s"])

    def test_usage_counts_do_not_claim_backend_timing(self):
        with patch("backend.discovery._open", return_value=sse({
                "choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 8}})):
            result = ask_model(OPENAI, "prompt")
        self.assertFalse(result["token_count_is_estimated"])
        self.assertTrue(result["decode_timing_is_estimated"])
        self.assertEqual(result["timing_source"], "backend_usage")
        self.assertIsNone(result["prefill_tokens_per_second"])

    def test_reasoning_only_is_not_an_answer_but_has_timing(self):
        progress = []
        with patch("backend.discovery._open", return_value=sse(
                {"choices": [{"delta": {"reasoning": "private"}, "finish_reason": "length"}]})):
            with self.assertRaises(AdapterError) as caught:
                ask_model(OPENAI, "prompt", progress_callback=progress.append)
        self.assertEqual(caught.exception.code, "EMPTY_RESPONSE")
        self.assertTrue(caught.exception.truncated)
        self.assertTrue(progress[-1]["truncated"])
        self.assertGreater(progress[-1]["output_tokens"], 0)
        self.assertIsNotNone(progress[-1]["ttft_s"])

    def test_reasoning_starts_decode_clock_before_visible_answer(self):
        with patch("backend.discovery._open", return_value=sse(
                {"choices": [{"delta": {"reasoning_content": "private"}}]},
                {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]})), \
                patch("backend.discovery.time.perf_counter", side_effect=[10, 10.2, 11]):
            result = ask_model(OPENAI, "prompt")
        self.assertEqual(result["text"], "answer")
        self.assertAlmostEqual(result["ttft_s"], 0.2)
        self.assertAlmostEqual(result["generation_s"], 0.8)
        self.assertIsNone(result["prefill_tokens_per_second"])

    def test_long_answers_not_silently_clipped(self):
        text = "a" * 60000
        for model, response in (
                (OPENAI, sse({"choices": [{"delta": {"content": text}, "finish_reason": "length"}]})),
                (OLLAMA, Response([(json.dumps({"message": {"content": text}, "done": True})
                                    + "\n").encode()]))):
            with self.subTest(backend=model["backend"]), patch("backend.discovery._open", return_value=response):
                self.assertEqual(len(ask_model(model, "prompt")["text"]), 60000)

    def test_response_limit_remains_enforced(self):
        with patch("backend.discovery._open", return_value=Response([b"x" * (MAX_MODEL_RESPONSE_BYTES + 1)])):
            with self.assertRaises(AdapterError) as caught:
                ask_model(OPENAI, "prompt")
        self.assertEqual(caught.exception.code, "RESPONSE_TOO_LARGE")

    def test_invalid_backend_counters_do_not_become_measurements(self):
        with patch("backend.discovery._open", return_value=sse({
                "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": -5, "completion_tokens": float("nan")},
                "timings": {"prompt_ms": -1, "predicted_per_second": float("inf")}})):
            result = ask_model(OPENAI, "prompt")
        self.assertTrue(result["token_count_is_estimated"])
        self.assertIsNone(result["prefill_tokens_per_second"])
