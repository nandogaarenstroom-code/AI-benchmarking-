import json
import unittest
from unittest.mock import Mock, patch

from backend.discovery import (
    AdapterError, MAX_MODEL_RESPONSE_BYTES, MODEL_READ_TIMEOUT_SECONDS, REQUEST_CAPABILITY_MATRIX,
    ask_model, map_request_options, normalize_generation_options, request_options_snapshot,
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
        for options, field in (({"max_tokens": 999999}, "max_tokens"),
                               ({"num_ctx": 999999}, "num_ctx"),
                               ({"num_gpu": -5}, "num_gpu"),
                               ({"num_thread": 1000}, "num_thread"),
                               ({"unknown": 1}, "unknown")):
            with self.subTest(options=options), self.assertRaises(ValueError) as caught:
                normalize_generation_options(options)
            self.assertEqual(caught.exception.field, field)
        self.assertEqual(normalize_generation_options({"max_tokens": 5000})["max_tokens"], 5000)
        self.assertEqual(normalize_generation_options(
            {"max_tokens": 512}, include_defaults=False), {"max_tokens": 512})

    def test_every_parameter_maps_to_exact_backend_request_body_or_is_omitted(self):
        models = {
            "ollama": OLLAMA,
            "lm_studio": {"id": "model", "backend": "LM Studio", "endpoint": "http://localhost/v1"},
            "openai_compatible": OPENAI,
            "llama_cpp": {**OPENAI, "llama_cpp": True},
        }
        values = {
            "temperature": 0.5, "top_p": 0.8, "top_k": 20, "min_p": 0.1,
            "repeat_penalty": 1.2, "max_tokens": 128, "seed": 42,
            "num_ctx": 4096, "num_gpu": 2, "num_thread": 4,
            "presence_penalty": 0.4, "frequency_penalty": 0.3, "stop": ["STOP"],
            "system_prompt": "system", "cache_prompt": False,
            "request_timeout_seconds": 60,
        }
        for backend, model in models.items():
            mappings = REQUEST_CAPABILITY_MATRIX[backend]
            for parameter, value in values.items():
                captured = {}

                def opened(request, timeout):
                    captured.update(body=json.loads(request.data), timeout=timeout)
                    if backend == "ollama":
                        return Response([b'{"message":{"content":"ok"},"done":true}\n'])
                    return sse({"choices": [{"delta": {"content": "ok"},
                                             "finish_reason": "stop"}]})

                with self.subTest(backend=backend, parameter=parameter), patch(
                        "backend.discovery._open", side_effect=opened):
                    ask_model(model, "prompt", options={parameter: value})
                mapping = mappings.get(parameter)
                base = ({"model": "model",
                         "messages": [{"role": "user", "content": "prompt"}],
                         "stream": True, "options": {}}
                        if backend == "ollama" else
                        {"model": "model",
                         "messages": [{"role": "user", "content": "prompt"}],
                         "stream": True, "stream_options": {"include_usage": True}})
                if backend == "llama_cpp":
                    base["timings_per_token"] = True
                if mapping:
                    mapped = map_request_options(model, {parameter: value})
                    if mapping.startswith("options."):
                        base["options"][mapping.split(".", 1)[1]] = value
                    elif mapping == "messages.system":
                        base["messages"].insert(0, {"role": "system", "content": value})
                    elif mapping != "transport.timeout":
                        base[mapping] = value
                self.assertEqual(captured["body"], base)
                self.assertEqual(captured["timeout"],
                                 value if mapping == "transport.timeout" else 300)

    def test_request_option_snapshot_records_mapped_fields_and_transport(self):
        snapshot = request_options_snapshot(
            {**OPENAI, "llama_cpp": True},
            {"temperature": 0.5, "top_k": 10, "request_timeout_seconds": 40})
        self.assertEqual(snapshot, {
            "body": {"temperature": 0.5, "top_k": 10},
            "transport": {"timeout_seconds": 40},
        })

    def test_runtime_settings_only_reach_ollama(self):
        for model in (OPENAI, {**OPENAI, "llama_cpp": True}, OLLAMA):
            captured = {}

            def opened(request, timeout):
                self.assertEqual(timeout, MODEL_READ_TIMEOUT_SECONDS)
                self.assertGreater(timeout, 60)
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
        self.assertEqual(result["generation_s"], 3)
        self.assertEqual(result["decode_token_count"], 12)
        self.assertEqual(result["decode_measurement_s"], 3)
        self.assertFalse(result["decode_token_count_is_estimated"])
        self.assertEqual(result["timing_source"], "ollama")
        self.assertFalse(result["token_count_is_estimated"])
        self.assertFalse(result["timing_is_estimated"])
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
        self.assertEqual(result["generation_s"], 4)
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
        self.assertIsNone(result["decode_tokens_per_second"])
        self.assertIsNone(result["decode_token_count"])
        self.assertIsNone(result["decode_measurement_s"])

    def test_usage_counts_do_not_claim_backend_timing(self):
        with patch("backend.discovery._open", return_value=sse({
                "choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 8}})):
            result = ask_model(OPENAI, "prompt")
        self.assertFalse(result["token_count_is_estimated"])
        self.assertTrue(result["decode_timing_is_estimated"])
        self.assertTrue(result["timing_is_estimated"])
        self.assertEqual(result["timing_source"], "backend_usage")
        self.assertIsNone(result["prefill_tokens_per_second"])

    def test_backend_rate_and_count_define_decode_measurement_interval(self):
        with patch("backend.discovery._open", return_value=sse({
                "choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}],
                "timings": {"prompt_n": 100, "prompt_per_second": 10,
                            "predicted_n": 20, "predicted_per_second": 5}})):
            result = ask_model(OPENAI, "prompt")
        self.assertEqual(result["decode_token_count"], 20)
        self.assertEqual(result["decode_measurement_s"], 4)
        self.assertFalse(result["decode_token_count_is_estimated"])

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

    def test_client_decode_estimate_excludes_first_chunk(self):
        with patch("backend.discovery._open", return_value=sse(
                {"choices": [{"delta": {"content": "a" * 400}}]},
                {"choices": [{"delta": {"content": "b" * 40}, "finish_reason": "stop"}]})), \
                patch("backend.discovery.time.perf_counter", side_effect=[10, 10.2, 12.2]):
            result = ask_model(OPENAI, "prompt")
        self.assertEqual(result["output_tokens"], 110)
        self.assertAlmostEqual(result["decode_tokens_per_second"], 5)
        self.assertEqual(result["decode_token_count"], 10)
        self.assertAlmostEqual(result["decode_measurement_s"], 2)
        self.assertTrue(result["decode_token_count_is_estimated"])

    def test_buffered_sse_content_events_do_not_form_client_decode_sample(self):
        response = Response([
            b'data: {"choices":[{"delta":{"content":"' + b"a" * 400 + b'"}}]}\n',
            b'\n\ndata: {"choices":[{"delta":{"content":"' + b"b" * 40 +
            b'"},"finish_reason":"stop"}]}\n\n',
        ])
        with patch("backend.discovery._open", return_value=response), \
                patch("backend.discovery.time.perf_counter", side_effect=[10, 10.2, 12]):
            result = ask_model(OPENAI, "prompt")
        self.assertIsNone(result["decode_tokens_per_second"])
        self.assertIsNone(result["decode_token_count"])
        self.assertIsNone(result["decode_measurement_s"])

    def test_near_simultaneous_sse_events_do_not_form_client_decode_sample(self):
        with patch("backend.discovery._open", return_value=sse(
                {"choices": [{"delta": {"content": "a" * 400}}]},
                {"choices": [{"delta": {"content": "b" * 40}, "finish_reason": "stop"}]})), \
                patch("backend.discovery.time.perf_counter",
                      side_effect=[10, 10.2, 10.200001]):
            result = ask_model(OPENAI, "prompt")
        self.assertIsNone(result["decode_tokens_per_second"])
        self.assertIsNone(result["decode_measurement_s"])

    def test_client_decode_endpoint_excludes_trailing_metadata_wait(self):
        with patch("backend.discovery._open", return_value=sse(
                {"choices": [{"delta": {"content": "a" * 400}}]},
                {"choices": [{"delta": {"content": "b" * 40}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 110}})), \
                patch("backend.discovery.time.perf_counter",
                      side_effect=[0, 1, 1.1, 1.35, 1.6]):
            result = ask_model(OPENAI, "prompt")
        self.assertAlmostEqual(result["generation_s"], 0.1)
        self.assertAlmostEqual(result["client_generation_s"], 0.1)
        self.assertAlmostEqual(result["decode_tokens_per_second"], 100)

    def test_finish_marker_preserves_answer_when_optional_metadata_times_out(self):
        class TimeoutResponse(Response):
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"complete"},"finish_reason":"stop"}]}\n\n'
                raise TimeoutError("optional usage never arrived")

        response = TimeoutResponse([])
        response.fp = Mock()
        with patch("backend.discovery._open", return_value=response):
            result = ask_model(OPENAI, "prompt")
        self.assertEqual(result["text"], "complete")
        self.assertEqual(result["finish_reason"], "stop")
        response.fp.raw._sock.settimeout.assert_called_once_with(0.25)

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
