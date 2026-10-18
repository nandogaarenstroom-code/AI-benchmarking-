"""Local endpoint discovery and model adapters."""
from __future__ import annotations

import json
import http.client
import urllib.request
import urllib.error
import time
import codecs
import math
import socket
from .hardware import hardware_info, recommend_settings
from .llama_server import llama_server_manager

MAX_MODEL_RESPONSE_BYTES = 16_777_216
MAX_DISCOVERY_RESPONSE_BYTES = 1_048_576


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _open(request, timeout):
    return urllib.request.build_opener(_NoRedirectHandler()).open(request, timeout=timeout)


GENERATION_DEFAULTS = {
    "temperature": 0.0, "top_p": 1.0, "top_k": 40, "min_p": 0.05,
    "repeat_penalty": 1.1, "max_tokens": 2048, "seed": 12345,
}
GENERATION_LIMITS = {
    "temperature": (0.0, 2.0), "top_p": (0.01, 1.0), "top_k": (0, 1000),
    "min_p": (0.0, 1.0), "repeat_penalty": (0.5, 2.0),
    "max_tokens": (1, 32768), "seed": (0, 2147483647),
    "num_ctx": (512, 131072), "num_gpu": (0, 999), "num_thread": (1, 256),
}
INTEGER_GENERATION_OPTIONS = {"top_k", "max_tokens", "seed", "num_ctx", "num_gpu", "num_thread"}


def normalize_generation_options(options=None):
    """Validate generation settings supplied by the local dashboard or API."""
    options = options if isinstance(options, dict) else {}
    normalized = {}
    defaults = dict(GENERATION_DEFAULTS)
    for name in ("num_ctx", "num_gpu", "num_thread"):
        if name in options:
            defaults[name] = GENERATION_LIMITS[name][0]
    for name, default in defaults.items():
        try:
            value = float(options.get(name, default))
            if not math.isfinite(value):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            value = default
        lower, upper = GENERATION_LIMITS[name]
        value = min(upper, max(lower, value))
        normalized[name] = int(value) if name in INTEGER_GENERATION_OPTIONS else value
    return normalized


class AdapterError(RuntimeError):
    """Typed local-adapter failure retained in benchmark results."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.diagnostic = message


def _estimate_tokens(text):
    return max(1, len(text.encode("utf-8")) // 4) if text else 0


def _response_lines(response):
    """Decode bounded response chunks into complete UTF-8 lines."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    response_size = 0
    for raw in response:
        response_size += len(raw)
        if response_size > MAX_MODEL_RESPONSE_BYTES:
            raise AdapterError(
                "RESPONSE_TOO_LARGE", "The local model response exceeded the configured size limit.")
        buffer += decoder.decode(raw)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line.rstrip("\r")
    buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer.rstrip("\r")


def _sse_data_events(response):
    """Yield data payloads from SSE, including events split across read chunks."""
    data_lines = []

    def finish_event():
        if not data_lines:
            return None
        return "\n".join(data_lines)

    for line in _response_lines(response):
        if not line:
            event = finish_event()
            data_lines.clear()
            if event is not None:
                yield event
        elif line.startswith(":"):
            continue
        elif line.startswith("data:"):
            if data_lines:
                previous = finish_event()
                try:
                    json.loads(previous)
                    previous_is_complete = True
                except json.JSONDecodeError:
                    previous_is_complete = previous.strip() == "[DONE]"
                if previous_is_complete:
                    yield previous
                    data_lines.clear()
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(" ") else value)
    event = finish_event()
    if event is not None:
        yield event


def _get(url, timeout=0.7):
    try:
        with _open(url, timeout=timeout) as response:
            body = response.read(MAX_DISCOVERY_RESPONSE_BYTES + 1)
            if len(body) > MAX_DISCOVERY_RESPONSE_BYTES:
                return None
            value = json.loads(body)
            return value if isinstance(value, dict) else None
    except Exception:
        return None


def _llama_runtime_details(endpoint):
    base_endpoint = endpoint[:-3] if endpoint.endswith("/v1") else endpoint
    props = _get(base_endpoint + "/props")
    if not isinstance(props, dict):
        return {}
    settings = props.get("default_generation_settings") or {}
    details = {}
    details["llama_cpp"] = True
    setting_names = {
        "temperature": "temperature", "top_p": "top_p", "top_k": "top_k",
        "min_p": "min_p", "repeat_penalty": "repeat_penalty", "max_tokens": "n_predict",
        "seed": "seed",
    }
    reported_options = {}
    for option, prop_name in setting_names.items():
        value = settings.get(prop_name)
        if value is None and option == "max_tokens":
            value = settings.get("max_tokens")
        if value is not None:
            if isinstance(value, bool):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            lower, upper = GENERATION_LIMITS[option]
            if (not math.isfinite(value) or not lower <= value <= upper or
                    (option in INTEGER_GENERATION_OPTIONS and not value.is_integer())):
                continue
            reported_options[option] = value
    if reported_options:
        details["generation_defaults"] = {
            key: value for key, value in normalize_generation_options(reported_options).items()
            if key in reported_options
        }
    for key, value in (
        ("backend_version", props.get("version") or props.get("server_version") or props.get("build_info")),
        ("context_length", settings.get("n_ctx") if settings.get("n_ctx") is not None else props.get("n_ctx")),
        ("gpu_offload", props.get("n_gpu_layers") if props.get("n_gpu_layers") is not None else props.get("gpu_layers")),
    ):
        if value is not None:
            details[key] = value
    return details


def discover_models():
    models = [{
        "id": "demo-local",
        "name": "Demo Local Model",
        "backend": "Demo",
        "endpoint": "local://demo",
        "size": "Unknown",
        "capabilities": {"streaming": True, "structured_output": True, "tools": False},
        "demo": True,
    }]
    ollama = _get("http://localhost:11434/api/tags")
    if ollama:
        for item in ollama.get("models", []):
            models.append({"id": item.get("name"), "name": item.get("name"),
                           "backend": "Ollama", "endpoint": "http://localhost:11434",
                           "size": item.get("size", "Unknown"), "digest": item.get("digest"),
                           "modified": item.get("modified_at"),
                           "capabilities": {"streaming": True, "structured_output": "Unknown",
                                            "tools": "Unknown"}, "demo": False})
    for name, endpoint in (("LM Studio", "http://localhost:1234/v1"),
                           ("OpenAI-compatible", "http://127.0.0.1:8080/v1")):
        data = _get(endpoint + "/models")
        if data:
            runtime_details = _llama_runtime_details(endpoint)
            for item in data.get("data", []):
                model = {"id": item.get("id"), "name": item.get("id"),
                         "backend": name, "endpoint": endpoint, "size": "Unknown",
                         **runtime_details,
                         "capabilities": {"streaming": True, "structured_output": "Unknown",
                                          "tools": "Unknown"}, "demo": False}
                managed_launch = llama_server_manager.managed_launch(endpoint)
                if managed_launch:
                    model["managed_launch"] = managed_launch
                models.append(model)
    return models


def resolve_model(requested):
    """Resolve request model metadata to an endpoint discovered by this server."""
    if not isinstance(requested, dict):
        raise ValueError("invalid model")
    model_id = requested.get("id")
    backend = requested.get("backend")
    endpoint = requested.get("endpoint")
    if not all(isinstance(value, str) and value for value in (model_id, backend, endpoint)):
        raise ValueError("invalid model")
    if (model_id, backend, endpoint) == ("demo-local", "Demo", "local://demo"):
        return {
            "id": "demo-local", "name": "Demo Local Model", "backend": "Demo",
            "endpoint": "local://demo", "size": "Unknown",
            "capabilities": {"streaming": True, "structured_output": True, "tools": False},
            "demo": True,
        }
    for model in discover_models():
        if (model.get("id"), model.get("backend"), model.get("endpoint")) == (
                model_id, backend, endpoint):
            return model
    raise ValueError("model endpoint is not an available local model")


def model_info(model):
    """Return explicit metadata; unknown values are never fabricated."""
    return {
        "name": model.get("name", "Unknown"), "family": "Unknown",
        "parameters": "Unknown", "quantization": "Unknown", "architecture": "Unknown",
        "context_length": model.get("context_length", "Unknown"), "vocabulary_size": "Unknown",
        "file_size": model.get("size", "Unknown"), "backend": model.get("backend", "Unknown"),
        "endpoint": model.get("endpoint", "Unknown"), "backend_version": model.get("backend_version", "Unknown"),
        "gpu_offload": model.get("gpu_offload", "Unknown"),
        "llama_cpp": model.get("llama_cpp", False),
        "generation_defaults": model.get("generation_defaults", {}),
        "capabilities": model.get("capabilities", {}),
        "managed_launch": model.get("managed_launch"),
    }


def _nonnegative(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


class _StreamTelemetry:
    def __init__(self, messages, callback, backend):
        self.started = time.perf_counter()
        self.first_token_at = None
        self.callback = callback
        self.backend = backend
        self.prompt_estimate = _estimate_tokens("\n".join(
            str(message.get("content", "")) for message in messages if isinstance(message, dict)))
        self.generated_bytes = 0
        self.prompt_tokens = self.output_tokens = None
        self.prefill_s = self.decode_s = None
        self.prefill_rate = self.decode_rate = None
        self.timing_source = "client_estimate"
        self.finish_reason = None
        self.backend_metrics = {}
        self.emit("prefill", self.started)

    def ingest(self, item, text="", reasoning=""):
        if text or reasoning:
            if self.first_token_at is None:
                self.first_token_at = time.perf_counter()
            self.generated_bytes += len((text + reasoning).encode("utf-8"))
        if self.backend == "Ollama":
            for source, target in (("prompt_eval_count", "prompt_tokens"), ("eval_count", "output_tokens")):
                value = _nonnegative(item.get(source))
                if value is not None:
                    setattr(self, target, int(value))
                    self.backend_metrics[source] = value
            for source, target in (("prompt_eval_duration", "prefill_s"), ("eval_duration", "decode_s")):
                value = _nonnegative(item.get(source))
                if value is not None:
                    setattr(self, target, value / 1e9)
                    self.backend_metrics[source] = value
            if item.get("done_reason") is not None:
                self.finish_reason = str(item["done_reason"])
            if self.prefill_s is not None or self.decode_s is not None:
                self.timing_source = "ollama"
        usage = item.get("usage")
        if isinstance(usage, dict):
            for source, target in (("prompt_tokens", "prompt_tokens"), ("completion_tokens", "output_tokens")):
                value = _nonnegative(usage.get(source))
                if value is not None:
                    setattr(self, target, int(value))
                    self.backend_metrics[source] = value
        timings = item.get("timings")
        if isinstance(timings, dict):
            for source, target in (("prompt_n", "prompt_tokens"), ("predicted_n", "output_tokens"),
                                   ("prompt_ms", "prefill_s"), ("predicted_ms", "decode_s"),
                                   ("prompt_per_second", "prefill_rate"),
                                   ("predicted_per_second", "decode_rate")):
                value = _nonnegative(timings.get(source))
                if value is not None:
                    setattr(self, target, value / 1000 if source.endswith("_ms") else value)
                    self.backend_metrics[source] = value
            if self.prefill_s is not None or self.decode_s is not None or self.prefill_rate is not None or self.decode_rate is not None:
                self.timing_source = "llama.cpp"
        if self.callback:
            self.emit("generating" if self.first_token_at is not None else "prefill",
                      time.perf_counter())

    def snapshot(self, phase, now):
        elapsed = max(0.0, now - (self.first_token_at if self.first_token_at is not None else now))
        prompt_tokens = self.prompt_tokens if self.prompt_tokens is not None else self.prompt_estimate
        output_tokens = self.output_tokens if self.output_tokens is not None else (
            max(1, self.generated_bytes // 4) if self.generated_bytes else 0)
        prefill_rate = self.prefill_rate
        if prefill_rate is None and self.prompt_tokens is not None and self.prefill_s:
            prefill_rate = self.prompt_tokens / self.prefill_s
        decode_rate = self.decode_rate
        if decode_rate is None and self.output_tokens is not None and self.decode_s:
            decode_rate = self.output_tokens / self.decode_s
        source = self.timing_source
        if source == "client_estimate" and self.prompt_tokens is not None and self.output_tokens is not None:
            source = "backend_usage"
        return {
            "phase": phase, "prompt_tokens": int(prompt_tokens), "output_tokens": int(output_tokens),
            "ttft_s": None if self.first_token_at is None else self.first_token_at - self.started,
            "generation_s": elapsed, "prefill_s": self.prefill_s, "decode_s": self.decode_s,
            "prefill_tokens_per_second": prefill_rate,
            "decode_tokens_per_second": decode_rate if decode_rate is not None else (
                output_tokens / elapsed if elapsed > 0 else None),
            "token_count_is_estimated": self.prompt_tokens is None or self.output_tokens is None,
            "timing_source": source,
            "decode_timing_is_estimated": decode_rate is None,
            "finish_reason": self.finish_reason,
            "truncated": self.finish_reason in ("length", "max_tokens", "max_output_tokens"),
            "backend_metrics": dict(self.backend_metrics),
        }

    def emit(self, phase, now):
        metrics = self.snapshot(phase, now)
        if self.callback:
            self.callback(metrics)
        return metrics

    def finish(self, answer, completed, backend_name):
        metrics = self.emit("completed" if completed else "incomplete", time.perf_counter())
        if not completed:
            raise AdapterError("INCOMPLETE_STREAM",
                               f"{backend_name} ended the response without a completion marker.")
        if not answer:
            error = AdapterError("EMPTY_RESPONSE",
                                 "The model server completed the request without generated text.")
            error.finish_reason = self.finish_reason
            error.truncated = metrics["truncated"]
            raise error
        return {"text": answer, **metrics}


def ask_model(model, prompt, history=None, options=None, progress_callback=None):
    """Stream an answer and optionally publish telemetry without exposing reasoning.

    Prefill throughput is None until reported by the backend, never derived from
    TTFT. Token counts and client-side decode estimates are explicitly marked.
    Ollama runtime options are not sent to external OpenAI-compatible servers.
    """
    options = normalize_generation_options(options)
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    try:
        if model.get("backend") == "Ollama":
            request = urllib.request.Request(
                model["endpoint"] + "/api/chat",
                data=json.dumps({
                    "model": model["id"], "messages": messages, "stream": True,
                    "options": {
                        "temperature": options["temperature"], "top_p": options["top_p"],
                        "top_k": options["top_k"], "min_p": options["min_p"],
                        "repeat_penalty": options["repeat_penalty"], "num_predict": options["max_tokens"],
                        "seed": options["seed"],
                        **{key: options[key] for key in ("num_ctx", "num_gpu", "num_thread")
                           if key in options},
                    },
                }).encode(),
                headers={"Content-Type": "application/json"})
            telemetry = _StreamTelemetry(messages, progress_callback, "Ollama")
            with _open(request, timeout=60) as response:
                chunks = []
                completed = False
                for line in _response_lines(response):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("invalid Ollama response")
                    if item.get("error"):
                        raise AdapterError("SERVER_ERROR", "Local model server reported a generation error.")
                    message = item.get("message", {})
                    text = message.get("content", "") if isinstance(message, dict) else None
                    if not isinstance(text, str):
                        raise ValueError("invalid Ollama response content")
                    reasoning = message.get("thinking", "") if isinstance(message, dict) else ""
                    if not isinstance(reasoning, str):
                        raise ValueError("invalid Ollama reasoning content")
                    telemetry.ingest(item, text, reasoning)
                    chunks.append(text)
                    if item.get("done"):
                        completed = True
                        break
                return telemetry.finish("".join(chunks).strip(), completed, "Ollama")
        body = {
            "model": model["id"], "messages": messages,
            "temperature": options["temperature"], "top_p": options["top_p"],
            "max_tokens": options["max_tokens"], "seed": options["seed"], "stream": True,
            "stream_options": {"include_usage": True},
        }
        if model.get("llama_cpp"):
            body.update({"top_k": options["top_k"], "min_p": options["min_p"],
                         "repeat_penalty": options["repeat_penalty"], "timings_per_token": True})
        request = urllib.request.Request(
            model["endpoint"] + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        telemetry = _StreamTelemetry(messages, progress_callback, "OpenAI-compatible")
        with _open(request, timeout=60) as response:
            chunks = []
            completed = False
            for event in _sse_data_events(response):
                if event.strip() == "[DONE]":
                    completed = True
                    break
                item = json.loads(event)
                if not isinstance(item, dict):
                    raise ValueError("invalid OpenAI-compatible response")
                if item.get("error"):
                    raise AdapterError("SERVER_ERROR", "Local model server reported a generation error.")
                choices = item.get("choices", [])
                if not isinstance(choices, list):
                    raise ValueError("invalid OpenAI-compatible choices")
                if not choices:
                    telemetry.ingest(item)
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise ValueError("invalid OpenAI-compatible choice")
                content_source = choice.get("delta", choice.get("message", {}))
                if isinstance(content_source, str):
                    text = content_source
                elif isinstance(content_source, dict):
                    text = content_source.get("content", choice.get("text", ""))
                else:
                    raise ValueError("invalid OpenAI-compatible content")
                if isinstance(text, list):
                    text = "".join(part.get("text", "") for part in text
                                   if isinstance(part, dict) and isinstance(part.get("text"), str))
                if text is None:
                    text = ""
                if not isinstance(text, str):
                    raise ValueError("invalid OpenAI-compatible content")
                reasoning = (content_source.get("reasoning_content") or
                             content_source.get("reasoning") or "") if isinstance(content_source, dict) else ""
                if not isinstance(reasoning, str):
                    raise ValueError("invalid reasoning content")
                chunks.append(text)
                if choice.get("finish_reason") is not None:
                    telemetry.finish_reason = str(choice["finish_reason"])
                    completed = True
                telemetry.ingest(item, text, reasoning)
            return telemetry.finish("".join(chunks).strip(), completed, "The model server")
    except AdapterError:
        raise
    except urllib.error.HTTPError as exc:
        raise AdapterError("HTTP_ERROR", f"Local model server returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise AdapterError(
                "TIMEOUT", "The local model server timed out before completing the response.") from exc
        raise AdapterError(
            "CONNECTION_ERROR", "Could not connect to the local model server.") from exc
    except TimeoutError as exc:
        raise AdapterError(
            "TIMEOUT", "The local model server timed out before completing the response.") from exc
    except (ConnectionError, OSError, http.client.HTTPException) as exc:
        raise AdapterError(
            "CONNECTION_ERROR", "The connection to the local model server failed during streaming.") from exc
    except (ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AdapterError(
            "SERVER_ERROR", "The local model server returned a malformed response stream.") from exc
