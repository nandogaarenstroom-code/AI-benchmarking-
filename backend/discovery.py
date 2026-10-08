"""Local endpoint discovery and model adapters."""
from __future__ import annotations

import json
import urllib.request
import urllib.error
import time
import codecs
import math
from datetime import datetime, timezone

MAX_MODEL_RESPONSE_BYTES = 1_048_576


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _open(request, timeout):
    return urllib.request.build_opener(_NoRedirectHandler()).open(request, timeout=timeout)


GENERATION_DEFAULTS = {
    "temperature": 0.0, "top_p": 1.0, "top_k": 40, "min_p": 0.05,
    "repeat_penalty": 1.1, "max_tokens": 128, "seed": 12345,
}
GENERATION_LIMITS = {
    "temperature": (0.0, 2.0), "top_p": (0.01, 1.0), "top_k": (0, 1000),
    "min_p": (0.0, 1.0), "repeat_penalty": (0.5, 2.0),
    "max_tokens": (1, 4096), "seed": (0, 2147483647),
}
INTEGER_GENERATION_OPTIONS = {"top_k", "max_tokens", "seed"}


def normalize_generation_options(options=None):
    """Validate generation settings supplied by the local dashboard or API."""
    options = options if isinstance(options, dict) else {}
    normalized = {}
    for name, default in GENERATION_DEFAULTS.items():
        try:
            value = float(options.get(name, default))
            if not math.isfinite(value):
                raise ValueError
        except (TypeError, ValueError):
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


def _estimate_tokens(text):
    return max(1, len(text.encode("utf-8")) // 4) if text else 0


def _get(url, timeout=0.7):
    try:
        with _open(url, timeout=timeout) as response:
            return json.loads(response.read())
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
                           ("OpenAI-compatible", "http://localhost:8080/v1")):
        data = _get(endpoint + "/models")
        if data:
            runtime_details = _llama_runtime_details(endpoint)
            for item in data.get("data", []):
                models.append({"id": item.get("id"), "name": item.get("id"),
                               "backend": name, "endpoint": endpoint, "size": "Unknown",
                               **runtime_details,
                               "capabilities": {"streaming": True, "structured_output": "Unknown",
                                                "tools": "Unknown"}, "demo": False})
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
    }


def hardware_info():
    import platform
    return {"cpu": platform.processor() or "Unknown", "cores": __import__("os").cpu_count(),
            "threads": __import__("os").cpu_count(), "architecture": platform.machine() or "Unknown",
            "gpu": "Unknown", "gpu_vram": "Unknown", "ram": "Unknown",
            "os": platform.platform(), "driver": "Unknown", "cuda": "Unknown", "rocm": "Unknown",
            "collected_at": datetime.now(timezone.utc).isoformat()}


def ask_model(model, prompt, history=None, options=None):
    """Send one isolated prompt, or an explicit history for context accumulation tests."""
    options = normalize_generation_options(options)
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    try:
        if model.get("backend") == "Ollama":
            request = urllib.request.Request(
                model["endpoint"] + "/api/generate",
                data=json.dumps({
                    "model": model["id"], "prompt": prompt, "stream": True,
                    "options": {
                        "temperature": options["temperature"], "top_p": options["top_p"],
                        "top_k": options["top_k"], "min_p": options["min_p"],
                        "repeat_penalty": options["repeat_penalty"], "num_predict": options["max_tokens"],
                        "seed": options["seed"],
                    },
                }).encode(),
                headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            with _open(request, timeout=60) as response:
                chunks, first_token_at = [], None
                response_size = 0
                for line in response:
                    response_size += len(line)
                    if response_size > MAX_MODEL_RESPONSE_BYTES:
                        raise ValueError("model response too large")
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    text = item.get("response", "")
                    if text and first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(text)
                    if item.get("done"):
                        break
                answer = "".join(chunks).strip()[:50_000]
                return {"text": answer, "prompt_tokens": _estimate_tokens(prompt),
                        "output_tokens": _estimate_tokens(answer),
                        "ttft_s": (first_token_at or time.perf_counter()) - started,
                        "generation_s": time.perf_counter() - (first_token_at or started)}
        body = {
            "model": model["id"], "messages": messages,
            "temperature": options["temperature"], "top_p": options["top_p"],
            "max_tokens": options["max_tokens"], "seed": options["seed"], "stream": True,
        }
        if model.get("llama_cpp"):
            body.update({"top_k": options["top_k"], "min_p": options["min_p"],
                         "repeat_penalty": options["repeat_penalty"]})
        request = urllib.request.Request(
            model["endpoint"] + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with _open(request, timeout=60) as response:
            decoder, chunks, first_token_at = codecs.getincrementaldecoder("utf-8")(), [], None
            response_size = 0
            for raw in response:
                response_size += len(raw)
                if response_size > MAX_MODEL_RESPONSE_BYTES:
                    raise ValueError("model response too large")
                for line in decoder.decode(raw).splitlines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    item = json.loads(line[5:].strip())
                    text = item.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                    if text and first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(text)
            answer = "".join(chunks).strip()[:50_000]
            return {"text": answer, "prompt_tokens": _estimate_tokens(prompt),
                    "output_tokens": _estimate_tokens(answer),
                    "ttft_s": (first_token_at or time.perf_counter()) - started,
                    "generation_s": time.perf_counter() - (first_token_at or started)}
    except TimeoutError as exc:
        raise AdapterError("TIMEOUT", str(exc)) from exc
    except urllib.error.URLError as exc:
        raise AdapterError("CONNECTION_ERROR", str(exc)) from exc
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AdapterError("SERVER_ERROR", str(exc)) from exc
