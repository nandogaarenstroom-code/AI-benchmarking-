"""Local endpoint discovery and model adapters."""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone


class AdapterError(RuntimeError):
    """Typed local-adapter failure retained in benchmark results."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _get(url, timeout=0.7):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


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
            for item in data.get("data", []):
                models.append({"id": item.get("id"), "name": item.get("id"),
                               "backend": name, "endpoint": endpoint, "size": "Unknown",
                               "capabilities": {"streaming": True, "structured_output": "Unknown",
                                                "tools": "Unknown"}, "demo": False})
    return models


def model_info(model):
    """Return explicit metadata; unknown values are never fabricated."""
    return {
        "name": model.get("name", "Unknown"), "family": "Unknown",
        "parameters": "Unknown", "quantization": "Unknown", "architecture": "Unknown",
        "context_length": model.get("context_length", "Unknown"), "vocabulary_size": "Unknown",
        "file_size": model.get("size", "Unknown"), "backend": model.get("backend", "Unknown"),
        "endpoint": model.get("endpoint", "Unknown"), "backend_version": model.get("backend_version", "Unknown"),
        "capabilities": model.get("capabilities", {}),
    }


def hardware_info():
    import platform
    return {"cpu": platform.processor() or "Unknown", "cores": __import__("os").cpu_count(),
            "threads": __import__("os").cpu_count(), "architecture": platform.machine() or "Unknown",
            "gpu": "Unknown", "gpu_vram": "Unknown", "ram": "Unknown",
            "os": platform.platform(), "driver": "Unknown", "cuda": "Unknown", "rocm": "Unknown",
            "collected_at": datetime.now(timezone.utc).isoformat()}


def ask_model(model, prompt, history=None):
    """Send one isolated prompt, or an explicit history for context accumulation tests."""
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    try:
        if model.get("backend") == "Ollama":
            request = urllib.request.Request(
                model["endpoint"] + "/api/generate",
                data=json.dumps({"model": model["id"], "prompt": prompt, "stream": False}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read()).get("response", "").strip()[:50_000]
        request = urllib.request.Request(
            model["endpoint"] + "/chat/completions",
            data=json.dumps({"model": model["id"], "messages": messages,
                             "temperature": 0, "max_tokens": 128}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read())
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()[:50_000]
    except TimeoutError as exc:
        raise AdapterError("TIMEOUT", str(exc)) from exc
    except urllib.error.URLError as exc:
        raise AdapterError("CONNECTION_ERROR", str(exc)) from exc
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AdapterError("SERVER_ERROR", str(exc)) from exc
