"""Local endpoint discovery and model adapters."""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone


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
        "context_length": "Unknown", "vocabulary_size": "Unknown",
        "file_size": model.get("size", "Unknown"), "backend": model.get("backend", "Unknown"),
        "endpoint": model.get("endpoint", "Unknown"), "capabilities": model.get("capabilities", {}),
    }


def hardware_info():
    import platform
    return {"cpu": platform.processor() or "Unknown", "cores": __import__("os").cpu_count(),
            "threads": __import__("os").cpu_count(), "architecture": platform.machine() or "Unknown",
            "gpu": "Unknown", "gpu_vram": "Unknown", "ram": "Unknown",
            "os": platform.platform(), "driver": "Unknown", "cuda": "Unknown", "rocm": "Unknown",
            "collected_at": datetime.now(timezone.utc).isoformat()}
