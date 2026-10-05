"""Local endpoint discovery and model adapters."""
from __future__ import annotations

import json
import urllib.request
import urllib.error
import time
import codecs
import socket
from datetime import datetime, timezone


class AdapterError(RuntimeError):
    """Typed local-adapter failure retained in benchmark results."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _estimate_tokens(text):
    return max(1, len(text.encode("utf-8")) // 4) if text else 0


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
                data=json.dumps({"model": model["id"], "prompt": prompt, "stream": True}).encode(),
                headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=60) as response:
                chunks, first_token_at = [], None
                for line in response:
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
        request = urllib.request.Request(
            model["endpoint"] + "/chat/completions",
            data=json.dumps({"model": model["id"], "messages": messages,
                             "temperature": 0, "max_tokens": 128}).encode(),
            headers={"Content-Type": "application/json"})
        request.data = json.dumps({"model": model["id"], "messages": messages,
                                   "temperature": 0, "max_tokens": 128, "stream": True}).encode()
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=60) as response:
            decoder, chunks, first_token_at = codecs.getincrementaldecoder("utf-8")(), [], None
            pending = ""
            for raw in response:
                pending += decoder.decode(raw)
                lines = pending.splitlines(keepends=True)
                pending = lines.pop() if lines and not lines[-1].endswith(("\n", "\r")) else ""
                for line in lines:
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    item = json.loads(line[5:].strip())
                    text = item.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                    if text and first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(text)
            pending += decoder.decode(b"", final=True)
            if pending.startswith("data:") and pending.strip() != "data: [DONE]":
                item = json.loads(pending[5:].strip())
                chunks.append(item.get("choices", [{}])[0].get("delta", {}).get("content", "") or "")
            answer = "".join(chunks).strip()[:50_000]
            return {"text": answer, "prompt_tokens": _estimate_tokens(prompt),
                    "output_tokens": _estimate_tokens(answer),
                    "ttft_s": (first_token_at or time.perf_counter()) - started,
                    "generation_s": time.perf_counter() - (first_token_at or started)}
    except (TimeoutError, socket.timeout) as exc:
        raise AdapterError("TIMEOUT", str(exc)) from exc
    except urllib.error.HTTPError as exc:
        raise AdapterError("SERVER_ERROR", f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise AdapterError("CONNECTION_ERROR", str(exc)) from exc
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AdapterError("SERVER_ERROR", str(exc)) from exc
