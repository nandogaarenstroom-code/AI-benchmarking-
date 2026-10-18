"""Zero-interaction local AI benchmark suite.

The server intentionally uses only the Python standard library so it can run on
an offline workstation with ``python app.py``.
"""
from __future__ import annotations

import atexit
import hmac
import json
import os
import secrets
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from backend.benchmark import BenchmarkManager
from backend.discovery import (
    GenerationOptionsError, REQUEST_CAPABILITY_MATRIX, discover_models, hardware_info,
    normalize_generation_options, recommend_settings, resolve_model,
)
from backend.llama_server import LlamaServerConflict, LlamaServerError, llama_server_manager

ROOT = Path(__file__).parent
manager = BenchmarkManager(ROOT / "results")
CSRF_TOKEN = secrets.token_urlsafe(32)
MAX_REQUEST_SIZE = 1_048_576
MAX_REQUEST_TARGET_SIZE = 4096
ALLOWED_HOSTNAMES = {"localhost", "127.0.0.1"}


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, *args, **kwargs):
        self._request_slots = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self._request_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def parse_request(self):
        if not super().parse_request():
            return False
        if len(self.path.encode("utf-8")) > MAX_REQUEST_TARGET_SIZE:
            self.close_connection = True
            self.send_error(414)
            return False
        if self.headers.get("Transfer-Encoding") is not None:
            self.close_connection = True
            self.send_error(400, "Transfer-Encoding is not supported")
            return False
        raw_size = self.headers.get("Content-Length")
        if raw_size is not None:
            if not raw_size.isdigit():
                self.close_connection = True
                self.send_error(400, "Invalid Content-Length")
                return False
            if len(raw_size) > len(str(MAX_REQUEST_SIZE)) or int(raw_size) > MAX_REQUEST_SIZE:
                self.close_connection = True
                self.send_error(413, "Request body too large")
                return False
        return True

    def _local_host(self, value):
        try:
            parsed = urlparse("//" + value)
            return (parsed.hostname in ALLOWED_HOSTNAMES and
                    parsed.port == self.server.server_port and
                    parsed.username is None and parsed.password is None and
                    not parsed.path and not parsed.query and not parsed.fragment)
        except ValueError:
            return False

    def _valid_host(self):
        return self._local_host(self.headers.get("Host", ""))

    def _valid_origin(self):
        origin = self.headers.get("Origin")
        if not origin:
            return False
        try:
            parsed = urlparse(origin)
            host = urlparse("//" + self.headers.get("Host", ""))
            return (parsed.scheme == "http" and parsed.hostname in ALLOWED_HOSTNAMES and
                    parsed.hostname == host.hostname and parsed.port == self.server.server_port and
                    host.port == self.server.server_port and not parsed.username and
                    not parsed.password and parsed.path in ("", "/") and
                    not parsed.query and not parsed.fragment)
        except ValueError:
            return False

    def _security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")

    def _json(self, value, status=200):
        body = json.dumps(value, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def _static(self, name):
        if name != "index.html":
            self.send_error(404)
            return
        path = ROOT / "frontend" / name
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes().replace(b"__CSRF_TOKEN__", CSRF_TOKEN.encode())
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._valid_host():
            return self.send_error(403)
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path == "/api/models":
            return self._json({"models": discover_models(),
                               "parameter_capabilities": REQUEST_CAPABILITY_MATRIX})
        if path == "/api/model-info":
            query = parse_qs(urlparse(self.path).query)
            requested = {key: query.get(key, [""])[0]
                         for key in ("id", "backend", "endpoint")}
            try:
                return self._json({"model": resolve_model(requested)})
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
        if path == "/api/hardware":
            hardware = hardware_info()
            return self._json({"hardware": hardware, "recommendations": recommend_settings(hardware)})
        if path == "/api/llama-server/status":
            return self._json(llama_server_manager.status())
        if path == "/api/status":
            return self._json(manager.status())
        if path == "/api/results":
            return self._json(manager.list_results())
        if path == "/api/compare":
            ids = parse_qs(urlparse(self.path).query).get("id", [])
            return self._json(manager.compare(ids))
        if path.startswith("/api/results/"):
            result = manager.load_result(path.rsplit("/", 1)[-1])
            return self._json(result, 200 if result else 404)
        if path.startswith("/results/"):
            filename = path[len("/results/"):]
            data = manager.read_artifact(filename)
            if data is not None:
                self.send_response(200)
                self.send_header(
                    "Content-Type", "application/pdf" if filename.endswith(".pdf")
                    else "text/html; charset=utf-8" if filename.endswith(".html")
                    else "application/octet-stream")
                if filename.endswith(".html"):
                    self.send_header("Content-Security-Policy",
                                     "sandbox allow-scripts; default-src 'none'; "
                                     "script-src 'unsafe-inline'; style-src 'unsafe-inline'")
                self.send_header("Content-Length", str(len(data)))
                self._security_headers()
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in (
                "/api/benchmark", "/api/resume", "/api/cancel",
                "/api/llama-server/start", "/api/llama-server/stop"):
            return self.send_error(404)
        if not self._valid_host() or not self._valid_origin():
            return self._json({"error": "forbidden origin"}, 403)
        if not hmac.compare_digest(self.headers.get("X-CSRF-Token", ""), CSRF_TOKEN):
            return self._json({"error": "missing or invalid CSRF token"}, 403)
        if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin"):
            return self._json({"error": "cross-site request rejected"}, 403)
        if self.headers.get_content_type() != "application/json":
            return self._json({"error": "application/json required"}, 415)
        try:
            raw_size = self.headers.get("Content-Length")
            if raw_size is None or not raw_size.isdigit():
                raise ValueError
            size = int(raw_size)
            if size > MAX_REQUEST_SIZE:
                return self._json({"error": "request too large"}, 413)
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError
            payload = json.loads(body or b"{}")
            if not isinstance(payload, dict):
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "invalid JSON"}, 400)
        if path == "/api/benchmark":
            requested_model = payload.get("model")
            if not isinstance(requested_model, dict):
                return self._json({"error": "select a model explicitly"}, 400)
            try:
                model = resolve_model(requested_model)
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            profile = payload.get("profile", "STANDARD")
            if not isinstance(profile, str) or profile.upper() not in (
                    "QUICK", "STANDARD", "FULL", "PERFORMANCE", "QUALITY", "REASONING",
                    "CODING", "LONG CONTEXT"):
                return self._json({"error": "invalid benchmark profile"}, 400)
            clean_payload = {
                "model": model, "profile": profile.upper(),
                "seed": payload.get("seed", 12345),
                "generation_options": payload.get("generation_options", {}),
                "repeat_runs": payload.get("repeat_runs", 1),
            }
            try:
                clean_payload["generation_options"] = normalize_generation_options(
                    clean_payload["generation_options"], include_defaults=False)
            except GenerationOptionsError as exc:
                return self._json({"error": f"{exc.field}: {exc}", "field": exc.field}, 400)
            try:
                clean_payload["seed"] = int(clean_payload["seed"])
            except (TypeError, ValueError, OverflowError):
                return self._json({"error": "invalid benchmark seed"}, 400)
            if not 0 <= clean_payload["seed"] <= 2_147_483_647:
                return self._json({"error": "invalid benchmark seed"}, 400)
            if not manager.start(clean_payload):
                return self._json({"error": "benchmark already running"}, 409)
            return self._json({"started": True})
        if path == "/api/resume":
            if manager.resume():
                return self._json({"started": True})
            return self._json({"error": "no resumable benchmark"}, 404)
        if path == "/api/cancel":
            return self._json({"cancelled": manager.cancel()})
        if path == "/api/llama-server/start":
            try:
                status = llama_server_manager.start(
                    payload.get("source"), payload.get("executable_path"),
                    payload.get("package_id"),
                    payload.get("model_path"), payload.get("context_size"),
                    payload.get("gpu_layers", 0))
            except LlamaServerConflict as exc:
                return self._json({"error": str(exc)}, 409)
            except LlamaServerError as exc:
                return self._json({"error": str(exc)}, 400)
            return self._json(status)
        if path == "/api/llama-server/stop":
            try:
                return self._json(llama_server_manager.stop())
            except LlamaServerConflict as exc:
                return self._json({"error": str(exc)}, 409)
        self.send_error(404)

    def log_message(self, *_):
        return


if __name__ == "__main__":
    atexit.register(llama_server_manager.stop)
    port = int(os.environ.get("PORT", "8000"))
    print(f"Local AI Benchmark running at http://localhost:{port}")
    BoundedThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
