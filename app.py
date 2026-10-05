"""Zero-interaction local AI benchmark suite.

The server intentionally uses only the Python standard library so it can run on
an offline workstation with ``python app.py``.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from backend.benchmark import BenchmarkManager
from backend.discovery import discover_models

ROOT = Path(__file__).parent
manager = BenchmarkManager(ROOT / "results")


class Handler(BaseHTTPRequestHandler):
    def _json(self, value, status=200):
        body = json.dumps(value, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name):
        path = ROOT / "frontend" / name
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path == "/api/models":
            return self._json({"models": discover_models()})
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
            file_path = manager.root / Path(path).name
            if file_path.exists() and file_path.is_file():
                data = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf" if file_path.suffix == ".pdf" else "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            size = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(size) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "invalid JSON"}, 400)
        if path == "/api/benchmark":
            if manager.running:
                return self._json({"error": "benchmark already running"}, 409)
            threading.Thread(target=manager.run, args=(payload,), daemon=True).start()
            return self._json({"started": True})
        if path == "/api/resume":
            if manager.resume():
                return self._json({"started": True})
            return self._json({"error": "no resumable benchmark"}, 404)
        if path == "/api/cancel":
            return self._json({"cancelled": manager.cancel()})
        self.send_error(404)

    def log_message(self, *_):
        return


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Local AI Benchmark running at http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
