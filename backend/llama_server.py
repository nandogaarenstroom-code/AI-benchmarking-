"""Managed, loopback-only llama.cpp server process lifecycle."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
WINGET_PACKAGE_ID = "ggml.llamacpp"
STARTUP_TIMEOUT = 180


class LlamaServerError(ValueError):
    """A requested llama.cpp server operation could not be completed."""


class LlamaServerConflict(LlamaServerError):
    """A managed server is already running."""


class LlamaServerManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._configuration = None
        self._output_lines = deque(maxlen=20)
        self._output_thread = None
        self._status = {"state": "stopped", "message": "No app-managed llama-server is running."}

    def status(self):
        with self._lock:
            if self._process is not None and self._process.poll() is not None:
                self._process = None
                self._configuration = None
                self._status = {"state": "stopped", "message": "The managed llama-server exited."}
            return {**self._status}

    def managed_launch(self, endpoint):
        with self._lock:
            if (endpoint != f"http://{SERVER_HOST}:{SERVER_PORT}/v1" or
                    self._process is None or self._process.poll() is not None):
                return None
            return {
                "source": self._configuration["source"],
                "model_file": self._configuration["model_file"],
                "context_size_requested": self._configuration["context_size"],
                "gpu_layers_requested": self._configuration["gpu_layers"],
                "host": SERVER_HOST,
                "port": SERVER_PORT,
            }

    def start(self, source, executable_path, model_path, context_size, gpu_layers):
        if sys.platform != "win32":
            raise LlamaServerError("Starting llama-server from the dashboard is supported on Windows only.")
        source = source if source in ("path", "winget") else None
        if source is None:
            raise LlamaServerError("Choose either an executable path or the winget install option.")
        model = self._validate_model_path(model_path)
        context = self._integer(context_size, "context size", 512, 2_147_483_647)
        layers = self._integer(gpu_layers, "GPU layers", 0, 1000)

        with self._lock:
            if (self._status["state"] in ("starting", "stopping") or
                    (self._process is not None and self._process.poll() is None)):
                raise LlamaServerConflict("An app-managed llama-server is already running.")
            self._process = None
            self._configuration = None
            self._status = {"state": "starting", "message": "Preparing llama-server."}

        process = None
        try:
            self._ensure_port_available()
            if source == "winget":
                executable = self._install_with_winget()
            else:
                executable = self._validate_executable_path(executable_path)
            command = [
                str(executable), "--host", SERVER_HOST, "--port", str(SERVER_PORT),
                "--model", str(model), "--ctx-size", str(context),
                "--n-gpu-layers", str(layers),
            ]
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, shell=False,
                **({"creationflags": subprocess.CREATE_NO_WINDOW}
                   if hasattr(subprocess, "CREATE_NO_WINDOW") else {}),
            )
            with self._lock:
                self._process = process
                self._output_lines.clear()
                self._output_thread = threading.Thread(
                    target=self._drain_output, args=(process,), daemon=True)
                self._output_thread.start()
                self._configuration = {
                    "source": source, "model_file": model.name,
                    "context_size": context, "gpu_layers": layers,
                }
                self._status = {
                    "state": "starting",
                    "message": "llama-server started; waiting for its health endpoint.",
                    "configuration": self.managed_launch(
                        f"http://{SERVER_HOST}:{SERVER_PORT}/v1"),
                }
            if not self._wait_until_ready(process):
                self._stop_process(process)
                if self._output_thread:
                    self._output_thread.join(timeout=1)
                detail = "\n".join(self._output_lines).strip()
                message = "llama-server did not become healthy before the startup timeout."
                if detail:
                    message += f" Recent server output: {detail[-2000:]}"
                with self._lock:
                    self._process = None
                    self._configuration = None
                    self._status = {"state": "error", "message": message}
                process = None
                raise LlamaServerError(self._status["message"])
            with self._lock:
                self._status = {
                    "state": "running", "message": "llama-server is ready.",
                    "configuration": self.managed_launch(
                        f"http://{SERVER_HOST}:{SERVER_PORT}/v1"),
                }
                return {**self._status}
        except Exception as exc:
            if process is not None:
                self._stop_process(process)
            with self._lock:
                if self._process is process:
                    self._process = None
                    self._configuration = None
                if self._process is None:
                    self._status = {"state": "error", "message": str(exc)[:500]}
            if isinstance(exc, LlamaServerError):
                raise
            raise LlamaServerError(f"Could not start llama-server: {exc}") from exc

    def stop(self):
        with self._lock:
            process = self._process
            if process is None and self._status["state"] == "starting":
                raise LlamaServerConflict("llama-server startup is still in progress.")
            if process is None or process.poll() is not None:
                self._process = None
                self._configuration = None
                self._status = {"state": "stopped", "message": "No app-managed llama-server is running."}
                return {**self._status}
            self._status = {"state": "stopping", "message": "Stopping the app-managed llama-server."}
        self._stop_process(process)
        with self._lock:
            self._process = None
            self._configuration = None
            self._status = {"state": "stopped", "message": "The app-managed llama-server was stopped."}
            return {**self._status}

    @staticmethod
    def _integer(value, label, minimum, maximum):
        try:
            if isinstance(value, bool) or str(value).strip() != str(int(value)):
                raise ValueError
            result = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise LlamaServerError(f"{label.capitalize()} must be a whole number.") from exc
        if not minimum <= result <= maximum:
            raise LlamaServerError(f"{label.capitalize()} must be between {minimum} and {maximum}.")
        return result

    @staticmethod
    def _validate_executable_path(value):
        if not isinstance(value, str) or not value.strip():
            raise LlamaServerError("Enter the full path to llama-server.exe.")
        path = Path(value.strip()).expanduser()
        if not path.is_absolute() or ".." in path.parts:
            raise LlamaServerError("Use an absolute llama-server.exe path without parent-directory segments.")
        try:
            path = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise LlamaServerError("The llama-server.exe path does not exist.") from exc
        if not path.is_file() or path.name.lower() != "llama-server.exe":
            raise LlamaServerError("Select an existing file named llama-server.exe.")
        return path

    @staticmethod
    def _validate_model_path(value):
        if not isinstance(value, str) or not value.strip():
            raise LlamaServerError("Enter the full path to a GGUF model file.")
        path = Path(value.strip()).expanduser()
        if not path.is_absolute() or ".." in path.parts:
            raise LlamaServerError("Use an absolute model path without parent-directory segments.")
        try:
            path = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise LlamaServerError("The model file path does not exist.") from exc
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise LlamaServerError("Select an existing .gguf model file.")
        return path

    @staticmethod
    def _ensure_port_available():
        try:
            with socket.socket() as probe:
                probe.bind((SERVER_HOST, SERVER_PORT))
        except OSError as exc:
            raise LlamaServerError(
                f"Port {SERVER_PORT} is already in use. Stop the existing server or choose another setup."
            ) from exc

    @staticmethod
    def _install_with_winget():
        winget = shutil.which("winget")
        if not winget:
            raise LlamaServerError("winget was not found. Install App Installer or choose an executable path.")
        command = [
            winget, "install", "--exact", "--id", WINGET_PACKAGE_ID,
            "--accept-source-agreements", "--accept-package-agreements",
            "--disable-interactivity",
        ]
        try:
            result = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=600, check=False,
                shell=False,
                **({"creationflags": subprocess.CREATE_NO_WINDOW}
                   if hasattr(subprocess, "CREATE_NO_WINDOW") else {}),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LlamaServerError(f"winget installation failed: {exc}") from exc
        if result.returncode != 0:
            output = (result.stdout or "").strip()[-500:]
            raise LlamaServerError(
                "winget could not install llama.cpp." + (f" {output}" if output else "")
            )
        executable = LlamaServerManager._find_installed_executable()
        if executable is None:
            raise LlamaServerError(
                "winget completed, but llama-server.exe was not found. Try Refresh or enter its path."
            )
        return LlamaServerManager._validate_executable_path(str(executable))

    @staticmethod
    def _find_installed_executable():
        found = shutil.which("llama-server.exe")
        if found:
            return Path(found)
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            return None
        winget_root = Path(local_app_data) / "Microsoft" / "WinGet"
        link = winget_root / "Links" / "llama-server.exe"
        if link.is_file():
            return link
        packages = winget_root / "Packages"
        if packages.is_dir():
            for package_dir in packages.glob(f"{WINGET_PACKAGE_ID}_*"):
                for candidate in package_dir.rglob("llama-server.exe"):
                    if candidate.is_file():
                        return candidate
        return None

    @staticmethod
    def _wait_until_ready(process):
        deadline = time.monotonic() + STARTUP_TIMEOUT
        health_url = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(health_url, timeout=1) as response:
                    if response.status == 200:
                        return True
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.5)
        return False

    @staticmethod
    def _stop_process(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _drain_output(self, process):
        if process.stdout is None:
            return
        for line in process.stdout:
            with self._lock:
                self._output_lines.append(line.rstrip())


llama_server_manager = LlamaServerManager()
