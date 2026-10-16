"""Managed, loopback-only llama.cpp server process lifecycle."""
from __future__ import annotations

import ctypes
import os
import re
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
        self._job_handle = None
        self._status = {"state": "stopped", "message": "No app-managed llama-server is running."}

    def status(self):
        exited_process = None
        with self._lock:
            if self._process is not None and self._process.poll() is not None:
                exited_process = self._process
                self._status = {
                    "state": "stopping", "message": "Cleaning up the exited managed llama-server."}
        if exited_process is not None:
            self._cleanup_owned_process(exited_process)
            with self._lock:
                if self._process is exited_process:
                    self._process = None
                    self._configuration = None
                    self._status = {
                        "state": "stopped", "message": "The managed llama-server exited."}
        with self._lock:
            return {**self._status}

    def managed_launch(self, endpoint):
        with self._lock:
            if (endpoint != f"http://{SERVER_HOST}:{SERVER_PORT}/v1" or
                    self._process is None or self._process.poll() is not None):
                return None
            return {
                "source": self._configuration["source"],
                "package_id": self._configuration["package_id"],
                "model_file": self._configuration["model_file"],
                "context_size_requested": self._configuration["context_size"],
                "gpu_layers_requested": self._configuration["gpu_layers"],
                "host": SERVER_HOST,
                "port": SERVER_PORT,
            }

    def start(self, source, executable_path, package_id, model_path, context_size, gpu_layers):
        if sys.platform != "win32":
            raise LlamaServerError("Starting llama-server from the dashboard is supported on Windows only.")
        source = source if source in ("path", "winget") else None
        if source is None:
            raise LlamaServerError("Choose an executable path or an installed WinGet package.")
        package_id = self._validate_package_id(package_id) if source == "winget" else None
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
                executable = self._find_installed_executable(package_id)
                if executable is None:
                    raise LlamaServerError(
                        f"llama-server.exe was not found in the installed WinGet package "
                        f"{package_id!r}. Check the package ID or use an explicit executable path."
                    )
                executable = self._validate_executable_path(str(executable))
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
                self._job_handle = self._assign_kill_on_close_job(process)
            with self._lock:
                self._process = process
                self._output_lines.clear()
                self._output_thread = threading.Thread(
                    target=self._drain_output, args=(process,), daemon=True)
                self._output_thread.start()
                self._configuration = {
                    "source": "winget-installed" if source == "winget" else "path",
                    "package_id": package_id, "model_file": model.name,
                    "context_size": context, "gpu_layers": layers,
                }
                self._status = {
                    "state": "starting",
                    "message": "llama-server started; waiting for its health endpoint.",
                    "configuration": self.managed_launch(
                        f"http://{SERVER_HOST}:{SERVER_PORT}/v1"),
                }
            if not self._wait_until_ready(process):
                self._cleanup_owned_process(process)
                with self._lock:
                    self._process = None
                    self._configuration = None
                    self._status = {
                        "state": "error",
                        "message": "llama-server did not become healthy before the startup timeout.",
                    }
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
                self._cleanup_owned_process(process)
            with self._lock:
                if self._process is process:
                    self._process = None
                    self._configuration = None
                if self._process is None:
                    self._status = {
                        "state": "error",
                        "message": (str(exc)[:500] if isinstance(exc, LlamaServerError)
                                    else "Could not start llama-server; internal details were not exposed."),
                    }
            if isinstance(exc, LlamaServerError):
                raise
            raise LlamaServerError("Could not start llama-server; internal details were not exposed.") from exc

    def stop(self):
        with self._lock:
            process = self._process
            if process is None and self._status["state"] == "starting":
                raise LlamaServerConflict("llama-server startup is still in progress.")
            if process is None:
                self._process = None
                self._configuration = None
                self._status = {"state": "stopped", "message": "No app-managed llama-server is running."}
                return {**self._status}
            if process.poll() is not None:
                self._process = None
                self._configuration = None
                self._status = {"state": "stopped", "message": "The managed llama-server exited."}
                already_exited = True
            else:
                already_exited = False
                self._status = {"state": "stopping", "message": "Stopping the app-managed llama-server."}
        self._cleanup_owned_process(process)
        with self._lock:
            self._process = None
            self._configuration = None
            self._status = {
                "state": "stopped",
                "message": ("The managed llama-server exited." if already_exited else
                            "The app-managed llama-server was stopped."),
            }
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
    def _validate_package_id(value):
        if (not isinstance(value, str) or
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value.strip())):
            raise LlamaServerError("Enter a valid installed WinGet package ID.")
        return value.strip()

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
    def _find_installed_executable(package_id):
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            return None
        winget_root = Path(local_app_data) / "Microsoft" / "WinGet"
        packages = winget_root / "Packages"
        if packages.is_dir():
            package_dirs = [path for path in packages.iterdir()
                            if path.is_dir() and path.name.casefold().startswith(
                                package_id.casefold() + "_")]
            for package_dir in package_dirs:
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

    @staticmethod
    def _assign_kill_on_close_job(process):
        if not hasattr(ctypes, "WinDLL"):
            return None
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = 0x2000
        try:
            if (not kernel32.SetInformationJobObject(
                    job, 9, ctypes.byref(info), ctypes.sizeof(info)) or
                    not kernel32.AssignProcessToJobObject(
                        job, wintypes.HANDLE(process._handle))):
                raise ctypes.WinError(ctypes.get_last_error())
            return job
        except Exception:
            kernel32.CloseHandle(job)
            raise

    def _cleanup_owned_process(self, process):
        self._stop_process(process)
        with self._lock:
            output_thread = self._output_thread
            job_handle = self._job_handle
            self._job_handle = None
        if output_thread is not None and output_thread is not threading.current_thread():
            output_thread.join(timeout=2)
        if process.stdout is not None and (output_thread is None or not output_thread.is_alive()):
            process.stdout.close()
        if job_handle is not None and hasattr(ctypes, "WinDLL"):
            try:
                from ctypes import wintypes
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.CloseHandle(job_handle)
            except OSError:
                pass
        with self._lock:
            if self._output_thread is output_thread:
                self._output_thread = None

    def _drain_output(self, process):
        if process.stdout is None:
            return
        with process.stdout:
            for line in process.stdout:
                with self._lock:
                    self._output_lines.append(line.rstrip())


llama_server_manager = LlamaServerManager()
