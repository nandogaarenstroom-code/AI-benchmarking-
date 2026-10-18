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
HELP_TIMEOUT_SECONDS = 10
HELP_OUTPUT_LIMIT_BYTES = 262_144
CACHE_TYPE_NAMES = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
CHAT_TEMPLATE_NAMES = {
    "chatml", "llama2", "llama3", "mistral", "gemma", "phi3", "deepseek",
    "command-r", "qwen2", "vicuna", "zephyr",
}
DENIED_EXTRA_FLAGS = {
    "--host", "--port", "--model", "-m", "--api-key", "--api-key-file", "--path",
    "--ssl-key-file", "--ssl-cert-file", "--log-file", "--hf-repo", "--model-url",
}
LAUNCH_FLAG_NAMES = {
    "threads": "--threads", "threads_batch": "--threads-batch",
    "batch_size": "--batch-size", "ubatch_size": "--ubatch-size",
    "flash_attn": "--flash-attn", "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v", "parallel": "--parallel", "mlock": "--mlock",
    "no_mmap": "--no-mmap", "tensor_split": "--tensor-split", "main_gpu": "--main-gpu",
    "n_cpu_moe": "--n-cpu-moe", "jinja": "--jinja", "chat_template": "--chat-template",
    "rope_scaling": "--rope-scaling", "rope_scale": "--rope-scale",
}


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
        self._output_process = None
        self._job_handle = None
        self._job_process = None
        self._help_cache = {}
        self._selected_executable = None
        self._selected_source = None
        self._selected_package_id = None
        self._port = SERVER_PORT
        self._status = {"state": "stopped", "message": "No app-managed llama-server is running."}

    def status(self):
        exited_process = None
        with self._lock:
            if self._status["state"] == "stopping":
                return {**self._status}
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
            return {**self._status, "output_lines": self._sanitized_output_lines()}

    def managed_launch(self, endpoint):
        with self._lock:
            if (endpoint != f"http://{SERVER_HOST}:{self._port}/v1" or
                    self._process is None or self._process.poll() is not None):
                return None
            return {
                "source": self._configuration["source"],
                "package_id": self._configuration["package_id"],
                "model_file": self._configuration["model_file"],
                "context_size_requested": self._configuration["context_size"],
                "gpu_layers_requested": self._configuration["gpu_layers"],
                "host": SERVER_HOST,
                "port": self._port,
                "argv": list(self._configuration.get("argv", [])),
                "parsed_from_log": dict(self._configuration.get("parsed_from_log", {})),
            }

    @property
    def endpoint(self):
        with self._lock:
            return f"http://{SERVER_HOST}:{self._port}/v1"

    def select_executable(self, source, executable_path, package_id):
        source = source if source in ("path", "winget") else None
        if source is None:
            raise LlamaServerError("Choose an executable path or an installed WinGet package.")
        package_id = self._validate_package_id(package_id) if source == "winget" else None
        if source == "winget":
            executable = self._find_installed_executable(package_id)
            if executable is None:
                raise LlamaServerError("llama-server.exe was not found in the selected WinGet package.")
            executable = self._validate_executable_path(str(executable))
        else:
            executable = self._validate_executable_path(executable_path)
        with self._lock:
            self._selected_executable = executable
            self._selected_source = source
            self._selected_package_id = package_id
        return {"selected": True}

    def capabilities(self):
        with self._lock:
            executable = self._selected_executable
        if executable is None:
            raise LlamaServerError("Choose a valid llama-server executable first.")
        return self._help_capabilities(executable)

    def _help_capabilities(self, executable):
        try:
            stat = executable.stat()
        except OSError as exc:
            raise LlamaServerError("The selected llama-server executable is unavailable.") from exc
        key = (str(executable), stat.st_mtime_ns)
        with self._lock:
            cached = self._help_cache.get(key)
        if cached is not None:
            return dict(cached)
        output = self._read_help(executable)
        parsed = self.parse_help(output)
        if not parsed["flags"]:
            raise LlamaServerError("Could not read supported flags from llama-server --help.")
        with self._lock:
            self._help_cache[key] = parsed
        return {"flags": dict(parsed["flags"]), "supported": list(parsed["supported"])}

    @staticmethod
    def parse_help(output):
        flags = {}
        for line in output.splitlines():
            matches = list(re.finditer(r"(?<![\w-])(--[A-Za-z0-9][A-Za-z0-9-]*|-[A-Za-z])", line))
            for match in matches:
                flag = match.group(0)
                tail = line[match.end():]
                choice_match = re.search(r"\[([A-Za-z0-9_|-]+)\]", tail)
                choices = choice_match.group(1).split("|") if choice_match else []
                argument_match = re.match(r"\s*(?:=)?\s*([A-Z][A-Z0-9_-]*)(?![a-z])", tail)
                takes_value = bool(choices or
                                   (argument_match and argument_match.group(1).isupper()))
                if re.search(
                        r"\b(?:FILE|PATH|URL|REPO|ADDRESS|HOST|PORT|IP|SOCKET|ENDPOINT)\b|"
                        r"\b(?:file|path|url|repository|address|host|port|listen|bind|socket)\b",
                        tail, re.I):
                    takes_value = True
                sensitive = bool(re.search(
                    r"\b(?:FILE|PATH|URL|REPO|ADDRESS|HOST|PORT|IP|SOCKET|ENDPOINT)\b|"
                    r"\b(?:file|path|url|repository|address|host|port|listen|bind|socket)\b",
                    tail, re.I))
                flags[flag] = {
                    "takes_value": takes_value, "choices": choices,
                    "sensitive": sensitive,
                }
        return {"flags": flags, "supported": sorted(flags)}

    @staticmethod
    def _read_help(executable):
        try:
            process = subprocess.Popen(
                [str(executable), "--help"], stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        except OSError as exc:
            raise LlamaServerError("Could not execute the selected llama-server --help command.") from exc
        collected = bytearray()
        overflow = threading.Event()

        def read_output():
            while len(collected) <= HELP_OUTPUT_LIMIT_BYTES:
                chunk = process.stdout.read(min(8192, HELP_OUTPUT_LIMIT_BYTES + 1 - len(collected)))
                if not chunk:
                    break
                collected.extend(chunk)
            if len(collected) > HELP_OUTPUT_LIMIT_BYTES:
                overflow.set()

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        reader.join(HELP_TIMEOUT_SECONDS)
        if reader.is_alive() or overflow.is_set():
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
            reader.join(timeout=1)
        elif process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
        if reader.is_alive():
            raise LlamaServerError("Reading llama-server --help exceeded the 10 second limit.")
        if overflow.is_set():
            collected = collected[:HELP_OUTPUT_LIMIT_BYTES]
        return bytes(collected).decode("utf-8", errors="replace")

    @classmethod
    def _build_launch_args(cls, launch_options, capabilities):
        launch_options = launch_options or {}
        if not isinstance(launch_options, dict):
            raise LlamaServerError("Launch options must be an object.")
        supported = capabilities["flags"]
        command_args = []

        def append(flag, value=None, *, required=False):
            if flag not in supported:
                raise LlamaServerError(f"This llama-server build does not support {flag}.")
            if value is None:
                return
            command_args.append(flag)
            if supported[flag]["takes_value"]:
                if value is None:
                    raise LlamaServerError(f"{flag} requires a value.")
                command_args.append(str(value))
            elif value not in (True, "on"):
                raise LlamaServerError(f"{flag} does not accept a value in this build.")

        for key, value in launch_options.items():
            if key == "extra_args":
                command_args.extend(cls._validate_extra_args(value, capabilities))
                continue
            flag = LAUNCH_FLAG_NAMES.get(key)
            if flag is None:
                raise LlamaServerError(f"Unknown llama-server launch option: {key}.")
            if value is None or value == "" or value is False:
                continue
            if key in {"threads", "threads_batch", "batch_size", "ubatch_size",
                       "parallel", "main_gpu", "n_cpu_moe"}:
                value = cls._integer(value, key.replace("_", " "), 0 if key in ("main_gpu", "n_cpu_moe") else 1, 65535)
            elif key in ("cache_type_k", "cache_type_v"):
                if value not in CACHE_TYPE_NAMES:
                    raise LlamaServerError(f"Unsupported {key.replace('_', ' ')} value.")
            elif key == "flash_attn":
                if value not in ("on", "off", "auto", True, False):
                    raise LlamaServerError("Flash attention must be on, off, or auto.")
                if supported.get(flag, {}).get("choices"):
                    value = "on" if value is True else "off" if value is False else value
                    if value not in supported[flag]["choices"]:
                        raise LlamaServerError(f"This build does not support flash attention mode {value}.")
                else:
                    if value == "auto":
                        raise LlamaServerError(
                            "This build exposes flash attention as a boolean flag; auto mode is unsupported.")
                    if value == "off":
                        raise LlamaServerError(
                            "This build exposes flash attention as a boolean flag and cannot explicitly disable it.")
                    value = value in ("on", True)
                    if value is False:
                        continue
            elif key in ("mlock", "no_mmap", "jinja"):
                if not isinstance(value, bool):
                    raise LlamaServerError(f"{key.replace('_', ' ')} must be a boolean.")
                if not value:
                    continue
            elif key == "tensor_split":
                if not isinstance(value, str) or not re.fullmatch(r"\d+(?:\.\d+)?(?:,\d+(?:\.\d+)?)*", value):
                    raise LlamaServerError("Tensor split must be comma-separated numeric values.")
            elif key == "chat_template":
                if value not in CHAT_TEMPLATE_NAMES:
                    raise LlamaServerError("Chat template must be a supported template name.")
            elif key == "rope_scaling":
                if value not in ("none", "linear", "yarn"):
                    raise LlamaServerError("RoPE scaling must be none, linear, or yarn.")
            elif key == "rope_scale":
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise LlamaServerError("RoPE scale must be a number from 0.01 to 100.") from exc
                if not 0.01 <= number <= 100:
                    raise LlamaServerError("RoPE scale must be a number from 0.01 to 100.")
                value = number
            append(flag, value)
        return command_args

    @classmethod
    def _validate_extra_args(cls, args, capabilities):
        if args in (None, ""):
            return []
        if not isinstance(args, list) or len(args) > 64:
            raise LlamaServerError("Extra arguments must be a list of up to 64 tokens.")
        result, index = [], 0
        flags = capabilities["flags"]
        while index < len(args):
            flag = args[index]
            if not isinstance(flag, str) or flag not in flags:
                raise LlamaServerError(f"Extra argument flag is not supported: {flag}.")
            lower = flag.lower()
            description = flags[flag]
            if (description.get("sensitive") or flag in DENIED_EXTRA_FLAGS or any(word in lower for word in
                    ("path", "file", "url", "repo", "address", "host", "port", "model",
                     "key", "cert", "log", "ip", "rpc", "socket", "endpoint", "proxy",
                     "bind", "listen"))):
                raise LlamaServerError(f"Extra argument flag is not allowed: {flag}.")
            if description["takes_value"]:
                index += 1
                if index >= len(args):
                    raise LlamaServerError(f"Extra argument {flag} requires a value.")
                value = args[index]
                if (not isinstance(value, str) or value.startswith("-") or
                        not re.fullmatch(r"[A-Za-z0-9_.:,=+-]{1,128}", value)):
                    raise LlamaServerError(f"Extra argument value for {flag} is invalid.")
                if (value.casefold() == "localhost" or
                        re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value) or
                        (re.fullmatch(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", value) and
                         not re.fullmatch(r"\d+(?:\.\d+)+", value))):
                    raise LlamaServerError(f"Extra argument value for {flag} cannot be a network address.")
                result.extend((flag, value))
            else:
                result.append(flag)
            index += 1
        return result

    def start(self, source, executable_path, package_id, model_path, context_size,
              gpu_layers, launch_options=None, port=SERVER_PORT):
        if sys.platform != "win32":
            raise LlamaServerError("Starting llama-server from the dashboard is supported on Windows only.")
        source = source if source in ("path", "winget") else None
        if source is None:
            raise LlamaServerError("Choose an executable path or an installed WinGet package.")
        package_id = self._validate_package_id(package_id) if source == "winget" else None
        model = self._validate_model_path(model_path)
        context = self._integer(context_size, "context size", 512, 2_147_483_647)
        layers = self._integer(gpu_layers, "GPU layers", 0, 1000)
        port = self._integer(port, "port", 1024, 65535)

        with self._lock:
            if (self._status["state"] in ("starting", "stopping") or
                    (self._process is not None and self._process.poll() is None)):
                raise LlamaServerConflict("An app-managed llama-server is already running.")
            previous_process = self._process
            self._status = {
                "state": "stopping" if previous_process is not None else "starting",
                "message": ("Cleaning up the exited managed llama-server." if previous_process is not None
                            else "Preparing llama-server."),
            }
        if previous_process is not None:
            self._cleanup_owned_process(previous_process)
            with self._lock:
                if self._process is previous_process:
                    self._process = None
                    self._configuration = None
                self._status = {"state": "starting", "message": "Preparing llama-server."}

        process = None
        try:
            self._ensure_port_available(port)
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
            capabilities = self._help_capabilities(executable)
            required_flags = ("--host", "--port", "--model", "--ctx-size", "--n-gpu-layers")
            missing = [flag for flag in required_flags if flag not in capabilities["flags"]]
            if missing:
                raise LlamaServerError(
                    "This llama-server build lacks required launch flags: " + ", ".join(missing))
            extra = self._build_launch_args(launch_options, capabilities)
            command = [str(executable), "--host", SERVER_HOST, "--port", str(port),
                       "--model", str(model), "--ctx-size", str(context),
                       "--n-gpu-layers", str(layers), *extra]
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, shell=False,
                **({"creationflags": subprocess.CREATE_NO_WINDOW}
                   if hasattr(subprocess, "CREATE_NO_WINDOW") else {}),
            )
            with self._lock:
                self._job_handle = self._assign_kill_on_close_job(process)
                self._job_process = process
            with self._lock:
                self._process = process
                self._port = port
                self._selected_executable = executable
                self._selected_source = source
                self._selected_package_id = package_id
                self._output_lines.clear()
                self._output_thread = threading.Thread(
                    target=self._drain_output, args=(process,), daemon=True)
                self._output_process = process
                self._output_thread.start()
                self._configuration = {
                    "source": "winget-installed" if source == "winget" else "path",
                    "package_id": package_id, "model_file": model.name,
                    "context_size": context, "gpu_layers": layers,
                    "launch_options": launch_options or {},
                    "argv": list(command),
                    "parsed_from_log": self._parsed_log_values(),
                }
                self._status = {
                    "state": "starting",
                    "message": "llama-server started; waiting for its health endpoint.",
                    "configuration": self.managed_launch(
                        f"http://{SERVER_HOST}:{port}/v1"),
                }
            if not self._wait_until_ready(process, port):
                self._cleanup_owned_process(process)
                output_lines = self._sanitized_output_lines()
                with self._lock:
                    self._process = None
                    self._configuration = None
                    self._status = {
                        "state": "error",
                        "message": "llama-server did not become healthy before the startup timeout.",
                        "output_lines": output_lines,
                    }
                process = None
                failure = LlamaServerError(self._status["message"])
                failure.output_lines = output_lines
                raise failure
            with self._lock:
                if self._configuration is not None:
                    self._configuration["parsed_from_log"] = self._parsed_log_values()
                self._status = {
                    "state": "running", "message": "llama-server is ready.",
                    "configuration": self.managed_launch(
                        f"http://{SERVER_HOST}:{port}/v1"),
                }
                return {**self._status}
        except Exception as exc:
            if isinstance(exc, LlamaServerError) and not hasattr(exc, "output_lines"):
                exc.output_lines = self._sanitized_output_lines()
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
                        "output_lines": getattr(exc, "output_lines", self._sanitized_output_lines()),
                    }
            if isinstance(exc, LlamaServerError):
                raise
            raise LlamaServerError("Could not start llama-server; internal details were not exposed.") from exc

    def stop(self):
        with self._lock:
            if self._status["state"] == "stopping":
                raise LlamaServerConflict("llama-server cleanup is already in progress.")
            process = self._process
            if process is None and self._status["state"] == "starting":
                raise LlamaServerConflict("llama-server startup is still in progress.")
            if process is None:
                self._process = None
                self._configuration = None
                self._status = {"state": "stopped", "message": "No app-managed llama-server is running."}
                return {**self._status}
            if process.poll() is not None:
                already_exited = True
                self._status = {
                    "state": "stopping", "message": "Cleaning up the exited managed llama-server."}
            else:
                already_exited = False
                self._status = {"state": "stopping", "message": "Stopping the app-managed llama-server."}
        self._cleanup_owned_process(process)
        with self._lock:
            if self._process is process:
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
    def _ensure_port_available(port=SERVER_PORT):
        try:
            with socket.socket() as probe:
                probe.bind((SERVER_HOST, port))
        except OSError as exc:
            raise LlamaServerError(
                f"Port {port} is already in use. Stop the existing server or choose another setup."
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
    def _wait_until_ready(process, port=SERVER_PORT):
        deadline = time.monotonic() + STARTUP_TIMEOUT
        health_url = f"http://{SERVER_HOST}:{port}/health"
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
            if self._output_process is process:
                output_thread = self._output_thread
                self._output_thread = None
                self._output_process = None
            else:
                output_thread = None
            if self._job_process is process:
                job_handle = self._job_handle
                self._job_handle = None
                self._job_process = None
            else:
                job_handle = None
        if output_thread is not None and output_thread is not threading.current_thread():
            output_thread.join(timeout=2)
        if process.stdout is not None and (output_thread is None or not output_thread.is_alive()):
            process.stdout.close()
        if job_handle is not None:
            self._close_job_object(job_handle)

    @staticmethod
    def _close_job_object(job_handle):
        if not hasattr(ctypes, "WinDLL"):
            return
        try:
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle(job_handle)
        except OSError:
            pass

    def _drain_output(self, process):
        if process.stdout is None:
            return
        with process.stdout:
            for line in process.stdout:
                with self._lock:
                    self._output_lines.append(line.rstrip())

    def _sanitized_output_lines(self):
        with self._lock:
            output = list(self._output_lines)[-20:]
            configuration = dict(self._configuration or {})
            executable = str(self._selected_executable or "")
        replacements = [path for path in (executable,) if path]
        if configuration.get("model_file"):
            replacements.append(configuration["model_file"])
        argv = configuration.get("argv", [])
        if "--model" in argv:
            model_index = argv.index("--model") + 1
            if model_index < len(argv):
                replacements.append(argv[model_index])
        sanitized = []
        for line in output:
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(line))
            for value in replacements:
                text = text.replace(value, "<local-path>")
            sanitized.append(text[:1000])
        return sanitized

    def _parsed_log_values(self):
        result = {}
        for line in self._sanitized_output_lines():
            offload = re.search(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers", line, re.I)
            if offload:
                result["offloaded_layers"] = {
                    "value": int(offload.group(1)), "total": int(offload.group(2))}
            if re.search(r"\b(?:build|version)\b", line, re.I) and "build" not in result:
                result["build"] = line[:500]
        return result


llama_server_manager = LlamaServerManager()
