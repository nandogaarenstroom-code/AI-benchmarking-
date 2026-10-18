import io
import tempfile
import threading
import unittest
import ctypes
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.llama_server import (
    HELP_OUTPUT_LIMIT_BYTES, LAUNCH_FLAG_NAMES, LlamaServerConflict, LlamaServerError, LlamaServerManager,
    SERVER_HOST, SERVER_PORT,
)


class LlamaServerManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.executable = self.root / "llama-server.exe"
        self.executable.touch()
        self.model = self.root / "model.gguf"
        self.model.touch()
        self.manager = LlamaServerManager()
        flags = {flag: {"takes_value": flag not in {
            "--mlock", "--no-mmap", "--jinja", "--flash-attn"},"choices": []}
                 for flag in ("--host", "--port", "--model", "--ctx-size", "--n-gpu-layers",
                              *LAUNCH_FLAG_NAMES.values(), "--threads", "--cache-type-k")}
        flags["--flash-attn"] = {"takes_value": False, "choices": []}
        self.manager._help_capabilities = MagicMock(
            return_value={"flags": flags, "supported": sorted(flags)})

    def tearDown(self):
        self.manager.stop()
        self.temp.cleanup()

    def test_start_uses_validated_arguments_and_stops_only_managed_process(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.LlamaServerManager._ensure_port_available"), \
                patch("backend.llama_server.LlamaServerManager._wait_until_ready", return_value=True), \
                patch.object(self.manager, "_assign_kill_on_close_job",
                             return_value="job-handle"), \
                patch.object(self.manager, "_close_job_object") as close_job, \
                patch("backend.llama_server.subprocess.Popen", return_value=process) as popen:
            status = self.manager.start(
                "path", str(self.executable), None, str(self.model), "65536", 24)
            self.assertEqual(status["state"], "running")
            args = popen.call_args.args[0]
            self.assertEqual(args, [
                str(self.executable.resolve()), "--host", SERVER_HOST, "--port", str(SERVER_PORT),
                "--model", str(self.model.resolve()), "--ctx-size", "65536",
                "--n-gpu-layers", "24",
            ])
            self.assertFalse(popen.call_args.kwargs["shell"])
            launch = self.manager.managed_launch(f"http://{SERVER_HOST}:{SERVER_PORT}/v1")
            self.assertEqual(launch["context_size_requested"], 65536)
            self.assertEqual(launch["model_file"], "model.gguf")
            self.assertEqual(launch["argv"], args)
            stopped = self.manager.stop()
        self.assertEqual(stopped["state"], "stopped")
        process.terminate.assert_called_once()
        process.wait.assert_called()
        close_job.assert_called_once_with("job-handle")

    def test_help_parser_handles_older_and_newer_layouts(self):
        older = """Options:
  --threads N             number of threads
  --flash-attn            enable flash attention
  --mlock                 lock model in memory
  --model FILE            model path
"""
        newer = """--threads N                 Set threads
--flash-attn [off|on|auto]   Flash attention mode
--cache-type-k TYPE          K cache quantization
--jinja                      Enable Jinja
"""
        old = LlamaServerManager.parse_help(older)["flags"]
        new = LlamaServerManager.parse_help(newer)["flags"]
        self.assertTrue(old["--threads"]["takes_value"])
        self.assertFalse(old["--flash-attn"]["takes_value"])
        self.assertTrue(old["--model"]["takes_value"])
        self.assertEqual(new["--flash-attn"]["choices"], ["off", "on", "auto"])
        self.assertTrue(new["--flash-attn"]["takes_value"])
        self.assertFalse(new["--jinja"]["takes_value"])

    def test_help_command_is_argv_only_and_output_is_capped(self):
        process = MagicMock()
        process.stdout = io.BytesIO(b"--threads N\n" + b"x" * 64)
        process.poll.return_value = 0
        with patch("backend.llama_server.HELP_OUTPUT_LIMIT_BYTES", 12), \
                patch("backend.llama_server.subprocess.Popen", return_value=process) as popen:
            output = LlamaServerManager._read_help(self.executable)
        self.assertEqual(len(output.encode()), 12)
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0], [str(self.executable), "--help"])
        self.assertFalse(popen.call_args.kwargs["shell"])

    def test_boolean_flash_attention_rejects_unavailable_modes(self):
        flags = {"--flash-attn": {"takes_value": False, "choices": []}}
        for mode in ("off", "auto"):
            with self.subTest(mode=mode), self.assertRaises(LlamaServerError):
                self.manager._build_launch_args({"flash_attn": mode}, {"flags": flags})

    def test_help_is_cached_by_executable_path_and_mtime(self):
        manager = LlamaServerManager()
        with patch.object(manager, "_read_help", return_value="--threads N") as read:
            first = manager._help_capabilities(self.executable)
            second = manager._help_capabilities(self.executable)
            self.assertEqual(first, second)
            read.assert_called_once_with(self.executable)

    def test_launch_options_generate_argv_for_each_allowlisted_flag(self):
        flags = {
            "--threads": {"takes_value": True, "choices": []},
            "--threads-batch": {"takes_value": True, "choices": []},
            "--batch-size": {"takes_value": True, "choices": []},
            "--ubatch-size": {"takes_value": True, "choices": []},
            "--flash-attn": {"takes_value": True, "choices": ["off", "on", "auto"]},
            "--cache-type-k": {"takes_value": True, "choices": []},
            "--cache-type-v": {"takes_value": True, "choices": []},
            "--parallel": {"takes_value": True, "choices": []},
            "--mlock": {"takes_value": False, "choices": []},
            "--no-mmap": {"takes_value": False, "choices": []},
            "--tensor-split": {"takes_value": True, "choices": []},
            "--main-gpu": {"takes_value": True, "choices": []},
            "--n-cpu-moe": {"takes_value": True, "choices": []},
            "--jinja": {"takes_value": False, "choices": []},
            "--chat-template": {"takes_value": True, "choices": []},
            "--rope-scaling": {"takes_value": True, "choices": []},
            "--rope-scale": {"takes_value": True, "choices": []},
            "--threads-extra": {"takes_value": True, "choices": []},
        }
        options = {
            "threads": 4, "threads_batch": 3, "batch_size": 512, "ubatch_size": 128,
            "flash_attn": "auto", "cache_type_k": "q8_0", "cache_type_v": "f16",
            "parallel": 2, "mlock": True, "no_mmap": True, "tensor_split": "0.5,0.5",
            "main_gpu": 0, "n_cpu_moe": 2, "jinja": True, "chat_template": "llama3",
            "rope_scaling": "yarn", "rope_scale": 2,
            "extra_args": ["--threads-extra", "5"],
        }
        result = self.manager._build_launch_args(options, {"flags": flags})
        self.assertEqual(result, [
            "--threads", "4", "--threads-batch", "3", "--batch-size", "512",
            "--ubatch-size", "128", "--flash-attn", "auto", "--cache-type-k", "q8_0",
            "--cache-type-v", "f16", "--parallel", "2", "--mlock", "--no-mmap",
            "--tensor-split", "0.5,0.5", "--main-gpu", "0", "--n-cpu-moe", "2",
            "--jinja", "--chat-template", "llama3", "--rope-scaling", "yarn",
            "--rope-scale", "2.0", "--threads-extra", "5",
        ])

    def test_launch_options_reject_unsupported_and_injection_attempts(self):
        with self.assertRaisesRegex(LlamaServerError, "does not support"):
            self.manager._build_launch_args(
                {"threads": 4}, {"flags": {}})
        for extra in (
                ["--model", "evil.gguf"], ["--custom", "C:\\secret"],
                ["--custom", "127.0.0.1"], ["--custom", "1;touch"],
                ["--custom", "--host"],
                ["--unknown"]):
            flags = {
                "--model": {"takes_value": True, "choices": []},
                "--custom": {"takes_value": True, "choices": []},
            }
            with self.subTest(extra=extra), self.assertRaises(LlamaServerError):
                self.manager._build_launch_args({"extra_args": extra}, {"flags": flags})

    def test_failure_output_is_limited_and_sanitizes_full_model_path(self):
        process = MagicMock()
        process.poll.return_value = None

        def fail_ready(*_):
            self.manager._output_lines.extend(
                [f"failed to load {self.model.resolve()}"] * 24)
            return False

        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.LlamaServerManager._ensure_port_available"), \
                patch("backend.llama_server.LlamaServerManager._wait_until_ready",
                      side_effect=fail_ready), \
                patch("backend.llama_server.subprocess.Popen", return_value=process):
            with self.assertRaises(LlamaServerError) as raised:
                self.manager.start(
                    "path", str(self.executable), None, str(self.model), 4096, 0)
        self.assertEqual(len(raised.exception.output_lines), 20)
        self.assertTrue(all("<local-path>" in line for line in raised.exception.output_lines))

    def test_configurable_port_is_bounded_and_recorded(self):
        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.LlamaServerManager._ensure_port_available"), \
                patch("backend.llama_server.LlamaServerManager._wait_until_ready", return_value=True), \
                patch("backend.llama_server.subprocess.Popen") as popen:
            process = MagicMock()
            process.poll.return_value = None
            popen.return_value = process
            status = self.manager.start(
                "path", str(self.executable), None, str(self.model), 4096, 0, port=9123)
            argv = popen.call_args.args[0]
            self.assertEqual(argv[argv.index("--port") + 1], "9123")
            self.assertEqual(status["configuration"]["port"], 9123)
        with patch("backend.llama_server.sys.platform", "win32"):
            with self.assertRaisesRegex(LlamaServerError, "between 1024 and 65535"):
                LlamaServerManager().start(
                    "path", str(self.executable), None, str(self.model), 4096, 0, port=80)

    def test_path_validation_rejects_non_server_exe_and_non_gguf_model(self):
        wrong_exe = self.root / "other.exe"
        wrong_exe.touch()
        wrong_model = self.root / "model.bin"
        wrong_model.touch()
        with patch("backend.llama_server.sys.platform", "win32"):
            with self.assertRaisesRegex(LlamaServerError, "absolute"):
                self.manager.start("path", "llama-server.exe", None, str(self.model), 4096, 0)
            with self.assertRaisesRegex(LlamaServerError, "llama-server.exe"):
                self.manager.start("path", str(wrong_exe), None, str(self.model), 4096, 0)
            with self.assertRaisesRegex(LlamaServerError, ".gguf"):
                self.manager.start("path", str(self.executable), None, str(wrong_model), 4096, 0)
            with self.assertRaisesRegex(LlamaServerError, "between"):
                self.manager.start("path", str(self.executable), None, str(self.model), 1, 0)
            with self.assertRaisesRegex(LlamaServerError, "between"):
                self.manager.start("path", str(self.executable), None, str(self.model), 4096, 0, port=65536)

    def test_start_is_windows_only(self):
        with patch("backend.llama_server.sys.platform", "linux"):
            with self.assertRaisesRegex(LlamaServerError, "Windows only"):
                self.manager.start("path", str(self.executable), None, str(self.model), 4096, 0)

    def test_startup_timeout_stops_child_and_clears_managed_state(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.LlamaServerManager._ensure_port_available"), \
                patch("backend.llama_server.LlamaServerManager._wait_until_ready", return_value=False), \
                patch("backend.llama_server.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(LlamaServerError, "startup timeout"):
                self.manager.start("path", str(self.executable), None, str(self.model), 4096, 0)
        self.assertEqual(self.manager.status()["state"], "error")
        self.assertIsNone(self.manager.managed_launch(f"http://{SERVER_HOST}:{SERVER_PORT}/v1"))
        process.terminate.assert_called_once()

    def test_unexpected_exit_closes_owned_output_and_job_resources(self):
        process = MagicMock()
        process.poll.return_value = 1
        self.manager._process = process
        self.manager._job_handle = "job-handle"
        self.manager._job_process = process
        with patch.object(self.manager, "_close_job_object") as close_job:
            status = self.manager.status()
        self.assertEqual(status["state"], "stopped")
        process.terminate.assert_not_called()
        process.stdout.close.assert_called_once()
        close_job.assert_called_once_with("job-handle")

    def test_start_cleans_exited_process_before_installing_replacement_job(self):
        old_process = MagicMock()
        old_process.poll.return_value = 1
        new_process = MagicMock()
        new_process.poll.return_value = None
        self.manager._process = old_process
        self.manager._job_handle = "old-job"
        self.manager._job_process = old_process
        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.LlamaServerManager._ensure_port_available"), \
                patch("backend.llama_server.LlamaServerManager._wait_until_ready", return_value=True), \
                patch.object(self.manager, "_assign_kill_on_close_job", return_value="new-job"), \
                patch.object(self.manager, "_close_job_object") as close_job, \
                patch("backend.llama_server.subprocess.Popen", return_value=new_process):
            self.manager.start("path", str(self.executable), None, str(self.model), 4096, 0)
            self.assertIs(self.manager._job_process, new_process)
            self.assertEqual(self.manager._job_handle, "new-job")
            self.manager.stop()
        close_job.assert_has_calls([unittest.mock.call("old-job"), unittest.mock.call("new-job")])
        old_process.stdout.close.assert_called_once()

    def test_start_is_blocked_while_stop_cleans_exited_process(self):
        process = MagicMock()
        process.poll.return_value = 1
        self.manager._process = process
        entered_cleanup = threading.Event()
        finish_cleanup = threading.Event()

        def wait_for_cleanup(_process):
            entered_cleanup.set()
            finish_cleanup.wait(timeout=2)

        with patch("backend.llama_server.sys.platform", "win32"), \
                patch.object(self.manager, "_cleanup_owned_process", side_effect=wait_for_cleanup):
            stopper = threading.Thread(target=self.manager.stop)
            stopper.start()
            self.assertTrue(entered_cleanup.wait(timeout=2))
            with self.assertRaisesRegex(LlamaServerConflict, "already running"):
                self.manager.start("path", str(self.executable), None, str(self.model), 4096, 0)
            self.assertIs(self.manager._process, process)
            finish_cleanup.set()
            stopper.join(timeout=2)
        self.assertFalse(stopper.is_alive())
        self.assertEqual(self.manager.status()["state"], "stopped")

    def test_windows_job_object_uses_kill_on_close_for_owned_process(self):
        process = MagicMock()
        process._handle = 1234
        kernel32 = MagicMock()
        kernel32.CreateJobObjectW.return_value = ctypes.c_void_p(5678)
        kernel32.SetInformationJobObject.return_value = 1
        kernel32.AssignProcessToJobObject.return_value = 1
        with patch.object(ctypes, "WinDLL", return_value=kernel32, create=True):
            handle = self.manager._assign_kill_on_close_job(process)
        self.assertEqual(handle.value, 5678)
        job_call = kernel32.SetInformationJobObject.call_args.args
        self.assertEqual(job_call[1], 9)
        self.assertEqual(ctypes.string_at(job_call[2], job_call[3])[16:20], b"\x00\x20\x00\x00")
        self.assertEqual(kernel32.AssignProcessToJobObject.call_args.args[1].value, 1234)

    def test_winget_source_launches_existing_package_executable(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.LlamaServerManager._find_installed_executable",
                      return_value=self.executable) as find_executable, \
                patch("backend.llama_server.LlamaServerManager._ensure_port_available"), \
                patch("backend.llama_server.LlamaServerManager._wait_until_ready", return_value=True), \
                patch("backend.llama_server.subprocess.Popen", return_value=process):
            result = self.manager.start("winget", "", "custom.fork", str(self.model), 32768, 0)
        self.assertEqual(result["state"], "running")
        find_executable.assert_called_once_with("custom.fork")
        self.assertEqual(result["configuration"]["package_id"], "custom.fork")

    def test_winget_source_requires_valid_id_and_finds_existing_package(self):
        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.LlamaServerManager._ensure_port_available"):
            with self.assertRaisesRegex(LlamaServerError, "package ID"):
                self.manager.start("winget", "", "../../fork", str(self.model), 32768, 0)
        self.assertEqual(LlamaServerManager._validate_package_id("ggml.llamacpp"), "ggml.llamacpp")
        self.assertEqual(LlamaServerManager._validate_package_id("vendor.custom-fork_3"), "vendor.custom-fork_3")

    def test_finds_server_executable_under_selected_winget_package(self):
        package_root = (
            self.root / "Microsoft" / "WinGet" / "Packages" /
            "vendor.custom-fork_1.0.0_x64")
        package_root.mkdir(parents=True)
        executable = package_root / "bin" / "llama-server.exe"
        executable.parent.mkdir()
        executable.touch()
        with patch.dict("os.environ", {"LOCALAPPDATA": str(self.root)}):
            found = LlamaServerManager._find_installed_executable("vendor.custom-fork")
        self.assertEqual(found, executable)


if __name__ == "__main__":
    unittest.main()
