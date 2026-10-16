import tempfile
import unittest
import ctypes
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.llama_server import (
    LlamaServerError, LlamaServerManager, SERVER_HOST, SERVER_PORT,
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
            stopped = self.manager.stop()
        self.assertEqual(stopped["state"], "stopped")
        process.terminate.assert_called_once()
        process.wait.assert_called()
        close_job.assert_called_once_with("job-handle")

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
        with patch.object(self.manager, "_close_job_object") as close_job:
            status = self.manager.status()
        self.assertEqual(status["state"], "stopped")
        process.terminate.assert_not_called()
        process.stdout.close.assert_called_once()
        close_job.assert_called_once_with("job-handle")

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
