import subprocess
import tempfile
import unittest
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
                patch("backend.llama_server.subprocess.Popen", return_value=process) as popen:
            status = self.manager.start(
                "path", str(self.executable), str(self.model), "65536", 24)
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

    def test_path_validation_rejects_non_server_exe_and_non_gguf_model(self):
        wrong_exe = self.root / "other.exe"
        wrong_exe.touch()
        wrong_model = self.root / "model.bin"
        wrong_model.touch()
        with patch("backend.llama_server.sys.platform", "win32"):
            with self.assertRaisesRegex(LlamaServerError, "llama-server.exe"):
                self.manager.start("path", str(wrong_exe), str(self.model), 4096, 0)
            with self.assertRaisesRegex(LlamaServerError, ".gguf"):
                self.manager.start("path", str(self.executable), str(wrong_model), 4096, 0)
            with self.assertRaisesRegex(LlamaServerError, "between"):
                self.manager.start("path", str(self.executable), str(self.model), 1, 0)

    def test_start_is_windows_only(self):
        with patch("backend.llama_server.sys.platform", "linux"):
            with self.assertRaisesRegex(LlamaServerError, "Windows only"):
                self.manager.start("path", str(self.executable), str(self.model), 4096, 0)

    def test_startup_timeout_stops_child_and_clears_managed_state(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.LlamaServerManager._ensure_port_available"), \
                patch("backend.llama_server.LlamaServerManager._wait_until_ready", return_value=False), \
                patch("backend.llama_server.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(LlamaServerError, "startup timeout"):
                self.manager.start("path", str(self.executable), str(self.model), 4096, 0)
        self.assertEqual(self.manager.status()["state"], "error")
        self.assertIsNone(self.manager.managed_launch(f"http://{SERVER_HOST}:{SERVER_PORT}/v1"))
        process.terminate.assert_called_once()

    def test_winget_installs_fixed_package_then_uses_located_executable(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.shutil.which", return_value="C:/Windows/winget.exe"), \
                patch("backend.llama_server.subprocess.run",
                      return_value=subprocess.CompletedProcess([], 0, "installed")) as run, \
                patch("backend.llama_server.LlamaServerManager._find_installed_executable",
                      return_value=self.executable), \
                patch("backend.llama_server.LlamaServerManager._ensure_port_available"), \
                patch("backend.llama_server.LlamaServerManager._wait_until_ready", return_value=True), \
                patch("backend.llama_server.subprocess.Popen", return_value=process):
            result = self.manager.start("winget", "", str(self.model), 32768, 0)
        self.assertEqual(result["state"], "running")
        self.assertIn("ggml.llamacpp", run.call_args.args[0])
        self.assertFalse(run.call_args.kwargs["shell"])

    def test_winget_requires_command_and_handles_install_failure(self):
        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.shutil.which", return_value=None):
            with self.assertRaisesRegex(LlamaServerError, "winget was not found"):
                self.manager.start("winget", "", str(self.model), 32768, 0)
        with patch("backend.llama_server.sys.platform", "win32"), \
                patch("backend.llama_server.shutil.which", return_value="winget"), \
                patch("backend.llama_server.subprocess.run",
                      return_value=subprocess.CompletedProcess([], 1, "package unavailable")):
            with self.assertRaisesRegex(LlamaServerError, "package unavailable"):
                self.manager.start("winget", "", str(self.model), 32768, 0)


if __name__ == "__main__":
    unittest.main()
