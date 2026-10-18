import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import hardware


class HardwareTests(unittest.TestCase):
    def test_fixed_commands_have_timeout_and_never_use_shell(self):
        with patch("backend.hardware.subprocess.run", return_value=subprocess.CompletedProcess(
                ["sysctl"], 0, "42", "")) as run:
            self.assertEqual(hardware._command(["sysctl", "-n", "hw.memsize"]), "42")
        self.assertEqual(run.call_args.kwargs["timeout"], 4)
        self.assertFalse(run.call_args.kwargs["shell"])
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("probe", 4)):
            with patch("backend.hardware.subprocess.run", side_effect=error):
                self.assertEqual(hardware._command(["nvidia-smi"]), "")
        with patch("backend.hardware.subprocess.run", return_value=subprocess.CompletedProcess(
                ["probe"], 0, None, None)):
            self.assertEqual(hardware._command(["nvidia-smi"]), "")

    def test_linux_cpu_ram_and_nvidia_vram_are_detected(self):
        cpu = "model name : Test CPU\nphysical id : 0\ncore id : 0\n\n" * 2

        def read(path):
            return {"/proc/cpuinfo": cpu, "/proc/meminfo": "MemTotal: 16777216 kB\n"}.get(str(path), "")

        with patch("backend.hardware.platform.system", return_value="Linux"), \
                patch("backend.hardware._read", side_effect=read), \
                patch("backend.hardware._command", return_value="Test GPU, 8192, 555.1"), \
                patch("backend.hardware.os.cpu_count", return_value=2):
            result = hardware.hardware_info()
        self.assertEqual(result["cpu"], "Test CPU")
        self.assertEqual(result["cores"], 1)
        self.assertEqual(result["threads"], 2)
        self.assertEqual(result["ram_bytes"], 16 * hardware.GIB)
        self.assertEqual(result["gpu_vram_bytes"], 8 * hardware.GIB)
        self.assertEqual(result["gpu"], "Test GPU")
        self.assertEqual(result["cuda"], "Unknown")

    def test_windows_cim_cpu_memory_gpu_without_unreliable_vram(self):
        data = {"cpu": [{"Name": "Windows CPU", "NumberOfCores": 8,
                         "NumberOfLogicalProcessors": 16}],
                "ram": 32 * hardware.GIB,
                "gpu": [{"Name": "Integrated GPU", "AdapterRAM": 4293918720,
                         "DriverVersion": "1.2"}]}

        def command(args):
            return json.dumps(data) if args[0] == "powershell.exe" else ""

        with patch("backend.hardware.platform.system", return_value="Windows"), \
                patch("backend.hardware._command", side_effect=command):
            result = hardware.hardware_info()
        self.assertEqual(result["cpu"], "Windows CPU")
        self.assertEqual(result["cores"], 8)
        self.assertEqual(result["threads"], 16)
        self.assertEqual(result["ram_bytes"], 32 * hardware.GIB)
        self.assertEqual(result["gpu"], "Integrated GPU")
        self.assertIsNone(result["gpu_vram_bytes"])

    def test_linux_drm_reports_amd_vram_without_vendor_tool(self):
        device = Path("/sys/class/drm/card0/device")

        def read(path):
            return {
                str(device / "product_name"): "AMD Radeon",
                str(device / "mem_info_vram_total"): str(16 * hardware.GIB),
            }.get(str(path), "")

        with patch("backend.hardware.platform.system", return_value="Linux"), \
                patch("backend.hardware._read", side_effect=read), \
                patch("backend.hardware._command", return_value=""), \
                patch("backend.hardware.Path.glob", return_value=[device]):
            result = hardware.hardware_info()
        self.assertEqual(result["gpu"], "AMD Radeon")
        self.assertEqual(result["gpu_vram_bytes"], 16 * hardware.GIB)
        self.assertEqual(result["rocm"], "Unknown")

    def test_mac_unified_memory_does_not_invent_dedicated_vram(self):
        def command(args):
            if args[0] == "sysctl":
                return {"machdep.cpu.brand_string": "Apple M4", "hw.memsize": str(24 * hardware.GIB),
                        "hw.physicalcpu": "10", "hw.logicalcpu": "10"}[args[-1]]
            if args[0] == "system_profiler":
                return json.dumps({"SPDisplaysDataType": [{"sppci_model": "Apple M4"}]})
            return ""

        with patch("backend.hardware.platform.system", return_value="Darwin"), \
                patch("backend.hardware._command", side_effect=command):
            result = hardware.hardware_info()
        self.assertEqual(result["ram_bytes"], 24 * hardware.GIB)
        self.assertEqual(result["gpu"], "Apple M4")
        self.assertEqual(result["gpu_vram"], "Unknown")
        self.assertTrue(any("shared" in note for note in result["detection_notes"]))

    def test_mac_discrete_vram_is_reported(self):
        info = {"gpus": [], "detection_notes": []}

        def command(args):
            return json.dumps({"SPDisplaysDataType": [
                {"sppci_model": "AMD Radeon", "spdisplays_vram": "8 GB"}]}) if args[0] == "system_profiler" else ""

        with patch("backend.hardware._command", side_effect=command):
            hardware._macos(info)
        self.assertEqual(info["gpus"][0]["vram_bytes"], 8 * hardware.GIB)

    def test_missing_tools_and_unknown_platform_are_honest(self):
        with patch("backend.hardware.platform.system", return_value="Other"), \
                patch("backend.hardware.platform.processor", return_value=""), \
                patch("backend.hardware._command", return_value=""), \
                patch("backend.hardware.os.cpu_count", return_value=None):
            result = hardware.hardware_info()
        for key in ("cpu", "cores", "threads", "ram", "gpu", "gpu_vram"):
            self.assertEqual(result[key], "Unknown")
        self.assertIsNone(result["ram_bytes"])
        self.assertIsNone(result["gpu_vram_bytes"])

    def test_multiple_gpus_do_not_pretend_their_memory_is_one_pool(self):
        with patch("backend.hardware.platform.system", return_value="Other"), \
                patch("backend.hardware._command", return_value="GPU 1, 8192, 1\nGPU 2, 8192, 1"):
            result = hardware.hardware_info()
        self.assertEqual(len(result["gpus"]), 2)
        self.assertEqual(result["gpu_vram_bytes"], 8 * hardware.GIB)

    def test_platform_probe_failures_degrade_to_unknown(self):
        with patch("backend.hardware.platform.processor", side_effect=AttributeError("unavailable")), \
                patch("backend.hardware.platform.platform", side_effect=OSError("unavailable")), \
                patch("backend.hardware.platform.system", return_value="Other"), \
                patch("backend.hardware._command", return_value=""):
            result = hardware.hardware_info()
        self.assertEqual(result["cpu"], "Unknown")
        self.assertEqual(result["os"], "Unknown")

    def test_recommendations_scale_output_but_never_guarantee_fit(self):
        small = hardware.recommend_settings({})
        medium = hardware.recommend_settings({"ram_bytes": 16 * hardware.GIB, "cores": 8})
        large = hardware.recommend_settings({"ram_bytes": 64 * hardware.GIB, "cores": 32,
                                             "gpu_vram_bytes": 24 * hardware.GIB, "gpus": [{}]})
        self.assertEqual(small["generation_options"]["max_tokens"], 2048)
        self.assertEqual(small["threads"], 1)
        self.assertLess(medium["generation_options"]["max_tokens"], large["generation_options"]["max_tokens"])
        self.assertTrue(large["advisory"])
        self.assertEqual(medium["generation_options"]["num_thread"], 8)
        self.assertEqual(medium["threads"], 8)
        self.assertEqual(large["context_size"], large["generation_options"]["num_ctx"])
        self.assertEqual(large["gpu_layers"], large["runtime_settings"]["gpu_layers"])
        self.assertGreater(large["runtime_settings"]["gpu_layers"], 0)
        self.assertTrue(any("quantization" in note for note in large["notes"]))
        self.assertTrue(any("external" in note for note in large["notes"]))
        for settings in (small, medium, large):
            self.assertGreater(settings["generation_options"]["num_ctx"],
                               settings["generation_options"]["max_tokens"])
