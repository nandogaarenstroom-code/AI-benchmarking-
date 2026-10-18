"""Best-effort local hardware inventory using only the standard library."""
from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

UNKNOWN = "Unknown"
GIB = 1024 ** 3


def _command(args):
    """Only callers' fixed argument lists are executed; missing tools are normal."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=4,
                                check=False, shell=False)
        if result.returncode == 0 and len(result.stdout) <= 1_048_576:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        pass
    return ""


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as source:
            return source.read(1_048_576)
    except OSError:
        return ""


def _positive(value):
    try:
        number = int(value)
        return number if number > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _memory_label(value):
    return f"{value / GIB:.1f} GiB" if value else UNKNOWN


def _nvidia_gpus():
    output = _command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                       "--format=csv,noheader,nounits"])
    gpus = []
    for row in csv.reader(output.splitlines()):
        if len(row) == 3 and row[0].strip():
            mib = _positive(row[1].strip())
            gpus.append({"name": row[0].strip(), "vram_bytes": mib * 1024 ** 2 if mib else None,
                         "driver": row[2].strip() or UNKNOWN})
    return gpus


def _linux(info):
    cpu = _read("/proc/cpuinfo")
    names = re.findall(r"^(?:model name|Hardware)\s*:\s*(.+)$", cpu, re.MULTILINE)
    if names:
        info["cpu"] = names[0].strip()
    physical = set()
    for block in cpu.split("\n\n"):
        fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if "physical id" in fields and "core id" in fields:
            physical.add((fields["physical id"], fields["core id"]))
    if physical:
        info["cores"] = len(physical)
    memory = re.search(r"^MemTotal:\s+(\d+)\s+kB", _read("/proc/meminfo"), re.MULTILINE)
    if memory:
        info["ram_bytes"] = int(memory[1]) * 1024
    if not info["gpus"]:
        for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
            name = _read(device / "product_name").strip()
            vendor = _read(device / "vendor").strip()
            device_id = _read(device / "device").strip()
            if not name and vendor and device_id:
                name = f"PCI {vendor}:{device_id}"
            if name:
                info["gpus"].append({
                    "name": name, "vram_bytes": _positive(_read(device / "mem_info_vram_total")),
                    "driver": UNKNOWN})
        if not info["gpus"]:
            for line in _command(["lspci", "-mm"]).splitlines():
                if any(kind in line for kind in ('"VGA compatible controller"',
                                                 '"3D controller"', '"Display controller"')):
                    parts = re.findall(r'"([^"]*)"', line)
                    if len(parts) >= 3:
                        info["gpus"].append({"name": " ".join(parts[1:3]),
                                             "vram_bytes": None, "driver": UNKNOWN})


def _windows(info):
    script = (
        "$ErrorActionPreference='Stop';"
        "$c=@(Get-CimInstance Win32_Processor|Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);"
        "$m=Get-CimInstance Win32_ComputerSystem;"
        "$g=@(Get-CimInstance Win32_VideoController|Select-Object Name,DriverVersion);"
        "[pscustomobject]@{cpu=$c;ram=$m.TotalPhysicalMemory;gpu=$g}|"
        "ConvertTo-Json -Depth 4 -Compress"
    )
    output = _command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])
    try:
        data = json.loads(output)
        cpus = data.get("cpu", [])
        cpus = cpus if isinstance(cpus, list) else [cpus]
        if cpus:
            info["cpu"] = ", ".join(str(cpu["Name"]).strip() for cpu in cpus if cpu.get("Name")) or UNKNOWN
            for field, target in (("NumberOfCores", "cores"),
                                  ("NumberOfLogicalProcessors", "threads")):
                values = [_positive(cpu.get(field)) for cpu in cpus]
                if values and all(values):
                    info[target] = sum(values)
        info["ram_bytes"] = _positive(data.get("ram"))
        if not info["gpus"]:
            gpus = data.get("gpu", [])
            for gpu in gpus if isinstance(gpus, list) else [gpus]:
                if gpu.get("Name"):
                    # CIM AdapterRAM is a 32-bit field and is unreliable for modern GPUs.
                    info["gpus"].append({"name": gpu["Name"], "vram_bytes": None,
                                         "driver": gpu.get("DriverVersion") or UNKNOWN})
    except (ValueError, TypeError, AttributeError, KeyError):
        info["detection_notes"].append("Windows hardware inventory was unavailable.")


def _macos(info):
    for key, target in (("machdep.cpu.brand_string", "cpu"), ("hw.memsize", "ram_bytes"),
                        ("hw.physicalcpu", "cores"), ("hw.logicalcpu", "threads")):
        value = _command(["sysctl", "-n", key])
        if value:
            info[target] = value if target == "cpu" else _positive(value) or UNKNOWN
    if not info["gpus"]:
        try:
            data = json.loads(_command(["system_profiler", "SPDisplaysDataType", "-json"]))
            for gpu in data.get("SPDisplaysDataType", []):
                vram = gpu.get("spdisplays_vram") or gpu.get("spdisplays_vram_shared") or ""
                match = re.search(r"([\d.]+)\s*(GB|MB)", vram, re.IGNORECASE)
                memory = int(float(match[1]) * (GIB if match[2].upper() == "GB" else 1024 ** 2)) if match else None
                info["gpus"].append({"name": gpu.get("sppci_model") or gpu.get("_name") or UNKNOWN,
                                     "vram_bytes": memory, "driver": UNKNOWN})
        except (ValueError, TypeError, AttributeError):
            pass
    if any("Apple" in gpu["name"] for gpu in info["gpus"]):
        info["detection_notes"].append("Apple GPUs use shared system memory; dedicated VRAM is not inferred from RAM.")


def hardware_info():
    """Return detected values, with Unknown/None rather than invented capacity."""
    info = {
        "cpu": platform.processor() or UNKNOWN, "cores": UNKNOWN,
        "threads": os.cpu_count() or UNKNOWN, "architecture": platform.machine() or UNKNOWN,
        "gpu": UNKNOWN, "gpu_vram": UNKNOWN, "ram": UNKNOWN, "os": platform.platform(),
        "driver": UNKNOWN, "cuda": UNKNOWN, "rocm": UNKNOWN,
        "ram_bytes": None, "gpu_vram_bytes": None, "gpus": _nvidia_gpus(),
        "detection_notes": [], "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    detector = {"Linux": _linux, "Windows": _windows, "Darwin": _macos}.get(platform.system())
    if detector:
        try:
            detector(info)
        except (OSError, ValueError, TypeError, AttributeError):
            info["detection_notes"].append("Some hardware details could not be read.")
    info["ram_bytes"] = _positive(info["ram_bytes"])
    info["ram"] = _memory_label(info["ram_bytes"])
    if info["gpus"]:
        info["gpu"] = ", ".join(gpu["name"] for gpu in info["gpus"])
        # A single model cannot generally pool all device memory.
        capacities = [gpu["vram_bytes"] for gpu in info["gpus"] if gpu["vram_bytes"]]
        info["gpu_vram_bytes"] = max(capacities) if capacities else None
        info["gpu_vram"] = _memory_label(info["gpu_vram_bytes"])
        info["driver"] = ", ".join(dict.fromkeys(gpu["driver"] for gpu in info["gpus"]))
    return info


def recommend_settings(hardware=None):
    """Suggest starting points, not a promise that an unspecified model will fit."""
    info = hardware if isinstance(hardware, dict) else hardware_info()
    ram = _positive(info.get("ram_bytes")) or 0
    vram = _positive(info.get("gpu_vram_bytes")) or 0
    capacity = max(ram, vram)
    max_tokens = 16384 if capacity >= 64 * GIB else 8192 if capacity >= 32 * GIB else 4096 if capacity >= 16 * GIB else 2048
    context = 16384 if capacity >= 64 * GIB else 8192 if capacity >= 32 * GIB else 4096
    context = max(context, max_tokens + 2048)
    threads = _positive(info.get("cores")) or _positive(info.get("threads"))
    generation = {"max_tokens": max_tokens, "num_ctx": context}
    if threads:
        generation["num_thread"] = min(threads, 32)
    gpu_layers = 24 if vram >= 16 * GIB else 12 if vram >= 8 * GIB else 4 if vram else 0
    if info.get("gpus"):
        generation["num_gpu"] = gpu_layers
    return {
        "generation_options": generation,
        "runtime_settings": {"context_length": context, "gpu_layers": gpu_layers},
        "advisory": True,
        "notes": [
            "Starting points only: model size, quantization, KV cache and free memory determine fit.",
            "Output allowance is not a requirement to generate that many tokens.",
            "Context must also fit the prompt and the model's supported context window.",
            "GPU layers are advisory; verify actual offload and memory use with the backend.",
            "Ollama accepts runtime options per request; external OpenAI-compatible servers require separate load/startup configuration.",
        ],
    }
