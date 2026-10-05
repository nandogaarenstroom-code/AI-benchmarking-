"""Startup checks for the dependency-free local benchmark."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def check_environment(results_dir):
    """Return actionable warnings without preventing the demo from starting."""
    warnings = []
    if sys.version_info < (3, 10):
        warnings.append("Python 3.10 or newer is recommended.")
    path = Path(results_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        warnings.append(f"Results directory is not writable: {exc}")
    if os.environ.get("BENCHMARK_USE_DOCKER") == "1":
        if not shutil.which("docker"):
            warnings.append("BENCHMARK_USE_DOCKER=1 but Docker is not available; using fallback execution.")
    return warnings
