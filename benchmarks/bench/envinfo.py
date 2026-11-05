"""What the numbers were measured on (written beside them, so that they can be read as
belonging to this machine and these versions)."""

from __future__ import annotations

import datetime
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _memory_gb() -> Any:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError):
        pass
    return None


def _version_line(command: str) -> Any:
    path = shutil.which(command)
    if not path:
        return None
    try:
        done = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=20)
        return done.stdout.splitlines()[0] if done.stdout else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> Dict[str, Any]:
    import numpy

    import seamsplit
    from seamsplit.audio import engine_info

    info: Dict[str, Any] = {
        "date": datetime.date.today().isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_threads": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "numpy": numpy.__version__,
        "seamsplit": getattr(seamsplit, "__version__", "unknown"),
        "audio_engine": engine_info(),
    }
    try:
        import pydub  # noqa: F401

        info["pydub"] = getattr(pydub, "__version__", "installed")
        info["ffmpeg_on_path"] = _version_line("ffmpeg")
    except ImportError:
        info["pydub"] = None
    return info
