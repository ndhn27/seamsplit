"""Separating files for the benchmarks: with a real model, run by ``audio-separator`` in its
own Python environment (:mod:`bench.worker`), and remembered by the file's SHA-256 so that
running the benchmarks again does not separate again.

Anything with a ``stems_for(paths)`` method and a ``name`` can stand in (the tests use a
fake)."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from .corpus import BENCH_DIR, sha256_of

DEFAULT_MODEL = "UVR-MDX-NET-Voc_FT.onnx"
DEFAULT_CACHE_DIR = os.path.join(BENCH_DIR, "cache")  # not in the repository (see .gitignore)
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")


class SeparationFailed(RuntimeError):
    pass


class LocalSeparator:
    """``audio-separator`` with one UVR model, in the Python environment ``python`` (the one
    that has ``audio-separator`` installed, which need not be the one SeamSplit runs in)."""

    def __init__(
        self,
        python: str,
        model: str = DEFAULT_MODEL,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ) -> None:
        self.python = python
        self.model = model
        self.cache_dir = cache_dir
        self.name = f"audio-separator / {model}"
        self.model_dir = os.path.join(cache_dir, "models")
        self.stems_dir = os.path.join(cache_dir, "stems", os.path.splitext(model)[0])
        self.loaded_s: Optional[float] = None

    def _paths(self, source: str) -> Dict[str, str]:
        base = os.path.join(self.stems_dir, sha256_of(source)[:24])
        return {
            "vocals": base + "_vocals.flac",
            "instrumental": base + "_instrumental.flac",
            "meta": base + ".json",
        }

    def stems_for(self, sources: Sequence[str]) -> List[Dict[str, Any]]:
        """For each file: ``{"vocals": path, "instrumental": path, "seconds": s}``, separating
        the ones that have not been separated yet (all in one run of the worker)."""
        todo: List[Dict[str, str]] = []
        queued = set()
        for source in sources:
            p = self._paths(source)
            if p["meta"] not in queued and not all(os.path.isfile(v) for v in p.values()):
                queued.add(p["meta"])
                todo.append({"input": source, **p})
        if todo:
            self._run(todo)
        out = []
        for source in sources:
            p = self._paths(source)
            with open(p["meta"], "r", encoding="utf-8") as f:
                meta = json.load(f)
            out.append({"vocals": p["vocals"], "instrumental": p["instrumental"], **meta})
        return out

    def _run(self, jobs: List[Dict[str, str]]) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.stems_dir, exist_ok=True)
        spec_path = os.path.join(self.cache_dir, "jobs.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump({"model": self.model, "model_dir": self.model_dir, "jobs": jobs}, f)
        print(f"  separating {len(jobs)} file(s) with {self.model} ...", flush=True)
        log_path = os.path.join(self.cache_dir, "worker.log")
        log = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            [self.python, WORKER, spec_path],
            stdout=subprocess.PIPE,
            stderr=log,  # a file, not a pipe: the model's progress bars must not fill a pipe
            text=True,
        )
        by_input = {job["input"]: job for job in jobs}
        assert process.stdout is not None
        for line in process.stdout:
            if not line.startswith("RESULT "):
                continue
            info = json.loads(line[len("RESULT ") :])
            if "loaded_s" in info:
                self.loaded_s = info["loaded_s"]
            elif info.get("input") in by_input:
                # written as soon as a file is done, so that a run that is stopped (or a
                # machine that goes away) can be taken up again where it was
                with open(by_input[info["input"]]["meta"], "w", encoding="utf-8") as f:
                    json.dump({"seconds": info.get("seconds")}, f)
        process.wait()
        log.close()
        if process.returncode != 0:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read()[-1500:]
            raise SeparationFailed(f"the separator failed ({process.returncode}):\n{tail}")


def model_fingerprint(separator: Any) -> Optional[str]:
    """SHA-256 of the model file, for the record, when there is one."""
    path = os.path.join(getattr(separator, "model_dir", ""), getattr(separator, "model", ""))
    return sha256_of(path) if os.path.isfile(path) else None


def separator_versions(python: str) -> Dict[str, str]:
    """The versions of ``audio-separator`` and what it runs on, from its own environment."""
    code = (
        "import json, importlib.metadata as m\n"
        "out = {}\n"
        "for n in ('audio-separator', 'onnxruntime', 'torch', 'numpy'):\n"
        "    try:\n"
        "        out[n] = m.version(n)\n"
        "    except m.PackageNotFoundError:\n"
        "        pass\n"
        "print(json.dumps(out))\n"
    )
    done = subprocess.run([python, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}


__all__ = [
    "DEFAULT_MODEL",
    "LocalSeparator",
    "SeparationFailed",
    "model_fingerprint",
    "separator_versions",
]
