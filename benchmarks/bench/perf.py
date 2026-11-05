"""Experiment 5, speed and memory of the audio layer and of the split and merge, on about
five minutes of real music, each in a fresh process (so that peak memory is its own).

The same phases can be run against another version of the package by pointing
``legacy_src`` at its ``src`` folder; without it only the current code is measured.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:  # not imported for real: a version under comparison may have no seamsplit.audio
    from .corpus import Song


HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(HERE)
RUN_PY = os.path.join(BENCH_DIR, "run.py")
TARGET_SECONDS = 300


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_mb() -> float:
    """The most memory this process has had, in MB. ``VmHWM`` and not ``ru_maxrss``: a process
    that is started by another one inherits the other's high-water mark in ``ru_maxrss``."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _audio_segment():
    try:
        from seamsplit.audio import AudioSegment  # the current code

        return AudioSegment
    except ImportError:
        from pydub import AudioSegment  # type: ignore[no-redef]  # no seamsplit.audio

        return AudioSegment


def build_inputs(songs: "List[Song]", folder: str) -> Dict[str, str]:
    """``long.wav``, ``long.flac`` and ``long.mp3`` (320 kbit/s): the songs of the corpus
    one after the other until there are about five minutes."""
    os.makedirs(folder, exist_ok=True)
    paths = {kind: os.path.join(folder, f"long.{kind}") for kind in ("wav", "flac", "mp3")}
    if all(os.path.isfile(p) for p in paths.values()):
        return paths
    from seamsplit.audio import AudioSegment

    parts = [song.mix.set_frame_rate(44100).set_channels(2).set_sample_width(2) for song in songs]
    total = AudioSegment.silent(0, 44100).set_channels(2)
    while len(total) < TARGET_SECONDS * 1000:
        for part in parts:
            total = total + part
    total = total[: TARGET_SECONDS * 1000]
    total.export(paths["wav"], format="wav")
    total.export(paths["flac"], format="flac")
    total.export(paths["mp3"], format="mp3", bitrate="320k")
    return paths


def _clock(fn: Callable[[], Any]) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def phase_decode(kind: str):
    def run(inputs: Dict[str, str], scratch: str) -> float:
        AudioSegment = _audio_segment()
        return _clock(lambda: AudioSegment.from_file(inputs[kind]).frame_count())

    return run


def phase_encode(kind: str):
    def run(inputs: Dict[str, str], scratch: str) -> float:
        AudioSegment = _audio_segment()
        audio = AudioSegment.from_file(inputs["wav"])
        out = os.path.join(scratch, f"encoded.{kind}")
        extra = {"bitrate": "320k"} if kind == "mp3" else {}
        return _clock(lambda: audio.export(out, format=kind, **extra))

    return run


def _split(inputs: Dict[str, str], scratch: str, source: str, fmt: str, **options: Any) -> float:
    from seamsplit.splitter import SplitConfig, split_audio

    config = SplitConfig(
        output_dir=os.path.join(scratch, f"chunks_{fmt}"), export_format=fmt, **options
    )
    return _clock(lambda: split_audio(inputs[source], config))


def phase_split_default(inputs: Dict[str, str], scratch: str) -> float:
    return _split(inputs, scratch, "mp3", "mp3")


def phase_split_voice_bar(inputs: Dict[str, str], scratch: str) -> Optional[float]:
    import dataclasses

    from seamsplit.splitter import SplitConfig

    if "snap_bar" not in {f.name for f in dataclasses.fields(SplitConfig)}:
        return None  # a version without these options cannot be timed on this phase
    return _split(inputs, scratch, "wav", "flac", cut_by="voice", snap_bar=True)


def phase_merge(inputs: Dict[str, str], scratch: str) -> Optional[float]:
    """Joining the chunks of ``inputs["merge_chunks"]`` (made beforehand, so that what is
    measured is the merge and nothing the split needed)."""
    import inspect

    from seamsplit.merger import merge_audio

    chunks = inputs["merge_chunks"]
    out = os.path.join(scratch, "merged.flac")
    extra = {"pairing": "off"} if "pairing" in inspect.signature(merge_audio).parameters else {}
    manifest = os.path.join(chunks, "manifest.json")
    return _clock(lambda: merge_audio(chunks, manifest, out, **extra))


PHASES: Dict[str, Callable[[Dict[str, str], str], Optional[float]]] = {
    "decode_wav": phase_decode("wav"),
    "decode_flac": phase_decode("flac"),
    "decode_mp3": phase_decode("mp3"),
    "encode_wav": phase_encode("wav"),
    "encode_flac": phase_encode("flac"),
    "encode_mp3": phase_encode("mp3"),
    "split_default": phase_split_default,
    "split_voice_bar": phase_split_voice_bar,
    "merge": phase_merge,
}


def run_phase_here(name: str, inputs: Dict[str, str], scratch: str) -> Dict[str, Any]:
    """In the child process: run the phase and report its seconds and the peak memory."""
    os.makedirs(scratch, exist_ok=True)
    seconds = PHASES[name](inputs, scratch)
    return {"seconds": seconds, "peak_rss_mb": _peak_rss_mb()}


def measure(
    name: str,
    inputs: Dict[str, str],
    scratch: str,
    legacy_src: Optional[str] = None,
    repeats: int = 3,
) -> Dict[str, Any]:
    env = dict(os.environ)
    if legacy_src:
        env["PYTHONPATH"] = legacy_src + os.pathsep + env.get("PYTHONPATH", "")
    runs = []
    for _ in range(repeats):
        done = subprocess.run(
            [sys.executable, RUN_PY, "_phase", name, json.dumps(inputs), scratch],
            capture_output=True,
            text=True,
            env=env,
        )
        if done.returncode != 0:
            return {"error": (done.stderr or done.stdout).strip().splitlines()[-1:]}
        run = json.loads(done.stdout.strip().splitlines()[-1])
        if run["seconds"] is None:
            return {"not_available": True}
        runs.append(run)
    return {
        "seconds_median": statistics.median(r["seconds"] for r in runs),
        "seconds_runs": [round(r["seconds"], 3) for r in runs],
        "peak_rss_mb": max(r["peak_rss_mb"] for r in runs),
    }


def run(
    songs: List[Song], folder: str, legacy_src: Optional[str] = None, repeats: int = 3
) -> Dict[str, Any]:
    inputs = build_inputs(songs, os.path.join(folder, "perf"))
    scratch = os.path.join(folder, "perf", "scratch")
    from seamsplit.audio import AudioSegment
    from seamsplit.splitter import SplitConfig, split_audio

    merge_chunks = os.path.join(folder, "perf", "merge_chunks")
    if not os.path.isfile(os.path.join(merge_chunks, "manifest.json")):
        split_audio(inputs["wav"], SplitConfig(output_dir=merge_chunks, export_format="flac"))
    inputs = dict(inputs, merge_chunks=merge_chunks)

    result: Dict[str, Any] = {
        "input_seconds": len(AudioSegment.from_file(inputs["wav"])) / 1000.0,
        "input_files": {k: os.path.getsize(v) for k, v in inputs.items() if k != "merge_chunks"},
        "repeats": repeats,
        "current": {},
    }
    for name in PHASES:
        result["current"][name] = measure(name, inputs, scratch, None, repeats)
    if legacy_src:
        result["legacy"] = {}
        for name in PHASES:
            result["legacy"][name] = measure(name, inputs, scratch, legacy_src, repeats)
    return result


def run_executable(exe: str, song: "Song", folder: str, repeats: int = 5) -> Dict[str, Any]:
    """The one-file executable: its size, how long it takes to start, and a split done
    with *nothing* on the path (no ffmpeg, no Python)."""
    scratch = os.path.join(folder, "exe_scratch")
    os.makedirs(scratch, exist_ok=True)
    bare = {"PATH": "/nonexistent", "HOME": scratch}

    def timed(args: List[str]) -> Dict[str, Any]:
        t = time.perf_counter()
        done = subprocess.run([exe] + args, capture_output=True, text=True, env=bare, timeout=600)
        return {
            "seconds": time.perf_counter() - t,
            "code": done.returncode,
            "out": done.stdout[-200:],
        }

    starts = [timed(["--version"]) for _ in range(repeats)]
    chunks = os.path.join(scratch, "chunks")
    split = timed(
        [
            "split",
            song.path,
            "--out",
            chunks,
            "--format",
            "flac",
            "--target",
            "8",
            "--before-slack",
            "3",
            "--after-slack",
            "1",
            "--min-final",
            "3",
        ]
    )
    return {
        "path": os.path.basename(exe),
        "sha256": _sha256(exe),
        "size_mb": os.path.getsize(exe) / 1024 / 1024,
        "start_seconds_median": statistics.median(s["seconds"] for s in starts),
        "start_seconds_first": starts[0]["seconds"],
        "start_ok": all(s["code"] == 0 for s in starts),
        "split_with_empty_path": {
            "seconds": split["seconds"],
            "exit_code": split["code"],
            "chunks_written": len([f for f in os.listdir(chunks)]) if os.path.isdir(chunks) else 0,
        },
    }
