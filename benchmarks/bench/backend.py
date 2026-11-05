"""Experiment 6, the whole program with a real local backend: ``seamsplit run`` with the
``audio-separator`` preset on one song, from the audio file to merged stems, with nothing
done by hand, and then run again to see that there is nothing left to do.

This is the check that the preset (the command line it builds, how it finds the files the
program writes, how the chunks are paired) works with the real program and not only with the
stand-in the tests use; and the numbers show what the one-command workflow delivers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from seamsplit.audio import AudioSegment

from . import metrics
from .corpus import Song

TARGET_S = 8
OVERLAP_S = 2
SYSTEM_MODEL_DIR = "/tmp/audio-separator-models"  # where the command line program looks first


def _command(song: Song, job: str, model: str) -> List[str]:
    return [
        sys.executable,
        "-m",
        "seamsplit",
        "run",
        song.path,
        "--job",
        job,
        "--backend",
        "audio-separator",
        "--backend-model",
        model,
        "--target",
        str(TARGET_S),
        "--before-slack",
        "3",
        "--after-slack",
        "1",
        "--overlap",
        str(OVERLAP_S),
        "--min-final",
        "3",
        "--format",
        "flac",
        "--out-format",
        "flac",
        "--anchor",
        "--report",
        os.path.join(job, "report.json"),
    ]


def _shown(command: List[str], song: Song, job: str) -> str:
    """The command as it can be copied: without the paths of this machine."""
    text = " ".join(command[3:])
    return "seamsplit " + text.replace(song.path, "mix.mp3").replace(job, "JOB")


def _find(folder: str, word: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    for name in sorted(os.listdir(folder)):
        if word in name.lower():
            return os.path.join(folder, name)
    return None


def run(
    songs: List[Song], separator: Any, cache_dir: str, separator_python: Optional[str]
) -> Dict[str, Any]:
    if separator_python is None or not hasattr(separator, "model_dir"):
        return {"skipped": "needs a real separator (--separator-python)"}
    candidates = [s for s in songs if s.vocals is not None]
    if not candidates:
        return {"skipped": "no song with ground truth among the selected songs"}
    song = candidates[0]
    model = separator.model
    os.makedirs(SYSTEM_MODEL_DIR, exist_ok=True)
    target = os.path.join(SYSTEM_MODEL_DIR, model)
    if not os.path.exists(target):
        shutil.copyfile(os.path.join(separator.model_dir, model), target)

    job = os.path.join(cache_dir, "backend_job", song.id)
    if os.path.isdir(job):
        shutil.rmtree(job)
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(os.path.abspath(separator_python)) + os.pathsep + env["PATH"]
    command = _command(song, job, model)

    t = time.perf_counter()
    first = subprocess.run(command, capture_output=True, text=True, env=env)
    first_s = time.perf_counter() - t
    t = time.perf_counter()
    second = subprocess.run(command, capture_output=True, text=True, env=env)
    second_s = time.perf_counter() - t

    result: Dict[str, Any] = {
        "song": song.id,
        "command": _shown(command, song, job),
        "first_run": {"seconds": first_s, "exit_code": first.returncode},
        "second_run": {"seconds": second_s, "exit_code": second.returncode},
    }
    if first.returncode != 0:
        result["error"] = (first.stderr or first.stdout).strip().splitlines()[-5:]
        return result
    out = os.path.join(job, "output")
    result["output_files"] = sorted(os.listdir(out)) if os.path.isdir(out) else []
    whole = separator.stems_for([song.path])[0]
    scores: Dict[str, Any] = {}
    for stem, truth in (("instrumental", song.instrumental), ("vocals", song.vocals)):
        path = _find(out, stem)
        if path is None:
            scores[stem] = {"error": "no output file with that name"}
            continue
        joined = AudioSegment.from_file(path)
        reference = AudioSegment.from_file(whole[stem])
        scores[stem] = {
            "vs_whole_db": metrics.snr_db(reference, joined),
            "vs_truth_db": metrics.snr_db(truth, joined),
            "whole_vs_truth_db": metrics.snr_db(truth, reference),
        }
    result["scores"] = scores
    report_path = os.path.join(job, "report.json")
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            result["merge_report"] = _summary_of(json.load(f))
    return result


def _summary_of(report: Dict[str, Any]) -> Dict[str, Any]:
    """What the merge said about the job: how many seams and chunks, how many it flagged."""
    boundaries = report.get("boundaries", [])
    chunks = report.get("chunks", [])
    gains = [c["gain_db"] for c in chunks if c.get("gain_db") is not None]
    return {
        "keys": sorted(report),
        "seams": len(boundaries),
        "seams_flagged": sum(1 for b in boundaries if b.get("flagged")),
        "chunks": len(chunks),
        "chunks_flagged": sum(1 for c in chunks if c.get("flagged")),
        "chunk_gain_db_range": [min(gains), max(gains)] if gains else None,
        "warnings": report.get("warnings", []),
    }
