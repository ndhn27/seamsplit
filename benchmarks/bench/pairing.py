"""Experiment 3, the pairing: when a service gives the files back without their names, are
they put in the right order by what they sound like?

Real separated chunks are used, so this measures what ``seamsplit.pairing`` could only be
checked on synthetic songs for (its thresholds were set there):

* the score of every chunk's stem against the original chunk it came from and against the
  others (the margin the thresholds have to fit in), and
* the trial itself: the files are renamed ``download (k).flac`` in a random order, merged
  with the pairing check on (``auto``) and off, and the result compared with the merge of
  the same files under their right names.
"""

from __future__ import annotations

import os
import random
import shutil
import statistics
from typing import Any, Dict, List, Optional

import numpy as np

from seamsplit import pairing as seam_pairing
from seamsplit.audio import AudioSegment
from seamsplit.merger import merge_audio
from seamsplit.report import MergeReport

from . import metrics
from .workset import STEM_NAMES, ChunkSet, SongWork


def _merge(folder: str, chunk_set: ChunkSet, out: str, pairing: str):
    report = MergeReport()
    try:
        merge_audio(
            folder,
            chunk_set.manifest_path,
            out,
            samplerate=44100,
            pairing=pairing,
            report=report,
        )
    except ValueError as exc:
        return None, report, str(exc)
    return AudioSegment.from_file(out), report, None


def score_real_stems(chunk_set: ChunkSet, stem: str) -> Dict[str, Any]:
    """How a real stem of each chunk scores against the originals."""
    files = [stems[stem] for stems in chunk_set.stems]
    scores, _ = seam_pairing.compare(files, chunk_set.manifest["paths"])
    n = len(files)
    true = [float(scores[k, k]) for k in range(n)]
    others = [float(max((scores[k, j] for j in range(n) if j != k), default=0.0)) for k in range(n)]
    return {
        "true": true,
        "runner_up": others,
        "right_part": sum(1 for k in range(n) if int(np.argmax(scores[k])) == k),
        "sure": sum(
            1
            for t, o in zip(true, others)
            if t >= seam_pairing.MIN_PAIR_NCC and t - o >= seam_pairing.MIN_PAIR_MARGIN
        ),
        "chunks": n,
    }


def trial(
    chunk_set: ChunkSet, stem: str, work_dir: str, perm: List[int], tag: str
) -> Dict[str, Any]:
    """One renaming: ``download (k)`` is the file of chunk ``perm[k]``."""
    base = os.path.join(work_dir, chunk_set.song_id, chunk_set.name, f"pairing_{tag}")
    if os.path.isdir(base):
        shutil.rmtree(base)
    right = os.path.join(base, "right")
    shuffled = os.path.join(base, "shuffled")
    os.makedirs(right)
    os.makedirs(shuffled)
    parts = chunk_set.manifest["parts"]
    for k, (part, stems) in enumerate(zip(parts, chunk_set.stems)):
        shutil.copyfile(stems[stem], os.path.join(right, os.path.splitext(part)[0] + ".flac"))
    for k, source in enumerate(perm):
        shutil.copyfile(
            chunk_set.stems[source][stem], os.path.join(shuffled, f"download ({k + 1:02d}).flac")
        )
    expected, _, _ = _merge(right, chunk_set, os.path.join(base, "right.flac"), "off")
    assert expected is not None
    out: Dict[str, Any] = {"order": perm, "in_order": perm == sorted(perm)}
    for mode in ("auto", "off"):
        got, report, error = _merge(shuffled, chunk_set, os.path.join(base, f"{mode}.flac"), mode)
        if got is None:
            out[mode] = {"result": "refused", "error": error}
            continue
        same = got.raw_data == expected.raw_data
        out[mode] = {
            "result": "identical" if same else "different",
            "vs_right_db": None if same else metrics.snr_db(expected, got),
            "warnings": len(report.warnings),
        }
    return out


def run_song(
    work: SongWork, work_dir: str, trials: int, seed: int, sets: Optional[List[str]] = None
):
    out: Dict[str, Any] = {}
    for set_name, chunk_set in work.sets.items():
        if chunk_set.manifest["overlap_ms"] == 0 or (sets and set_name not in sets):
            continue
        n = len(chunk_set.manifest["parts"])
        if n < 3:
            continue
        rng = random.Random(f"{seed}/{work.song.id}/{set_name}")
        entry: Dict[str, Any] = {"chunks": n, "stems": {}}
        for stem in STEM_NAMES:
            runs = []
            for t in range(trials):
                perm = list(range(n))
                while perm == list(range(n)):  # a renaming that leaves them alone proves nothing
                    rng.shuffle(perm)
                runs.append(trial(chunk_set, stem, work_dir, perm, f"{stem}{t}"))
            entry["stems"][stem] = {"scores": score_real_stems(chunk_set, stem), "trials": runs}
        out[set_name] = entry
    return out


def run(works: List[SongWork], work_dir: str, trials: int = 4, seed: int = 1):
    return {w.song.id: run_song(w, work_dir, trials, seed) for w in works}


def summarise(results: Dict[str, Any]) -> Dict[str, Any]:
    per_stem: Dict[str, Dict[str, Any]] = {}
    for song in results.values():
        for entry in song.values():
            for stem, data in entry["stems"].items():
                slot = per_stem.setdefault(
                    stem,
                    {
                        "true": [],
                        "runner_up": [],
                        "chunks": 0,
                        "right": 0,
                        "sure": 0,
                        "trials": 0,
                        "auto": {},
                        "off": {},
                    },
                )
                s = data["scores"]
                slot["true"].extend(s["true"])
                slot["runner_up"].extend(s["runner_up"])
                slot["chunks"] += s["chunks"]
                slot["right"] += s["right_part"]
                slot["sure"] += s["sure"]
                for t in data["trials"]:
                    slot["trials"] += 1
                    for mode in ("auto", "off"):
                        slot[mode][t[mode]["result"]] = slot[mode].get(t[mode]["result"], 0) + 1
    out = {}
    for stem, slot in per_stem.items():
        margins = [a - b for a, b in zip(slot["true"], slot["runner_up"])]
        out[stem] = {
            "chunks": slot["chunks"],
            "right_part_of_best_score": slot["right"],
            "sure_by_thresholds": slot["sure"],
            "true_score_min": min(slot["true"]),
            "true_score_median": float(statistics.median(slot["true"])),
            "runner_up_max": max(slot["runner_up"]),
            "margin_min": min(margins),
            "thresholds": {
                "min_ncc": seam_pairing.MIN_PAIR_NCC,
                "min_margin": seam_pairing.MIN_PAIR_MARGIN,
            },
            "shuffled_trials": slot["trials"],
            "pairing_auto": slot["auto"],
            "pairing_off": slot["off"],
        }
    return out
