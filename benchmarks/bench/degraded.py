"""Experiment 4, what a service does to its answers (SIMULATED).

The separation is real but the damage is not: the chunks come back from the local model as
they are, and here they are changed *the way services were seen to change them* (a level
that differs from chunk to chunk, an offset of some milliseconds, a lossy file), to see
what the merge does about each. No web service was used; whether a particular service
does any of this, and by how much, is for its user to find out with ``seamsplit merge``
and its report.

The merged result is compared with the merge of the same chunks *before* they were changed
(after the best single gain and the best single shift in time, so that a level or a
position that is off for the file as a whole, which no merge can know, does not count).
Nothing but the damage separates the two.
"""

from __future__ import annotations

import os
import random
import shutil
from typing import Any, Dict, List

import numpy as np

from seamsplit.audio import AudioSegment
from seamsplit.merger import merge_audio
from seamsplit.report import MergeReport

from . import metrics
from .chunks import seam_centres_ms
from .workset import STEM_NAMES, ChunkSet, SongWork, write_processed

MAX_GAIN_DB = 3.0  # each chunk is made louder or quieter by up to this much
MAX_OFFSET_MS = 30  # and later or earlier by up to this much

DAMAGE = {
    "gain": {"gain": True, "offset": False, "mp3": False},
    "offset": {"gain": False, "offset": True, "mp3": False},
    "mp3": {"gain": False, "offset": False, "mp3": True},
    "all": {"gain": True, "offset": True, "mp3": True},
}

# how the damaged chunks are merged: name -> merge_audio options
HANDLING = {
    "off": {"align": False, "match_level": False},
    "default": {},
    "anchor": {"anchor": True},
}


def shift(audio: AudioSegment, offset_ms: int) -> AudioSegment:
    """``audio`` delayed (positive) or advanced by ``offset_ms``, with the length kept."""
    n = int(round(audio.frame_rate * offset_ms / 1000.0))
    if n == 0:
        return audio
    x = audio.frames
    pad = np.zeros((abs(n), x.shape[1]), dtype=x.dtype)
    y = np.concatenate([pad, x[:-n]]) if n > 0 else np.concatenate([x[-n:], pad])
    return AudioSegment.from_samples(
        y.reshape(-1), frame_rate=audio.frame_rate, channels=audio.channels
    )


def damage_files(chunk_set: ChunkSet, folder: str, kind: str, seed: str) -> List[Dict[str, Any]]:
    """Write the chunks of ``chunk_set`` damaged as ``kind`` says into ``folder`` (the
    same change for both stems of a chunk, as a service would make it) and say what was done."""
    rng = random.Random(seed)
    what = DAMAGE[kind]
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    done = []
    for part, stems in zip(chunk_set.manifest["parts"], chunk_set.stems):
        gain = round(rng.uniform(-MAX_GAIN_DB, MAX_GAIN_DB), 2) if what["gain"] else 0.0
        offset = rng.randint(-MAX_OFFSET_MS, MAX_OFFSET_MS) if what["offset"] else 0
        for stem in STEM_NAMES:
            audio = AudioSegment.from_file(stems[stem])
            if what["gain"]:
                audio = audio.apply_gain(gain)
            if what["offset"]:
                audio = shift(audio, offset)
            base = os.path.join(folder, f"{os.path.splitext(part)[0]}_{stem}")
            if what["mp3"]:
                audio.export(base + ".mp3", format="mp3", bitrate="128k")
            else:
                audio.export(base + ".flac", format="flac")
        done.append({"part": part, "gain_db": gain, "offset_ms": offset})
    return done


def merge_two(chunk_set: ChunkSet, folder: str, out_dir: str, tag: str, options: Dict[str, Any]):
    out_main = os.path.join(out_dir, f"{tag}_instrumental.flac")
    out_other = os.path.join(out_dir, f"{tag}_vocals.flac")
    report = MergeReport()
    extra = dict(options)
    if extra.get("anchor"):
        extra["originals_dir"] = chunk_set.chunk_dir
    merge_audio(
        folder,
        chunk_set.manifest_path,
        out_main,
        match="instrumental",
        other_match="vocals",
        other_output=out_other,
        report=report,
        samplerate=44100,
        pairing="off",
        **extra,
    )
    return {
        "instrumental": AudioSegment.from_file(out_main),
        "vocals": AudioSegment.from_file(out_other),
    }, report


def run_song(work: SongWork, work_dir: str, out_dir: str, set_name: str = "quiet2", seed: int = 7):
    chunk_set = work.sets[set_name]
    centres = [float(c) for c in seam_centres_ms(chunk_set.manifest)]
    os.makedirs(out_dir, exist_ok=True)
    clean_dir = write_processed(chunk_set, work_dir, "clean")
    clean, _ = merge_two(chunk_set, clean_dir, out_dir, f"{work.song.id}_clean", {})
    result: Dict[str, Any] = {"set": set_name, "seams": len(centres), "damage": {}}
    for kind in DAMAGE:
        folder = os.path.join(work_dir, work.song.id, set_name, f"damaged_{kind}")
        applied = damage_files(chunk_set, folder, kind, f"{seed}/{work.song.id}/{kind}")
        entry: Dict[str, Any] = {"applied": applied, "handling": {}}
        for name, options in HANDLING.items():
            joined, report = merge_two(
                chunk_set, folder, out_dir, f"{work.song.id}_{kind}_{name}", options
            )
            scores = {}
            for stem in STEM_NAMES:
                whole, seams, shift_ms = metrics.snr_fitted(clean[stem], joined[stem], centres)
                scores[stem] = {
                    "vs_clean_db": whole,
                    "vs_clean_seams_db": seams,
                    "whole_file_shift_ms": round(shift_ms, 2),
                }
            scores["flagged_seams"] = sum(1 for b in report.boundaries if b.flagged)
            entry["handling"][name] = scores
        result["damage"][kind] = entry
    return result


def run(works: List[SongWork], work_dir: str, out_dir: str, set_name: str = "quiet2"):
    return {
        w.song.id: run_song(w, work_dir, out_dir, set_name)
        for w in works
        if set_name in w.sets and len(w.sets[set_name].manifest["parts"]) >= 3
    }


def summarise(results: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kind in DAMAGE:
        out[kind] = {}
        for name in HANDLING:
            whole, seams = [], []
            for song in results.values():
                for stem in STEM_NAMES:
                    s = song["damage"][kind]["handling"][name][stem]
                    whole.append(s["vs_clean_db"])
                    seams.append(s["vs_clean_seams_db"])
            out[kind][name] = {
                "stems_scored": len(whole),
                "vs_clean_db": sum(whole) / len(whole) if whole else float("nan"),
                "vs_clean_seams_db": sum(seams) / len(seams) if seams else float("nan"),
            }
    return out
