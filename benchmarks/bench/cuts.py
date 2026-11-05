"""Experiment 2, the cut points: is a cut at the quietest point of a window really in a
quiet place, and does it leave a smaller seam?

For every cut of every chunk set: the level of the *mix* around the cut (RMS over 200 ms),
as a number of dB below the level of the whole song, and the percentage of the song's 200 ms
windows that are quieter than it (0 % is the quietest place there is). From the seams
experiment: the error of the stems around that cut when the chunks are simply joined there
(the ``hardcut`` of the same cuts), which is what the choice of cut point alone is
responsible for.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List

import numpy as np

from . import metrics
from .chunks import seam_centres_ms
from .workset import SongWork

WINDOW_MS = 200.0


def cut_levels(work: SongWork, set_name: str) -> Dict[str, Any]:
    mix = work.song.mix
    centres = seam_centres_ms(work.sets[set_name].manifest)
    times, curve = metrics.level_curve_db(mix, window_ms=WINDOW_MS, hop_ms=50.0)
    whole = metrics.level_db(mix, len(mix) / 2.0, len(mix) / 2.0)
    out = []
    for c in centres:
        level = metrics.level_db(mix, c, WINDOW_MS / 2)
        quieter = float(np.mean(curve < level)) * 100.0
        out.append({"at_ms": c, "below_song_db": whole - level, "percent_of_song_quieter": quieter})
    return {"cuts": out}


def run(works: List[SongWork], seams: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for work in works:
        song = work.song.id
        result[song] = {}
        for set_name in work.sets:
            entry = cut_levels(work, set_name)
            hard = (
                seams.get(song, {})
                .get("sets", {})
                .get(set_name, {})
                .get("variants", {})
                .get("hardcut")
            )
            if hard is not None:
                entry["hardcut_seam_db"] = {
                    stem: scores["vs_whole_per_seam_db"] for stem, scores in hard["stems"].items()
                }
            result[song][set_name] = entry
    return result


def summarise(results: Dict[str, Any]) -> Dict[str, Any]:
    sets: Dict[str, Dict[str, List[float]]] = {}
    for song in results.values():
        for set_name, entry in song.items():
            slot = sets.setdefault(set_name, {"below": [], "quieter": [], "seam": []})
            for cut in entry["cuts"]:
                slot["below"].append(cut["below_song_db"])
                slot["quieter"].append(cut["percent_of_song_quieter"])
            for values in entry.get("hardcut_seam_db", {}).values():
                slot["seam"].extend(v for v in values if v == v)
    return {
        name: {
            "cuts": len(slot["below"]),
            "level_below_song_db_mean": (
                float(statistics.fmean(slot["below"])) if slot["below"] else None
            ),
            "percent_of_song_quieter_mean": (
                float(statistics.fmean(slot["quieter"])) if slot["quieter"] else None
            ),
            "hardcut_seam_db_mean": float(statistics.fmean(slot["seam"])) if slot["seam"] else None,
            "hardcut_seam_db_median": (
                float(statistics.median(slot["seam"])) if slot["seam"] else None
            ),
        }
        for name, slot in sets.items()
    }
