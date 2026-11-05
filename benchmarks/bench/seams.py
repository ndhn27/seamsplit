"""Experiment 1, the seams: the chunks of a song are separated for real and joined again in
the different ways, and what is left of the join is measured.

For every song, chunk set and way of joining, the merged stems are compared

* with the **whole song separated in one go** (the reference every chunked result is
  trying to equal), in the windows around the seams and everywhere else; and
* where the corpus has them, with the **ground-truth stems** of the song;

and, without any reference, how much the join stands out as a click.
"""

from __future__ import annotations

import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from seamsplit.audio import AudioSegment
from seamsplit.merger import merge_audio
from seamsplit.report import MergeReport

from . import metrics
from .chunks import DESCRIPTIONS, chunk_ranges_ms, seam_centres_ms
from .workset import STEM_NAMES, ChunkSet, SongWork, write_processed

HALF_WINDOW_MS = 1000.0  # the seam windows: the cut point +- 1 s (the whole overlap, with 2 s)

# name -> options of merge_audio (None: not made by merge_audio, see hard_cut)
VARIANTS: Dict[str, Optional[Dict[str, Any]]] = {
    "hardcut": None,
    "linear": {"method": "linear"},
    "equalpower": {"method": "equalpower"},
    "adaptive": {"method": "adaptive"},
    "adaptive+seam": {"method": "adaptive", "seam_search": True},
    "adaptive+anchor": {"method": "adaptive", "anchor": True},
}
DEFAULT_VARIANT = "adaptive"
BASELINE = "hardcut"


def load_stem(path: str) -> AudioSegment:
    return AudioSegment.from_file(path)


def hard_cut(chunk_set: ChunkSet, stem: str) -> AudioSegment:
    """The chunks of one stem cut at the seams and joined end to end: what is left when the
    overlap is thrown away (what a script does that cuts and joins)."""
    pieces = []
    for rng, stems in zip(chunk_ranges_ms(chunk_set.manifest), chunk_set.stems):
        audio = load_stem(stems[stem])
        pieces.append(audio[rng["from_ms"] - rng["start_ms"] : rng["to_ms"] - rng["start_ms"]])
    out = pieces[0]
    for piece in pieces[1:]:
        out = out + piece
    return out


def merged(
    chunk_set: ChunkSet, work_dir: str, variant: str, out_dir: str
) -> Tuple[Dict[str, AudioSegment], MergeReport]:
    """Both stems of ``chunk_set`` joined the way ``variant`` says."""
    options = VARIANTS[variant]
    report = MergeReport()
    if options is None:
        return {stem: hard_cut(chunk_set, stem) for stem in STEM_NAMES}, report
    processed = write_processed(chunk_set, work_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_main = os.path.join(
        out_dir, f"{chunk_set.song_id}_{chunk_set.name}_{variant}_instrumental.flac"
    )
    out_other = os.path.join(out_dir, f"{chunk_set.song_id}_{chunk_set.name}_{variant}_vocals.flac")
    kwargs: Dict[str, Any] = {k: v for k, v in options.items() if k != "anchor"}
    if options.get("anchor"):
        merge_audio(
            processed,
            chunk_set.manifest_path,
            out_main,
            match="instrumental",
            other_match="vocals",
            other_output=out_other,
            anchor=True,
            originals_dir=chunk_set.chunk_dir,
            report=report,
            samplerate=44100,
            **kwargs,
        )
    else:
        merge_audio(
            processed,
            chunk_set.manifest_path,
            out_main,
            match="instrumental",
            report=report,
            samplerate=44100,
            pairing="off",
            **kwargs,
        )
        merge_audio(
            processed,
            chunk_set.manifest_path,
            out_other,
            match="vocals",
            report=MergeReport(),
            samplerate=44100,
            pairing="off",
            **kwargs,
        )
    return {"instrumental": load_stem(out_main), "vocals": load_stem(out_other)}, report


def score_stem(
    joined: AudioSegment,
    reference: AudioSegment,
    truth: Optional[AudioSegment],
    centres: List[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "vs_whole_db": metrics.snr_db(reference, joined),
        "vs_whole_seams_db": metrics.snr_pooled_windows_db(
            reference, joined, centres, HALF_WINDOW_MS
        ),
        "vs_whole_away_db": metrics.snr_outside_windows_db(
            reference, joined, centres, HALF_WINDOW_MS
        ),
        "vs_whole_per_seam_db": metrics.snr_windows_db(reference, joined, centres, HALF_WINDOW_MS),
        "click_db": [metrics.click_db(joined, c) for c in centres],
    }
    if truth is not None:
        out["vs_truth_db"] = metrics.snr_db(truth, joined)
        out["vs_truth_seams_db"] = metrics.snr_pooled_windows_db(
            truth, joined, centres, HALF_WINDOW_MS
        )
    return out


def _total_seconds(stems: List[Dict[str, Any]]) -> Optional[float]:
    """Seconds the separator took on all of these (None if some run was not timed, as when a
    stopped run was taken up again)."""
    times = [s.get("seconds") for s in stems]
    return None if any(t is None for t in times) else round(sum(times), 1)


def bootstrap_ci(values: List[float], resamples: int = 2000, seed: int = 1) -> List[float]:
    """95 % interval of the mean of ``values`` by resampling them (seams of one song, and the
    two stems of a seam, are not independent, so the interval is narrower than it should be)."""
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    means = rng.choice(data, size=(resamples, len(data)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _mean(values: List[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def run_song(
    work: SongWork, work_dir: str, out_dir: str, variants: Optional[List[str]] = None
) -> Dict[str, Any]:
    song = work.song
    assert work.reference is not None
    whole = {stem: load_stem(work.reference[stem]) for stem in STEM_NAMES}
    truth = {"vocals": song.vocals, "instrumental": song.instrumental}
    result: Dict[str, Any] = {
        "title": song.title,
        "seconds": round(song.seconds, 2),
        "has_truth": song.vocals is not None,
        "reference": {
            "recon_db": metrics.reconstruction_db(song.mix, whole["vocals"], whole["instrumental"]),
        },
        "sets": {},
    }
    if song.vocals is not None:
        for stem in STEM_NAMES:
            result["reference"][f"{stem}_vs_truth_db"] = metrics.snr_db(truth[stem], whole[stem])
        result["reference"]["truth_recon_db"] = metrics.reconstruction_db(
            song.mix, song.vocals, song.instrumental
        )
    for set_name, chunk_set in work.sets.items():
        centres = [float(c) for c in seam_centres_ms(chunk_set.manifest)]
        names = [
            v
            for v in (variants or list(VARIANTS))
            if chunk_set.manifest["overlap_ms"] > 0 or v == BASELINE
        ]
        set_result: Dict[str, Any] = {
            "about": DESCRIPTIONS[set_name],
            "chunks": len(chunk_set.manifest["parts"]),
            "chunk_seconds": [round(s["duration_s"], 2) for s in chunk_set.manifest["segments"]],
            "overlap_ms": chunk_set.manifest["overlap_ms"],
            "seams_ms": centres,
            "separation_seconds": _total_seconds(chunk_set.stems),
            "variants": {},
        }
        for variant in names:
            joined, report = merged(chunk_set, work_dir, variant, out_dir)
            entry: Dict[str, Any] = {"stems": {}}
            for stem in STEM_NAMES:
                entry["stems"][stem] = score_stem(joined[stem], whole[stem], truth[stem], centres)
            entry["recon_db"] = metrics.reconstruction_db(
                song.mix, joined["vocals"], joined["instrumental"]
            )
            entry["flagged_seams"] = sum(1 for b in report.boundaries if b.flagged)
            entry["warnings"] = list(report.warnings)
            set_result["variants"][variant] = entry
        result["sets"][set_name] = set_result
    return result


def run(works: List[SongWork], work_dir: str, out_dir: str, variants: Optional[List[str]] = None):
    return {work.song.id: run_song(work, work_dir, out_dir, variants) for work in works}


def summarise(results: Dict[str, Any]) -> Dict[str, Any]:
    """Across the songs: for every chunk set and way of joining, the mean of each score, and,
    seam by seam, how often and by how much it beats cutting the same seams hard."""
    summary: Dict[str, Any] = {}
    set_names: List[str] = []
    for song in results.values():
        for name in song["sets"]:
            if name not in set_names:
                set_names.append(name)
    for set_name in set_names:
        per_variant: Dict[str, Any] = {}
        for song in results.values():
            for variant, entry in song["sets"].get(set_name, {}).get("variants", {}).items():
                slot = per_variant.setdefault(
                    variant,
                    {
                        "seams": 0,
                        "whole": [],
                        "seams_db": [],
                        "away": [],
                        "truth": [],
                        "truth_seams": [],
                        "click": [],
                        "recon": [],
                        "flagged": 0,
                        "per_seam": [],
                    },
                )
                for stem, scores in entry["stems"].items():
                    slot["whole"].append(scores["vs_whole_db"])
                    slot["seams_db"].append(scores["vs_whole_seams_db"])
                    slot["away"].append(scores["vs_whole_away_db"])
                    slot["click"].extend(c for c in scores["click_db"] if c == c)
                    if "vs_truth_db" in scores:
                        slot["truth"].append(scores["vs_truth_db"])
                        slot["truth_seams"].append(scores["vs_truth_seams_db"])
                    slot["per_seam"].append((song["title"], stem, scores["vs_whole_per_seam_db"]))
                slot["recon"].append(entry["recon_db"])
                slot["flagged"] += entry["flagged_seams"]
        base = per_variant.get(BASELINE)
        rows = {}
        for variant, slot in per_variant.items():
            row = {
                "stems_scored": len(slot["whole"]),
                "vs_whole_db": _mean(slot["whole"]),
                "vs_whole_seams_db": _mean(slot["seams_db"]),
                "vs_whole_away_db": _mean(slot["away"]),
                "click_db_mean": _mean(slot["click"]),
                "recon_db": _mean(slot["recon"]),
                "flagged_seams": slot["flagged"],
            }
            if slot["truth"]:
                row["vs_truth_db"] = _mean(slot["truth"])
                row["vs_truth_seams_db"] = _mean(slot["truth_seams"])
            if base is not None and variant != BASELINE:
                gains = []
                for mine, theirs in zip(slot["per_seam"], base["per_seam"]):
                    gains.extend(a - b for a, b in zip(mine[2], theirs[2]))
                row["seams_compared"] = len(gains)
                row["seams_better_than_hardcut"] = sum(1 for g in gains if g > 0)
                row["seam_gain_median_db"] = (
                    float(statistics.median(gains)) if gains else float("nan")
                )
                row["seam_gain_mean_db"] = _mean(gains)
                row["seam_gain_mean_ci95_db"] = bootstrap_ci(gains)
            rows[variant] = row
        summary[set_name] = rows
    return summary


def seam_gains(results: Dict[str, Any], variant: str = DEFAULT_VARIANT) -> List[Dict[str, Any]]:
    """Every seam (both stems, every chunk set) with how much lower its error is with
    ``variant`` than with a hard cut, best first by ``gain_db`` descending."""
    rows = []
    for song_id, song in results.items():
        for set_name, chunk_set in song["sets"].items():
            variants = chunk_set["variants"]
            if variant not in variants or BASELINE not in variants:
                continue
            for stem in STEM_NAMES:
                hard = variants[BASELINE]["stems"][stem]["vs_whole_per_seam_db"]
                mine = variants[variant]["stems"][stem]["vs_whole_per_seam_db"]
                for k, (h, m) in enumerate(zip(hard, mine)):
                    rows.append(
                        {
                            "song": song_id,
                            "set": set_name,
                            "stem": stem,
                            "at_s": round(chunk_set["seams_ms"][k] / 1000.0, 2),
                            "hardcut_db": h,
                            "variant_db": m,
                            "gain_db": m - h,
                        }
                    )
    return sorted(rows, key=lambda r: -r["gain_db"])


def worst_seams(results: Dict[str, Any], count: int = 6) -> List[Dict[str, Any]]:
    """The seams where ``DEFAULT_VARIANT`` did worse than a hard cut by the most: a median
    that is a gain can hide a few that are a loss."""
    return seam_gains(results)[::-1][:count]
