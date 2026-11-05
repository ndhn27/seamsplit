"""The ways a song is cut for the benchmarks.

Every set is cut with SeamSplit's own :func:`seamsplit.splitter.split_audio`, so the
manifest is the one the merge reads; they differ in where the cuts go and in the overlap:

``fixed0``  equal pieces, no overlap: what a script that cuts every N seconds and joins
            the pieces again does. The baseline.
``fixed2``  the same cut points, 2 s of overlap: separates the effect of the overlap
            and crossfade from the effect of choosing where to cut.
``quiet2``  cut at the quietest point of a window (``--cut-by rms``), 2 s of overlap.
``voice2``  cut where the voice band is quietest (``--cut-by voice``), 2 s of overlap.

The clips of the corpus are 10-30 s long, so the chunks are short (4-14 s) where the
services' limits make them 30-60 s for real songs; see the README of the benchmarks.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from seamsplit.splitter import SplitConfig, split_audio

TARGET_S = 8.0
OVERLAP_S = 2.0

CHUNK_SETS: Dict[str, Dict[str, Any]] = {
    "fixed0": {"overlap_s": 0.0, "before_slack_s": 0.0, "after_slack_s": 0.0},
    "fixed2": {"overlap_s": OVERLAP_S, "before_slack_s": 0.0, "after_slack_s": 0.0},
    "quiet2": {"overlap_s": OVERLAP_S, "before_slack_s": 3.0, "after_slack_s": 1.0},
    "voice2": {
        "overlap_s": OVERLAP_S,
        "before_slack_s": 3.0,
        "after_slack_s": 1.0,
        "cut_by": "voice",
    },
}

DESCRIPTIONS = {
    "fixed0": "equal pieces, no overlap (a plain cut-and-join script)",
    "fixed2": "the same equal pieces with 2 s of overlap",
    "quiet2": "cut at the quietest point, 2 s of overlap",
    "voice2": "cut where the voice band is quietest, 2 s of overlap",
}


def make_chunk_set(song_path: str, set_name: str, out_dir: str) -> Dict[str, Any]:
    """Cut ``song_path`` as ``set_name`` says into ``out_dir`` (lossless FLAC, so that the
    separator gets exactly the audio that is cut) and return the manifest, with the path
    of every chunk under ``paths`` and ``set`` added."""
    options = dict(CHUNK_SETS[set_name])
    config = SplitConfig(
        target_s=TARGET_S,
        min_final_s=3.0,
        output_dir=out_dir,
        export_format="flac",
        **options,
    )
    manifest = split_audio(song_path, config)
    manifest["paths"] = [os.path.join(out_dir, name) for name in manifest["parts"]]
    manifest["set"] = set_name
    return manifest


def seam_centres_ms(manifest: Dict[str, Any]) -> List[int]:
    """Where each seam is, in the song: the cut point between two neighbouring chunks (the
    middle of their overlap, when they have one)."""
    segments = manifest["segments"]
    lead = int(manifest["overlap_ms"]) // 2
    return [int(seg["start_ms"]) + lead for seg in segments[1:]]


def chunk_ranges_ms(manifest: Dict[str, Any]) -> List[Dict[str, int]]:
    """For each chunk: its span in the song and the span it owns (from the previous seam to
    the next)."""
    segments = manifest["segments"]
    centres = seam_centres_ms(manifest)
    total = int(manifest["source_length_ms"])
    out = []
    for i, seg in enumerate(segments):
        owns_from = 0 if i == 0 else centres[i - 1]
        owns_to = total if i == len(segments) - 1 else centres[i]
        out.append(
            {
                "start_ms": int(seg["start_ms"]),
                "end_ms": int(seg["end_ms"]),
                "from_ms": owns_from,
                "to_ms": owns_to,
            }
        )
    return out


def only_sets(names: Optional[List[str]]) -> List[str]:
    if not names:
        return list(CHUNK_SETS)
    unknown = [n for n in names if n not in CHUNK_SETS]
    if unknown:
        raise ValueError(
            f"unknown chunk set: {', '.join(unknown)} (known: {', '.join(CHUNK_SETS)})"
        )
    return list(names)
