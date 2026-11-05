"""
A merge that has been set up: which file is each chunk, what is done to it, and how it
is crossfaded. :func:`seamsplit.merger.merge_audio` builds one and runs it over every
chunk; :mod:`seamsplit.audition` runs the same one over just the two chunks of a seam,
with each of the ways to join them, so what you hear is what the merge did.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

from .audio import AudioSegment

from .anchor import DEFAULT_RECON_WARN_DB, AnchorOptions, prepare_chunk, split_stems
from .crossfade import (
    DEFAULT_MAX_GAIN_DB,
    DEFAULT_MAX_LAG_MS,
    DEFAULT_METHOD,
    DEFAULT_SEAM_FADE_MS,
    DEFAULT_SEAM_WARN_DB,
    _crossfade_merge,
)
from .report import MergeReport, _emit

DEFAULT_LENGTH_TOLERANCE_MS = 60.0


def _check_part_length(
    part: int,
    seg: AudioSegment,
    manifest: dict,
    tolerance_ms: float,
    report: Optional[MergeReport],
) -> None:
    """Warn when the processed chunk ``part`` (1-based) is not as long as the
    manifest says it should be."""
    planned = manifest.get("segments")
    if not isinstance(planned, list) or len(planned) != len(manifest["parts"]):
        return
    try:
        plan = planned[part - 1]
        expected = float(plan["end_ms"]) - float(plan["start_ms"])
    except (KeyError, TypeError, ValueError):
        return
    deviation = len(seg) - expected
    if abs(deviation) > tolerance_ms:
        _emit(
            report,
            f"part {part}: processed file is {len(seg) / 1000:.2f}s but the manifest "
            f"expects {expected / 1000:.2f}s ({deviation:+.0f} ms)",
        )


@dataclass
class MergePlan:
    """The files and the options of one merge (see :func:`seamsplit.merger.merge_audio`
    for what the options mean). ``other_paths`` are the files of the second stem and
    ``original_paths`` those of the original chunks, each ``None`` when not used."""

    manifest: dict
    overlap_ms: float
    paths: List[str]
    other_paths: Optional[List[str]] = None
    original_paths: Optional[List[str]] = None
    method: str = DEFAULT_METHOD
    align: bool = True
    max_lag_ms: float = DEFAULT_MAX_LAG_MS
    match_level: bool = True
    max_gain_db: float = DEFAULT_MAX_GAIN_DB
    seam_search: bool = False
    seam_fade_ms: float = DEFAULT_SEAM_FADE_MS
    seam_warn_db: Optional[float] = DEFAULT_SEAM_WARN_DB
    length_tolerance_ms: float = DEFAULT_LENGTH_TOLERANCE_MS
    recon_warn_db: Optional[float] = DEFAULT_RECON_WARN_DB

    def stream(
        self,
        indices: Iterable[int],
        report: Optional[MergeReport] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
        extra_db: float = 0.0,
    ) -> Iterator[AudioSegment]:
        """The chunks at ``indices`` (0-based, in order), decoded one at a time as they
        are reached and made ready for the merge: with the second stem stacked behind
        the first, and shifted and scaled to the original, as the plan says.
        ``extra_db`` is added to the first one (a seam being played on its own needs
        the level its chunk has in the whole)."""
        options = AnchorOptions(
            max_lag_ms=self.max_lag_ms if self.align else 0.0,
            level=self.match_level,
            recon_warn_db=self.recon_warn_db,
        )
        clipped = 0
        for position, index in enumerate(indices):
            part = index + 1
            if progress is not None:
                progress(part, len(self.paths), os.path.basename(self.paths[index]))
            seg = AudioSegment.from_file(self.paths[index])
            _check_part_length(part, seg, self.manifest, self.length_tolerance_ms, report)
            gain = extra_db if position == 0 else 0.0
            if self.other_paths is None and self.original_paths is None and not gain:
                yield seg  # one stem, nothing to anchor to: the chunk as it came back
                continue
            other = None
            if self.other_paths is not None:
                other = AudioSegment.from_file(self.other_paths[index])
            original = None
            if self.original_paths is not None:
                original = AudioSegment.from_file(self.original_paths[index])
            prepared, over = prepare_chunk(
                part, seg, other, original, options, report, self.length_tolerance_ms, gain
            )
            clipped += over
            yield prepared
        if clipped:
            _emit(
                report,
                f"{clipped} sample(s) exceeded full scale when the chunks were brought to "
                "the level of their originals, and were clipped",
            )

    def merge(
        self,
        indices: Optional[Iterable[int]] = None,
        report: Optional[MergeReport] = None,
        *,
        seam_search: Optional[bool] = None,
        align: Optional[bool] = None,
        match_level: Optional[bool] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
        extra_db: float = 0.0,
        chunks: Optional[Sequence[AudioSegment]] = None,
    ) -> AudioSegment:
        """Crossfade the chunks at ``indices`` (all of them by default) into one. The
        keyword arguments override the plan's own for this run; ``chunks`` are chunks
        already made ready by :meth:`stream`, to merge more than once without reading
        them again. With a second stem the result has both, stacked as channels (see
        :meth:`split`)."""
        use_align = self.align if align is None else align
        search = self.seam_search if seam_search is None else seam_search
        lag = self.max_lag_ms if use_align and self.overlap_ms > 0 else 0.0
        chosen = range(len(self.paths)) if indices is None else indices
        return _crossfade_merge(
            self.stream(chosen, report, progress, extra_db) if chunks is None else chunks,
            self.overlap_ms,
            self.method,
            lag,
            report,
            self.match_level if match_level is None else match_level,
            self.max_gain_db,
            search and self.overlap_ms > 0,
            self.seam_fade_ms,
            self.seam_warn_db,
        )

    def split(self, merged: AudioSegment) -> Tuple[AudioSegment, Optional[AudioSegment]]:
        """The main stem and (with a second stem) the other one, out of what
        :meth:`merge` returned."""
        if self.other_paths is None:
            return merged, None
        main, other = split_stems(merged, merged.channels // 2)
        return main, other
