"""What a merge found and did: one :class:`BoundaryInfo` per seam, collected in a
:class:`MergeReport` together with the warnings, as data (``to_dict``) or text
(``to_markdown``). :func:`_emit` is how the merge code raises a warning: into the
report when there is one, as a :class:`RuntimeWarning` otherwise."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BoundaryInfo:
    """What was measured and done at one chunk boundary (``boundary`` 1 is
    between part 1 and part 2).

    ``lag_ms`` is the shift for which ``next[k] ~ previous[k + lag]`` (the
    convention of :func:`estimate_lag`). It is **negative when the next chunk's
    audio arrives later** than the manifest says (a service put silence in
    front of it) and **positive when it arrives earlier** (a service trimmed
    its start). ``correlation`` is the peak normalized correlation of the two
    overlap regions; ``aligned`` is True when that lag was applied.
    ``gain_db`` is the level correction applied to the next chunk (positive =
    boosted) and ``fade_correlation`` the correlation that shaped an
    ``adaptive`` fade (measured over the fade itself).

    ``seam_ms`` is where the middle of the crossfade sits in the merged audio
    and ``fade_ms`` its length. ``seam_match_db`` / ``overlap_match_db`` say
    how well the two chunks agree (10·log10 of their power over the power of
    their difference, after alignment and level matching; capped at 99) at the
    seam and over the whole overlap. ``flagged`` is True when the boundary
    needs a listen: the match is below the threshold or the overlaps did not
    correspond at all."""

    boundary: int
    lag_ms: float
    correlation: float
    aligned: bool
    gain_db: float = 0.0
    fade_correlation: Optional[float] = None
    seam_ms: Optional[float] = None
    fade_ms: Optional[float] = None
    seam_match_db: Optional[float] = None
    overlap_match_db: Optional[float] = None
    flagged: bool = False


@dataclass
class ChunkInfo:
    """What comparing one processed chunk with the original chunk it was made from
    found (``part`` is 1-based; only with ``anchor``).

    ``lag_ms`` is the offset measured and cancelled, with the convention of
    :class:`BoundaryInfo` (``processed[k] ~ original[k + lag]``: positive when the
    service trimmed the start, negative when it added silence), or ``None`` when
    the chunk could not be anchored in time. ``correlation`` is the normalized
    correlation of the two at that offset. ``gain_db`` is the level correction
    applied to the chunk (``None``: the level was not taken from the original, which
    needs both stems, see :mod:`seamsplit.anchor`). ``recon_db`` says how well the
    two stems add up to the original, ``10·log10(power of the original / power of
    what is left after the best offset and gain)`` (capped at 99), and is ``None``
    with one stem. ``flagged`` is True when that is below the threshold."""

    part: int
    lag_ms: Optional[float] = None
    correlation: float = 0.0
    gain_db: Optional[float] = None
    recon_db: Optional[float] = None
    flagged: bool = False


def format_time(ms: float) -> str:
    """``m:ss.s`` (``h:mm:ss.s`` from one hour on)."""
    seconds = ms / 1000.0
    hours, rest = divmod(seconds, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"
    return f"{int(minutes)}:{secs:04.1f}"


@dataclass
class MergeReport:
    """Diagnostics collected by :func:`merge_audio` (pass one in to read it)."""

    boundaries: List[BoundaryInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # one per chunk, only when the merge was anchored to the original chunks
    chunks: List[ChunkInfo] = field(default_factory=list)
    # things that went as they should and are worth saying (not warnings)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "boundaries": [asdict(b) for b in self.boundaries],
            "warnings": list(self.warnings),
        }
        if self.chunks:  # only present when the merge was anchored
            data["chunks"] = [asdict(c) for c in self.chunks]
        if self.notes:
            data["notes"] = list(self.notes)
        return data

    def to_markdown(self) -> str:
        """A per-seam table plus the warnings, for keeping next to the output."""

        def num(value: Optional[float], spec: str) -> str:
            return "–" if value is None else format(value, spec)

        lines = [
            "# Seam report",
            "",
            "| Seam | At | Lag (ms) | Level (dB) | Match at seam (dB) | Match, whole overlap (dB)"
            " | Fade (ms) | Check |",
            "|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]
        for b in self.boundaries:
            at = "–" if b.seam_ms is None else format_time(b.seam_ms)
            lines.append(
                f"| {b.boundary}/{b.boundary + 1} | {at} | {b.lag_ms:+.1f} | {b.gain_db:+.1f} "
                f"| {num(b.seam_match_db, '.1f')} | {num(b.overlap_match_db, '.1f')} "
                f"| {num(b.fade_ms, '.0f')} | {'**listen**' if b.flagged else 'ok'} |"
            )
        if not self.boundaries:
            lines.append("| – | – | – | – | – | – | – | nothing was analysed |")
        if self.chunks:
            lines += [
                "",
                "## Chunks against the original chunks",
                "",
                "| Chunk | Lag (ms) | Correlation | Level (dB) | Stems add up to the original (dB)"
                " | Check |",
                "|---:|---:|---:|---:|---:|:---|",
            ]
            for c in self.chunks:
                lines.append(
                    f"| {c.part} | {num(c.lag_ms, '+.1f')} | {c.correlation:.3f} "
                    f"| {num(c.gain_db, '+.1f')} | {num(c.recon_db, '.1f')} "
                    f"| {'**check**' if c.flagged else 'ok'} |"
                )
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def _emit(report: Optional[MergeReport], message: str) -> None:
    if report is not None:
        report.warnings.append(message)
    else:
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _note(report: Optional[MergeReport], message: str) -> None:
    """Something that went as it should and is worth saying: into the report when there
    is one, and nowhere otherwise (it is not a warning)."""
    if report is not None:
        report.notes.append(message)
