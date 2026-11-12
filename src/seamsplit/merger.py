"""
Merge the chunks of a split back into one file: read the manifest, find each
part's processed file, crossfade neighbours across their overlap and write the
result, with a report of every seam.

The work is split over small modules, and this one is the entry point
(:func:`merge_audio`) that ties them together:

* :mod:`seamsplit.manifest` and :mod:`seamsplit.checks` read the manifest and
  check the options;
* :mod:`seamsplit.matching` pairs parts with the files that came back, and
  :mod:`seamsplit.pairing` confirms that pairing by comparing the files with the original
  chunks when their names were lost;
* :mod:`seamsplit.crossfade` merges (its docstring explains alignment, level
  matching, seam search and the fade curves) on top of :mod:`seamsplit.dsp`;
* :mod:`seamsplit.anchor` compares each chunk with the original it was made from and
  merges both stems of a service together, :mod:`seamsplit.plan` holds a merge that has
  been set up and :mod:`seamsplit.audition` plays its seams in the ways they can be joined;
* :mod:`seamsplit.report` holds what was found.

The public names of those modules that a merge uses are re-exported here (``__all__`` lists
them), so they can be imported from this one too. A name that starts with an underscore is
private to the module that defines it and is not re-exported: import it from there.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple

from .anchor import DEFAULT_RECON_WARN_DB, other_output_path
from .audio import AudioSegment, CouldntDecodeError, CouldntEncodeError
from .audition import AuditionOptions, export_auditions
from .checks import _check_merge_options, _check_number, _check_samplerate, _check_threshold
from .manifest import load_manifest
from .pairing import PAIRING_MODES, check_pairing
from .plan import MergePlan
from .report import _emit
from .splitter import export_segment

# Re-exports: the public names of the modules above, kept importable from here. Each is
# defined in the module it is imported from. ``__all__`` is the list; a name with a leading
# underscore is private to its module and is not part of it (import it from there).
from .crossfade import (  # noqa: F401
    DEFAULT_MAX_GAIN_DB,
    DEFAULT_MAX_LAG_MS,
    DEFAULT_METHOD,
    DEFAULT_SEAM_FADE_MS,
    DEFAULT_SEAM_WARN_DB,
    METHODS,
    adaptive_crossfade_merge,
    equal_power_crossfade_merge,
    linear_crossfade_merge,
)
from .dsp import (  # noqa: F401
    LEVEL_DEADBAND_DB,
    MIN_ALIGN_CORRELATION,
    estimate_lag,
    find_seam,
    match_db,
)
from .matching import (  # noqa: F401
    AUDIO_EXTENSIONS,
    PairedFiles,
    find_original_chunks,
    find_processed_file,
    find_processed_files,
    pair_processed_files,
)
from .plan import DEFAULT_LENGTH_TOLERANCE_MS  # noqa: F401
from .report import BoundaryInfo, ChunkInfo, MergeReport, format_time  # noqa: F401

__all__ = [
    "merge_audio",
    # report
    "BoundaryInfo",
    "ChunkInfo",
    "MergeReport",
    "format_time",
    # manifest
    "load_manifest",
    # matching
    "AUDIO_EXTENSIONS",
    "PairedFiles",
    "find_original_chunks",
    "find_processed_file",
    "find_processed_files",
    "pair_processed_files",
    # dsp
    "LEVEL_DEADBAND_DB",
    "MIN_ALIGN_CORRELATION",
    "estimate_lag",
    "find_seam",
    "match_db",
    # crossfade
    "DEFAULT_MAX_GAIN_DB",
    "DEFAULT_MAX_LAG_MS",
    "DEFAULT_METHOD",
    "DEFAULT_SEAM_FADE_MS",
    "DEFAULT_SEAM_WARN_DB",
    "METHODS",
    "adaptive_crossfade_merge",
    "equal_power_crossfade_merge",
    "linear_crossfade_merge",
    # plan
    "DEFAULT_LENGTH_TOLERANCE_MS",
    "MergePlan",
]

_HOW_TO_CHECK = (
    " (the original chunks, next to the manifest or in --originals, would let "
    "seamsplit check that by comparing them with the files)"
)


@dataclass(frozen=True)
class _Tuning:
    """The options that decide how the chunks are joined (they are the fields of
    :class:`MergePlan` of the same names), as one value instead of ten arguments."""

    method: str
    align: bool
    max_lag_ms: float
    match_level: bool
    max_gain_db: float
    seam_search: bool
    seam_fade_ms: float
    seam_warn_db: Optional[float]
    length_tolerance_ms: float
    recon_warn_db: Optional[float]


# The functions below are the stages of :func:`merge_audio`, in the order it runs them. They
# raise or return; what is to be reported is returned as text and emitted by ``merge_audio``
# itself, because without a ``report`` it becomes a warning, which ``_emit`` points at the
# caller of the function it is called from: that has to be the caller of ``merge_audio``.


def _check_numbers(
    tuning: _Tuning, samplerate: Optional[int], overlap_s: Optional[float]
) -> Optional[int]:
    """Check every number before anything is read or decoded, whatever the caller (CLI, GUI
    or library) did or did not check, and return ``samplerate`` as a whole number.
    ``max_lag_ms`` is checked even when align is off: a value that can never be valid is a
    mistake either way."""
    if tuning.method not in METHODS:
        raise ValueError(f"Unknown method: {tuning.method!r} (choose from {', '.join(METHODS)})")
    samplerate = _check_samplerate(samplerate)
    _check_merge_options(
        tuning.max_lag_ms, tuning.max_gain_db, tuning.seam_fade_ms, tuning.seam_warn_db
    )
    _check_number("length_tolerance_ms", tuning.length_tolerance_ms)
    _check_threshold("recon_warn_db", tuning.recon_warn_db)
    if overlap_s is not None:
        _check_number("overlap_s", overlap_s)
    return samplerate


def _check_outputs(
    output_path: str,
    other_match: Optional[str],
    other_output: Optional[str],
    pairing: str,
    originals_dir: Optional[str],
    anchor: bool,
    audition: Optional[AuditionOptions],
    report: Optional[MergeReport],
) -> Optional[str]:
    """Check the options that name files and stems, and return where the second stem is to
    go: ``other_output``, or (when only ``other_match`` is given) a name made from
    ``output_path``."""
    if other_match is not None and not other_match.strip():
        raise ValueError("other_match must not be empty")
    if other_match is None and other_output is not None:
        raise ValueError("other_output needs other_match: which files are the other stem?")
    if pairing not in PAIRING_MODES:
        raise ValueError(f"Unknown pairing: {pairing!r} (choose from {', '.join(PAIRING_MODES)})")
    if originals_dir is not None and not anchor and pairing == "off":
        raise ValueError(
            "originals_dir is only used with anchor=True or when the pairing is checked "
            "(pairing='auto' or 'verify')"
        )
    if audition is not None and report is None:
        raise ValueError("audition needs a MergeReport (report=...) to read the seams from")
    if other_match is not None and other_output is None:
        other_output = other_output_path(output_path, other_match)
    if other_output is not None and os.path.abspath(other_output) == os.path.abspath(output_path):
        raise ValueError("other_output must not be the same file as the output")
    return other_output


def _overlap_ms(manifest: dict, overlap_s: Optional[float]) -> Tuple[float, Optional[str]]:
    """The overlap to crossfade over, in ms, and a warning when it is not the one the
    chunks were cut with (``overlap_s`` overrides the manifest)."""
    if overlap_s is None:
        return float(manifest["overlap_ms"]), None
    overlap_ms = overlap_s * 1000
    if abs(overlap_ms - float(manifest["overlap_ms"])) > 0.5:
        return overlap_ms, (
            f"overlap override ({overlap_ms:.0f} ms) differs from the manifest "
            f"({float(manifest['overlap_ms']):.0f} ms); chunks were cut with the latter"
        )
    return overlap_ms, None


def _pair_stems(
    processed_dir: str, part_names: List[str], match: Optional[str], other_match: Optional[str]
) -> Tuple[PairedFiles, Optional[PairedFiles]]:
    """The processed file of every part for the stem ``match`` names and, when there is
    one, for the stem ``other_match`` names. The two must not pick out the same files."""
    paired = pair_processed_files(processed_dir, part_names, match)
    other_paired = None
    if other_match is not None:
        other_paired = pair_processed_files(processed_dir, part_names, other_match)
        if [os.path.abspath(p) for p in other_paired.paths] == [
            os.path.abspath(p) for p in paired.paths
        ]:
            raise ValueError(
                f"'{match}' and '{other_match}' pick out the same files: the two stems "
                "need names that tell them apart"
            )
    return paired, other_paired


def _stems_to_check(
    stems: List[Tuple[PairedFiles, str]],
    original_paths: Optional[List[str]],
    folder: str,
    part_names: List[str],
    pairing: str,
) -> Tuple[List[Tuple[PairedFiles, str]], Optional[List[str]]]:
    """Which of the paired stems are to be compared with the original chunks (see
    ``pairing`` in :func:`merge_audio`) and those chunks, or None when they are not there."""
    to_check = [
        (files, what)
        for files, what in stems
        if pairing == "verify" or (pairing == "auto" and files.by_order)
    ]
    reference = original_paths
    if to_check and reference is None:
        try:
            reference = find_original_chunks(folder, part_names)
        except FileNotFoundError:
            if pairing == "verify":
                raise
    return to_check, reference


def _order_notes(
    stems: List[Tuple[PairedFiles, str]], reference: Optional[List[str]], pairing: str
) -> List[str]:
    """What is to be said about stems that were paired with the parts by order and that
    nothing will check (no original chunks to compare with, or ``pairing='off'``)."""
    return [
        files.order_note + (_HOW_TO_CHECK if pairing == "auto" else "")
        for files, _ in stems
        if files.by_order and (reference is None or pairing == "off")
    ]


def _length_note(main: AudioSegment, manifest: dict, tolerance_ms: float) -> Optional[str]:
    """A warning when the merged audio is not as long as the source it was cut from."""
    source_len = manifest.get("source_length_ms")
    if isinstance(source_len, (int, float)) and abs(len(main) - source_len) > tolerance_ms:
        return (
            f"merged audio is {len(main) / 1000:.2f}s but the source was "
            f"{source_len / 1000:.2f}s ({len(main) - source_len:+.0f} ms) — "
            "a chunk may be missing, cut short or padded"
        )
    return None


def _export(
    main: AudioSegment,
    other: Optional[AudioSegment],
    output_path: str,
    other_output: Optional[str],
    samplerate: Optional[int],
    bitrate: str,
) -> AudioSegment:
    """Write the merged stem (and the second one, if there is one) and return the first as
    it was written."""
    fmt = os.path.splitext(output_path)[1].lstrip(".") or "mp3"
    # Without a samplerate the result keeps the rate the chunks have: nothing is resampled
    # behind the user's back.
    merged = main if samplerate is None else main.set_frame_rate(samplerate)
    export_segment(merged, output_path, fmt, bitrate)
    if other is not None and other_output is not None:
        fmt_other = os.path.splitext(other_output)[1].lstrip(".") or "mp3"
        export_segment(
            other if samplerate is None else other.set_frame_rate(samplerate),
            other_output,
            fmt_other,
            bitrate,
        )
    return merged


# The options are flat keyword arguments, one per CLI flag, so that every caller (the CLI,
# the GUI, jobs and library users) keeps its keyword arguments as they are. Inside they are
# grouped (:class:`_Tuning`) and each stage above takes only what it needs.
def merge_audio(
    processed_dir: str,
    manifest_path: str,
    output_path: str,
    overlap_s: Optional[float] = None,
    match: Optional[str] = None,
    method: str = DEFAULT_METHOD,
    bitrate: str = "320k",
    samplerate: Optional[int] = None,
    align: bool = True,
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
    match_level: bool = True,
    max_gain_db: float = DEFAULT_MAX_GAIN_DB,
    seam_search: bool = False,
    seam_fade_ms: float = DEFAULT_SEAM_FADE_MS,
    seam_warn_db: Optional[float] = DEFAULT_SEAM_WARN_DB,
    length_tolerance_ms: float = DEFAULT_LENGTH_TOLERANCE_MS,
    report: Optional[MergeReport] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    anchor: bool = False,
    originals_dir: Optional[str] = None,
    other_match: Optional[str] = None,
    other_output: Optional[str] = None,
    recon_warn_db: Optional[float] = DEFAULT_RECON_WARN_DB,
    audition: Optional[AuditionOptions] = None,
    pairing: str = "auto",
) -> AudioSegment:
    """Load the manifest written by
    :func:`seamsplit.splitter.split_audio`, locate each part's processed
    counterpart in ``processed_dir``, merge them with a crossfade, export
    to ``output_path``, and return the merged ``AudioSegment``.

    With ``align=True`` (default) any timing offset of up to ``max_lag_ms``
    between neighboring chunks is measured and cancelled. With
    ``match_level=True`` (default) each chunk is scaled so its overlap is as
    loud as the previous chunk's, by at most ``max_gain_db`` either way.
    ``method="adaptive"`` (default) shapes every crossfade from the measured
    correlation of the overlap. With ``seam_search=True`` (experimental, off by
    default; see the module docstring) the crossfade is ``seam_fade_ms`` long
    and placed where the two chunks agree best instead of spanning the whole
    overlap. Either way, seams where the chunks agree to less than
    ``seam_warn_db`` are flagged in the report. An overlap of 0 simply
    concatenates the chunks. Integrity problems (a chunk whose length
    differs from the manifest by more than ``length_tolerance_ms``, a merged
    length that doesn't match the source, chunks that can't be aligned) are
    appended to ``report.warnings`` if a :class:`MergeReport` is given, and
    raised as :class:`RuntimeWarning` otherwise.

    ``anchor=True`` compares every processed chunk with the original chunk it was
    made from (the files ``split`` wrote, in ``originals_dir``, by default next to the
    manifest) and takes its place in time from that, not from its neighbours; with both
    stems (``other_match``) it takes its level from it too, and says in
    ``report.chunks`` how well the stems add up to the original (flagged below
    ``recon_warn_db``). ``other_match`` names the second stem the way ``match`` names
    the first (``match="instrumental", other_match="vocals"``): the two are then
    merged together, with the same offsets, gains and fades, and the second goes to
    ``other_output`` (by default ``output_path`` with the word added). See
    :mod:`seamsplit.anchor`.

    ``pairing`` says what is done about which file is which chunk. Files are found by
    the part names in their names; when the names were lost they can only be paired with the
    parts by order, which is a guess. ``"auto"`` (default) then compares each file with
    the original chunk it was paired with (the files ``split`` wrote, found next to the
    manifest or in ``originals_dir`` if they are there), puts files that clearly belong
    elsewhere in their place and says what it did in ``report``; a set of files that
    cannot be fitted to the parts at all is a :class:`ValueError`. ``"verify"`` does that
    for files found by name as well (and needs the originals), ``"off"`` never does. See
    :mod:`seamsplit.pairing`.

    ``audition`` (see :class:`seamsplit.audition.AuditionOptions`) also writes, for the
    seams it selects, the audio around them joined in each of the three ways, so they can
    be compared by ear; it needs a ``report`` to read the seams from.

    Raises :class:`ValueError`, before any file is read, for an unknown
    ``method`` or a numeric option outside its range: ``samplerate`` must be ``None`` or a
    whole number greater than 0; ``overlap_s``, ``max_lag_ms``, ``max_gain_db``
    and ``length_tolerance_ms`` must be finite and not negative; ``seam_fade_ms``
    must be finite and greater than 0; ``seam_warn_db`` and ``recon_warn_db`` must not
    be NaN.

    ``progress(i, n, name)`` is called as chunk ``i`` of ``n`` (1-based, ``name``
    the file) is about to be read. An exception it raises stops the merge before
    anything is written, so it is also the way to cancel one."""
    tuning = _Tuning(
        method,
        align,
        max_lag_ms,
        match_level,
        max_gain_db,
        seam_search,
        seam_fade_ms,
        seam_warn_db,
        length_tolerance_ms,
        recon_warn_db,
    )
    samplerate = _check_numbers(tuning, samplerate, overlap_s)
    other_output = _check_outputs(
        output_path, other_match, other_output, pairing, originals_dir, anchor, audition, report
    )

    manifest = load_manifest(manifest_path)
    overlap_ms, note = _overlap_ms(manifest, overlap_s)
    if note is not None:
        _emit(report, note)

    part_names = manifest["parts"]
    paired, other_paired = _pair_stems(processed_dir, part_names, match, other_match)
    folder = originals_dir or os.path.dirname(os.path.abspath(manifest_path))
    original_paths = find_original_chunks(folder, part_names) if anchor else None
    stems = [(paired, match or "")] + ([(other_paired, other_match or "")] if other_paired else [])
    to_check, reference = _stems_to_check(stems, original_paths, folder, part_names, pairing)
    for note in _order_notes(stems, reference, pairing):
        _emit(report, note)
    if reference is not None:
        for files, what in to_check:
            files.paths = check_pairing(
                files.paths,
                reference,
                part_names,
                report,
                what if other_paired else "",
            )

    plan = MergePlan(
        manifest,
        overlap_ms,
        paired.paths,
        other_paired.paths if other_paired else None,
        original_paths,
        **asdict(tuning),
    )
    main, other = plan.split(plan.merge(report=report, progress=progress))

    note = _length_note(main, manifest, length_tolerance_ms)
    if note is not None:
        _emit(report, note)

    merged = _export(main, other, output_path, other_output, samplerate, bitrate)

    if audition is not None and report is not None:
        try:
            export_auditions(plan, report, audition, manifest.get("source_file"))
        except (OSError, ValueError, CouldntDecodeError, CouldntEncodeError) as exc:
            # the merged files are written and good: do not fail them over the extras
            _emit(report, f"could not prepare the listening kit: {exc}")
    return merged
