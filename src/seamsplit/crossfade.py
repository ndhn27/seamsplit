"""
Re-assemble audio chunks that were split with overlap by
``seamsplit.splitter`` (and then processed independently, e.g. by an AI
vocal/instrumental separator) using a real crossfade across each overlap
region — instead of a straight concatenation, which just glues PCM data
together and leaves any level or phase mismatch between the independently
processed chunks audible as a click or seam.

Alignment: an AI service (or an MP3 encoder/decoder) can add or trim a few
milliseconds at the start of a chunk. Blending two copies of the same audio
that are offset by even a millisecond causes comb filtering, so before each
crossfade the offset between the two overlap regions is measured with a
cross-correlation and the overlap is lengthened/shortened to cancel it.

Level matching: a service may also normalize every chunk differently, which
would leave a step in loudness at each seam. Both chunks carry the same
music in the overlap, so their RMS ratio there is the exact gain needed; it
is applied to the whole next chunk (limited to ``max_gain_db``).

Seam search (optional, ``seam_search=True``, off by default): the overlap is
long (seconds), and blending two independently separated versions of it over
its whole length smears their different artifacts across all of it. The idea
is that each chunk is at its worst near its own edges, so the overlap is
scanned for the stretch where the two chunks agree best (the difference
energy relative to the local signal energy, away from both edges) and only a
short crossfade is made there; the rest of the overlap comes from whichever
chunk is "inside" at that point.

That premise is a hypothesis, not a measured fact: it has not been checked
against real separator output, and it can lose. If both chunks carry about
the same error everywhere in the overlap, a fade over the whole overlap
averages the two errors while a short fade cannot. Listen to both before
relying on it.

Whether or not seam search is used, how well the two chunks agree at each
seam is reported (``MergeReport``), and seams where they differ a lot are
flagged.

Which fade curve? It depends on how alike the two overlap regions are,
measured by their normalized correlation ``rho``. For identical audio the
gains must sum to 1 (a *linear* crossfade) to keep the level constant; for
uncorrelated audio the *powers* must sum to 1 (an *equal-power* crossfade).
Real separator output sits in between, so the default ``adaptive`` method
interpolates spherically between the two chunks (``theta = arccos(rho)``)::

    gain_out = sin((1 - t) * theta) / sin(theta)
    gain_in  = sin(t * theta) / sin(theta)

This is exactly linear at ``rho = 1``, the sine/cosine equal-power law at
``rho = 0``, and keeps the power of the blend constant for every ``rho`` in
between (``gain_out**2 + gain_in**2 + 2*rho*gain_out*gain_in == 1``).
``linear`` and ``equalpower`` remain available as fixed curves.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import numpy as np
from .audio import AudioSegment

from .checks import _check_merge_options, _check_number
from .dsp import (
    _DTYPES,
    LEVEL_DEADBAND_DB,
    MIN_ALIGN_CORRELATION,
    _fade_gains,
    _level_ratio,
    _ncc,
    _to_int_array,
    _to_mono,
    estimate_lag,
    find_seam,
    match_db,
)
from .report import BoundaryInfo, MergeReport, _emit, format_time

METHODS = ("adaptive", "linear", "equalpower")


DEFAULT_METHOD = "adaptive"


DEFAULT_MAX_LAG_MS = 50.0


# Largest level correction (dB, either direction) applied to a chunk by default.
DEFAULT_MAX_GAIN_DB = 6.0


# Length of the crossfade made at the best-matching point of an overlap.
DEFAULT_SEAM_FADE_MS = 250.0


# A seam where the two chunks agree to less than this (dB, signal vs their
# difference) is flagged. An unvalidated heuristic (roughly rho < 0.985 for
# similar levels), picked without real separator output: tune it against what
# a given service's good seams look like. It only flags; it never changes audio.
DEFAULT_SEAM_WARN_DB = 15.0


def _crossfade_merge(
    segments: Iterable[AudioSegment],
    overlap_ms: float,
    method: str,
    max_lag_ms: float = 0.0,
    report: Optional[MergeReport] = None,
    match_level: bool = False,
    max_gain_db: float = DEFAULT_MAX_GAIN_DB,
    seam_search: bool = False,
    seam_fade_ms: float = DEFAULT_SEAM_FADE_MS,
    seam_warn_db: Optional[float] = DEFAULT_SEAM_WARN_DB,
) -> AudioSegment:
    if isinstance(segments, (list, tuple)) and not segments:
        raise ValueError("segments must not be empty")
    _check_number("overlap_ms", overlap_ms)
    _check_merge_options(max_lag_ms, max_gain_db, seam_fade_ms, seam_warn_db)

    # The segments are taken one at a time (a generator that decodes each file
    # when it is reached is fine), so only one chunk is in memory beside the
    # merged audio, not all of them.
    remaining = iter(segments)
    first = next(remaining, None)
    if first is None:
        raise ValueError("segments must not be empty")
    sr = first.frame_rate
    ch = first.channels
    sample_width = first.sample_width
    np_dtype = np.dtype(_DTYPES.get(first.array_type, np.int16))
    lo, hi = float(np.iinfo(np_dtype).min), float(np.iinfo(np_dtype).max)

    def as_samples(seg: AudioSegment) -> np.ndarray:
        if seg.frame_rate != sr or seg.channels != ch:
            seg = seg.set_frame_rate(sr).set_channels(ch)
        return _to_int_array(seg).astype(np.float64)

    overlap_frames = max(0, int(round(overlap_ms / 1000.0 * sr)))
    max_lag_frames = int(round(max_lag_ms / 1000.0 * sr))
    max_ratio = 10.0 ** (max_gain_db / 20.0)
    fade_frames = max(2, int(round(seam_fade_ms / 1000.0 * sr)))
    # The overlaps only need to be compared if something depends on the result.
    analyse = max_lag_frames > 0 or match_level or seam_search or method == "adaptive"

    # A boundary only ever reads or changes the last `overlap + max lag` frames of
    # the audio merged so far (the lag is at most max_lag_frames in either
    # direction). Everything before that is final: it is rounded to the sample
    # type of the chunks and kept as bytes (2 bytes per 16-bit sample) instead of
    # as float64 (8 bytes), and `prev` holds only the part that can still change.
    keep_samples = (overlap_frames + max_lag_frames) * ch
    settled: List[bytes] = []
    settled_frames = 0
    clipped = 0

    def settle(block: np.ndarray, keep: int = keep_samples) -> np.ndarray:
        """Store all of ``block`` but its last ``keep`` samples as final and
        return those (a copy, so the big array can be freed)."""
        nonlocal settled_frames, clipped
        cut = max(0, len(block) - keep)
        if cut:
            final = block[:cut]
            clipped += int(np.count_nonzero((final > hi + 0.5) | (final < lo - 0.5)))
            np.rint(final, out=final)
            np.clip(final, lo, hi, out=final)
            settled.append(final.astype(np_dtype).tobytes())
            settled_frames += cut // ch
            block = block[cut:].copy()
        return block

    prev = settle(as_samples(first))
    del first

    for i, seg in enumerate(remaining, start=1):
        nxt = as_samples(seg)
        del seg
        merged_frames = settled_frames + len(prev) // ch  # length of the audio merged so far

        frames = min(overlap_frames, len(prev) // ch, len(nxt) // ch)
        # Without a measurement (silent overlap, analysis off) the two sides are
        # assumed to be the same audio, which is the linear curve.
        fade_rho = 1.0
        info: Optional[BoundaryInfo] = None
        coherent = False  # the overlaps clearly are the same audio
        pad_frames = 0  # start of the overlap where the next chunk has no real audio yet

        if frames > 0 and analyse:
            tail_full = prev[-frames * ch :]
            head_full = nxt[: frames * ch]
            tail = _to_mono(tail_full, ch)
            head = _to_mono(head_full, ch)
            silent = min(np.sqrt(np.mean(tail**2)), np.sqrt(np.mean(head**2))) < 1.0
            if not silent:
                lag, corr = estimate_lag(tail, head, max_lag_frames)
                coherent = corr >= MIN_ALIGN_CORRELATION
                gain_db = 0.0

                if coherent:
                    fade_rho = corr
                    if max_lag_frames > 0 and abs(lag) >= max_lag_frames:
                        _emit(
                            report,
                            f"boundary {i}/{i + 1}: offset reached the search limit "
                            f"({1000.0 * lag / sr:+.1f} ms); try a larger --max-lag-ms",
                        )
                    if match_level:
                        ratio = _level_ratio(tail_full, head_full, lag * ch)
                        wanted_db = 20.0 * math.log10(ratio)
                        if ratio > max_ratio or ratio < 1.0 / max_ratio:
                            _emit(
                                report,
                                f"boundary {i}/{i + 1}: chunks differ by {-wanted_db:+.1f} dB "
                                f"in level; correction limited to ±{max_gain_db:g} dB "
                                "(see --max-gain-db)",
                            )
                            ratio = min(max(ratio, 1.0 / max_ratio), max_ratio)
                        if abs(20.0 * math.log10(ratio)) >= LEVEL_DEADBAND_DB:
                            nxt = nxt * ratio
                            gain_db = 20.0 * math.log10(ratio)
                    if max_lag_frames > 0:
                        pad_frames = max(0, -lag)
                        # b[k] ~ a[k + lag] => the true overlap is `lag` shorter
                        frames = max(0, min(frames - lag, len(prev) // ch, len(nxt) // ch))
                else:
                    # no shift is applied, so the fade is shaped by the
                    # correlation the two regions really have as they are
                    fade_rho = _ncc(tail, head, 0)
                    skipped = [
                        what
                        for what, enabled in (
                            ("alignment", max_lag_frames > 0),
                            ("level matching", match_level),
                            ("seam search", seam_search),
                        )
                        if enabled
                    ]
                    detail = f", so no {' or '.join(skipped)} was applied" if skipped else ""
                    _emit(
                        report,
                        f"boundary {i}/{i + 1}: overlap regions barely correlate "
                        f"({corr:.2f}){detail}",
                    )

                info = BoundaryInfo(
                    i,
                    1000.0 * lag / sr,
                    corr,
                    aligned=coherent and max_lag_frames > 0,
                    gain_db=gain_db,
                    flagged=not coherent,
                )

        this_overlap = frames * ch
        if this_overlap <= 0:
            prev = settle(np.concatenate([prev, nxt]))
            continue

        tail_samples = prev[-this_overlap:]
        head_samples = nxt[:this_overlap]

        # Where the crossfade goes: a short one at the best-matching point, or
        # (no seam search, or nothing to search on) one across the whole overlap.
        start, width = 0, frames
        if coherent and seam_search and fade_frames < frames:
            start, width = find_seam(tail_samples, head_samples, ch, fade_frames)
        if info is not None and coherent:
            sl = slice(start * ch, (start + width) * ch)
            fade_rho = _ncc(tail_samples[sl], head_samples[sl], 0) if width < frames else fade_rho
            real = slice(pad_frames * ch, None)
            info.overlap_match_db = match_db(tail_samples[real], head_samples[real])
            info.seam_match_db = match_db(tail_samples[sl], head_samples[sl])
            judged = info.seam_match_db if seam_search else info.overlap_match_db
            if seam_warn_db is not None and judged < seam_warn_db:
                info.flagged = True
                _emit(
                    report,
                    f"boundary {i}/{i + 1}: the chunks only agree to {judged:.1f} dB "
                    f"{'at the best seam' if seam_search else 'across the overlap'} "
                    f"(threshold {seam_warn_db:g} dB) near "
                    f"{format_time(1000.0 * ((merged_frames - frames) + start + width / 2) / sr)}"
                    " — worth a listen",
                )
        if info is not None:
            info.fade_ms = 1000.0 * width / sr
            info.seam_ms = 1000.0 * ((merged_frames - frames) + start + width / 2.0) / sr
            info.fade_correlation = fade_rho
            if report is not None:
                report.boundaries.append(info)

        t = np.linspace(0.0, 1.0, width, endpoint=True)
        fade_out, fade_in = _fade_gains(method, t, fade_rho)
        # before the window only the outgoing chunk is heard, after it only the incoming
        fade_out = np.concatenate([np.ones(start), fade_out, np.zeros(frames - start - width)])
        fade_in = np.concatenate([np.zeros(start), fade_in, np.ones(frames - start - width)])
        if ch > 1:
            fade_out = np.repeat(fade_out, ch)
            fade_in = np.repeat(fade_in, ch)

        blended = tail_samples * fade_out + head_samples * fade_in

        prev = settle(np.concatenate([prev[:-this_overlap], blended, nxt[this_overlap:]]))

    settle(prev, 0)
    if clipped:
        _emit(
            report,
            f"{clipped} sample(s) exceeded full scale and were clipped; "
            "lower the level of the source chunks or use a different --method",
        )
    data = b"".join(settled)
    settled.clear()
    return AudioSegment(data=data, sample_width=sample_width, frame_rate=sr, channels=ch)


def adaptive_crossfade_merge(
    segments: Iterable[AudioSegment],
    overlap_ms: float,
    max_lag_ms: float = 0.0,
    report: Optional[MergeReport] = None,
    *,
    match_level: bool = False,
    max_gain_db: float = DEFAULT_MAX_GAIN_DB,
    seam_search: bool = False,
    seam_fade_ms: float = DEFAULT_SEAM_FADE_MS,
    seam_warn_db: Optional[float] = DEFAULT_SEAM_WARN_DB,
) -> AudioSegment:
    """Concatenate ``segments`` in order, crossfading ``overlap_ms`` between
    every consecutive pair with a curve chosen from the measured correlation
    of the two overlap regions: linear for identical audio, equal-power for
    unrelated audio, and a constant-power blend in between (see the module
    docstring).

    ``max_lag_ms > 0`` first measures and cancels any timing offset (up to
    that many ms) between the two overlap regions. ``match_level=True`` also
    scales each chunk so its overlap is as loud as the previous one's, by at
    most ``max_gain_db`` in either direction. ``seam_search=True`` replaces the
    fade across the whole overlap by one of ``seam_fade_ms`` at the position
    where the two chunks agree best (see :func:`find_seam`); seams whose
    agreement is below ``seam_warn_db`` (``None`` = never) are flagged in the
    report."""
    return _crossfade_merge(
        segments,
        overlap_ms,
        "adaptive",
        max_lag_ms,
        report,
        match_level,
        max_gain_db,
        seam_search,
        seam_fade_ms,
        seam_warn_db,
    )


def linear_crossfade_merge(
    segments: Iterable[AudioSegment],
    overlap_ms: float,
    max_lag_ms: float = 0.0,
    report: Optional[MergeReport] = None,
    *,
    match_level: bool = False,
    max_gain_db: float = DEFAULT_MAX_GAIN_DB,
    seam_search: bool = False,
    seam_fade_ms: float = DEFAULT_SEAM_FADE_MS,
    seam_warn_db: Optional[float] = DEFAULT_SEAM_WARN_DB,
) -> AudioSegment:
    """Concatenate ``segments`` in order with a linear crossfade of
    ``overlap_ms`` between every consecutive pair (gains ``1 - t`` and
    ``t``, which sum to 1 — the right curve when both sides carry the same
    audio, as they do for chunks produced by ``seamsplit split``).

    ``max_lag_ms > 0`` first measures and cancels any timing offset (up to
    that many ms) between the two overlap regions; ``match_level`` as in
    :func:`adaptive_crossfade_merge`."""
    return _crossfade_merge(
        segments,
        overlap_ms,
        "linear",
        max_lag_ms,
        report,
        match_level,
        max_gain_db,
        seam_search,
        seam_fade_ms,
        seam_warn_db,
    )


def equal_power_crossfade_merge(
    segments: Iterable[AudioSegment],
    overlap_ms: float,
    max_lag_ms: float = 0.0,
    report: Optional[MergeReport] = None,
    *,
    match_level: bool = False,
    max_gain_db: float = DEFAULT_MAX_GAIN_DB,
    seam_search: bool = False,
    seam_fade_ms: float = DEFAULT_SEAM_FADE_MS,
    seam_warn_db: Optional[float] = DEFAULT_SEAM_WARN_DB,
) -> AudioSegment:
    """Concatenate ``segments`` in order with an equal-power crossfade
    (gains ``sqrt(1 - t)`` and ``sqrt(t)``) across ``overlap_ms``.

    Equal-power keeps *power* constant for **uncorrelated** signals. On the
    coherent overlaps ``seamsplit split`` produces it raises the level by up
    to 3 dB mid-fade, so prefer :func:`adaptive_crossfade_merge` or
    :func:`linear_crossfade_merge` unless the two sides genuinely differ."""
    return _crossfade_merge(
        segments,
        overlap_ms,
        "equalpower",
        max_lag_ms,
        report,
        match_level,
        max_gain_db,
        seam_search,
        seam_fade_ms,
        seam_warn_db,
    )
