"""
Merging against the original chunks, and merging both stems of a service as one.

**Why.** Without it, ``merge`` only ever compares a chunk with its neighbour, in the few
seconds they share. That is a *relative* measurement: what one chunk is corrected by
depends on the one before it, so a small bias at every seam (a separator that gives a
chunk a little less level near its edges, say) adds up along the track, and where the
overlaps do not correspond (a vocals stem that is silent in the overlap) there is nothing
to measure at all. But the original chunks are on this computer: they are what was sent
to the service. Comparing each processed chunk with *its own original* gives an
**absolute** reference for where it belongs in time and, if the service gives back both
stems, for how loud it should be, because the two stems add up to (about) the original::

    vocals + instrumental  ~  original chunk

*Time.* The offset of ``processed`` against ``original`` is measured with the same
cross-correlation as between neighbours, and cancelled (zeros are put in front of a
chunk whose start was trimmed, and the silence a service put in front of one is
dropped). Every chunk then sits where the manifest says it was cut, however its
neighbours came out, and the first chunk is aligned too.

*Level.* One gain per chunk, the least-squares amplitude at which the sum of the stems
reproduces the original, applied to **both** stems so their balance is untouched. It
cannot be taken from one stem: an instrumental alone is quieter than the mix it came
from by exactly what the voice carried, so matching it to the original would make it
too loud. With one stem only the timing is anchored, and the level is left to the
neighbour matching. The correction is skipped (and said so) for a chunk whose
stems do not add up to the original (correlation below ``ANCHOR_GAIN_MIN_CORRELATION``)
or that would need more than ``ANCHOR_MAX_GAIN_DB``, where it would be a guess.

*Quality.* What is left of the original after the best offset and gain, ``recon_db``,
is how well the stems account for it. It is a consistency check, **not** a separation
score: a separator can reproduce the mix perfectly and still leave the voice in the
instrumental, and one that is not meant to conserve energy can separate well and reproduce
it badly. What it does catch is a chunk where the service did something to the audio
that is neither of the two (a different normalization of each stem, a dropout, a
different take of the material), so that chunk is the first to listen to. The 15 dB
default threshold is a heuristic, like ``--seam-warn-db``: look at your service's
typical value first.

*Both stems, one set of decisions.* Merged one at a time, the two stems would each get
their own offsets, gains and fade positions from their own overlaps, and what they
measure differs (a stem that is silent for a while measures nothing there). The stems
are therefore merged **together**: the two are stacked as the channels of one
multi-channel signal for :func:`seamsplit.crossfade._crossfade_merge`, which measures
on the average of all channels (the sum of the stems, i.e. the music) and applies
what it finds to all of them. Both outputs then have the same timing, the same gains
and the same fades, and add up to the same thing the original did.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from .audio import AudioSegment

from .dsp import (
    _DTYPES,
    LEVEL_DEADBAND_DB,
    MIN_ALIGN_CORRELATION,
    _to_int_array,
    anchor_fit,
)
from .report import ChunkInfo, MergeReport, _emit

# Seconds from the middle of a chunk used to find its offset (see ``anchor_fit``).
ANCHOR_LAG_WINDOW_S = 20.0

# The level of a chunk is taken from the original only when its stems add up to the
# original at least this well (normalized correlation of the sum with the original).
ANCHOR_GAIN_MIN_CORRELATION = 0.9

# The largest level correction taken from the original, dB in either direction. Larger
# than ``--max-gain-db``, which limits the *difference between neighbours*: a service
# that normalizes every chunk to a peak can be far from the original, and the same
# distance from it at every chunk is exactly what this puts right.
ANCHOR_MAX_GAIN_DB = 24.0

# A chunk whose stems add up to the original less well than this (dB) is flagged. An
# unvalidated heuristic: it only flags, it never changes audio.
DEFAULT_RECON_WARN_DB = 15.0


@dataclass
class AnchorOptions:
    """What :func:`prepare_chunk` is to do. ``max_lag_ms`` 0 leaves the timing alone
    (``--no-align``); ``level`` False leaves the level alone (``--no-match-level``)."""

    max_lag_ms: float = 50.0
    level: bool = True
    recon_warn_db: Optional[float] = DEFAULT_RECON_WARN_DB


def other_output_path(output_path: str, other_match: str) -> str:
    """Where the second stem goes when none was named: next to ``output_path``, with
    the word that identified it added (``song.mp3`` + ``vocals`` -> ``song_vocals.mp3``)."""
    word = "".join(c if c.isalnum() or c in "-." else "_" for c in other_match.strip())
    word = word.strip("._-") or "other"
    root, ext = os.path.splitext(output_path)
    return f"{root}_{word}{ext}"


def _frames(seg: AudioSegment) -> np.ndarray:
    """The samples of ``seg`` as float64, one row per frame."""
    return _to_int_array(seg).astype(np.float64).reshape(-1, seg.channels)


def _conform(seg: AudioSegment, like: AudioSegment) -> AudioSegment:
    """``seg`` in the format of ``like`` (levels compared across different sample
    widths would not mean anything)."""
    if seg.sample_width != like.sample_width:
        seg = seg.set_sample_width(like.sample_width)
    if seg.frame_rate != like.frame_rate:
        seg = seg.set_frame_rate(like.frame_rate)
    if seg.channels != like.channels:
        seg = seg.set_channels(like.channels)
    return seg


def _pad_rows(x: np.ndarray, rows: int) -> np.ndarray:
    return x if len(x) >= rows else np.concatenate([x, np.zeros((rows - len(x), x.shape[1]))])


def _shift(x: np.ndarray, lag: int) -> np.ndarray:
    """Cancel an offset: ``x[k] ~ original[k + lag]`` becomes ``out[k] ~ original[k]``,
    by putting ``lag`` frames of silence in front (the service trimmed the start) or
    dropping ``-lag`` of them (it put silence there)."""
    if lag > 0:
        return np.concatenate([np.zeros((lag, x.shape[1])), x])
    if lag < 0:
        return x[-lag:]
    return x


def _pack(blocks: Sequence[np.ndarray], like: AudioSegment) -> Tuple[AudioSegment, int]:
    """The blocks side by side as the channels of one segment in the sample type of
    ``like``, and how many samples did not fit and were clipped."""
    dtype = np.dtype(_DTYPES.get(like.array_type, np.int16))
    lo, hi = float(np.iinfo(dtype).min), float(np.iinfo(dtype).max)
    joined = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
    clipped = int(np.count_nonzero((joined > hi + 0.5) | (joined < lo - 0.5)))
    data = np.clip(np.rint(joined), lo, hi).astype(dtype).tobytes()
    seg = AudioSegment(
        data=data,
        sample_width=like.sample_width,
        frame_rate=like.frame_rate,
        channels=joined.shape[1],
    )
    return seg, clipped


def split_stems(merged: AudioSegment, channels: int) -> Tuple[AudioSegment, AudioSegment]:
    """Undo the stacking: the first ``channels`` channels of ``merged`` are the main
    stem, the rest the other one."""
    dtype = np.dtype(_DTYPES.get(merged.array_type, np.int16))
    rows = np.frombuffer(merged.raw_data, dtype=dtype).reshape(-1, merged.channels)
    parts = []
    for block in (rows[:, :channels], rows[:, channels:]):
        parts.append(
            AudioSegment(
                data=np.ascontiguousarray(block).tobytes(),
                sample_width=merged.sample_width,
                frame_rate=merged.frame_rate,
                channels=channels,
            )
        )
    return parts[0], parts[1]


def _anchor(
    part: int,
    blocks: List[np.ndarray],
    original: np.ndarray,
    sr: int,
    both: bool,
    options: AnchorOptions,
    report: Optional[MergeReport],
) -> Tuple[List[np.ndarray], ChunkInfo]:
    """Cancel the offset of the processed chunk (whose stems are ``blocks``) against
    ``original`` and, with both stems, bring it to the original's level."""
    info = ChunkInfo(part)
    reply = np.sum(blocks, axis=0).mean(axis=1)  # the stems added up, as mono
    max_lag = int(round(options.max_lag_ms / 1000.0 * sr))
    fit = anchor_fit(original.mean(axis=1), reply, max_lag, window=int(ANCHOR_LAG_WINDOW_S * sr))
    info.correlation = fit.correlation
    if not np.any(reply) or not np.any(original):
        return blocks, info  # silence: nothing to measure

    if fit.correlation < MIN_ALIGN_CORRELATION:
        _emit(
            report,
            f"part {part}: the processed chunk barely correlates with the original "
            f"({fit.correlation:.2f}), so it was not anchored to it",
        )
        return blocks, info

    if max_lag > 0:
        if abs(fit.lag) >= max_lag:
            _emit(
                report,
                f"part {part}: offset to the original reached the search limit "
                f"({1000.0 * fit.lag / sr:+.1f} ms); try a larger --max-lag-ms",
            )
        info.lag_ms = 1000.0 * fit.lag / sr
        blocks = [_shift(b, fit.lag) for b in blocks]

    if not both:
        return blocks, info  # one stem says nothing about the level of the mix

    info.recon_db = fit.recon_db
    if options.recon_warn_db is not None and fit.recon_db < options.recon_warn_db:
        info.flagged = True
        _emit(
            report,
            f"part {part}: the stems add up to the original only to {fit.recon_db:.1f} dB "
            f"(threshold {options.recon_warn_db:g} dB) — worth a listen",
        )
    if not options.level:
        return blocks, info

    gain_db = 20.0 * math.log10(fit.gain) if fit.gain > 0 else math.inf
    reason = None
    if fit.correlation < ANCHOR_GAIN_MIN_CORRELATION:
        reason = f"the stems do not add up to it (correlation {fit.correlation:.2f})"
    elif abs(gain_db) > ANCHOR_MAX_GAIN_DB:
        reason = f"it would take {gain_db:+.1f} dB (limit {ANCHOR_MAX_GAIN_DB:g} dB)"
    if reason is not None:
        info.flagged = True
        _emit(report, f"part {part}: the level was not taken from the original: {reason}")
        return blocks, info

    info.gain_db = 0.0
    if abs(gain_db) >= LEVEL_DEADBAND_DB:
        info.gain_db = gain_db
        blocks = [b * fit.gain for b in blocks]
    return blocks, info


def prepare_chunk(
    part: int,
    main: AudioSegment,
    other: Optional[AudioSegment],
    original: Optional[AudioSegment],
    options: AnchorOptions,
    report: Optional[MergeReport] = None,
    length_tolerance_ms: float = 60.0,
    extra_db: float = 0.0,
) -> Tuple[AudioSegment, int]:
    """One chunk made ready for the merge: ``main`` (the stem being merged), with the
    ``other`` stem stacked behind it as more channels when there is one, both shifted
    and scaled to ``original`` when that is given (see the module docstring), and
    ``extra_db`` more on top of that. ``part`` is 1-based and names the chunk in
    warnings; what was found is added to ``report.chunks``. Returns the chunk and how
    many of its samples had to be clipped (a chunk brought to the level of an original
    that peaks at full scale can go a few samples over)."""
    blocks = [_frames(main)]
    if other is not None:
        other = _conform(other, main)
        difference_ms = 1000.0 * abs(len(other.raw_data) - len(main.raw_data))
        difference_ms /= main.frame_width * main.frame_rate
        if difference_ms > length_tolerance_ms:
            _emit(
                report,
                f"part {part}: the two stems differ in length by {difference_ms:.0f} ms; "
                "the shorter one was padded with silence",
            )
        blocks.append(_frames(other))
        rows = max(len(b) for b in blocks)
        blocks = [_pad_rows(b, rows) for b in blocks]

    if original is not None:
        orig = _frames(_conform(original, main))
        blocks, info = _anchor(
            part, blocks, orig, main.frame_rate, other is not None, options, report
        )
        if report is not None:
            report.chunks.append(info)
    if extra_db:
        blocks = [b * 10.0 ** (extra_db / 20.0) for b in blocks]
    if original is None and other is None and not extra_db:
        return main, 0
    return _pack(blocks, main)
