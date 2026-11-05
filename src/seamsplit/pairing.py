"""Telling which chunk a processed file belongs to by what it sounds like.

A service hands back files whose names may have been lost (``download (3).mp3``), and
pairing those with the parts by order is a guess. The original chunks ``split`` wrote
are still here, and a processed file is, whatever the service did to it (a stem, a
different level, an offset of some milliseconds), a piece of the chunk it came from, so
it correlates with that chunk and with no other. This module measures that:
:func:`identify` says, for each file, which part it is most like and how sure that is, and
:func:`check_pairing` uses it to confirm (or put right) the order the files were paired in.

How the comparison is made: every file is brought to 2 kHz mono and **whitened** (every
frequency band given the same weight, in bands of 2 Hz), then the peak of the normalised
cross-correlation within ±1 s is taken. Whitening matters: without it a sustained bass
note, which every chunk of a song shares, dominates and chunks look alike. What it
cannot do is tell apart chunks whose *content* repeats exactly (a loop played unchanged
through the whole song); those come out as "not sure" rather than as a wrong answer,
and :func:`check_pairing` then leaves the order alone and says so.

The thresholds (:data:`MIN_PAIR_NCC`, :data:`MIN_PAIR_MARGIN`) were set on synthetic
songs, one and two channels, instrumental and vocal stems, against the other chunks of the
same song: ``score`` of a true pair 0.2-0.7, of a wrong one up to about 0.15, margins at
least 0.12.

They were then measured on real stems (``benchmarks/``: five songs, 47 chunks, both stems of each,
separated by a UVR model): the part a file correlated with best was the right one for all 94, a
true pair scored 0.11-1.0 (instrumental stems 0.82 and up, vocal stems 0.11-0.8, the lowest where
there is hardly any singing) and a wrong one up to 0.06 for a vocal stem and 0.28 for an
instrumental stem of a looped track. 90 of the 94 cleared the thresholds; the other four were
left "not sure", which is the cautious side. Nothing was ever paired wrongly with them. Five
songs are few: the thresholds were not moved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from .audio import AudioSegment, CouldntDecodeError

from .report import MergeReport, _emit, _note

SIGNATURE_RATE = 2000
WHITEN_HZ = 2.0
WHITEN_FLOOR = 0.1
DEFAULT_PAIR_LAG_MS = 1000.0
# A file is identified when it is at least this much like its best part, and ahead of the
# next best by at least this much (see the module docstring for where they come from).
MIN_PAIR_NCC = 0.15
MIN_PAIR_MARGIN = 0.08

PAIRING_MODES = ("auto", "verify", "off")


@dataclass(frozen=True)
class Match:
    """Which part a file is most like. ``part`` is 0-based, ``None`` when the file could
    not be read (``error`` says why) or holds nothing but silence. ``score`` is the peak
    normalised correlation with that part and ``runner_up`` the best with any other;
    ``confident`` is whether ``score`` and ``score - runner_up`` both clear the
    thresholds."""

    file: str
    part: Optional[int]
    score: float
    runner_up: float
    confident: bool
    error: Optional[str] = None


def _whiten(x: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    width = max(3, int(WHITEN_HZ / (SIGNATURE_RATE / len(x))) | 1)
    magnitude = np.abs(spectrum)
    smooth = np.convolve(magnitude, np.ones(width) / width, mode="same")
    flat: np.ndarray = np.fft.irfft(
        spectrum / (smooth + WHITEN_FLOOR * smooth.mean() + 1e-12), len(x)
    )
    return flat


def signature(path: str) -> np.ndarray:
    """The audio of ``path`` as the comparison sees it (see the module docstring); an
    empty array for a file that holds no audio. Raises ``CouldntDecodeError`` or
    ``OSError`` for one that cannot be read."""
    segment = AudioSegment.from_file(path)
    if segment.frame_count() < 2 or segment.sample_width == 0:
        return np.zeros(0)
    mono = segment.frames.astype(np.float64).mean(axis=1)
    mono /= float(1 << (8 * segment.sample_width - 1))
    box = max(1, int(round(segment.frame_rate / SIGNATURE_RATE)))
    if box > 1:  # a plain average is the low-pass; the rate only has to be the same everywhere
        mono = np.convolve(mono, np.ones(box) / box, mode="same")
    seconds = len(mono) / segment.frame_rate
    grid = np.arange(int(seconds * SIGNATURE_RATE)) / SIGNATURE_RATE
    if len(grid) < 16:
        return np.zeros(0)
    x = np.interp(grid, np.arange(len(mono)) / segment.frame_rate, mono)
    x -= x.mean()
    if not np.any(x):
        return np.zeros(0)
    return _whiten(x)


def score_matrix(
    candidates: Sequence[np.ndarray],
    originals: Sequence[np.ndarray],
    max_lag_ms: float = DEFAULT_PAIR_LAG_MS,
) -> np.ndarray:
    """``scores[i, j]``: the peak normalised cross-correlation of ``candidates[i]`` with
    ``originals[j]`` within ``±max_lag_ms`` (0 when either holds no audio)."""
    scores = np.zeros((len(candidates), len(originals)))
    sizes = [len(v) for v in list(candidates) + list(originals) if len(v)]
    if not sizes:
        return scores
    size = 1 << int(np.ceil(np.log2(2 * max(sizes))))
    reach = int(max_lag_ms / 1000.0 * SIGNATURE_RATE)
    spectra = [np.fft.rfft(v, size) if len(v) else None for v in originals]
    for i, cand in enumerate(candidates):
        if not len(cand):
            continue
        cand_spectrum = np.fft.rfft(cand, size)
        for j, orig_spectrum in enumerate(spectra):
            if orig_spectrum is None:
                continue
            cross = np.fft.irfft(cand_spectrum * np.conj(orig_spectrum), size)
            peak = max(cross[: reach + 1].max(), cross[-reach:].max() if reach else -np.inf)
            scale = float(np.linalg.norm(cand) * np.linalg.norm(originals[j]))
            scores[i, j] = max(0.0, float(peak) / scale) if scale > 0 else 0.0
    return scores


def _is_sure(score: float, runner_up: float) -> bool:
    return score >= MIN_PAIR_NCC and score - runner_up >= MIN_PAIR_MARGIN


def _load(
    paths: Sequence[str],
    progress: Optional[Callable[[int, int, str], None]],
    offset: int,
    total: int,
) -> "tuple[List[np.ndarray], List[Optional[str]]]":
    signatures: List[np.ndarray] = []
    errors: List[Optional[str]] = []
    for i, path in enumerate(paths):
        if progress is not None:
            progress(offset + i + 1, total, os.path.basename(path))
        try:
            signatures.append(signature(path))
            errors.append(None)
        except (OSError, CouldntDecodeError) as exc:
            signatures.append(np.zeros(0))
            errors.append(str(exc) or exc.__class__.__name__)
    return signatures, errors


def compare(
    candidates: Sequence[str],
    originals: Sequence[str],
    max_lag_ms: float = DEFAULT_PAIR_LAG_MS,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> "tuple[np.ndarray, List[Optional[str]]]":
    """The score matrix of ``candidates`` against ``originals`` (files), and for every
    candidate the reason it could not be read (``None`` if it could). An original that
    cannot be read is an error: nothing can be said about a file without it."""
    total = len(candidates) + len(originals)
    orig_sigs, orig_errors = _load(originals, progress, 0, total)
    for path, error in zip(originals, orig_errors):
        if error is not None:
            raise OSError(f"could not read the original chunk '{path}': {error}")
    cand_sigs, cand_errors = _load(candidates, progress, len(originals), total)
    return score_matrix(cand_sigs, orig_sigs, max_lag_ms), cand_errors


def matches_from(
    candidates: Sequence[str], scores: np.ndarray, errors: Sequence[Optional[str]]
) -> List[Match]:
    out = []
    for path, row, error in zip(candidates, scores, errors):
        if error is not None or not np.any(row > 0):
            out.append(Match(path, None, 0.0, 0.0, False, error))
            continue
        ranked = np.argsort(row)[::-1]
        best = float(row[ranked[0]])
        second = float(row[ranked[1]]) if len(ranked) > 1 else 0.0
        out.append(Match(path, int(ranked[0]), best, second, _is_sure(best, second)))
    return out


def identify(
    candidates: Sequence[str],
    originals: Sequence[str],
    max_lag_ms: float = DEFAULT_PAIR_LAG_MS,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[Match]:
    """For every file of ``candidates``, the part (index into ``originals``) it is most
    like. Several files may name the same part (both stems of a chunk, a duplicate);
    that is for the caller to make sense of."""
    scores, errors = compare(candidates, originals, max_lag_ms, progress)
    return matches_from(candidates, scores, errors)


@dataclass
class Resolution:
    """The order a set of files, paired with the parts as ``order[k]`` = the file at part
    ``k``, should be in: ``order`` lists file indices in part order, ``moved`` the files
    that were not where they belong, ``unsure`` those nothing could be said about, and
    ``problem`` (set instead of a usable ``order``) why the files do not fit the parts."""

    order: List[int]
    moved: List[int]
    unsure: List[int]
    problem: Optional[str] = None


def resolve_order(matches: Sequence[Match], names: Sequence[str]) -> Resolution:
    """Reconcile files paired with the parts by order (``matches[k]`` is the file that was
    paired with part ``k``, ``names[k]`` that part's name) with what they are like.

    A file that is clearly another part goes there; a file nothing could be said about
    stays where it was paired. The result is used only if it is a proper pairing: each
    part gets exactly one file. Otherwise ``problem`` says which files disagree."""
    n = len(matches)
    target = [m.part if (m.confident and m.part is not None) else k for k, m in enumerate(matches)]
    unsure = [k for k, m in enumerate(matches) if not m.confident]
    claimed: dict = {}
    for k, part in enumerate(target):
        claimed.setdefault(part, []).append(k)
    clashes = {part: ks for part, ks in claimed.items() if len(ks) > 1}
    if clashes:
        lines = []
        for part, ks in sorted(clashes.items()):
            files = ", ".join(f"'{os.path.basename(matches[k].file)}'" for k in ks)
            lines.append(f"{files} all belong to part {part + 1} ('{names[part]}')")
        empty = sorted(set(range(n)) - set(claimed))
        if empty:
            lines.append("nothing was found for " + ", ".join(f"part {p + 1}" for p in empty))
        return Resolution(list(range(n)), [], unsure, "; ".join(lines))
    order = [0] * n
    for k, part in enumerate(target):
        order[part] = k
    moved = [k for k, part in enumerate(target) if part != k]
    return Resolution(order, moved, unsure)


def check_pairing(
    paths: Sequence[str],
    originals: Sequence[str],
    names: Sequence[str],
    report: Optional[MergeReport] = None,
    what: str = "",
    max_lag_ms: float = DEFAULT_PAIR_LAG_MS,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[str]:
    """Check ``paths`` (``paths[k]`` was paired with part ``k``, ``names[k]``) against the
    original chunks ``originals`` and return them in the right order.

    Everything it finds goes to ``report``: a note when every file was confirmed, a
    warning when the order had to be changed or some files could not be checked. Raises
    :class:`ValueError` when the files cannot be fitted to the parts at all (two of them
    are the same part, so another has none): merging them would produce nonsense. ``what``
    names the set of files in messages ("vocals", ...)."""
    label = f" ({what})" if what else ""
    scores, errors = compare(paths, originals, max_lag_ms, progress)
    matches = matches_from(paths, scores, errors)
    resolution = resolve_order(matches, names)
    if resolution.problem is not None:
        raise ValueError(
            f"the processed files{label} do not fit the parts: {resolution.problem}. "
            "'seamsplit status' shows which chunk each file is; --pairing off merges "
            "them in the order they were paired anyway"
        )

    ordered = [paths[k] for k in resolution.order]
    if resolution.moved:
        moves = []
        for k in resolution.moved:
            part = matches[k].part
            assert part is not None  # only a file that is sure of its part is moved
            moves.append(f"'{os.path.basename(matches[k].file)}' → part {part + 1}")
        weakest = min(matches[k].score - matches[k].runner_up for k in resolution.moved)
        more = len(moves) - 6
        _emit(
            report,
            f"{len(moves)} processed file(s){label} were not in part order: compared with the "
            "original chunks, "
            + ", ".join(moves[:6])
            + (f" and {more} more" if more > 0 else "")
            + f" (every one is ahead of the next best match by at least {weakest:.2f}); "
            "they were put in order by that",
        )
    if resolution.unsure:
        shown = ", ".join(
            f"'{os.path.basename(matches[k].file)}' (part {k + 1}, best similarity "
            f"{matches[k].score:.2f})"
            for k in resolution.unsure[:6]
        )
        more = len(resolution.unsure) - 6
        _emit(
            report,
            f"{len(resolution.unsure)} processed file(s){label} could not be told apart from "
            f"the other chunks by comparison with the originals, and stay where they were "
            f"paired: {shown}" + (f" and {more} more" if more > 0 else ""),
        )
    elif not resolution.moved:
        lowest = min(m.score for m in matches)
        _note(
            report,
            f"the {len(paths)} processed files{label} were each compared with the original "
            f"chunk they were paired with, and all are that chunk (lowest similarity "
            f"{lowest:.2f})",
        )
    return ordered
