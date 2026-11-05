"""The measures: signal-to-error ratios of a stem against another, in the whole file or
in windows around the seams, and levels around a point.

The ratio is the plain one, ``10 log10(sum(reference^2) / sum((reference - estimate)^2))``
over all samples of all channels. It is not the BSS-eval SDR of the separation
literature (which lets the estimate be filtered first) and the numbers are not to be
compared with those; it is used here to compare the *same* separator's output cut in
different ways, where the plain ratio is the stricter of the two and needs no
assumptions.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from seamsplit.audio import AudioSegment


def as_float(seg: AudioSegment) -> np.ndarray:
    """The samples as float, one row per frame, scaled so that full scale is 1.0."""
    scale = float(1 << (8 * seg.sample_width - 1))
    return seg.frames.astype(np.float64) / scale


def aligned(reference: AudioSegment, estimate: AudioSegment):
    """Both as float arrays, cut to the shorter, with the same number of channels."""
    if reference.frame_rate != estimate.frame_rate:
        estimate = estimate.set_frame_rate(reference.frame_rate)
    if estimate.channels != reference.channels:
        estimate = estimate.set_channels(reference.channels)
    a, b = as_float(reference), as_float(estimate)
    n = min(len(a), len(b))
    return a[:n], b[:n]


def ratio_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Reference energy over the energy of the difference, in dB (99 when they are equal)."""
    error = float(np.sum((reference - estimate) ** 2))
    signal = float(np.sum(reference**2))
    if error <= 1e-18:
        return 99.0
    if signal <= 1e-18:
        return -99.0
    return float(10.0 * np.log10(signal / error))


def snr_db(reference: AudioSegment, estimate: AudioSegment) -> float:
    a, b = aligned(reference, estimate)
    return ratio_db(a, b)


def window_slices(centres_ms: Sequence[float], half_ms: float, rate: int, frames: int):
    """``(start, stop)`` frame ranges of the windows ``centre +- half`` inside the file."""
    out = []
    for c in centres_ms:
        lo = max(0, int(round((c - half_ms) * rate / 1000.0)))
        hi = min(frames, int(round((c + half_ms) * rate / 1000.0)))
        if hi > lo:
            out.append((lo, hi))
    return out


def snr_windows_db(
    reference: AudioSegment, estimate: AudioSegment, centres_ms: Sequence[float], half_ms: float
) -> List[float]:
    """The ratio in each window around the given times."""
    a, b = aligned(reference, estimate)
    return [
        ratio_db(a[lo:hi], b[lo:hi])
        for lo, hi in window_slices(centres_ms, half_ms, reference.frame_rate, len(a))
    ]


def snr_pooled_windows_db(
    reference: AudioSegment, estimate: AudioSegment, centres_ms: Sequence[float], half_ms: float
) -> float:
    """One ratio over all the windows together (so that a quiet window does not count as
    much as a loud one: it is the error energy near the seams against the signal energy
    there)."""
    a, b = aligned(reference, estimate)
    spans = window_slices(centres_ms, half_ms, reference.frame_rate, len(a))
    if not spans:
        return float("nan")
    return ratio_db(
        np.concatenate([a[lo:hi] for lo, hi in spans]),
        np.concatenate([b[lo:hi] for lo, hi in spans]),
    )


def level_db(seg: AudioSegment, centre_ms: float, half_ms: float) -> float:
    """The RMS level (dB, full scale = 0) in the window around ``centre_ms``."""
    x = as_float(seg)
    lo = max(0, int(round((centre_ms - half_ms) * seg.frame_rate / 1000.0)))
    hi = min(len(x), int(round((centre_ms + half_ms) * seg.frame_rate / 1000.0)))
    if hi <= lo:
        return -120.0
    rms = float(np.sqrt(np.mean(x[lo:hi] ** 2)))
    return float(20.0 * np.log10(max(rms, 1e-6)))


def level_curve_db(seg: AudioSegment, window_ms: float = 200.0, hop_ms: float = 50.0):
    """Times (ms) and RMS levels (dB) of the file, window by window."""
    x = as_float(seg)
    rate = seg.frame_rate
    w = max(1, int(rate * window_ms / 1000.0))
    h = max(1, int(rate * hop_ms / 1000.0))
    mono = np.mean(x**2, axis=1)
    starts = np.arange(0, max(1, len(mono) - w + 1), h)
    cumulative = np.concatenate([[0.0], np.cumsum(mono)])
    power = (cumulative[starts + w] - cumulative[starts]) / w
    times = 1000.0 * (starts + w / 2) / rate
    return times, 10.0 * np.log10(np.maximum(power, 1e-12))


def snr_outside_windows_db(
    reference: AudioSegment, estimate: AudioSegment, centres_ms: Sequence[float], half_ms: float
) -> float:
    """One ratio over everything *but* the windows around the seams (what chunking costs
    away from a join: each chunk was separated without the music that follows it)."""
    a, b = aligned(reference, estimate)
    keep = np.ones(len(a), dtype=bool)
    for lo, hi in window_slices(centres_ms, half_ms, reference.frame_rate, len(a)):
        keep[lo:hi] = False
    if not keep.any():
        return float("nan")
    return ratio_db(a[keep], b[keep])


def click_db(seg: AudioSegment, centre_ms: float, near_ms: float = 10.0, around_ms: float = 1000.0):
    """How much the sample-to-sample change at a join stands out, without any reference: the
    energy of the first difference in the loudest millisecond within ``near_ms`` of the join,
    over the median of that energy in every millisecond within ``around_ms``, in dB.
    Near 0 or below: nothing to hear; clearly positive: a step or a click."""
    x = as_float(seg)
    rate = seg.frame_rate
    d = np.diff(x, axis=0)
    energy = np.sum(d**2, axis=1)
    block = max(1, int(rate / 1000))
    usable = (len(energy) // block) * block
    if usable < block:
        return float("nan")
    blocks = energy[:usable].reshape(-1, block).sum(axis=1)
    c = int(centre_ms)  # one block per millisecond
    near = blocks[max(0, c - int(near_ms)) : min(len(blocks), c + int(near_ms) + 1)]
    wide = blocks[max(0, c - int(around_ms)) : min(len(blocks), c + int(around_ms) + 1)]
    if len(near) == 0 or len(wide) == 0:
        return float("nan")
    typical = float(np.median(wide))
    return float(10.0 * np.log10(max(float(near.max()), 1e-18) / max(typical, 1e-18)))


def reconstruction_db(mix: AudioSegment, first: AudioSegment, second: AudioSegment) -> float:
    """How well two stems add up to the mix they were taken from, in dB."""
    a, b = aligned(mix, first)
    _, c = aligned(mix, second)
    n = min(len(a), len(c))
    return ratio_db(a[:n], b[:n] + c[:n])


def fit_gain(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """``estimate`` multiplied by the one number that brings it closest to ``reference`` (least
    squares), so that a level that is off as a whole does not count as an error: what is left
    is the part that is wrong *within* the file, which is what joining chunks can get wrong."""
    energy = float(np.sum(estimate**2))
    if energy <= 1e-18:
        return estimate
    return estimate * (float(np.sum(reference * estimate)) / energy)


def fit_lag(reference: np.ndarray, estimate: np.ndarray, rate: int, max_ms: float = 100.0):
    """``estimate`` moved in time by the one shift (up to ``max_ms``) that makes it most like
    ``reference``, and that shift in milliseconds (negative when the estimate was late). A file
    that is late or early as a whole is not an error a merge could know about; chunks that are
    late or early by different amounts are."""
    n = min(len(reference), len(estimate))
    a = reference[:n].mean(axis=1)
    b = estimate[:n].mean(axis=1)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    cross = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    reach = int(max_ms / 1000.0 * rate)
    lags = np.arange(-reach, reach + 1)
    values = cross[lags % size]
    shift = int(lags[int(np.argmax(values))])  # estimate[t] ~ reference[t + shift]
    moved = np.zeros_like(estimate)
    if shift > 0:
        moved[shift:] = estimate[: len(estimate) - shift]
    elif shift < 0:
        moved[: len(estimate) + shift] = estimate[-shift:]
    else:
        moved = estimate.copy()
    return moved, 1000.0 * shift / rate


def snr_fitted(
    reference: AudioSegment,
    estimate: AudioSegment,
    centres_ms: Sequence[float] = (),
    half_ms: float = 1000.0,
):
    """``(whole file, around the seams, shift in ms)``: ratios after :func:`fit_lag` and
    :func:`fit_gain` (the seams one is nan without ``centres_ms``)."""
    a, b = aligned(reference, estimate)
    b, shift_ms = fit_lag(a, b, reference.frame_rate)
    b = fit_gain(a, b)
    whole = ratio_db(a, b)
    spans = window_slices(centres_ms, half_ms, reference.frame_rate, len(a))
    if not spans:
        return whole, float("nan"), shift_ms
    seams = ratio_db(
        np.concatenate([a[lo:hi] for lo, hi in spans]),
        np.concatenate([b[lo:hi] for lo, hi in spans]),
    )
    return whole, seams, shift_ms
