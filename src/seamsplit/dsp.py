"""The signal processing of a merge, on plain arrays: measuring the offset between
two overlaps (``estimate_lag``), how alike they are (``match_db``, normalized
correlation), the gain to bring one to the level of the other, the fade curves,
and the search for the stretch where two chunks agree best (``find_seam``). Nothing
here reads a file or knows about reports; :mod:`seamsplit.crossfade` puts it to
work."""

from __future__ import annotations

import math
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
from .audio import AudioSegment

from .checks import _check_number

# Below this normalized correlation the two overlap regions are not similar
# enough for the measured offset to be trusted, so no shift is applied.
MIN_ALIGN_CORRELATION = 0.5


# Level differences smaller than this are left alone.
LEVEL_DEADBAND_DB = 0.05


# Below this angle (radians) the adaptive curve is indistinguishable from linear.
_SLERP_EPS = 1e-4


# Fraction of the overlap kept free of the seam at each end (chunk edges).
_SEAM_GUARD_FRAC = 0.1


# Quiet stretches are judged against this floor (dB below the overlap's mean
# power) so near-silence with tiny absolute differences is not penalized.
_SEAM_FLOOR_DB = -40.0


# Positions scoring within this factor of the best count as ties; the one
# closest to the middle of the overlap wins.
_SEAM_TIE = 1.05


_MATCH_CAP_DB = 99.0


_DTYPES: Dict[str, Any] = {"b": np.int8, "h": np.int16, "i": np.int32, "l": np.int32}


def _to_int_array(segment: AudioSegment) -> np.ndarray:
    return segment.samples


def _to_mono(interleaved: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return interleaved
    return interleaved.reshape(-1, channels).mean(axis=1)


def estimate_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> Tuple[int, float]:
    """Measure how far ``b`` is offset from ``a``.

    Returns ``(lag, correlation)`` where ``lag`` (in samples, within
    ``±max_lag``) is the shift for which ``b[k] ≈ a[k + lag]`` and
    ``correlation`` is the normalized cross-correlation at that shift
    (1.0 = identical shape, ~0 = unrelated). If several shifts score within
    0.01 of the best (periodic signals), the one closest to 0 wins. ``a`` and ``b`` are compared
    over the part where they overlap at each candidate shift, so a large
    lag doesn't get an unfair advantage or penalty.

    ``max_lag`` must be finite and not negative (``ValueError`` otherwise); a
    negative range is a mistake, not "nothing to search", and must not come
    back looking like two unrelated signals.
    """
    _check_number("max_lag", max_lag)
    n = min(len(a), len(b))
    max_lag = min(int(max_lag), n - 1)
    if n < 2 or max_lag < 0:
        return 0, 0.0
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)

    size = 1 << int(2 * n - 1).bit_length()
    # circular cross-correlation: raw[m] = sum_k a[k + m] * b[k]; zero padding
    # to >= 2n makes it equal to the linear one for every |m| < n
    raw_full = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    lags = np.arange(-max_lag, max_lag + 1)
    raw = raw_full[lags % size]

    cs_a = np.concatenate(([0.0], np.cumsum(a * a)))
    cs_b = np.concatenate(([0.0], np.cumsum(b * b)))
    absl = np.abs(lags)
    pos = lags >= 0
    energy_a = np.where(pos, cs_a[n] - cs_a[absl], cs_a[n - absl])
    energy_b = np.where(pos, cs_b[n - absl], cs_b[n] - cs_b[absl])
    denom = np.sqrt(np.maximum(energy_a * energy_b, 0.0))
    ncc = np.divide(raw, denom, out=np.zeros_like(raw), where=denom > 1e-9)

    best = float(ncc.max())
    # Periodic material (a held note, a pure tone) correlates equally well at
    # every multiple of its period; among near-equal peaks take the smallest
    # shift instead of an arbitrary one.
    near_best = np.flatnonzero(ncc >= best - 0.01)
    k = int(near_best[np.argmin(absl[near_best])])
    return int(lags[k]), float(ncc[k])


def _linear_gains(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return 1.0 - t, t


def _equal_power_gains(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.sqrt(1.0 - t), np.sqrt(t)


def _adaptive_gains(t: np.ndarray, correlation: float) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-power crossfade for two signals whose normalized correlation
    is ``correlation``: a spherical interpolation between them with
    ``theta = arccos(rho)``. ``rho = 1`` gives the linear curve, ``rho = 0``
    the sine/cosine equal-power curve. Anti-correlation is treated as 0."""
    rho = min(max(float(correlation), 0.0), 1.0)
    theta = math.acos(rho)
    if theta < _SLERP_EPS:
        return _linear_gains(t)
    sin_theta = math.sin(theta)
    return np.sin((1.0 - t) * theta) / sin_theta, np.sin(t * theta) / sin_theta


def _fade_gains(method: str, t: np.ndarray, correlation: float) -> Tuple[np.ndarray, np.ndarray]:
    if method == "linear":
        return _linear_gains(t)
    if method == "equalpower":
        return _equal_power_gains(t)
    return _adaptive_gains(t, correlation)


def _overlapping_parts(a: np.ndarray, b: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """The parts of ``a`` and ``b`` that line up when ``b[k]`` is paired with
    ``a[k + lag]`` (the convention of :func:`estimate_lag`)."""
    n = min(len(a), len(b))
    if lag >= 0:
        return a[lag:n], b[: n - lag]
    return a[: n + lag], b[-lag:n]


def _ncc(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Normalized cross-correlation of ``a`` and ``b`` at one shift."""
    x, y = _overlapping_parts(a, b, lag)
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y)) / denom if denom > 1e-9 else 0.0


def _level_ratio(tail: np.ndarray, head: np.ndarray, shift: int) -> float:
    """Gain that makes ``head`` as loud as ``tail`` over the stretch where the
    two carry the same audio (RMS ratio at ``shift``). Pass the interleaved
    samples and ``shift`` in samples (frames * channels)."""
    a, b = _overlapping_parts(tail, head, shift)
    energy_a = float(np.dot(a, a))
    energy_b = float(np.dot(b, b))
    if energy_a <= 0.0 or energy_b <= 0.0:
        return 1.0
    return math.sqrt(energy_a / energy_b)


def match_db(a: np.ndarray, b: np.ndarray) -> float:
    """How well two equally long stretches of the same audio agree:
    ``10·log10(power / power of the difference)`` in dB, where *power* is the
    mean of the two signals' powers. Identical audio gives 99 (the cap), two
    independent signals about −3."""
    diff = a - b
    d = float(np.dot(diff, diff))
    e = 0.5 * (float(np.dot(a, a)) + float(np.dot(b, b)))
    if e <= 0.0 or d <= e * 10.0 ** (-_MATCH_CAP_DB / 10.0):
        return _MATCH_CAP_DB
    return min(_MATCH_CAP_DB, 10.0 * math.log10(e / d))


def find_seam(a: np.ndarray, b: np.ndarray, channels: int, fade_frames: int) -> Tuple[int, int]:
    """Where in an overlap should a crossfade of ``fade_frames`` frames go?

    ``a`` and ``b`` are the interleaved, already aligned (and level matched)
    samples of the same overlap from the outgoing and the incoming chunk.
    Every position of the fade window is scored by the energy of
    ``a - b`` inside it relative to the signal energy inside it (with a floor
    so near-silence with tiny differences does not look bad), and the lowest
    score wins. The first and last 10 % of the overlap are never used: that
    is where each chunk is closest to its own edge, which separators handle
    worst. Positions within 5 % of the best score tie, and the one nearest
    the middle of the overlap wins (so identical chunks give a centered fade).

    Returns ``(start_frame, width_frames)`` of the fade window; the width is
    ``fade_frames`` limited to the length of the overlap."""
    frames = min(len(a), len(b)) // channels
    if frames <= 0:
        return 0, 0
    width = int(min(max(fade_frames, 1), frames))
    last = frames - width  # the last position a window can start at
    if last <= 0:
        return 0, width

    a2 = np.asarray(a[: frames * channels], dtype=np.float64).reshape(frames, channels)
    b2 = np.asarray(b[: frames * channels], dtype=np.float64).reshape(frames, channels)
    diff_power = ((a2 - b2) ** 2).sum(axis=1)
    signal_power = 0.5 * ((a2**2).sum(axis=1) + (b2**2).sum(axis=1))
    cum_diff = np.concatenate(([0.0], np.cumsum(diff_power)))
    cum_signal = np.concatenate(([0.0], np.cumsum(signal_power)))

    starts = np.arange(last + 1)
    diff_win = cum_diff[starts + width] - cum_diff[starts]
    signal_win = cum_signal[starts + width] - cum_signal[starts]
    floor = 10.0 ** (_SEAM_FLOOR_DB / 10.0) * (cum_signal[-1] / frames) * width
    score = diff_win / (signal_win + floor)

    guard = int(round(_SEAM_GUARD_FRAC * frames))
    lo, hi = guard, last - guard
    if hi < lo:  # overlap too short to keep clear of both edges
        lo = hi = last // 2
    candidates = score[lo : hi + 1]
    near_best = np.flatnonzero(candidates <= candidates.min() * _SEAM_TIE + 1e-12)
    k = int(near_best[np.argmin(np.abs(lo + near_best - last / 2.0))])
    return lo + k, width


class AnchorFit(NamedTuple):
    """How a processed chunk compares with the original chunk it was made from."""

    lag: int  # processed[k] ~ original[k + lag], in samples
    correlation: float  # normalized correlation of the two at that lag
    gain: float  # what to multiply the processed signal by to best reproduce the original
    recon_db: float  # 10*log10(power of the original / power of what is left), capped


def anchor_fit(
    original: np.ndarray, processed: np.ndarray, max_lag: int, window: Optional[int] = None
) -> AnchorFit:
    """Compare ``processed`` with the ``original`` it was made from (mono arrays).

    The offset is found as in :func:`estimate_lag` (``processed[k] ~ original[k + lag]``,
    within ``+-max_lag``), on the middle ``window`` samples when given: the offset of
    a chunk is one number, and a stretch of twenty seconds says as much as sixty while
    needing a fraction of the memory. The gain and the rest are then measured over the
    whole of the two, where they line up: ``gain`` is the least-squares amplitude that
    makes ``processed`` best reproduce ``original`` (so noise in ``processed`` that is
    not in the original does not pull it up), and ``recon_db`` is the power of the
    original over the power of what is left after that, i.e. how much of the original
    the processed signal accounts for. For a processed signal that *is* the original
    (up to a gain and an offset) that is the cap, 99 dB. Silence gives ``gain`` 1 and
    ``recon_db`` 0: there is nothing to measure."""
    n = min(len(original), len(processed))
    if window is not None and window > 0 and n > window:
        start = (n - window) // 2
        lag, _ = estimate_lag(
            original[start : start + window], processed[start : start + window], max_lag
        )
    else:
        lag, _ = estimate_lag(original, processed, max_lag)
    a, b = _overlapping_parts(
        np.asarray(original, dtype=np.float64), np.asarray(processed, dtype=np.float64), lag
    )
    aa, bb, ab = float(np.dot(a, a)), float(np.dot(b, b)), float(np.dot(a, b))
    if aa <= 0.0 or bb <= 0.0:
        return AnchorFit(lag, 0.0, 1.0, 0.0)
    gain = ab / bb
    rest = a - gain * b
    left = float(np.dot(rest, rest))
    recon = (
        _MATCH_CAP_DB
        if left <= aa * 10.0 ** (-_MATCH_CAP_DB / 10.0)
        else 10.0 * math.log10(aa / left)
    )
    return AnchorFit(lag, ab / math.sqrt(aa * bb), gain, min(_MATCH_CAP_DB, recon))
