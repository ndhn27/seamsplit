"""
What the cut-point search can know about a track besides how loud it is.

The default search (see :mod:`seamsplit.splitter`) looks for the quietest stretch
of the *whole* signal. In a mix that is often the wrong place to cut: a bass
line or a kick drum sets the overall level, so the quietest moment can fall in
the middle of a sung phrase, and nothing says it is anywhere near the beat.
This module gives the search two more things to go on, both measured from the
audio with numpy alone (no model, no extra dependency):

**Voice-band activity** (:func:`analyze_audio`, ``voice``). The energy between
300 Hz and 3.4 kHz, where the fundamentals and formants of a singing voice sit,
instead of the energy of everything. For a stereo track only the part that is the
same in both channels counts (the real part of the cross-spectrum), because lead
vocals are usually panned to the centre while guitars, keyboards and reverb are
spread wider; for a mono track it is the plain band energy.

That common part is an average with a sign: for two unrelated signals it is
zero *on average* but positive or negative from one frame to the next. It is
therefore kept as a signed power (``voice_power``), to be averaged over time
first and only then cut off at zero and turned into an amplitude
(:func:`voice_amplitude`); cutting it off frame by frame would turn every negative
fluctuation into a false silence.

**Beats** (:func:`track_beats`). The onset strength (spectral flux) gives the
tempo through its autocorrelation, and dynamic programming then lays beats along
it that follow the onsets while keeping a steady spacing, so they carry on
through a rest and follow a slow drift of tempo. (This is the approach of
Ellis, "Beat Tracking by Dynamic Programming", 2007.)

Neither is a detector. The voice-band curve cannot tell a voice from a guitar
that plays in the same range and the same place in the stereo image, and a
track that is not rhythmic, or whose tempo changes quickly, gets few reliable
beats. They are better evidence than overall loudness, not proof; ``seamsplit
split --dry-run`` shows where the cuts would go, and the manifest records how far
each one is from a beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .audio import AudioSegment

CUT_BY = ("rms", "voice")
VOICE_BAND_HZ = (300.0, 3400.0)
# Onsets are looked for between these frequencies, in this many bands spaced evenly
# on a log scale (so the low end, where kick drums are, counts as much as the top).
_FLUX_BAND_HZ = (30.0, 8000.0)
_FLUX_BANDS = 24
# The bands whose upper edge is at or below this make the "low" onset curve: the kick drum
# and the bass, whose pattern says where a bar starts.
LOW_BAND_HZ = 150.0
# A tempo is only believed when the onset strength repeats at that period this much
# (its autocorrelation there, relative to lag 0): unrelated noise stays under 0.06,
# drum loops and click tracks reach 0.2 to 1, a weak but real beat under a loud mix 0.2.
MIN_BEAT_STRENGTH = 0.1
# A steady tone has no onsets at all, but the way its frames fall against the tone makes
# the little there is repeat very regularly. Music has a mean onset strength of 0.06 or
# more on this scale, a pure tone about 0.01.
MIN_ONSET_LEVEL = 0.03
# Frames analysed per block; bounds the size of the spectra held at once.
_BLOCK_FRAMES = 512
# Magnitudes are compared on a log scale with this floor (-80 dB of full scale).
_LOG_FLOOR = 1e-4

DEFAULT_BEAT_TOLERANCE_DB = 6.0
DEFAULT_BEATS_PER_BAR = 4
# A bar line is believed when the beats that would start the bars are stronger than the rest
# by this many standard errors (see :func:`bar_phase`). Random beat strengths reach it in
# under 1 % of simulated tracks (best of four phases, with or without slow swells).
MIN_BAR_CONFIDENCE = 3.0
# Fewer bars than this say too little about which beat of the bar is the strong one.
_MIN_BARS = 6
BPM_RANGE = (60.0, 200.0)
_PRIOR_BPM = 120.0
# How hard the beat tracker holds to a steady spacing; see track_beats.
_TIGHTNESS = 100.0
# Fewer seconds of audio than this has no tempo worth estimating.
_MIN_TRACK_S = 6.0


@dataclass
class CutFeatures:
    """Per-frame measurements on the same frames as ``compute_rms_curve``: frame
    ``k`` covers ``window_ms`` of audio starting at ``times_ms[k]``."""

    times_ms: np.ndarray
    # mean square of the voice-band component, in sample units squared (RMS squared);
    # signed for stereo, see the module docstring
    voice_power: np.ndarray
    onset: np.ndarray  # spectral flux
    window_ms: float
    hop_ms: float
    # the rise in level of the bands below ``LOW_BAND_HZ`` (kick drum, bass notes), on a
    # linear scale: the broadband ``onset`` is logarithmic, so a kick twice as loud adds
    # little to it, and how hard a beat is struck is what tells the bar's first beat
    low_onset: Optional[np.ndarray] = None


def analyze_audio(
    audio: AudioSegment,
    window_ms: float = 30,
    hop_ms: float = 10,
    band_hz: Tuple[float, float] = VOICE_BAND_HZ,
) -> CutFeatures:
    """Measure the voice-band activity and the onset strength of ``audio`` on the
    frames :func:`seamsplit.splitter.compute_rms_curve` uses (same window, hop and
    times), so the curves can be swapped for one another."""
    sr = audio.frame_rate
    channels = audio.channels
    window = max(1, int(sr * window_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    pcm = audio.samples
    n = len(pcm) // channels
    n_frames = max(0, (n - window) // hop + 1)
    times_ms = 1000.0 * (np.arange(n_frames) * hop) / sr
    voice_power = np.zeros(n_frames)
    onset = np.zeros(n_frames)
    low_onset = np.zeros(n_frames)
    if n_frames == 0:
        return CutFeatures(times_ms, voice_power, onset, window_ms, hop_ms, low_onset)

    full_scale = float(2 ** (8 * audio.sample_width - 1))
    samples = pcm[: n * channels].reshape(n, channels)
    n_fft = 1 << (window - 1).bit_length()
    taper = np.hanning(window)
    energy = float((taper**2).sum())
    amplitude = 0.5 * float(taper.sum())  # the peak of a sine of amplitude 1 in the spectrum
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    in_voice = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    flux_bins = np.flatnonzero((freqs >= _FLUX_BAND_HZ[0]) & (freqs <= _FLUX_BAND_HZ[1]))
    edges = np.unique(
        np.round(np.geomspace(flux_bins[0], flux_bins[-1] + 1, _FLUX_BANDS + 1)).astype(int)
    )
    edges = (
        np.concatenate([edges[:-1], [flux_bins[-1] + 1]]) if edges[-1] <= flux_bins[-1] else edges
    )

    low_bands = np.flatnonzero(freqs[edges[1:] - 1] <= LOW_BAND_HZ)  # the bands of the kick
    previous: Optional[np.ndarray] = None  # last frame's log spectrum, for the flux
    previous_low = 0.0  # last frame's level in the low bands
    for k0 in range(0, n_frames, _BLOCK_FRAMES):
        k1 = min(n_frames, k0 + _BLOCK_FRAMES)
        span = samples[k0 * hop : (k1 - 1) * hop + window].astype(np.float64) / full_scale

        def spectra(channel: np.ndarray) -> np.ndarray:
            frames = sliding_window_view(channel, window)[::hop] * taper
            return np.fft.rfft(frames, n=n_fft, axis=1)

        left = spectra(span[:, 0])
        if channels >= 2:
            right = spectra(span[:, 1])
            # what the two channels have in common; what is out of phase or
            # hard-panned to one side adds nothing
            common = (left[:, in_voice] * np.conj(right[:, in_voice])).real.sum(axis=1)
            power = common
            mixed = 0.5 * (left + right)
        else:
            power = (np.abs(left[:, in_voice]) ** 2).sum(axis=1)
            mixed = left
        voice_power[k0:k1] = 2.0 * power / (n_fft * energy) * full_scale**2

        magnitude = np.abs(mixed) / amplitude
        bands = np.stack(
            [np.sqrt((magnitude[:, a:b] ** 2).mean(axis=1)) for a, b in zip(edges[:-1], edges[1:])],
            axis=1,
        )
        log_spectrum = np.log1p(bands / _LOG_FLOOR)
        stacked = log_spectrum if previous is None else np.vstack([previous, log_spectrum])
        first = k0 + (1 if previous is None else 0)
        onset[first:k1] = np.maximum(np.diff(stacked, axis=0), 0.0).mean(axis=1)
        if len(low_bands):
            low_level = np.sqrt((bands[:, low_bands] ** 2).sum(axis=1))
            rise = np.maximum(np.diff(np.concatenate([[previous_low], low_level])), 0.0)
            low_onset[k0:k1] = rise
            previous_low = float(low_level[-1])
            if previous is None:
                low_onset[0] = 0.0  # no frame before the first
        previous = log_spectrum[-1:]

    return CutFeatures(times_ms, voice_power, onset, window_ms, hop_ms, low_onset)


def voice_amplitude(voice_power: np.ndarray) -> np.ndarray:
    """The voice-band RMS from the (smoothed) signed power: none where the power is
    not above zero."""
    return np.sqrt(np.maximum(voice_power, 0.0))


def _tempo_period(onset: np.ndarray, fps: float) -> Optional[Tuple[float, float]]:
    """``(period, strength)``: the beat period, in frames, whose multiples line up best
    with the onsets, and how strongly the onsets repeat at it (0 to 1). The period is the
    peak of the autocorrelation of the onset strength, weighted towards 120 BPM (a tempo
    is more likely near the middle of the range than at its ends)."""
    n = len(onset)
    lag_min = int(np.ceil(fps * 60.0 / BPM_RANGE[1]))
    lag_max = int(np.floor(fps * 60.0 / BPM_RANGE[0]))
    if lag_max + 2 >= n // 2 or lag_min < 2:
        return None
    # slow swells and fades are not rhythm: take the local average (about a second) away
    width = max(3, int(round(fps)) | 1)
    centred = onset - np.convolve(onset, np.ones(width) / width, mode="same")
    spectrum = np.fft.rfft(centred, 2 * n)
    auto = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    if auto[0] <= 0:
        return None
    auto = auto / np.arange(n, 0, -1)  # fewer products at long lags: undo the bias
    lags = np.arange(lag_min - 1, lag_max + 2)
    bpm = fps * 60.0 / lags[1:-1]
    score = auto[lags[1:-1]] * np.exp(-0.5 * np.log2(bpm / _PRIOR_BPM) ** 2)
    best = int(np.argmax(score))
    if score[best] <= 0:
        return None
    period = float(lags[1:-1][best])
    below, here, above = auto[lags[best]], auto[lags[best + 1]], auto[lags[best + 2]]
    curvature = below - 2.0 * here + above
    if curvature < 0:  # a peak: put the period between the lags, where the parabola peaks
        period += 0.5 * (below - above) / curvature
    return period, float(here / auto[0])


def track_beats(onset: np.ndarray, fps: float) -> Tuple[np.ndarray, Optional[float]]:
    """Beat positions as frame indices, and the tempo in BPM, from an onset
    strength curve sampled ``fps`` times a second. ``([], None)`` when there is too
    little audio or no rhythm to follow (the onsets do not repeat at any tempo more
    than noise would).

    Every frame gets the best score of a chain of beats ending there: its own onset
    strength plus the best predecessor between half and twice a beat earlier, less
    ``_TIGHTNESS * log(gap / period)**2`` for straying from the estimated period.
    Following the best chain back from its end gives the beats. The penalty is what
    lets the chain carry on through a rest (no onsets, but the spacing is kept) and
    what keeps it from locking onto every strong onset."""
    onset = np.asarray(onset, dtype=np.float64)
    n = len(onset)
    if n < fps * _MIN_TRACK_S or onset.std() <= 1e-12 or onset.mean() < MIN_ONSET_LEVEL:
        return np.empty(0, dtype=int), None
    scaled = onset / onset.std()
    found = _tempo_period(scaled, fps)
    if found is None or found[1] < MIN_BEAT_STRENGTH:
        return np.empty(0, dtype=int), None
    period = found[0]

    sigma = max(1.0, period / 32.0)
    half = int(3 * sigma)
    bump = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    local = np.convolve(scaled, bump, mode="same")

    lo = max(1, int(round(period / 2)))
    hi = int(round(period * 2))
    gaps = np.arange(lo, hi + 1)
    # indexed like a slice of the score array, oldest predecessor first
    penalty = (-_TIGHTNESS * np.log(gaps / period) ** 2)[::-1]
    best = local.copy()
    link = np.full(n, -1)
    first = True  # a chain starts at the first onset of any weight
    floor = 0.01 * local.max()
    for i in range(lo, n):
        start = max(0, i - hi)
        candidates = best[start : i - lo + 1] + penalty[len(penalty) - (i - lo + 1 - start) :]
        j = int(np.argmax(candidates))
        best[i] = local[i] + candidates[j]
        if first and local[i] < floor:
            continue
        link[i] = start + j
        first = False

    peaks = np.flatnonzero((best[1:-1] > best[:-2]) & (best[1:-1] >= best[2:])) + 1
    if len(peaks) == 0:
        return np.empty(0, dtype=int), None
    good = peaks[best[peaks] >= 0.5 * np.median(best[peaks])]
    tail = int(good[-1])
    beats: List[int] = [tail]
    while link[beats[-1]] >= 0:
        beats.append(int(link[beats[-1]]))
    frames = np.array(beats[::-1], dtype=int)
    if len(frames) < 3:
        return np.empty(0, dtype=int), None
    tempo = 60.0 * fps / float(np.median(np.diff(frames)))
    return frames, tempo


def beat_times_ms(features: CutFeatures) -> Tuple[np.ndarray, Optional[float]]:
    """Beat times in milliseconds (the middle of the frame each beat falls in, which is
    where a transient shows up in the onset curve) and the tempo in BPM."""
    fps = 1000.0 / features.hop_ms
    frames, tempo = track_beats(features.onset, fps)
    if len(frames) == 0:
        return np.empty(0), None
    return features.times_ms[frames] + features.window_ms / 2.0, tempo


@dataclass
class BeatGrid:
    """The beats of a track and, when the track says so clearly enough, where its bars
    start. ``bars_ms`` is empty when no bar line could be told (``bar_confidence`` is
    then the evidence found, or ``None`` when there were too few beats to look)."""

    beats_ms: np.ndarray
    tempo_bpm: Optional[float]
    bars_ms: np.ndarray
    bar_confidence: Optional[float]
    beats_per_bar: int


def bar_phase(
    frames: np.ndarray, features: CutFeatures, beats_per_bar: int = DEFAULT_BEATS_PER_BAR
) -> Optional[Tuple[int, float]]:
    """Which of every ``beats_per_bar`` beats starts a bar, and how sure that is.

    The first beat of a bar is usually the one that is struck hardest: the kick drum lands
    on it, the bass plays a new note, the chord changes. So the strength of the onset at
    each beat (broadband and in the low bands alone, the highest of the frames within 30
    ms) is standardized, the slow swells of the arrangement (what the neighbouring bars
    do) are taken away, and the phase whose beats are strongest is the bar line. The
    confidence is how many standard errors that phase stands above the others. In a simulation
    with random beat strengths, chance reaches 3 (``MIN_BAR_CONFIDENCE``) in under 1 % of
    tracks, an accent of one standard deviation on the first beat passes it from about 20 bars
    on, and one of half a standard deviation in about two thirds of tracks of 40 bars. (Real
    music is not random beat strengths: how large its accents are is for the dry run to
    show.) ``None`` when there are fewer than six bars of beats or the onsets are flat.

    It is a heuristic with known blind spots. A drum pattern that strikes beats 1 and 3
    alike cannot tell them apart (either can come out as the bar line, which is still a
    half-bar line); a track whose first beat of the bar is not the strongest one has the
    wrong phase, with confidence to match; and a beat tracker that skips a beat shifts every
    later bar line by one (they are then still beats)."""
    b = int(beats_per_bar)
    n = len(frames)
    if b < 2 or n < _MIN_BARS * b or features.low_onset is None:
        return None
    reach = int(round(30.0 / features.hop_ms))
    last = len(features.onset)

    def at_beats(curve: np.ndarray) -> np.ndarray:
        return np.array([curve[max(0, f - reach) : min(last, f + reach + 1)].max() for f in frames])

    def standardize(x: np.ndarray) -> Optional[np.ndarray]:
        spread = float(x.std())
        return None if spread <= 1e-12 else (x - x.mean()) / spread

    parts = [standardize(at_beats(features.onset)), standardize(at_beats(features.low_onset))]
    usable = [x for x in parts if x is not None]
    if not usable:
        return None
    accent = np.sum(usable, axis=0)
    width = np.ones(2 * b + 1)
    trend = np.convolve(accent, width, mode="same") / np.convolve(np.ones(n), width, mode="same")
    flat = standardize(accent - trend)
    if flat is None:
        return None
    means = np.array([flat[p::b].mean() for p in range(b)])
    best = int(np.argmax(means))
    in_best = len(flat[best::b])
    others = n - in_best
    rest = (flat.sum() - flat[best::b].sum()) / others
    error = np.sqrt(1.0 / in_best + 1.0 / others)  # of a difference of means, with unit spread
    return best, float((means[best] - rest) / error)


def beat_grid(features: CutFeatures, beats_per_bar: int = DEFAULT_BEATS_PER_BAR) -> BeatGrid:
    """The beats of ``features`` and, when :func:`bar_phase` is sure of it, the bar lines."""
    fps = 1000.0 / features.hop_ms
    frames, tempo = track_beats(features.onset, fps)
    if len(frames) == 0:
        return BeatGrid(np.empty(0), None, np.empty(0), None, beats_per_bar)
    beats = features.times_ms[frames] + features.window_ms / 2.0
    found = bar_phase(frames, features, beats_per_bar)
    if found is None:
        return BeatGrid(beats, tempo, np.empty(0), None, beats_per_bar)
    phase, confidence = found
    bars = beats[phase::beats_per_bar] if confidence >= MIN_BAR_CONFIDENCE else np.empty(0)
    return BeatGrid(beats, tempo, bars, confidence, beats_per_bar)


def distance_to_beat_ms(times_ms: np.ndarray, beats_ms: np.ndarray) -> np.ndarray:
    """For each time, how far it is from the nearest beat (milliseconds)."""
    times_ms = np.asarray(times_ms, dtype=np.float64)
    if len(beats_ms) == 0:
        return np.full(times_ms.shape, np.inf)
    right = np.searchsorted(beats_ms, times_ms)
    left_beat = beats_ms[np.clip(right - 1, 0, len(beats_ms) - 1)]
    right_beat = beats_ms[np.clip(right, 0, len(beats_ms) - 1)]
    return np.minimum(np.abs(times_ms - left_beat), np.abs(right_beat - times_ms))
