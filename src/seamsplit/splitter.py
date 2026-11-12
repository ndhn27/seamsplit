"""
Split a long audio file into shorter, overlapping chunks at stable quiet
points, suitable for feeding one at a time into an AI vocal/instrumental
separation service that limits how long a single upload can be.

Why not just cut at the single quietest sample and concatenate later?

1. **Cut point selection.** A single instantaneous RMS minimum can land in
   a very short, unstable dip. Smoothing the loudness curve with a moving
   average before searching favors a *sustained* quiet passage instead,
   which gives the separation model more stable context on both sides of
   the cut.
2. **No overlap.** Without any shared audio between neighboring chunks
   there is nothing to crossfade later, so a naive concatenation (e.g.
   `ffmpeg -f concat`) is the only option — and that does not blend
   anything, so any mismatch between two independent AI inference passes
   is left audible as a click or seam at the join.

This module exports chunks *with* overlap on their shared edges so that
``seamsplit.merger`` can blend across that overlap with a real crossfade
after the chunks have been processed separately. See ``seamsplit.merger``
for the merge side of the workflow.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .audio import AudioSegment, duration_s

from .cutpoints import (
    CUT_BY,
    DEFAULT_BEAT_TOLERANCE_DB,
    DEFAULT_BEATS_PER_BAR,
    analyze_audio,
    beat_grid,
    beat_times_ms,
    distance_to_beat_ms,
    voice_amplitude,
)

# Frames of the RMS curve evaluated per block. Bounds the temporary memory
# (a block holds about ``_RMS_CHUNK_FRAMES * hop`` squared samples, ~15 MB at
# 44.1 kHz) so a multi-hour file does not need a squared copy of itself.
_RMS_CHUNK_FRAMES = 4096


def compute_rms_curve(
    audio: AudioSegment, window_ms: float = 30, hop_ms: float = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(times_ms, rms)``: the RMS loudness of ~``window_ms``
    slices, stepped every ``hop_ms``, across the whole track (downmixed to
    mono for analysis only — the original audio is untouched).

    Vectorized. The squared samples are summed once per hop-sized block
    (``reshape`` + ``sum``); a frame's window is then a short run of
    neighboring block sums, plus the few leftover samples when the window is
    not a whole number of hops (e.g. 22.05 kHz: 661 vs 220). Only positive
    terms are ever added — no running total that is subtracted from itself —
    so near-silent stretches, which are exactly what the cut-point search
    looks for, keep full precision next to loud ones."""
    mono = audio.set_channels(1)
    # integer samples, read in place; converted to float one block at a time
    samples = mono.samples
    sr = mono.frame_rate

    window = max(1, int(sr * window_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    n = len(samples)

    n_frames = max(0, (n - window) // hop + 1)
    times_ms = 1000.0 * (np.arange(n_frames) * hop) / sr
    rms_vals = np.empty(n_frames)

    whole, rem = divmod(window, hop)  # window = whole * hop + rem
    if whole == 0:
        # window shorter than the hop (hop_ms > window_ms): frames don't
        # touch, so there are no blocks to share — sum each window directly
        for k in range(n_frames):
            seg = samples[k * hop : k * hop + window].astype(np.float64)
            rms_vals[k] = np.sqrt(np.mean(seg * seg))
        return times_ms, rms_vals

    for k0 in range(0, n_frames, _RMS_CHUNK_FRAMES):
        k1 = min(n_frames, k0 + _RMS_CHUNK_FRAMES)
        frames = k1 - k0
        start = k0 * hop

        # whole hop-blocks 0 .. frames + whole - 2 of this stretch
        n_blocks = frames - 1 + whole
        sq = samples[start : start + n_blocks * hop].astype(np.float64) ** 2
        block_sums = sq.reshape(n_blocks, hop).sum(axis=1)
        window_sums = (
            block_sums if whole == 1 else np.convolve(block_sums, np.ones(whole), mode="valid")
        )

        if rem:
            tail_start = start + whole * hop
            tail_end = tail_start + (frames - 1) * hop + rem
            tail = samples[tail_start:tail_end].astype(np.float64) ** 2
            window_sums = window_sums + sliding_window_view(tail, rem)[::hop].sum(axis=1)

        rms_vals[k0:k1] = np.sqrt(window_sums / window)

    return times_ms, rms_vals


def smooth_curve(rms_values: np.ndarray, hop_ms: float, smooth_ms: float) -> np.ndarray:
    """Moving-average smoothing of the RMS curve, so the cut-point search
    favors a sustained quiet passage rather than a single noisy sample.
    ``smooth_ms <= 0`` disables smoothing and returns the input unchanged."""
    if smooth_ms <= 0:
        return rms_values
    win = max(1, int(round(smooth_ms / hop_ms)))
    if win <= 1:
        return rms_values
    kernel = np.ones(win) / win
    # 'valid' after edge-padding keeps the array length unchanged without
    # skewing the curve near the two ends.
    padded = np.pad(rms_values, (win // 2, win - 1 - win // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# A point this close to a bar line (ms) counts as being on it.
BAR_REACH_MS = 50.0


def find_quietest_point(
    times_ms: np.ndarray,
    rms_values: np.ndarray,
    window_start_ms: float,
    window_end_ms: float,
    beats_ms: Optional[np.ndarray] = None,
    tolerance_db: float = DEFAULT_BEAT_TOLERANCE_DB,
    bars_ms: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Return the timestamp (ms) of the lowest (smoothed) RMS value within
    ``[window_start_ms, window_end_ms]``, or ``None`` if the window
    contains no analysis samples.

    With ``beats_ms`` (beat times, ascending) the answer is, instead, the point
    nearest to a beat among those that are at most ``tolerance_db`` louder than the
    quietest tenth of the window: a rest is rarely a single instant, so some of its points
    sit on a beat, but a beat is not worth a cut in the middle of a loud passage.

    With ``bars_ms`` (the beats that start a bar) a point on a bar line, within
    ``BAR_REACH_MS`` of one, is preferred to any point that is only on a beat; when the
    quiet stretch has no bar line, the beat rule decides."""
    mask = (times_ms >= window_start_ms) & (times_ms <= window_end_ms)
    if not np.any(mask):
        return None
    sub_times = times_ms[mask]
    sub_rms = rms_values[mask]
    idx = int(np.argmin(sub_rms))
    has_beats = beats_ms is not None and len(beats_ms) > 0
    has_bars = bars_ms is not None and len(bars_ms) > 0
    if not has_beats and not has_bars:
        return float(sub_times[idx])
    # The level of the quiet to measure the tolerance from. The single lowest value of a
    # curve that is itself noisy (the voice-band one reaches exactly 0 in a few places
    # even where the singer is still singing) is a poor measure of it, so the lowest tenth of
    # the window counts as one level, and anything 60 dB under the window's median as silence.
    floor = max(
        float(sub_rms[idx]), float(np.percentile(sub_rms, 10)), float(np.median(sub_rms)) * 1e-3
    )
    limit = max(floor * 10.0 ** (tolerance_db / 20.0), floor + 1e-9)
    near = np.flatnonzero(sub_rms <= limit)
    if has_bars:
        from_bar = distance_to_beat_ms(sub_times[near], bars_ms)  # type: ignore[arg-type]
        on_bar = from_bar <= BAR_REACH_MS
        if on_bar.any():
            keep = near[on_bar]
            order = np.lexsort((keep, sub_rms[keep], from_bar[on_bar]))
            return float(sub_times[keep[order[0]]])
    if not has_beats:
        return float(sub_times[idx])
    away = distance_to_beat_ms(sub_times[near], beats_ms)  # type: ignore[arg-type]
    # nearest to a beat; the quieter one, then the earlier, on a tie
    order = np.lexsort((near, sub_rms[near], away))
    return float(sub_times[near[order[0]]])


# Formats whose files last exactly as long as the audio inside them.
LOSSLESS_FORMATS = frozenset({"wav", "flac", "aiff", "aif"})
# A lossy encoder pads the stream (an MP3 of exactly 59.000 s reads 59.037 s at
# 44.1 kHz and 59.063 s at 22.05 kHz, as older ffprobe versions and many decoders
# report it), so a length limit needs some room. 0.25 s is several times the padding
# seen at common rates.
DEFAULT_LOSSY_MARGIN_S = 0.25


# ffmpeg knows the container as "aiff"; "aif" is only a file extension.
_FORMAT_ALIASES = {"aif": "aiff"}


def normalize_export_format(export_format: str) -> str:
    """Canonical form of a format name: lower case, no leading dot (so
    ``".WAV"`` and ``"wav"`` give the same files), ``aif`` -> ``aiff``."""
    name = export_format.strip().lower().lstrip(".")
    if not name:
        raise ValueError("export format must not be empty")
    return _FORMAT_ALIASES.get(name, name)


def is_lossless(export_format: str) -> bool:
    """True for formats that store the samples exactly (wav, flac, aiff):
    no quality loss, no encoder delay or padding, and no bitrate."""
    return normalize_export_format(export_format) in LOSSLESS_FORMATS


def default_margin_s(export_format: str) -> float:
    """Room to leave under a service's length limit for an export format."""
    return 0.0 if is_lossless(export_format) else DEFAULT_LOSSY_MARGIN_S


def export_segment(
    segment: AudioSegment, path: str, export_format: str, bitrate: Optional[str] = None
) -> None:
    """Write ``segment`` to ``path``. ``bitrate`` only applies to lossy
    formats; for wav/flac/aiff it is not passed to ffmpeg at all (a lossless
    stream has no bitrate setting, and the option has no meaning there)."""
    fmt = normalize_export_format(export_format)
    if is_lossless(fmt) or not bitrate:
        segment.export(path, format=fmt)
    else:
        segment.export(path, format=fmt, bitrate=bitrate)


def probe_duration_s(path: str) -> Optional[float]:
    """Duration of an audio file as the container reports it (what ``ffprobe`` shows as
    ``format.duration``), which includes the padding a lossy encoder adds. ``None`` if it
    can't be read."""
    return duration_s(path)


def _check_geometry(
    target: float, before_slack: float, after_slack: float, min_final: float
) -> None:
    """Reject chunk-length settings that cannot produce a valid split
    (units don't matter as long as all four values share them).

    ``before_slack >= target`` would let the cut-point search return the
    current position again, so the split loop would never advance."""
    if not target > 0:
        raise ValueError("target must be greater than 0")
    if not before_slack >= 0 or not after_slack >= 0:
        raise ValueError("before-slack and after-slack must not be negative")
    if before_slack >= target:
        raise ValueError(
            "before-slack must be smaller than target (a chunk would be allowed to "
            "have zero length)"
        )
    if not min_final >= 0:
        raise ValueError("min-final must not be negative")
    if min_final > (target + after_slack) / 2:
        raise ValueError("min-final must be at most half of (target + after-slack)")


def choose_cut_points(
    times_ms: np.ndarray,
    rms_values: np.ndarray,
    total_len_ms: float,
    target_ms: float,
    before_slack_ms: float,
    after_slack_ms: float,
    min_final_ms: float,
    beats_ms: Optional[np.ndarray] = None,
    beat_tolerance_db: float = DEFAULT_BEAT_TOLERANCE_DB,
    bars_ms: Optional[np.ndarray] = None,
) -> List[float]:
    """Walk through the track choosing cut points inside
    ``[target - before_slack, target + after_slack]`` windows, at the
    quietest (smoothed) point available in each window. Falls back to a
    hard cut at ``target_ms`` if a window contains no analysis data.
    Always returns a list starting at ``0.0`` and ending at
    ``total_len_ms`` in which no segment is longer than
    ``target_ms + after_slack_ms``.

    If the last segment would be shorter than ``min_final_ms`` it is *not*
    merged into its neighbour (that always produced a chunk longer than the
    limit the caller asked for). Instead the last two segments are
    re-divided at the quietest point that keeps both within the limit and
    the final one at least ``min_final_ms`` long.

    With ``beats_ms`` each cut is, within ``beat_tolerance_db`` of the quietest
    point of its window, moved onto a beat, and with ``bars_ms`` onto a bar line where
    there is one in reach (see :func:`find_quietest_point`).

    Raises ``ValueError`` for settings that cannot produce a valid split."""
    _check_geometry(target_ms, before_slack_ms, after_slack_ms, min_final_ms)
    min_len_ms = target_ms - before_slack_ms
    max_len_ms = target_ms + after_slack_ms

    cuts = [0.0]
    pos = 0.0

    while total_len_ms - pos > max_len_ms:
        window_start = pos + min_len_ms
        window_end = pos + max_len_ms
        cut = find_quietest_point(
            times_ms, rms_values, window_start, window_end, beats_ms, beat_tolerance_db, bars_ms
        )
        if cut is None or cut <= pos:
            cut = pos + target_ms  # no analysis data in range (or no progress)
        cuts.append(cut)
        pos = cut

    cuts.append(float(total_len_ms))

    if len(cuts) > 2 and (cuts[-1] - cuts[-2]) < min_final_ms:
        prev_start = cuts[-3]
        combined = cuts[-1] - prev_start  # always > max_len_ms here
        lo = max(cuts[-1] - max_len_ms, prev_start + min_final_ms)
        hi = min(prev_start + max_len_ms, cuts[-1] - min_final_ms)
        new_cut = find_quietest_point(
            times_ms, rms_values, lo, hi, beats_ms, beat_tolerance_db, bars_ms
        )
        if new_cut is None:
            new_cut = prev_start + combined / 2
        cuts[-2] = new_cut

    return cuts


# The moving average applied to the curve before the quiet point is searched for. The
# voice-band curve is a noisier measurement (see seamsplit.cutpoints) and needs more of
# it: on synthetic mixes about a second finds a rest from half a second long up, where
# 250 ms stops finding it as soon as the wide instruments are as loud as the voice.
DEFAULT_SMOOTH_MS = {"rms": 250.0, "voice": 1000.0}


@dataclass
class SplitConfig:
    """Options for :func:`split_audio`. Mirrors the ``seamsplit split`` CLI
    flags; see docs/splitting.md for what each one does."""

    target_s: float = 52.0
    before_slack_s: float = 20.0
    after_slack_s: float = 3.0
    overlap_s: float = 4.0
    # None: 250 ms for the loudness curve, 1000 ms for the voice-band curve (see
    # DEFAULT_SMOOTH_MS); a number is used as it is for either
    smooth_ms: Optional[float] = None
    fade_ms: float = 8.0
    min_final_s: float = 5.0
    output_dir: str = "chunks"
    export_format: str = (
        "flac"  # lossless; "mp3" only if a service needs it (see docs/splitting.md)
    )
    bitrate: str = "320k"  # lossy formats only; ignored for wav / flac / aiff
    # How the quiet point of each window is judged: "rms" is the loudness of the whole
    # signal; "voice" the activity in the voice band (see seamsplit.cutpoints).
    cut_by: str = "rms"
    snap_beat: bool = False  # among nearly-as-quiet points, cut on a beat
    beat_tolerance_db: float = DEFAULT_BEAT_TOLERANCE_DB
    # ... and, when the track shows where its bars start, on a bar line (implies the beat
    # tracking; ``bar_beats`` is how many beats make a bar)
    snap_bar: bool = False
    bar_beats: int = DEFAULT_BEATS_PER_BAR

    def __post_init__(self) -> None:
        self.export_format = normalize_export_format(self.export_format)
        if self.cut_by not in CUT_BY:
            raise ValueError(f"cut_by must be one of: {', '.join(CUT_BY)}")
        if not self.beat_tolerance_db >= 0:
            raise ValueError("beat tolerance must not be negative")
        if isinstance(self.bar_beats, bool) or not isinstance(self.bar_beats, int):
            raise ValueError("bar_beats must be a whole number of beats")
        if not 2 <= self.bar_beats <= 16:
            raise ValueError("a bar has between 2 and 16 beats")
        _check_geometry(self.target_s, self.before_slack_s, self.after_slack_s, self.min_final_s)
        if not self.overlap_s >= 0:
            raise ValueError("overlap must not be negative")
        if self.overlap_s >= self.target_s - self.before_slack_s:
            raise ValueError(
                "overlap must be shorter than the shortest allowed chunk " "(target - before-slack)"
            )
        if (self.smooth_ms is not None and not self.smooth_ms >= 0) or not self.fade_ms >= 0:
            raise ValueError("smooth-ms and fade-ms must not be negative")

    @property
    def effective_smooth_ms(self) -> float:
        return DEFAULT_SMOOTH_MS[self.cut_by] if self.smooth_ms is None else self.smooth_ms


@dataclass
class CutPlan:
    """Where a track is cut and how that was decided."""

    cuts_ms: List[int]  # 0, every cut, the length of the track
    cut_by: str
    snap_beat: bool
    tempo_bpm: Optional[float] = None  # None: not asked for, or no beat to follow
    beat_offsets_ms: Optional[List[int]] = None  # of each cut but the ends, from the nearest beat
    snap_bar: bool = False
    # how clearly the track marks its bars (see cutpoints.bar_phase); None: not looked for,
    # or too few beats; the bar lines were used only when it reaches MIN_BAR_CONFIDENCE
    bar_confidence: Optional[float] = None
    bar_offsets_ms: Optional[List[int]] = (
        None  # of each cut but the ends, from the nearest bar line
    )


def plan_cuts(audio: AudioSegment, config: SplitConfig) -> CutPlan:
    """Choose the cut points of ``audio`` as ``config`` asks: by the loudness of
    the whole signal, or by the activity in the voice band, and, with
    ``config.snap_beat``, on a beat among the points nearly as quiet, or, with
    ``config.snap_bar``, on a bar line where the track shows one."""
    beats: Optional[np.ndarray] = None
    bars: Optional[np.ndarray] = None
    tempo: Optional[float] = None
    confidence: Optional[float] = None
    if config.cut_by == "rms" and not (config.snap_beat or config.snap_bar):
        times_ms, curve = compute_rms_curve(audio)
    else:
        features = analyze_audio(audio)
        times_ms = features.times_ms
        curve = features.voice_power if config.cut_by == "voice" else compute_rms_curve(audio)[1]
        if config.snap_bar:
            grid = beat_grid(features, config.bar_beats)
            beats, tempo, confidence = grid.beats_ms, grid.tempo_bpm, grid.bar_confidence
            bars = grid.bars_ms if len(grid.bars_ms) else None
        elif config.snap_beat:
            beats, tempo = beat_times_ms(features)
    smoothed = smooth_curve(curve, hop_ms=10, smooth_ms=config.effective_smooth_ms)
    if config.cut_by == "voice":
        smoothed = voice_amplitude(smoothed)  # average the signed power first, then cut it at zero

    cuts = choose_cut_points(
        times_ms,
        smoothed,
        len(audio),
        target_ms=config.target_s * 1000,
        before_slack_ms=config.before_slack_s * 1000,
        after_slack_ms=config.after_slack_s * 1000,
        min_final_ms=config.min_final_s * 1000,
        beats_ms=beats,
        beat_tolerance_db=config.beat_tolerance_db,
        bars_ms=bars,
    )
    cuts_ms = [int(round(c)) for c in cuts]
    inner = np.array(cuts_ms[1:-1])
    offsets = bar_offsets = None
    if beats is not None and len(beats):
        offsets = [int(round(d)) for d in distance_to_beat_ms(inner, beats)]
    if bars is not None and len(bars):
        bar_offsets = [int(round(d)) for d in distance_to_beat_ms(inner, bars)]
    return CutPlan(
        cuts_ms,
        config.cut_by,
        config.snap_beat,
        tempo,
        offsets,
        config.snap_bar,
        confidence,
        bar_offsets,
    )


def split_audio(
    input_file: str,
    config: SplitConfig,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Split ``input_file`` into overlapping chunks according to
    ``config``, write them (plus ``manifest.json``) into
    ``config.output_dir``, and return the manifest dict.

    The manifest (returned and written to disk) contains ``source_file``,
    ``source_length_ms``, ``overlap_ms``, ``parts`` (an ordered list of
    chunk filenames) and ``max_export_ms`` (the longest exported chunk —
    compare it with your AI service's limit), plus a ``segments`` breakdown
    with each chunk's exact ``start_ms`` / ``end_ms`` in the source.

    With ``dry_run=True`` the audio is analysed and the plan is returned,
    but nothing is written. ``progress(i, n, name)`` is called after each
    chunk is exported (or planned, in a dry run).
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Whole milliseconds everywhere: chunk i's tail and chunk i+1's head must
    # share *exactly* ``overlap_ms`` of audio, because the merge step
    # crossfades a fixed-length region.
    overlap_ms = int(round(max(0.0, config.overlap_s * 1000)))
    lead_ms = overlap_ms // 2  # taken from the end of the previous chunk
    tail_ms = overlap_ms - lead_ms

    audio = AudioSegment.from_file(input_file)
    total_len_ms = len(audio)

    plan = plan_cuts(audio, config)
    cuts_ms = plan.cuts_ms

    if not dry_run:
        os.makedirs(config.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_file))[0]

    manifest_parts: List[str] = []
    segments: List[dict] = []
    n = len(cuts_ms) - 1
    max_export_len_ms = 0

    for i in range(n):
        seg_start_ms, seg_end_ms = cuts_ms[i], cuts_ms[i + 1]

        # Extend each chunk into its neighbors (lead_ms + tail_ms == overlap_ms),
        # except at the true start/end of the whole track.
        export_start = max(0, seg_start_ms - (lead_ms if i > 0 else 0))
        export_end = min(total_len_ms, seg_end_ms + (tail_ms if i < n - 1 else 0))

        out_name = f"{base}_part{i + 1:02d}.{config.export_format}"

        if not dry_run:
            chunk = audio[export_start:export_end]

            # Only fade at the true start/end of the whole track — internal
            # chunk edges are left raw since they will be crossfaded on merge.
            fade_start = (i == 0) and config.fade_ms > 0
            fade_end = (i == n - 1) and config.fade_ms > 0
            if fade_start or fade_end:
                fade = min(config.fade_ms, len(chunk) / 2)
                if fade_start:
                    chunk = chunk.fade_in(int(fade))
                if fade_end:
                    chunk = chunk.fade_out(int(fade))

            out_path = os.path.join(config.output_dir, out_name)
            export_segment(chunk, out_path, config.export_format, config.bitrate)

        max_export_len_ms = max(max_export_len_ms, export_end - export_start)
        manifest_parts.append(out_name)
        segments.append(
            {
                "name": out_name,
                "start_ms": export_start,
                "end_ms": export_end,
                "start_s": round(export_start / 1000, 2),
                "end_s": round(export_end / 1000, 2),
                "duration_s": round((export_end - export_start) / 1000, 2),
            }
        )
        if progress is not None:
            progress(i + 1, n, out_name)

    manifest = {
        "source_file": os.path.basename(input_file),
        "source_length_ms": total_len_ms,
        "overlap_ms": overlap_ms,
        "parts": manifest_parts,
        "segments": segments,
        "max_export_ms": max_export_len_ms,
        "cut_by": plan.cut_by,
        "snap_beat": plan.snap_beat,
        "tempo_bpm": None if plan.tempo_bpm is None else round(plan.tempo_bpm, 1),
        "cut_beat_offsets_ms": plan.beat_offsets_ms,
        "snap_bar": plan.snap_bar,
        "bar_beats": config.bar_beats,
        "downbeat_confidence": (
            None if plan.bar_confidence is None else round(plan.bar_confidence, 1)
        ),
        "cut_bar_offsets_ms": plan.bar_offsets_ms,
    }
    if not dry_run:
        manifest_path = os.path.join(config.output_dir, "manifest.json")
        # all at once: the manifest is what says the split is complete, so a kill while
        # it is being written must not leave half of one
        with open(manifest_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(manifest_path + ".tmp", manifest_path)
    return manifest
