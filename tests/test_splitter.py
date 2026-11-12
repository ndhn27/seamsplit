import numpy as np
import pytest
from seamsplit.audio import AudioSegment

from seamsplit import splitter
from seamsplit.splitter import (
    DEFAULT_LOSSY_MARGIN_S,
    SplitConfig,
    choose_cut_points,
    compute_rms_curve,
    default_margin_s,
    export_segment,
    find_quietest_point,
    is_lossless,
    normalize_export_format,
    smooth_curve,
)


def test_the_default_export_format_is_lossless():
    from seamsplit.splitopts import DEFAULT_FORMAT

    assert DEFAULT_FORMAT == "flac" and is_lossless(DEFAULT_FORMAT)
    assert SplitConfig().export_format == DEFAULT_FORMAT
    assert default_margin_s(SplitConfig().export_format) == 0.0


def test_smooth_curve_reduces_spread():
    rms = np.array([1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0], dtype=float)
    smoothed = smooth_curve(rms, hop_ms=10, smooth_ms=50)
    assert len(smoothed) == len(rms)
    assert smoothed.max() - smoothed.min() < rms.max() - rms.min()


def test_smooth_curve_noop_when_disabled():
    rms = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(smooth_curve(rms, hop_ms=10, smooth_ms=0), rms)


def test_find_quietest_point_picks_the_minimum():
    times = np.array([0, 100, 200, 300, 400], dtype=float)
    rms = np.array([5.0, 1.0, 4.0, 0.5, 3.0])
    assert find_quietest_point(times, rms, 100, 400) == 300


def test_find_quietest_point_empty_window_returns_none():
    times = np.array([0, 100, 200], dtype=float)
    rms = np.array([5.0, 1.0, 4.0])
    assert find_quietest_point(times, rms, 1000, 2000) is None


def test_choose_cut_points_lands_near_planted_quiet_points():
    times = np.arange(0, 200_000, 10, dtype=float)
    rms = np.ones_like(times)
    quiet_points = [50_000, 100_000, 150_000]
    for qp in quiet_points:
        idx = int(np.searchsorted(times, qp))
        rms[max(0, idx - 5) : idx + 5] = 0.01

    cuts = choose_cut_points(
        times,
        rms,
        total_len_ms=200_000,
        target_ms=50_000,
        before_slack_ms=10_000,
        after_slack_ms=10_000,
        min_final_ms=5_000,
    )

    assert cuts[0] == 0.0
    assert cuts[-1] == 200_000.0
    for cut in cuts[1:-1]:
        assert min(abs(cut - qp) for qp in quiet_points) < 300


def test_choose_cut_points_rebalances_short_final_segment():
    times = np.arange(0, 59_000, 10, dtype=float)
    rms = np.ones_like(times)
    idx = int(np.searchsorted(times, 45_000))
    rms[max(0, idx - 5) : idx + 5] = 0.01

    cuts = choose_cut_points(
        times,
        rms,
        total_len_ms=59_000,
        target_ms=50_000,
        before_slack_ms=10_000,
        after_slack_ms=5_000,
        min_final_ms=20_000,
    )

    segments = np.diff(cuts)
    assert cuts[0] == 0.0 and cuts[-1] == 59_000.0
    # a ~14s tail is too short, but folding it into the previous chunk would
    # give 59s > target + after_slack (55s): it must be re-divided instead
    assert len(cuts) == 3
    assert segments.max() <= 55_000
    assert segments[-1] >= 20_000


def test_choose_cut_points_never_exceeds_max_chunk_length():
    times = np.arange(0, 114_900, 10, dtype=float)
    rms = -times  # quietest point is always at the far edge of the window

    cuts = choose_cut_points(
        times,
        rms,
        total_len_ms=114_900,
        target_ms=52_000,
        before_slack_ms=20_000,
        after_slack_ms=3_000,
        min_final_ms=5_000,
    )

    segments = np.diff(cuts)
    assert segments.max() <= 55_000  # stays clear of the 60 s limit
    assert segments[-1] >= 5_000


def test_choose_cut_points_short_final_without_analysis_data_splits_in_half():
    cuts = choose_cut_points(
        np.array([], dtype=float),
        np.array([], dtype=float),
        total_len_ms=114_900,
        target_ms=52_000,
        before_slack_ms=20_000,
        after_slack_ms=3_000,
        min_final_ms=5_000,
    )
    assert np.diff(cuts).max() <= 55_000


@pytest.mark.parametrize(
    "target, before, after, min_final",
    [
        (50_000, 50_000, 3_000, 5_000),  # before_slack == target: no chunk could have a length
        (50_000, 60_000, 3_000, 5_000),
        (0, 0, 3_000, 5_000),  # target 0: no chunk could have a length
        (50_000, -1, 3_000, 5_000),
        (50_000, 20_000, -1, 5_000),
        (50_000, 20_000, 3_000, 40_000),  # min_final larger than half a chunk
    ],
)
def test_choose_cut_points_rejects_impossible_settings(target, before, after, min_final):
    times = np.arange(0, 200_000, 10, dtype=float)
    with pytest.raises(ValueError):
        choose_cut_points(times, np.ones_like(times), 200_000, target, before, after, min_final)


def test_split_config_defaults_are_valid():
    SplitConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"before_slack_s": 52.0},
        {"before_slack_s": 60.0},
        {"target_s": 0},
        {"after_slack_s": -1},
        {"overlap_s": -1},
        {"overlap_s": 40.0},  # longer than the shortest allowed chunk (32s)
        {"smooth_ms": -5},
        {"fade_ms": -5},
    ],
)
def test_split_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        SplitConfig(**kwargs)


def test_choose_cut_points_falls_back_to_hard_cut_without_analysis_data():
    times = np.array([], dtype=float)
    rms = np.array([], dtype=float)

    cuts = choose_cut_points(
        times,
        rms,
        total_len_ms=150_000,
        target_ms=50_000,
        before_slack_ms=10_000,
        after_slack_ms=10_000,
        min_final_ms=5_000,
    )

    assert cuts == [0.0, 50_000.0, 100_000.0, 150_000.0]


# --- vectorized RMS curve ----------------------------------------------------


def _reference_rms_curve(audio, window_ms=30, hop_ms=10):
    """The original frame-by-frame loop, kept here as the ground truth."""
    mono = audio.set_channels(1)
    samples = np.array(mono.get_array_of_samples()).astype(np.float64)
    sr = mono.frame_rate
    window = max(1, int(sr * window_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    n_frames = max(0, (len(samples) - window) // hop + 1)
    times = np.empty(n_frames)
    rms = np.empty(n_frames)
    for k in range(n_frames):
        seg = samples[k * hop : k * hop + window]
        rms[k] = np.sqrt(np.mean(seg * seg))
        times[k] = 1000.0 * k * hop / sr
    return times, rms


def _noise(sr, seconds, channels, width=2, amp=0.2):
    rng = np.random.default_rng(7)
    full = 2 ** (8 * width - 1) - 1
    dtype = {2: np.int16, 4: np.int32}[width]
    x = (rng.standard_normal((int(sr * seconds), channels)) * amp * full).astype(dtype)
    return AudioSegment(x.tobytes(), sample_width=width, frame_rate=sr, channels=channels)


@pytest.mark.parametrize(
    "sr, channels, width, kwargs",
    [
        (44100, 2, 2, {}),  # window = 3 hops exactly
        (22050, 1, 2, {}),  # window is 3 hops + 1 sample (661 vs 220)
        (48000, 2, 4, {}),  # 32-bit samples
        (22050, 1, 2, {"window_ms": 250, "hop_ms": 33}),
        (44100, 1, 2, {"window_ms": 5, "hop_ms": 10}),  # window shorter than the hop
    ],
)
@pytest.mark.parametrize("chunk_frames", [4096, 7])
def test_compute_rms_curve_matches_the_frame_by_frame_loop(
    monkeypatch, sr, channels, width, kwargs, chunk_frames
):
    monkeypatch.setattr(splitter, "_RMS_CHUNK_FRAMES", chunk_frames)  # force many blocks
    audio = _noise(sr, 8, channels, width)

    times, rms = compute_rms_curve(audio, **kwargs)
    ref_times, ref_rms = _reference_rms_curve(audio, **kwargs)

    assert np.array_equal(times, ref_times)
    assert np.allclose(rms, ref_rms, rtol=1e-12, atol=0)


def test_compute_rms_curve_of_audio_shorter_than_one_window_is_empty():
    times, rms = compute_rms_curve(AudioSegment.silent(10, frame_rate=44100))
    assert len(times) == 0 and len(rms) == 0


def test_compute_rms_curve_keeps_digital_silence_exactly_zero_after_loud_audio():
    # a running total that is subtracted from itself would leave rounding
    # residue here and blur the quiet point the cut search is looking for
    audio = _noise(44100, 3, 1, amp=0.5) + AudioSegment.silent(2000, frame_rate=44100)
    times, rms = compute_rms_curve(audio)
    assert np.all(rms[(times > 3100) & (times < 4900)] == 0)


# --- export formats ----------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [("wav", "wav"), (".WAV", "wav"), (" Flac ", "flac"), ("aif", "aiff"), ("mp3", "mp3")],
)
def test_normalize_export_format(name, expected):
    assert normalize_export_format(name) == expected


@pytest.mark.parametrize("name", ["", "  ", "."])
def test_normalize_export_format_rejects_empty(name):
    with pytest.raises(ValueError, match="export format"):
        normalize_export_format(name)


def test_lossless_formats_are_recognised_and_need_no_margin():
    for fmt in ("wav", ".FLAC", "aiff", "aif"):
        assert is_lossless(fmt)
        assert default_margin_s(fmt) == 0.0
    for fmt in ("mp3", "ogg", "m4a"):
        assert not is_lossless(fmt)
        assert default_margin_s(fmt) == DEFAULT_LOSSY_MARGIN_S


class _RecordingSegment:
    def __init__(self):
        self.calls = []

    def export(self, path, **kwargs):
        self.calls.append((path, kwargs))


def test_export_segment_passes_bitrate_only_to_lossy_formats():
    seg = _RecordingSegment()
    export_segment(seg, "a.mp3", "mp3", "192k")
    export_segment(seg, "a.flac", ".FLAC", "192k")
    export_segment(seg, "a.wav", "wav", "192k")
    assert seg.calls == [
        ("a.mp3", {"format": "mp3", "bitrate": "192k"}),
        ("a.flac", {"format": "flac"}),
        ("a.wav", {"format": "wav"}),
    ]


def test_split_config_normalises_the_export_format():
    assert SplitConfig(export_format=".FLAC").export_format == "flac"
    with pytest.raises(ValueError, match="export format"):
        SplitConfig(export_format="")
