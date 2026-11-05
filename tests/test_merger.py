import json
import os
import shutil
import tracemalloc

import numpy as np
import pytest
from seamsplit.audio import AudioSegment

from seamsplit.merger import (
    MergeReport,
    _adaptive_gains,
    _equal_power_gains,
    _linear_gains,
    adaptive_crossfade_merge,
    equal_power_crossfade_merge,
    find_seam,
    format_time,
    estimate_lag,
    find_processed_file,
    find_processed_files,
    linear_crossfade_merge,
    match_db,
    load_manifest,
    merge_audio,
)


def _tone(duration_ms, freq=440, frame_rate=44100):
    """A full-scale mono 16 bit sine (the samples cut towards zero)."""
    n = int(frame_rate * (duration_ms / 1000.0))
    wave = np.sin(2 * np.pi * freq / frame_rate * np.arange(n)) * 32767
    return AudioSegment(
        wave.astype(np.int16).tobytes(), sample_width=2, frame_rate=frame_rate, channels=1
    )


@pytest.mark.filterwarnings("ignore:.*clipped:RuntimeWarning")
def test_equal_power_crossfade_merge_length():
    a, b = _tone(1000), _tone(1000)
    overlap_ms = 200
    merged = equal_power_crossfade_merge([a, b], overlap_ms)
    expected_len = len(a) + len(b) - overlap_ms
    assert abs(len(merged) - expected_len) <= 1  # rounding tolerance


def test_equal_power_crossfade_merge_single_segment_passthrough():
    a = _tone(500)
    merged = equal_power_crossfade_merge([a], 100)
    assert len(merged) == len(a)


def test_equal_power_crossfade_no_gross_clipping():
    a, b = _tone(500), _tone(500)
    with pytest.warns(RuntimeWarning, match="clipped"):  # identical full-scale audio: +3 dB
        merged = equal_power_crossfade_merge([a, b], 100)
    samples = np.array(merged.get_array_of_samples())
    max_amp = max(
        np.abs(np.array(a.get_array_of_samples())).max(),
        np.abs(np.array(b.get_array_of_samples())).max(),
    )
    assert np.abs(samples).max() <= max_amp + 1


def test_linear_crossfade_merge_length():
    a, b = _tone(1000), _tone(1000)
    merged = linear_crossfade_merge([a, b], 200)
    assert len(merged) == len(a) + len(b) - 200


def test_find_processed_file_matches_by_stem(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "song_part01_instrumental.mp3").write_bytes(b"fake")

    found = find_processed_file(str(processed_dir), "song_part01.mp3")
    assert found.endswith("song_part01_instrumental.mp3")


def test_find_processed_file_disambiguates_with_match(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "song_part01_instrumental.mp3").write_bytes(b"fake")
    (processed_dir / "song_part01_vocals.mp3").write_bytes(b"fake")

    with pytest.raises(ValueError):
        find_processed_file(str(processed_dir), "song_part01.mp3")

    found = find_processed_file(str(processed_dir), "song_part01.mp3", match_substr="instrumental")
    assert found.endswith("song_part01_instrumental.mp3")


def test_find_processed_file_raises_when_missing(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        find_processed_file(str(processed_dir), "song_part99.mp3")


def test_merge_audio_end_to_end(tmp_path):
    part1, part2 = _tone(1000), _tone(1000)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    part1.export(str(processed_dir / "song_part01.wav"), format="wav")
    part2.export(str(processed_dir / "song_part02.wav"), format="wav")

    manifest = {
        "source_file": "song.wav",
        "overlap_ms": 200,
        "parts": ["song_part01.wav", "song_part02.wav"],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    out_path = tmp_path / "merged.wav"
    merged = merge_audio(
        processed_dir=str(processed_dir),
        manifest_path=str(manifest_path),
        output_path=str(out_path),
    )

    assert out_path.exists()
    assert abs(len(merged) - 1800) <= 1  # 1000 + 1000 - 200 overlap


def test_merge_audio_zero_overlap_concatenates(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _tone(500).export(str(processed_dir / "song_part01.wav"), format="wav")
    _tone(700).export(str(processed_dir / "song_part02.wav"), format="wav")

    manifest = {
        "source_file": "song.wav",
        "overlap_ms": 0,
        "parts": ["song_part01.wav", "song_part02.wav"],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    merged = merge_audio(
        processed_dir=str(processed_dir),
        manifest_path=str(manifest_path),
        output_path=str(tmp_path / "out.wav"),
    )
    assert abs(len(merged) - 1200) <= 1


def _noise(duration_ms, seed=0, frame_rate=44100):
    rng = np.random.default_rng(seed)
    n = int(frame_rate * duration_ms / 1000)
    pcm = (rng.standard_normal(n) * 3000).astype(np.int16)
    return AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=frame_rate, channels=1)


def _split_with_overlap(source, cut_ms, overlap_ms):
    """Two chunks of the same audio that share exactly ``overlap_ms``."""
    lead = overlap_ms // 2
    return source[: cut_ms + (overlap_ms - lead)], source[cut_ms - lead :]


def test_linear_crossfade_reconstructs_coherent_overlap():
    source = _noise(3000)
    a, b = _split_with_overlap(source, 1500, 400)

    merged = linear_crossfade_merge([a, b], 400)

    ref = np.array(source.get_array_of_samples()).astype(float)
    out = np.array(merged.get_array_of_samples()).astype(float)
    assert len(out) == len(ref)
    assert np.abs(out - ref).max() <= 1  # only rounding


def test_equal_power_crossfade_boosts_coherent_overlap():
    """Documents *why* linear is the default: on identical audio the
    equal-power curve is ~3 dB too loud in the middle of the fade."""
    source = _noise(3000)
    a, b = _split_with_overlap(source, 1500, 400)

    merged = equal_power_crossfade_merge([a, b], 400)

    sr = source.frame_rate
    mid = slice(int(1.45 * sr), int(1.55 * sr))
    ref = np.array(source.get_array_of_samples()).astype(float)[mid]
    out = np.array(merged.get_array_of_samples()).astype(float)[mid]
    gain_db = 20 * np.log10(np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(ref**2)))
    assert 2.5 < gain_db < 3.5


def test_merge_audio_default_method_reconstructs_a_coherent_overlap(tmp_path):
    source = _noise(3000)
    a, b = _split_with_overlap(source, 1500, 400)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    a.export(str(processed_dir / "song_part01.wav"), format="wav")
    b.export(str(processed_dir / "song_part02.wav"), format="wav")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_file": "song.wav",
                "overlap_ms": 400,
                "parts": ["song_part01.wav", "song_part02.wav"],
            }
        )
    )

    merged = merge_audio(str(processed_dir), str(manifest_path), str(tmp_path / "out.wav"))

    ref = np.array(source.get_array_of_samples()).astype(float)
    out = np.array(merged.get_array_of_samples()).astype(float)
    assert np.abs(out - ref).max() <= 1


_NAN, _INF = float("nan"), float("inf")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_lag_ms": -1}, "max_lag_ms must not be negative"),
        ({"max_lag_ms": -0.001}, "max_lag_ms must not be negative"),
        ({"max_lag_ms": _NAN}, "max_lag_ms must be a finite number"),
        ({"max_lag_ms": _INF}, "max_lag_ms must be a finite number"),
        ({"max_lag_ms": "50"}, "max_lag_ms must be a number"),
        ({"max_lag_ms": True}, "max_lag_ms must be a number"),
        ({"align": False, "max_lag_ms": -1}, "max_lag_ms must not be negative"),
        ({"samplerate": 0}, "samplerate must be a positive whole number"),
        ({"samplerate": -44100}, "samplerate must be a positive whole number"),
        ({"samplerate": 44100.5}, "samplerate must be a positive whole number"),
        ({"samplerate": _NAN}, "samplerate must be a positive whole number"),
        ({"samplerate": True}, "samplerate must be a positive whole number"),
        ({"samplerate": None}, "samplerate must be a positive whole number"),
        ({"max_gain_db": _NAN}, "max_gain_db must be a finite number"),
        ({"max_gain_db": -1}, "max_gain_db must not be negative"),
        ({"seam_fade_ms": 0}, "seam_fade_ms must be positive"),
        ({"seam_fade_ms": _NAN}, "seam_fade_ms must be a finite number"),
        ({"seam_warn_db": _NAN}, "seam_warn_db must not be NaN"),
        ({"seam_warn_db": "15"}, "seam_warn_db must be a number or None"),
        ({"length_tolerance_ms": -1}, "length_tolerance_ms must not be negative"),
        ({"length_tolerance_ms": _NAN}, "length_tolerance_ms must be a finite number"),
        ({"overlap_s": -1}, "overlap_s must not be negative"),
        ({"overlap_s": _NAN}, "overlap_s must be a finite number"),
        ({"overlap_s": _INF}, "overlap_s must be a finite number"),
    ],
)
def test_merge_audio_rejects_numbers_outside_their_range_before_any_work(tmp_path, kwargs, message):
    # The manifest names a chunk that does not exist: if a bad number were only
    # noticed after the chunks are loaded, the error would be a FileNotFoundError.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"overlap_ms": 200, "parts": ["a.wav"]}))
    with pytest.raises(ValueError, match=message):
        merge_audio(str(tmp_path), str(manifest_path), str(tmp_path / "o.wav"), **kwargs)


@pytest.mark.parametrize(
    "merge", [adaptive_crossfade_merge, linear_crossfade_merge, equal_power_crossfade_merge]
)
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"overlap_ms": -1}, "overlap_ms must not be negative"),
        ({"overlap_ms": _NAN}, "overlap_ms must be a finite number"),
        ({"max_lag_ms": -5}, "max_lag_ms must not be negative"),
        ({"max_lag_ms": _NAN}, "max_lag_ms must be a finite number"),
        ({"max_gain_db": _NAN}, "max_gain_db must be a finite number"),
        ({"seam_warn_db": _NAN}, "seam_warn_db must not be NaN"),
    ],
)
def test_crossfade_merges_reject_numbers_outside_their_range(merge, kwargs, message):
    a, b = _noise(2000, seed=1), _noise(2000, seed=2)
    kwargs = {"overlap_ms": 400, **kwargs}
    with pytest.raises(ValueError, match=message):
        merge([a, b], **kwargs)


def test_a_negative_max_lag_is_an_error_not_a_low_correlation_warning():
    source = _noise(3000)
    a, b = _split_with_overlap(source, 1500, 400)
    report = MergeReport()
    with pytest.raises(ValueError, match="max_lag_ms"):
        adaptive_crossfade_merge([a, b], 400, max_lag_ms=-50, report=report)
    assert report.warnings == []


@pytest.mark.parametrize("max_lag", [-1, -500, _NAN, _INF, "10"])
def test_estimate_lag_rejects_a_search_range_that_is_not_valid(max_lag):
    x = np.random.default_rng(0).standard_normal(2000)
    with pytest.raises(ValueError, match="max_lag"):
        estimate_lag(x, x.copy(), max_lag)


def test_the_range_checks_still_accept_the_edges():
    source = _noise(3000)
    a, b = _split_with_overlap(source, 1500, 400)
    # 0 switches alignment off; numpy integers and floats are numbers too
    for lag in (0, 0.0, 25, np.float64(25.0), np.int64(25)):
        merged = adaptive_crossfade_merge([a, b], 400, max_lag_ms=lag)
        assert len(merged) == len(source)
    assert estimate_lag(np.zeros(1), np.zeros(1), 0) == (0, 0.0)  # too short is still not an error


def test_merge_audio_rejects_unknown_method(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"overlap_ms": 200, "parts": ["a.wav"]}))
    with pytest.raises(ValueError):
        merge_audio(str(tmp_path), str(manifest_path), str(tmp_path / "o.wav"), method="cubic")


# ---- alignment ---------------------------------------------------------------


@pytest.mark.parametrize("lag", [0, 37, -37, 500, -500])
def test_estimate_lag_recovers_known_offset(lag):
    rng = np.random.default_rng(5)
    a = rng.standard_normal(8000)
    # b[k] = a[k + lag]
    b = np.zeros_like(a)
    if lag >= 0:
        b[: len(a) - lag] = a[lag:]
    else:
        b[-lag:] = a[: len(a) + lag]

    found, corr = estimate_lag(a, b, max_lag=1000)

    assert found == lag
    assert corr > 0.9


def test_estimate_lag_of_unrelated_signals_has_low_correlation():
    rng = np.random.default_rng(6)
    _, corr = estimate_lag(rng.standard_normal(8000), rng.standard_normal(8000), 500)
    assert corr < 0.2


def test_estimate_lag_prefers_zero_for_periodic_signals():
    t = np.arange(8000) / 44100
    tone = np.sin(2 * np.pi * 441 * t)  # period is exactly 100 samples
    lag, corr = estimate_lag(tone, tone.copy(), max_lag=1000)
    assert lag == 0
    assert corr > 0.99


def _write_parts(tmp_path, parts, overlap_ms, source_len_ms=None, segments=None):
    processed = tmp_path / "processed"
    processed.mkdir()
    names = []
    for i, part in enumerate(parts, start=1):
        name = f"song_part{i:02d}.wav"
        part.export(str(processed / name), format="wav")
        names.append(name)
    manifest = {"source_file": "song.wav", "overlap_ms": overlap_ms, "parts": names}
    if source_len_ms is not None:
        manifest["source_length_ms"] = source_len_ms
    if segments is not None:
        manifest["segments"] = segments
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return str(processed), str(manifest_path)


def _snr_db(ref, out):
    n = min(len(ref), len(out))
    err = out[:n] - ref[:n]
    return 10 * np.log10(np.mean(ref[:n] ** 2) / max(np.mean(err**2), 1e-12))


@pytest.mark.parametrize("offset_ms", [25, -25])
def test_merge_audio_cancels_a_timing_offset_between_chunks(tmp_path, offset_ms):
    """A service that pads (+) or trims (-) the start of a chunk must not
    misalign the result."""
    source = _noise(4000)
    a, b = _split_with_overlap(source, 2000, 600)
    if offset_ms > 0:
        b = AudioSegment.silent(offset_ms, frame_rate=source.frame_rate) + b
    else:
        b = b[-offset_ms:]
    processed, manifest = _write_parts(tmp_path, [a, b], 600, source_len_ms=len(source))
    ref = np.array(source.get_array_of_samples()).astype(float)

    aligned_report = MergeReport()
    aligned = merge_audio(
        processed, manifest, str(tmp_path / "a.wav"), report=aligned_report, samplerate=44100
    )
    with pytest.warns(RuntimeWarning, match="barely correlate"):
        plain = merge_audio(
            processed, manifest, str(tmp_path / "p.wav"), align=False, samplerate=44100
        )

    assert len(aligned) == len(source)
    # a padded chunk has no real audio in its first `offset` ms, so a small residual
    # remains there; without alignment the result is destroyed
    assert _snr_db(ref, np.array(aligned.get_array_of_samples()).astype(float)) > 45
    assert _snr_db(ref, np.array(plain.get_array_of_samples()).astype(float)) < 20
    assert aligned_report.boundaries[0].aligned
    assert abs(aligned_report.boundaries[0].lag_ms + offset_ms) < 0.1
    assert not aligned_report.warnings


def test_merge_audio_warns_when_overlaps_do_not_match(tmp_path):
    a = _noise(2000, seed=1)
    b = _noise(2000, seed=2)  # unrelated audio: nothing to align on
    processed, manifest = _write_parts(tmp_path, [a, b], 500)

    report = MergeReport()
    merge_audio(processed, manifest, str(tmp_path / "o.wav"), report=report)

    assert not report.boundaries[0].aligned
    assert any("barely correlate" in w for w in report.warnings)


def test_merge_audio_without_report_raises_runtime_warning(tmp_path):
    a = _noise(2000, seed=1)
    b = _noise(2000, seed=2)
    processed, manifest = _write_parts(tmp_path, [a, b], 500)
    with pytest.warns(RuntimeWarning, match="barely correlate"):
        merge_audio(processed, manifest, str(tmp_path / "o.wav"))


def test_merge_audio_reports_chunk_length_and_total_length_problems(tmp_path):
    source = _noise(4000)
    a, b = _split_with_overlap(source, 2000, 400)
    segments = [
        {"name": "song_part01.wav", "start_ms": 0, "end_ms": len(a)},
        {"name": "song_part02.wav", "start_ms": 1800, "end_ms": 1800 + len(b) + 500},  # wrong
    ]
    processed, manifest = _write_parts(
        tmp_path, [a, b], 400, source_len_ms=len(source) + 900, segments=segments
    )

    report = MergeReport()
    merge_audio(processed, manifest, str(tmp_path / "o.wav"), report=report)

    assert any(w.startswith("part 2:") for w in report.warnings)
    assert any("merged audio is" in w for w in report.warnings)


def test_merge_audio_warns_about_overlap_override(tmp_path):
    source = _noise(3000)
    a, b = _split_with_overlap(source, 1500, 400)
    processed, manifest = _write_parts(tmp_path, [a, b], 400)

    report = MergeReport()
    merge_audio(processed, manifest, str(tmp_path / "o.wav"), overlap_s=1.0, report=report)

    assert any("overlap override" in w for w in report.warnings)


# ---- manifest + file matching --------------------------------------------------


def test_find_processed_file_does_not_confuse_part10_with_part100(tmp_path):
    (tmp_path / "song_part10.wav").write_bytes(b"x")
    (tmp_path / "song_part100_vocals.wav").write_bytes(b"x")

    assert find_processed_file(str(tmp_path), "song_part10.mp3").endswith("song_part10.wav")
    assert find_processed_file(str(tmp_path), "song_part100.mp3").endswith(
        "song_part100_vocals.wav"
    )


def test_load_manifest_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_manifest(str(bad))

    no_parts = tmp_path / "no_parts.json"
    no_parts.write_text(json.dumps({"overlap_ms": 100}))
    with pytest.raises(ValueError, match="parts"):
        load_manifest(str(no_parts))

    no_overlap = tmp_path / "no_overlap.json"
    no_overlap.write_text(json.dumps({"parts": ["a.wav"]}))
    with pytest.raises(ValueError, match="overlap_ms"):
        load_manifest(str(no_overlap))


def _manifest_with(**changes):
    """A manifest in the shape ``split`` writes, with some fields replaced."""
    manifest = {
        "source_file": "song.wav",
        "source_length_ms": 3800,
        "overlap_ms": 200,
        "parts": ["song_part01.wav", "song_part02.wav"],
        "segments": [
            {"name": "song_part01.wav", "start_ms": 0, "end_ms": 2100},
            {"name": "song_part02.wav", "start_ms": 1900, "end_ms": 3800},
        ],
        "max_export_ms": 2100,
    }
    manifest.update(changes)
    return manifest


def _write_manifest(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    # json.dumps writes NaN / Infinity, and json.load reads them back by default
    path.write_text(json.dumps(manifest))
    return str(path)


def test_load_manifest_accepts_what_split_writes(tmp_path):
    manifest = _manifest_with()
    assert load_manifest(_write_manifest(tmp_path, manifest)) == manifest


def test_load_manifest_accepts_floats_zero_overlap_and_null_optional_fields(tmp_path):
    manifest = _manifest_with(
        overlap_ms=0,
        source_file=None,
        source_length_ms=None,
        max_export_ms=None,
        segments=[
            {"start_ms": 0.0, "end_ms": 2100.5},
            {"start_ms": 1900, "end_ms": 1900},  # an empty chunk is odd, not a type error
        ],
    )
    assert load_manifest(_write_manifest(tmp_path, manifest)) == manifest

    minimal = {"parts": ["a.wav"], "overlap_ms": 150.5}
    assert load_manifest(_write_manifest(tmp_path, minimal)) == minimal


@pytest.mark.parametrize(
    "manifest, message",
    [
        # parts
        (_manifest_with(parts="song_part01.wav"), r"'parts' must be a list.*string"),
        (_manifest_with(parts={"a": 1}), r"'parts' must be a list.*object"),
        (_manifest_with(parts=None), r"'parts' must be a list.*null"),
        (_manifest_with(parts=[]), r"'parts' must not be empty"),
        # an int where a file name belongs
        ({"parts": [1], "overlap_ms": 0}, r"'parts\[0\]' must be a file name.*number 1"),
        (_manifest_with(parts=["a.wav", None]), r"'parts\[1\]'.*null"),
        (_manifest_with(parts=["a.wav", True]), r"'parts\[1\]'.*boolean True"),
        (_manifest_with(parts=[["a.wav"], "b.wav"]), r"'parts\[0\]'.*array"),
        (_manifest_with(parts=[{"name": "a.wav"}, "b.wav"]), r"'parts\[0\]'.*object"),
        (_manifest_with(parts=[""]), r"'parts\[0\]' is not a file name"),
        (_manifest_with(parts=["   "]), r"'parts\[0\]' is not a file name"),
        (_manifest_with(parts=[".."]), r"'parts\[0\]' is not a file name"),
        (_manifest_with(parts=["sub/a.wav"]), r"'parts\[0\]'.*without folders"),
        (_manifest_with(parts=["..\\a.wav"]), r"'parts\[0\]'.*without folders"),
        (_manifest_with(parts=["a\x00b.wav"]), r"'parts\[0\]'.*null character"),
        # overlap_ms
        (_manifest_with(overlap_ms="200"), r"'overlap_ms' must be a number.*string '200'"),
        (_manifest_with(overlap_ms=None), r"'overlap_ms' must be a number.*null"),
        (_manifest_with(overlap_ms=True), r"'overlap_ms' must be a number.*boolean"),
        (_manifest_with(overlap_ms=[200]), r"'overlap_ms' must be a number.*array"),
        (_manifest_with(overlap_ms=-1), r"'overlap_ms' must not be negative"),
        (_manifest_with(overlap_ms=float("nan")), r"'overlap_ms' must be a finite number"),
        (_manifest_with(overlap_ms=float("inf")), r"'overlap_ms' must be a finite number"),
        (_manifest_with(overlap_ms=float("-inf")), r"'overlap_ms' must be a finite number"),
        # an int too large for a float is reported like any other bad number
        (_manifest_with(overlap_ms=10**400), r"'overlap_ms' must be a finite number"),
        # optional fields
        (_manifest_with(source_file=5), r"'source_file' must be a string.*number 5"),
        (_manifest_with(source_length_ms="3800"), r"'source_length_ms' must be a number.*string"),
        (_manifest_with(source_length_ms=-1), r"'source_length_ms' must not be negative"),
        (_manifest_with(source_length_ms=float("nan")), r"'source_length_ms' must be a finite"),
        (_manifest_with(source_length_ms=False), r"'source_length_ms' must be a number.*boolean"),
        (_manifest_with(max_export_ms="long"), r"'max_export_ms' must be a number.*string"),
        # segments
        (_manifest_with(segments="none"), r"'segments' must be a list.*string"),
        (_manifest_with(segments={}), r"'segments' must be a list.*object"),
        (_manifest_with(segments=[1, 2]), r"'segments\[0\]' must be an object.*number 1"),
        (_manifest_with(segments=[{"start_ms": 0, "end_ms": 5}]), r"1 entries but 'parts' has 2"),
        (_manifest_with(segments=[{}, {}]), r"'segments\[0\]\.start_ms' is missing"),
        (
            _manifest_with(segments=[{"start_ms": 0}, {"start_ms": 0, "end_ms": 5}]),
            r"'segments\[0\]\.end_ms' is missing",
        ),
        (
            _manifest_with(segments=[{"start_ms": 0, "end_ms": 5}, {"start_ms": "0", "end_ms": 5}]),
            r"'segments\[1\]\.start_ms' must be a number.*string '0'",
        ),
        (
            _manifest_with(segments=[{"start_ms": 0, "end_ms": float("nan")}] * 2),
            r"'segments\[0\]\.end_ms' must be a finite number",
        ),
        (
            _manifest_with(segments=[{"start_ms": -5, "end_ms": 5}] * 2),
            r"'segments\[0\]\.start_ms' must not be negative",
        ),
        (
            _manifest_with(segments=[{"start_ms": 0, "end_ms": 5}, {"start_ms": 9, "end_ms": 4}]),
            r"'segments\[1\]' ends before it starts \(start_ms 9, end_ms 4\)",
        ),
    ],
)
def test_load_manifest_rejects_wrong_types_with_a_clear_valueerror(tmp_path, manifest, message):
    path = _write_manifest(tmp_path, manifest)
    # ValueError specifically: a TypeError / OverflowError would escape the CLI as a traceback
    with pytest.raises(ValueError, match=message) as excinfo:
        load_manifest(path)
    assert path in str(excinfo.value)  # the message says which file is wrong


@pytest.mark.parametrize("content", [[], "just a string", 5, None, True])
def test_load_manifest_rejects_a_json_value_that_is_not_an_object(tmp_path, content):
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_manifest(_write_manifest(tmp_path, content))


def test_load_manifest_turns_unreadable_content_into_valueerrors(tmp_path):
    not_utf8 = tmp_path / "not_utf8.json"
    not_utf8.write_bytes(b"\xff\xfe{}")
    with pytest.raises(ValueError, match="not UTF-8"):
        load_manifest(str(not_utf8))

    # nesting far deeper than any manifest. Whether the JSON parser gives up on it (a
    # RecursionError, "nested too deeply") or copes (Python 3.14 sizes its limit from the real
    # stack and parses 100 000 levels) depends on the interpreter; either way the answer is a
    # ValueError, never a traceback
    deep = tmp_path / "deep.json"
    deep.write_text('{"parts": ' + "[" * 100_000 + "]" * 100_000 + ', "overlap_ms": 0}')
    with pytest.raises(ValueError, match="nested too deeply|not a valid seamsplit manifest"):
        load_manifest(str(deep))


def test_load_manifest_says_so_when_the_json_parser_runs_out_of_stack(tmp_path, monkeypatch):
    """The ``RecursionError`` branch on its own, whatever the interpreter's limit is."""

    def overflow(_file):
        raise RecursionError("maximum recursion depth exceeded")

    path = _write_manifest(tmp_path, {"parts": ["a.wav"], "overlap_ms": 0})
    monkeypatch.setattr(json, "load", overflow)
    with pytest.raises(ValueError, match="nested too deeply") as excinfo:
        load_manifest(path)
    assert path in str(excinfo.value)


def test_merge_audio_rejects_a_part_that_is_not_a_file_name(tmp_path):
    """A part that is not a string is rejected up front, not in the middle of the merge
    with ``TypeError: expected str, bytes or os.PathLike object, not int``."""
    processed = tmp_path / "processed"
    processed.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"parts": [1], "overlap_ms": 0}))

    with pytest.raises(ValueError, match=r"'parts\[0\]' must be a file name"):
        merge_audio(str(processed), str(manifest), str(tmp_path / "o.wav"))
    assert not (tmp_path / "o.wav").exists()


# ---- adaptive crossfade ---------------------------------------------------------


def test_adaptive_gains_are_linear_for_identical_and_equal_power_for_unrelated():
    t = np.linspace(0.0, 1.0, 501)

    out, inn = _adaptive_gains(t, 1.0)
    lin_out, lin_in = _linear_gains(t)
    assert np.allclose(out, lin_out) and np.allclose(inn, lin_in)

    out, inn = _adaptive_gains(t, 0.0)
    assert np.allclose(out**2 + inn**2, 1.0)  # the equal-power law
    eq_out, eq_in = _equal_power_gains(t)
    assert np.allclose(eq_out**2 + eq_in**2, 1.0)


@pytest.mark.parametrize("rho", [0.0, 0.2, 0.5, 0.9, 0.999, 1.0])
def test_adaptive_gains_keep_power_constant_for_any_correlation(rho):
    t = np.linspace(0.0, 1.0, 501)
    out, inn = _adaptive_gains(t, rho)

    assert (out[0], inn[0]) == pytest.approx((1.0, 0.0))
    assert (out[-1], inn[-1]) == pytest.approx((0.0, 1.0))
    # power of a*out + b*in for unit-power a, b with correlation rho
    assert np.allclose(out**2 + inn**2 + 2 * rho * out * inn, 1.0)


def test_adaptive_gains_clamp_the_correlation():
    t = np.linspace(0.0, 1.0, 101)
    for rho, expected in ((-0.7, 0.0), (1.3, 1.0)):
        got = _adaptive_gains(t, rho)
        want = _adaptive_gains(t, expected)
        assert np.allclose(got[0], want[0]) and np.allclose(got[1], want[1])


def test_adaptive_crossfade_reconstructs_a_coherent_overlap():
    source = _noise(3000)
    a, b = _split_with_overlap(source, 1500, 400)

    report = MergeReport()
    merged = adaptive_crossfade_merge([a, b], 400, 50, report, match_level=True)

    ref = np.array(source.get_array_of_samples()).astype(float)
    out = np.array(merged.get_array_of_samples()).astype(float)
    assert len(out) == len(ref)
    assert np.abs(out - ref).max() <= 1
    assert report.boundaries[0].fade_correlation > 0.999
    assert report.boundaries[0].gain_db == 0.0  # identical chunks are left untouched
    assert not report.warnings


def _noisy_copies(shared_noise_ratio=1.0, seconds=12, overlap_s=4, sr=44100):
    """Two chunks of ``s + n_i`` (independent n_i) whose overlaps have a
    correlation of 1 / (1 + shared_noise_ratio**2)."""
    rng = np.random.default_rng(3)
    n, ov = sr * seconds, sr * overlap_s
    s = rng.standard_normal(n) * 3000
    n1, n2 = n // 2 + ov // 2, n - (n // 2 - ov // 2)
    a = s[:n1] + rng.standard_normal(n1) * 3000 * shared_noise_ratio
    b = s[n // 2 - ov // 2 :] + rng.standard_normal(n2) * 3000 * shared_noise_ratio

    def seg(x):
        pcm = np.clip(x, -32768, 32767).astype(np.int16)
        return AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=sr, channels=1)

    return seg(a), seg(b), n // 2


def _mid_fade_level_db(merged, centre, sr=44100):
    x = np.array(merged.get_array_of_samples()).astype(float)
    mid = x[centre - sr // 20 : centre + sr // 20]
    edge = np.concatenate([x[sr : 2 * sr], x[-2 * sr : -sr]])
    return 10 * np.log10(np.mean(mid**2) / np.mean(edge**2))


def test_adaptive_crossfade_keeps_the_level_flat_when_overlaps_only_partly_match():
    """With rho = 0.5 a linear fade dips ~1.25 dB mid-fade and an equal-power
    fade rises ~1.76 dB; the adaptive curve stays flat."""
    a, b, centre = _noisy_copies()
    report = MergeReport()

    adaptive = adaptive_crossfade_merge([a, b], 4000, report=report)
    linear = linear_crossfade_merge([a, b], 4000)
    equal = equal_power_crossfade_merge([a, b], 4000)

    assert report.boundaries[0].fade_correlation == pytest.approx(0.5, abs=0.05)
    assert abs(_mid_fade_level_db(adaptive, centre)) < 0.4
    assert _mid_fade_level_db(linear, centre) < -0.8
    assert _mid_fade_level_db(equal, centre) > 1.2


def test_adaptive_crossfade_uses_equal_power_for_unrelated_overlaps():
    a, b = _noise(4000, seed=1), _noise(4000, seed=2)
    report = MergeReport()

    merged = adaptive_crossfade_merge([a, b], 1000, 50, report)

    assert abs(report.boundaries[0].fade_correlation) < 0.2
    assert any("barely correlate" in w for w in report.warnings)
    # the power of two independent signals stays level only with an equal-power law
    assert abs(_mid_fade_level_db(merged, len(a) - 500)) < 0.5


# ---- level matching --------------------------------------------------------------


def _scaled(segment, factor):
    x = np.array(segment.get_array_of_samples()).astype(float) * factor
    return AudioSegment(
        np.rint(x).astype(np.int16).tobytes(),
        sample_width=2,
        frame_rate=segment.frame_rate,
        channels=segment.channels,
    )


def _three_chunks(source, overlap_ms=600):
    lead = overlap_ms // 2
    c1 = source[: 3000 + (overlap_ms - lead)]
    c2 = source[3000 - lead : 6000 + (overlap_ms - lead)]
    c3 = source[6000 - lead :]
    return c1, c2, c3


def test_level_matching_removes_the_loudness_step_between_chunks():
    """A service that normalizes every chunk differently must not leave a
    step in loudness at the seams. The correction chains: chunk 3 is matched
    to chunk 2 *after* chunk 2 was corrected."""
    source = _noise(9000)
    c1, c2, c3 = _three_chunks(source)
    chunks = [c1, _scaled(c2, 1.4), _scaled(c3, 0.8)]
    ref = np.array(source.get_array_of_samples()).astype(float)

    with pytest.warns(RuntimeWarning, match="agree"):  # the step is what the report flags
        off = adaptive_crossfade_merge(chunks, 600, 50)
    report = MergeReport()
    on = adaptive_crossfade_merge(chunks, 600, 50, report, match_level=True)

    out_off = np.array(off.get_array_of_samples()).astype(float)
    out_on = np.array(on.get_array_of_samples()).astype(float)
    assert len(out_on) == len(ref)
    assert _snr_db(ref, out_on) > 60
    assert _snr_db(ref, out_off) < 20
    assert [b.gain_db for b in report.boundaries] == pytest.approx(
        [-20 * np.log10(1.4), -20 * np.log10(0.8)], abs=0.02
    )
    assert not report.warnings


def test_level_matching_is_limited_to_max_gain_db():
    source = _noise(6000)
    a, b = _split_with_overlap(source, 3000, 600)
    report = MergeReport()

    adaptive_crossfade_merge(
        [a, _scaled(b, 0.25)], 600, 50, report, match_level=True, max_gain_db=6.0
    )  # needs +12 dB

    assert report.boundaries[0].gain_db == pytest.approx(6.0, abs=0.01)
    assert any("limited" in w for w in report.warnings)


def test_level_matching_leaves_unrelated_overlaps_alone():
    """Different material in the overlap says nothing about a level mismatch."""
    a = _noise(2000, seed=1)
    b = _scaled(_noise(2000, seed=2), 0.3)
    report = MergeReport()

    adaptive_crossfade_merge([a, b], 500, 50, report, match_level=True)

    assert report.boundaries[0].gain_db == 0.0
    assert any("level matching" in w for w in report.warnings)


def test_level_matching_ignores_the_padding_added_to_a_chunk():
    """Silence a service put in front of a chunk must not be mistaken for a
    quiet chunk."""
    source = _noise(6000)
    a, b = _split_with_overlap(source, 3000, 600)
    pad = AudioSegment.silent(25, frame_rate=source.frame_rate)
    report = MergeReport()

    adaptive_crossfade_merge([a, pad + b], 600, 50, report, match_level=True)

    assert report.boundaries[0].lag_ms == pytest.approx(-25, abs=0.1)
    assert report.boundaries[0].gain_db == 0.0


def test_stereo_alignment_level_matching_and_adaptive_fade_together():
    sr = 44100
    rng = np.random.default_rng(11)
    n = sr * 9
    left = (rng.standard_normal(n) * 3000).astype(np.int16)
    right = (rng.standard_normal(n) * 2000).astype(np.int16)
    pcm = np.empty(2 * n, dtype=np.int16)
    pcm[0::2], pcm[1::2] = left, right
    source = AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=sr, channels=2)
    c1, c2, c3 = _three_chunks(source)
    pad = AudioSegment.silent(25, frame_rate=sr).set_channels(2)

    report = MergeReport()
    merged = adaptive_crossfade_merge(
        [c1, pad + _scaled(c2, 0.7), c3], 600, 50, report, match_level=True
    )

    ref = np.array(source.get_array_of_samples()).astype(float)
    out = np.array(merged.get_array_of_samples()).astype(float)
    assert merged.channels == 2 and len(out) == len(ref)
    assert _snr_db(ref, out) > 50
    assert report.boundaries[0].lag_ms == pytest.approx(-25, abs=0.1)
    assert report.boundaries[0].gain_db == pytest.approx(-20 * np.log10(0.7), abs=0.02)
    assert not report.warnings


def test_merge_audio_matches_level_by_default_and_can_be_switched_off(tmp_path):
    source = _noise(6000)
    a, b = _split_with_overlap(source, 3000, 600)
    processed, manifest = _write_parts(tmp_path, [a, _scaled(b, 0.7)], 600)
    ref = np.array(source.get_array_of_samples()).astype(float)

    on_report = MergeReport()
    on = merge_audio(processed, manifest, str(tmp_path / "on.wav"), report=on_report)
    with pytest.warns(RuntimeWarning, match="agree"):
        off = merge_audio(processed, manifest, str(tmp_path / "off.wav"), match_level=False)

    assert _snr_db(ref, np.array(on.get_array_of_samples()).astype(float)) > 50
    assert _snr_db(ref, np.array(off.get_array_of_samples()).astype(float)) < 25
    assert on_report.boundaries[0].gain_db == pytest.approx(-20 * np.log10(0.7), abs=0.02)


def test_merge_audio_rejects_a_negative_max_gain(tmp_path):
    source = _noise(3000)
    a, b = _split_with_overlap(source, 1500, 400)
    processed, manifest = _write_parts(tmp_path, [a, b], 400)
    with pytest.raises(ValueError, match="max_gain_db"):
        merge_audio(processed, manifest, str(tmp_path / "o.wav"), max_gain_db=-1)


def test_clipping_is_reported_instead_of_silent():
    a, b = _tone(500), _tone(500)  # full-scale, identical
    report = MergeReport()

    equal_power_crossfade_merge([a, b], 100, report=report)

    assert any("clipped" in w for w in report.warnings)


def test_offset_at_the_search_limit_is_reported():
    source = _noise(6000)
    a, b = _split_with_overlap(source, 3000, 600)
    b = AudioSegment.silent(50, frame_rate=source.frame_rate) + b
    report = MergeReport()

    adaptive_crossfade_merge([a, b], 600, 50, report)

    assert any("search limit" in w for w in report.warnings)


def test_no_align_does_not_claim_the_search_limit_was_hit():
    """With alignment off the measured lag is always 0 == the (zero) limit."""
    source = _noise(6000)
    a, b = _split_with_overlap(source, 3000, 600)
    report = MergeReport()

    adaptive_crossfade_merge([a, b], 600, 0.0, report, match_level=True)

    assert not report.warnings
    assert report.boundaries[0].aligned is False
    assert report.boundaries[0].fade_correlation > 0.999


# ---- seam search ------------------------------------------------------------------

SR = 44100


def _frames(ms):
    return int(round(ms / 1000 * SR))


def test_match_db_scale():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    assert match_db(x, x) == 99.0
    assert match_db(np.zeros(10), np.zeros(10)) == 99.0
    assert match_db(x, rng.standard_normal(20000)) == pytest.approx(-3.0, abs=0.3)
    # y = 1.1 x: difference power 0.01, mean signal power (1 + 1.21) / 2 = 1.105
    assert match_db(x, x * 1.1) == pytest.approx(10 * np.log10(1.105 / 0.01), abs=0.1)
    # noise at 1 % amplitude (1e-4 power) on top of the signal: 40 dB
    assert match_db(x, x + rng.standard_normal(20000) * 0.01) == pytest.approx(40, abs=0.5)


def test_find_seam_is_centered_for_identical_overlaps():
    x = np.random.default_rng(1).standard_normal(_frames(4000))
    start, width = find_seam(x, x.copy(), 1, _frames(250))
    assert width == _frames(250)
    assert abs(start - (len(x) - width) / 2) <= 1


def _overlap_with_one_clean_stretch(clean_from_ms, clean_to_ms, channels=1, total_ms=4000):
    """Overlap halves that differ everywhere except in one stretch."""
    rng = np.random.default_rng(5)
    n = _frames(total_ms)
    base = rng.standard_normal((n, channels)) * 3000
    a = base + rng.standard_normal((n, channels)) * 3000
    b = base + rng.standard_normal((n, channels)) * 3000
    lo, hi = _frames(clean_from_ms), _frames(clean_to_ms)
    b[lo:hi] = a[lo:hi]
    return a.reshape(-1), b.reshape(-1), (lo, hi)


@pytest.mark.parametrize("channels", [1, 2])
def test_find_seam_goes_to_the_stretch_where_the_chunks_agree(channels):
    a, b, (lo, hi) = _overlap_with_one_clean_stretch(2400, 3000, channels)  # 60-75 % in

    start, width = find_seam(a, b, channels, _frames(250))

    assert lo <= start and start + width <= hi


def test_find_seam_keeps_clear_of_the_overlap_edges():
    # the only clean stretch is right at the very start of the overlap (a chunk edge)
    a, b, _ = _overlap_with_one_clean_stretch(0, 300)
    n = len(a)

    start, width = find_seam(a, b, 1, _frames(250))

    assert start >= round(0.1 * n)
    assert start + width <= n - round(0.1 * n)


def test_find_seam_handles_overlaps_shorter_than_the_fade():
    x = np.random.default_rng(2).standard_normal(1000)
    assert find_seam(x, x, 1, 5000) == (0, 1000)
    assert find_seam(x[:0], x[:0], 1, 100) == (0, 0)


def _edge_artifact_chunks(sr=SR, seconds=12, overlap_s=4, baseline=15):
    """Chunks built on the *premise of seam search*: each version of the overlap
    is the true signal plus an error that grows towards that chunk's own edge
    (the end of the first chunk, the start of the second).

    That is an assumption, not a measurement of any real separator. Tests using
    this data show that the algorithm exploits such an error profile if it is
    there, not that real separator output has it."""
    rng = np.random.default_rng(9)
    n, ov = sr * seconds, sr * overlap_s
    src = rng.standard_normal(n) * 3000
    n1 = n // 2 + ov // 2
    first_edge = np.clip((np.arange(n1) - (n1 - sr * 1.5)) / (sr * 1.5), 0, 1)  # last 1.5 s
    n2 = n - (n // 2 - ov // 2)
    second_edge = np.clip(1 - np.arange(n2) / (sr * 1.5), 0, 1)  # first 1.5 s

    def seg(x):
        pcm = np.clip(np.rint(x), -32768, 32767).astype(np.int16)
        return AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=sr, channels=1)

    a = src[:n1] + rng.standard_normal(n1) * 3000 * first_edge
    a = a + rng.standard_normal(n1) * baseline  # what even a clean stretch differs by
    b = src[n // 2 - ov // 2 :] + rng.standard_normal(n2) * 3000 * second_edge
    b = b + rng.standard_normal(n2) * baseline
    return seg(a), seg(b), seg(src), n // 2


def test_seam_search_does_not_beat_a_whole_overlap_fade_when_the_error_is_uniform():
    """The flip side of the test below, and why seam search is opt-in: with the
    same independent error everywhere in the overlap there is no better place to
    cut to, and the fade over the whole overlap averages the two errors."""
    rng = np.random.default_rng(3)
    n, ov = SR * 12, SR * 4
    src = rng.standard_normal(n) * 3000
    n1, n2 = n // 2 + ov // 2, n - (n // 2 - ov // 2)

    def seg(x):
        pcm = np.clip(np.rint(x), -32768, 32767).astype(np.int16)
        return AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=SR, channels=1)

    a = seg(src[:n1] + rng.standard_normal(n1) * 600)
    b = seg(src[n // 2 - ov // 2 :] + rng.standard_normal(n2) * 600)
    ref = src[n // 2 - ov // 2 : n // 2 + ov // 2]
    lo, hi = n // 2 - ov // 2, n // 2 + ov // 2

    def overlap_snr(**kwargs):
        with pytest.warns(RuntimeWarning, match="worth a listen"):  # 11 dB < the 15 dB flag
            out = adaptive_crossfade_merge([a, b], 4000, **kwargs)
        return _snr_db(ref, np.array(out.get_array_of_samples()).astype(float)[lo:hi])

    assert overlap_snr() >= overlap_snr(seam_search=True)


def test_seam_search_beats_a_whole_overlap_fade_when_the_chunk_edges_are_bad():
    a, b, src, centre = _edge_artifact_chunks()
    ref = np.array(src.get_array_of_samples()).astype(float)
    lo, hi = centre - 2 * SR, centre + 2 * SR  # the overlap: all of the difference is in here

    full_report, seam_report = MergeReport(), MergeReport()
    full = adaptive_crossfade_merge([a, b], 4000, report=full_report)
    seam = adaptive_crossfade_merge([a, b], 4000, report=seam_report, seam_search=True)

    full_out = np.array(full.get_array_of_samples()).astype(float)
    seam_out = np.array(seam.get_array_of_samples()).astype(float)
    assert len(seam_out) == len(full_out) == len(ref)
    assert _snr_db(ref[lo:hi], seam_out[lo:hi]) > _snr_db(ref[lo:hi], full_out[lo:hi]) + 10

    info = seam_report.boundaries[0]
    assert info.fade_ms == pytest.approx(250, abs=1)
    assert full_report.boundaries[0].fade_ms == pytest.approx(4000, abs=1)
    # the seam avoids the two bad edges, so the chunks agree there far better
    assert info.seam_match_db > info.overlap_match_db + 6
    assert abs(info.seam_ms - 1000 * centre / SR) < 1000  # still in the middle of the overlap


def test_seam_position_is_reported_in_the_merged_timeline():
    source = _noise(3000)
    a, b = _split_with_overlap(source, 1500, 400)  # overlap 1300-1700 ms, middle 1500
    report = MergeReport()

    adaptive_crossfade_merge([a, b], 400, 50, report, seam_search=True)

    info = report.boundaries[0]
    assert info.seam_ms == pytest.approx(1500, abs=5)
    assert info.fade_ms == pytest.approx(250, abs=1)
    assert info.seam_match_db == 99.0 and not info.flagged


def test_seam_goes_where_the_chunks_agree_in_a_real_merge():
    """The position reported for the seam lies inside the clean stretch."""
    sr = SR
    rng = np.random.default_rng(4)
    ov_ms, cut = 4000, 6000
    ov = _frames(ov_ms)
    src = rng.standard_normal(sr * 12) * 3000
    first = src[: _frames(cut) + ov // 2].copy()
    second = src[_frames(cut) - ov // 2 :].copy()
    # inside the overlap (second half of `first` is the overlap's tail) spoil the
    # second chunk everywhere except 2.7-3.3 s into the overlap
    keep = slice(_frames(2700), _frames(3300))
    spoiled = second[:ov] + rng.standard_normal(ov) * 3000
    spoiled[keep] = second[:ov][keep]
    second[:ov] = spoiled

    def seg(x):
        pcm = np.clip(np.rint(x), -32768, 32767).astype(np.int16)
        return AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=sr, channels=1)

    report = MergeReport()
    adaptive_crossfade_merge([seg(first), seg(second)], ov_ms, 0.0, report, seam_search=True)

    start_of_overlap_ms = cut - ov_ms / 2
    info = report.boundaries[0]
    assert 2700 <= info.seam_ms - start_of_overlap_ms <= 3300


def test_seam_fade_as_long_as_the_overlap_is_the_old_whole_overlap_fade():
    source = _noise(3000)
    a, b = _split_with_overlap(source, 1500, 400)
    report = MergeReport()

    adaptive_crossfade_merge([a, b], 400, report=report, seam_search=True, seam_fade_ms=900)

    assert report.boundaries[0].fade_ms == pytest.approx(400, abs=1)


def test_seam_fade_must_be_positive():
    a, b = _noise(2000, seed=1), _noise(2000, seed=2)
    with pytest.raises(ValueError, match="seam_fade_ms"):
        adaptive_crossfade_merge([a, b], 400, seam_search=True, seam_fade_ms=0)


def test_unrelated_overlaps_fall_back_to_the_whole_overlap_and_are_flagged():
    a, b = _noise(4000, seed=1), _noise(4000, seed=2)
    report = MergeReport()

    adaptive_crossfade_merge([a, b], 1000, 50, report, seam_search=True)

    info = report.boundaries[0]
    assert info.flagged and info.seam_match_db is None
    assert info.fade_ms == pytest.approx(1000, abs=1)
    assert any("seam search" in w for w in report.warnings)


# ---- seam report -----------------------------------------------------------------


def test_poorly_matching_seams_are_flagged_and_located():
    a, b, _, centre = _edge_artifact_chunks()
    # make the two versions disagree everywhere, not only at the edges
    rng = np.random.default_rng(1)

    def noisier(seg):
        x = (
            np.array(seg.get_array_of_samples()).astype(float)
            + rng.standard_normal(len(seg.get_array_of_samples())) * 2500
        )
        return AudioSegment(
            np.clip(x, -32768, 32767).astype(np.int16).tobytes(),
            sample_width=2,
            frame_rate=SR,
            channels=1,
        )

    report = MergeReport()
    adaptive_crossfade_merge([noisier(a), noisier(b)], 4000, report=report, seam_search=True)

    info = report.boundaries[0]
    assert info.flagged
    assert info.seam_match_db < 15
    warning = next(w for w in report.warnings if "agree" in w)
    assert "best seam" in warning and format_time(info.seam_ms) in warning


def test_seam_warn_threshold_is_adjustable_and_can_be_disabled():
    a, b, _ = _noisy_copies()  # rho ~ 0.5: agree only to ~0 dB

    quiet = MergeReport()
    adaptive_crossfade_merge([a, b], 4000, report=quiet, seam_search=True, seam_warn_db=None)
    assert not quiet.boundaries[0].flagged and not quiet.warnings

    strict = MergeReport()
    adaptive_crossfade_merge([a, b], 4000, report=strict, seam_search=True, seam_warn_db=-10)
    assert not strict.boundaries[0].flagged


def test_without_seam_search_the_whole_overlap_is_judged():
    a, b, _ = _noisy_copies()
    report = MergeReport()

    adaptive_crossfade_merge([a, b], 4000, report=report)

    assert report.boundaries[0].flagged
    assert any("across the overlap" in w for w in report.warnings)


def test_overlap_match_ignores_the_padding_a_service_added():
    source = _noise(6000)
    a, b = _split_with_overlap(source, 3000, 600)
    pad = AudioSegment.silent(40, frame_rate=source.frame_rate)
    report = MergeReport()

    adaptive_crossfade_merge([a, pad + b], 600, 50, report)  # no seam search

    assert report.boundaries[0].overlap_match_db == 99.0 and not report.warnings


def test_report_serializes_to_json_and_markdown():
    a, b, _ = _noisy_copies()
    report = MergeReport()
    adaptive_crossfade_merge([a, b], 4000, report=report, seam_search=True)

    data = json.loads(json.dumps(report.to_dict()))
    assert data["boundaries"][0]["boundary"] == 1
    assert set(data["boundaries"][0]) >= {"seam_ms", "seam_match_db", "flagged", "gain_db"}
    assert data["warnings"] == report.warnings

    md = report.to_markdown()
    assert md.startswith("# Seam report")
    assert "| 1/2 |" in md and "**listen**" in md and "## Warnings" in md
    assert "nothing was analysed" in MergeReport().to_markdown()


def test_format_time():
    assert format_time(0) == "0:00.0"
    assert format_time(151_400) == "2:31.4"
    assert format_time(3_725_000) == "1:02:05.0"


def test_merge_audio_leaves_seam_search_off_by_default_and_can_switch_it_on(tmp_path):
    # Seam search is experimental (its premise is unverified on real separator
    # output), so the default is the fade across the whole overlap.
    source = _noise(6000)
    a, b = _split_with_overlap(source, 3000, 800)
    processed, manifest = _write_parts(tmp_path, [a, b], 800)

    default, on = MergeReport(), MergeReport()
    merge_audio(processed, manifest, str(tmp_path / "d.wav"), report=default)
    merge_audio(processed, manifest, str(tmp_path / "on.wav"), seam_search=True, report=on)
    short = MergeReport()
    merge_audio(
        processed,
        manifest,
        str(tmp_path / "s.wav"),
        seam_search=True,
        seam_fade_ms=100,
        report=short,
    )
    # the fade length alone does not turn seam search on
    ignored = MergeReport()
    merge_audio(processed, manifest, str(tmp_path / "i.wav"), seam_fade_ms=100, report=ignored)

    assert default.boundaries[0].fade_ms == pytest.approx(800, abs=1)
    assert ignored.boundaries[0].fade_ms == pytest.approx(800, abs=1)
    assert on.boundaries[0].fade_ms == pytest.approx(250, abs=1)
    assert short.boundaries[0].fade_ms == pytest.approx(100, abs=1)
    assert on.boundaries[0].seam_ms == pytest.approx(3000, abs=5)


@pytest.mark.parametrize("offset_ms, expected_sign", [(13, -1), (-13, +1)])
def test_lag_ms_is_negative_when_the_next_chunk_arrives_late_and_positive_when_early(
    offset_ms, expected_sign
):
    """The documented convention of ``BoundaryInfo.lag_ms``: a service that put
    silence in front of a chunk (its audio arrives later) gives a negative lag; one
    that trimmed the start of the chunk gives a positive lag."""
    source = _noise(6000)
    a, b = _split_with_overlap(source, 3000, 600)
    if offset_ms > 0:
        b = AudioSegment.silent(offset_ms, frame_rate=source.frame_rate) + b
    else:
        b = b[-offset_ms:]
    report = MergeReport()

    merged = adaptive_crossfade_merge([a, b], 600, 50, report)

    info = report.boundaries[0]
    assert info.aligned
    assert info.lag_ms * expected_sign > 0
    assert info.lag_ms == pytest.approx(-offset_ms, abs=0.1)
    assert len(merged) == len(source)  # the offset is cancelled, not just measured


def test_adaptive_fade_follows_the_correlation_at_the_seam_not_the_whole_overlap():
    a, b, _, _ = _edge_artifact_chunks()
    whole, seam = MergeReport(), MergeReport()

    adaptive_crossfade_merge([a, b], 4000, report=whole)
    adaptive_crossfade_merge([a, b], 4000, report=seam, seam_search=True)

    # over the whole overlap the bad edges pull the correlation down; the seam sits
    # in the clean stretch, where the two chunks are practically the same audio
    assert whole.boundaries[0].fade_correlation < 0.9
    assert seam.boundaries[0].fade_correlation > 0.99


def test_find_seam_centers_the_fade_when_the_overlap_has_no_room_for_the_edge_guard():
    x = np.random.default_rng(6).standard_normal(1000)
    # a fade of 900 of 1000 frames leaves 100 frames of play: less than the 10 % + 10 % guard
    assert find_seam(x, x.copy(), 1, 900) == (50, 900)


# ---- renamed files: pairing by order ------------------------------------------


def _touch(folder, *names):
    folder.mkdir(exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x")
    return str(folder)


def _part_names(n):
    return [f"song_part{i:02d}.wav" for i in range(1, n + 1)]


def test_find_processed_files_by_name_needs_no_warning(tmp_path):
    folder = _touch(tmp_path / "p", "song_part02_instrumental.wav", "song_part01_instrumental.wav")
    report = MergeReport()
    found = find_processed_files(folder, _part_names(2), report=report)
    assert [os.path.basename(f) for f in found] == [
        "song_part01_instrumental.wav",
        "song_part02_instrumental.wav",
    ]
    assert report.warnings == []


def test_find_processed_files_pairs_renamed_files_by_order_not_alphabetically(tmp_path):
    # `Song - 10` sorts before `Song - 2` as text; it must still be part 10
    folder = _touch(tmp_path / "p", *[f"Song - {i}.wav" for i in range(1, 11)])
    report = MergeReport()
    found = find_processed_files(folder, _part_names(10), report=report)
    assert [os.path.basename(f) for f in found] == [f"Song - {i}.wav" for i in range(1, 11)]
    assert len(report.warnings) == 1
    assert "by order" in report.warnings[0]
    assert "'Song - 1.wav' is part 1" in report.warnings[0]
    assert "'Song - 10.wav' is part 10" in report.warnings[0]


def test_find_processed_files_order_pairing_without_a_report_warns(tmp_path):
    folder = _touch(tmp_path / "p", "a1.wav", "a2.wav")
    with pytest.warns(RuntimeWarning, match="by order"):
        find_processed_files(folder, _part_names(2))


def test_find_processed_files_order_pairing_ignores_files_that_are_not_audio(tmp_path):
    folder = _touch(tmp_path / "p", "Song - 1.flac", "Song - 2.flac", "manifest.json", "notes.txt")
    found = find_processed_files(folder, _part_names(2), report=MergeReport())
    assert [os.path.basename(f) for f in found] == ["Song - 1.flac", "Song - 2.flac"]


def test_find_processed_files_order_pairing_respects_match(tmp_path):
    names = []
    for i in (1, 2, 3):
        names += [f"Song - {i} (Vocals).wav", f"Song - {i} (Instrumental).wav"]
    folder = _touch(tmp_path / "p", *names)
    found = find_processed_files(folder, _part_names(3), "instrumental", MergeReport())
    assert [os.path.basename(f) for f in found] == [
        f"Song - {i} (Instrumental).wav" for i in (1, 2, 3)
    ]


def test_find_processed_files_renamed_with_the_wrong_count_says_so(tmp_path):
    folder = _touch(tmp_path / "p", *[f"Song - {i}.wav" for i in (1, 3, 4)])
    with pytest.raises(FileNotFoundError, match=r"found 3 for 5 parts; some chunks may be missing"):
        find_processed_files(folder, _part_names(5))


def test_find_processed_files_renamed_with_extra_files_points_at_match(tmp_path):
    folder = _touch(tmp_path / "p", "Song - 1 (v).wav", "Song - 1 (i).wav", "Song - 2 (v).wav")
    with pytest.raises(FileNotFoundError, match="--match"):
        find_processed_files(folder, _part_names(2))


def test_find_processed_files_names_every_missing_part(tmp_path):
    folder = _touch(tmp_path / "p", "song_part01.wav", "song_part03.wav", "song_part04.wav")
    with pytest.raises(FileNotFoundError) as exc:
        find_processed_files(folder, _part_names(6))
    message = str(exc.value)
    assert "3 of 6 parts" in message
    for stem in ("song_part02", "song_part05", "song_part06"):
        assert stem in message


def test_find_processed_files_one_missing_part_keeps_the_single_file_message(tmp_path):
    folder = _touch(tmp_path / "p", "song_part01.wav", "song_part03.wav")
    with pytest.raises(FileNotFoundError, match="matches 'song_part02'$"):
        find_processed_files(folder, _part_names(3))


def test_find_processed_files_does_not_pair_by_order_when_only_some_parts_are_found(tmp_path):
    # part 1 is there by name; "Song - 2" cannot be assumed to be part 2
    folder = _touch(tmp_path / "p", "song_part01.wav", "Song - 2.wav")
    with pytest.raises(FileNotFoundError, match="song_part02"):
        find_processed_files(folder, _part_names(2))


def test_find_processed_files_still_rejects_several_files_for_one_part(tmp_path):
    folder = _touch(tmp_path / "p", "song_part01_a.wav", "song_part01_b.wav", "song_part02.wav")
    with pytest.raises(ValueError, match="use --match"):
        find_processed_files(folder, _part_names(2))


def test_merge_audio_with_renamed_chunks_gives_the_same_audio(tmp_path):
    source = _noise(9000)
    cuts = [(0, 3400), (3000, 6400), (6000, 9000)]
    parts = [source[a:b] for a, b in cuts]
    processed, manifest = _write_parts(tmp_path, parts, 400)
    by_name = merge_audio(processed, manifest, str(tmp_path / "by_name.wav"))

    renamed = tmp_path / "renamed"
    renamed.mkdir()
    for i in (1, 2, 3):
        shutil.copy(f"{processed}/song_part{i:02d}.wav", renamed / f"Song - {i}.wav")
    report = MergeReport()
    by_order = merge_audio(str(renamed), manifest, str(tmp_path / "by_order.wav"), report=report)

    assert by_order.raw_data == by_name.raw_data
    assert any("by order" in w for w in report.warnings)


# ---- memory: the merged audio is not held as float64 ---------------------------


def _tone_chunks(n_chunks, chunk_s, overlap_s, sr=16000):
    """Chunks of one continuous tone, made one at a time as they are asked for."""
    step = chunk_s - overlap_s
    for k in range(n_chunks):
        t = (np.arange(int(chunk_s * sr)) + k * int(step * sr)) / sr
        samples = (np.sin(2 * np.pi * 330 * t) * 8000).astype(np.int16)
        yield AudioSegment(samples.tobytes(), sample_width=2, frame_rate=sr, channels=1)


def test_crossfade_merge_accepts_chunks_one_at_a_time():
    make = lambda: _tone_chunks(5, 4, 1)  # noqa: E731
    from_list = adaptive_crossfade_merge(list(make()), 1000, 50, match_level=True)
    from_generator = adaptive_crossfade_merge(make(), 1000, 50, match_level=True)
    assert from_generator.raw_data == from_list.raw_data
    assert len(from_generator) == 5 * 4000 - 4 * 1000


def test_crossfade_merge_of_an_empty_generator_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        adaptive_crossfade_merge(iter(()), 500)


def test_crossfade_merge_peak_memory_stays_near_the_size_of_the_output():
    n_chunks, chunk_s, overlap_s, sr = 60, 10, 1, 16000
    output_bytes = (n_chunks * (chunk_s - overlap_s) + overlap_s) * sr * 2  # 16-bit mono

    tracemalloc.start()
    try:
        merged = adaptive_crossfade_merge(
            _tone_chunks(n_chunks, chunk_s, overlap_s, sr), overlap_s * 1000, 50, match_level=True
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(merged.raw_data) == output_bytes
    # The result and the copy made when it is joined are about 2x the output; holding the
    # whole song as float64 (4x) and copying it at every seam would peak at 8x or more.
    assert peak < 4 * output_bytes


# ---- progress / cancelling -----------------------------------------------------


def _three_parts(tmp_path):
    source = _noise(9000)
    parts = [source[a:b] for a, b in [(0, 3400), (3000, 6400), (6000, 9000)]]
    return _write_parts(tmp_path, parts, 400)


def test_merge_audio_reports_progress_for_every_chunk(tmp_path):
    processed, manifest = _three_parts(tmp_path)
    seen = []
    merge_audio(
        processed,
        manifest,
        str(tmp_path / "out.wav"),
        progress=lambda i, n, name: seen.append((i, n, name)),
    )
    assert seen == [(1, 3, "song_part01.wav"), (2, 3, "song_part02.wav"), (3, 3, "song_part03.wav")]


def test_an_exception_from_progress_stops_the_merge_before_anything_is_written(tmp_path):
    processed, manifest = _three_parts(tmp_path)
    out = tmp_path / "out.wav"
    reads = []

    def progress(i, n, name):
        reads.append(i)
        if i == 2:
            raise KeyboardInterrupt("stop")  # any exception, not only Exception subclasses

    with pytest.raises(KeyboardInterrupt):
        merge_audio(processed, manifest, str(out), progress=progress)
    assert reads == [1, 2]  # chunk 3 was never read
    assert not out.exists()
