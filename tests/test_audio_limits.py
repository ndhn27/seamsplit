"""How much audio is read.

All of a file's audio is held in memory, and what it takes is its length times its rate,
channels and sample width, not its size on disk: 15 MB of FLAC silence is 24 hours, 16 GB
decoded. A file is read only up to a limit (see ``seamsplit.audio_io``), checked against what
the header says before decoding and again as the audio comes out of the decoder.
"""

import json
import os
import tracemalloc
import wave

import numpy as np
import pytest
from seamsplit import audio_io, pairing
from seamsplit.audio import AudioSegment, AudioTooLongError, CouldntDecodeError
from seamsplit.audio_io import DEFAULT_MAX_DURATION_S, check_max_duration, read_audio
from seamsplit.cli import main
from seamsplit.merger import merge_audio
from seamsplit.pairing import identify, signature
from seamsplit.splitter import SplitConfig, split_audio
from seamsplit.status import compute_status

RATE = 8000


def _silent(seconds, rate=RATE, channels=1, width=2):
    return AudioSegment(
        bytes(int(seconds * rate) * channels * width),
        sample_width=width,
        frame_rate=rate,
        channels=channels,
    )


@pytest.fixture
def ten_minutes(tmp_path):
    """Ten minutes of silence as FLAC: about 100 kB on disk, 9.6 MB decoded."""
    path = tmp_path / "silence.flac"
    _silent(600).export(str(path), format="flac")
    assert os.path.getsize(path) < 200_000  # against 9.6 MB decoded
    return str(path)


@pytest.fixture
def no_header(tmp_path, ten_minutes):
    """The same file with the length left out of its header, as a file that lies about it
    would: nothing before the decoder says how long it is."""
    data = bytearray(open(ten_minutes, "rb").read())
    assert data[:4] == b"fLaC" and data[4] & 0x7F == 0  # STREAMINFO comes first
    data[21] &= 0xF0  # its 36 bits of total samples: the low four of this byte ...
    data[22:26] = b"\0\0\0\0"  # ... and these four
    path = tmp_path / "no_header.flac"
    path.write_bytes(bytes(data))
    return str(path)


# -- the limit itself ----------------------------------------------------------------


def test_the_default_is_two_hours():
    assert DEFAULT_MAX_DURATION_S == 7200.0


@pytest.mark.parametrize("value", [0, -1, float("nan"), True, "60", [60]])
def test_a_limit_that_is_not_a_number_of_seconds_is_refused(value):
    with pytest.raises(ValueError, match="max_duration_s"):
        check_max_duration(value)


def test_no_limit_is_none_or_infinity():
    assert check_max_duration(None) is None
    assert check_max_duration(float("inf")) is None
    assert check_max_duration(60) == 60.0


# -- refused before it is decoded ------------------------------------------------------


def test_a_file_that_says_it_is_too_long_is_refused_before_anything_is_decoded(
    ten_minutes, monkeypatch
):
    def decoded(*args, **kwargs):
        raise AssertionError("the decoder was started")

    monkeypatch.setattr(audio_io.av, "AudioResampler", decoded)  # made at the first frame
    with pytest.raises(AudioTooLongError, match="about 10 min long") as caught:
        AudioSegment.from_file(ten_minutes, max_duration_s=60)
    assert "--max-input-duration" in str(caught.value) and "1 min" in str(caught.value)
    assert isinstance(caught.value, CouldntDecodeError)  # what the CLI and the GUI report


@pytest.mark.parametrize(
    "seconds, text",
    [(7200, "2 h"), (5400, "1 h 30 min"), (90, "1 min 30 s"), (600, "10 min"), (45, "45 s")],
)
def test_durations_are_said_the_way_a_person_says_them(seconds, text):
    assert audio_io._hms(seconds) == text


def test_a_file_object_is_limited_too(ten_minutes):
    with open(ten_minutes, "rb") as f, pytest.raises(AudioTooLongError, match="the audio is about"):
        read_audio(f, max_duration_s=60)


def test_a_file_inside_the_limit_is_read_as_before(ten_minutes):
    seg = AudioSegment.from_file(ten_minutes, max_duration_s=601)
    assert len(seg) == 600_000 and seg.frame_rate == RATE
    assert len(AudioSegment.from_file(ten_minutes)) == 600_000  # and by default


def test_no_limit_reads_anything(ten_minutes):
    assert len(AudioSegment.from_file(ten_minutes, max_duration_s=None)) == 600_000


# -- stopped by the decoder, whatever the header says -----------------------------------


def test_a_file_that_does_not_say_how_long_it_is_stops_at_the_limit(no_header):
    import av

    container = av.open(no_header)
    assert container.streams.audio[0].duration is None and container.duration is None
    container.close()
    with pytest.raises(AudioTooLongError, match="holds more audio than that"):
        read_audio(no_header, max_duration_s=60)


def test_stopping_at_the_limit_keeps_memory_at_the_limit_not_at_the_file(no_header):
    tracemalloc.start()
    try:
        with pytest.raises(AudioTooLongError):
            read_audio(no_header, max_duration_s=60)  # 60 s: 960 kB; the file: 9.6 MB
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 3 * 960_000


def test_an_unusual_format_cannot_get_round_a_limit_in_seconds(tmp_path):
    """Eight channels at 96 kHz are 0.6 s here, inside a limit of 2 s, and 921,600 bytes:
    more than 2 s of the heaviest ordinary audio (48 kHz, stereo, 32 bit: 768,000 bytes)."""
    path = tmp_path / "heavy.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(8)
        w.setsampwidth(2)
        w.setframerate(96_000)
        w.writeframes(bytes(int(0.6 * 96_000) * 8 * 2))
    assert len(AudioSegment.from_file(str(path), max_duration_s=None)) == 600
    with pytest.raises(AudioTooLongError, match="holds more audio than that"):
        read_audio(str(path), max_duration_s=2)


# -- copies --------------------------------------------------------------------------


def _traced_peak(function):
    tracemalloc.start()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def test_decoding_holds_the_audio_about_once(tmp_path):
    """The blocks, their join and its conversion made three copies at the peak."""
    path = tmp_path / "stereo.wav"
    seg = AudioSegment(
        np.random.default_rng(1).integers(-9000, 9000, 60 * 44_100 * 2).astype("<i2").tobytes(),
        sample_width=2,
        frame_rate=44_100,
        channels=2,
    )
    seg.export(str(path), format="wav")
    size = len(seg.raw_data)  # 10.6 MB
    del seg
    decoded, peak = _traced_peak(lambda: AudioSegment.from_file(str(path)))
    assert len(decoded.raw_data) == size
    assert peak < 1.6 * size


def test_mixing_down_does_not_hold_the_mono_audio_twice(monkeypatch):
    # the float temporaries are one block, a fixed size; a small block shows the rest
    monkeypatch.setattr("seamsplit.audio._DOWNMIX_BLOCK", 1 << 12)
    stereo = AudioSegment(
        np.random.default_rng(2).integers(-9000, 9000, 30 * 44_100 * 2).astype("<i2").tobytes(),
        sample_width=2,
        frame_rate=44_100,
        channels=2,
    )
    mono, peak = _traced_peak(lambda: stereo.set_channels(1))
    assert len(mono.raw_data) == len(stereo.raw_data) // 2
    assert peak < 1.4 * len(mono.raw_data)


@pytest.mark.parametrize("channels, width", [(1, 2), (2, 2), (3, 2), (2, 4), (2, 1)])
def test_mono_samples_are_the_mixdown_of_a_stretch(channels, width):
    rng = np.random.default_rng(3)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
    info = np.iinfo(dtype)
    data = rng.integers(info.min, info.max, 5000 * channels, dtype=dtype)
    seg = AudioSegment(data.tobytes(), sample_width=width, frame_rate=8000, channels=channels)
    whole = seg.set_channels(1).samples
    assert np.array_equal(seg.mono_samples(), whole)
    for lo, hi in ((0, 10), (17, 4000), (4990, 5000), (100, None), (0, 5000)):
        assert np.array_equal(seg.mono_samples(lo, hi), whole[lo:hi])


def test_the_rms_curve_is_what_it_was_without_the_mono_copy():
    from seamsplit.splitter import compute_rms_curve

    rng = np.random.default_rng(4)
    stereo = AudioSegment(
        (rng.standard_normal(20 * 22_050 * 2) * 6000).astype("<i2").tobytes(),
        sample_width=2,
        frame_rate=22_050,
        channels=2,
    )
    times, via_mono_copy = compute_rms_curve(stereo.set_channels(1))
    times2, curve = compute_rms_curve(stereo)
    assert np.array_equal(times, times2) and np.array_equal(curve, via_mono_copy)


# -- split and merge -------------------------------------------------------------------


def test_split_refuses_a_file_that_is_too_long_and_writes_nothing(ten_minutes, tmp_path):
    out = tmp_path / "chunks"
    with pytest.raises(AudioTooLongError):
        split_audio(ten_minutes, SplitConfig(output_dir=str(out), max_input_s=60))
    assert not out.exists()


def test_a_split_config_with_a_limit_that_cannot_be_one_is_refused():
    with pytest.raises(ValueError, match="max_duration_s"):
        SplitConfig(max_input_s=0)
    assert SplitConfig(max_input_s=None).max_input_s is None
    assert SplitConfig().max_input_s == DEFAULT_MAX_DURATION_S


def _chunks(tmp_path, seconds, count, overlap_ms=0):
    processed = tmp_path / "processed"
    processed.mkdir()
    names = []
    for i in range(count):
        name = f"song_part{i + 1:02d}.wav"
        _silent(seconds).export(str(processed / name), format="wav")
        names.append(name)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"parts": names, "overlap_ms": overlap_ms}))
    return str(processed), str(manifest)


def test_merge_refuses_a_chunk_that_is_longer_than_the_limit(tmp_path):
    processed, manifest = _chunks(tmp_path, seconds=40, count=2)
    out = tmp_path / "merged.wav"
    with pytest.raises(AudioTooLongError, match="about 40 s long"):
        merge_audio(processed, manifest, str(out), pairing="off", max_input_s=30)
    assert not out.exists()


def test_merge_refuses_chunks_that_add_up_to_more_than_the_limit(tmp_path):
    """Each chunk is inside the limit; the merged audio, which is held whole, is not."""
    processed, manifest = _chunks(tmp_path, seconds=40, count=3)
    out = tmp_path / "merged.wav"
    with pytest.raises(AudioTooLongError, match="the merged audio reaches 120 s at part 3"):
        merge_audio(processed, manifest, str(out), pairing="off", max_input_s=100)
    assert not out.exists()
    merged = merge_audio(processed, manifest, str(out), pairing="off", max_input_s=120)
    assert len(merged) == 120_000  # as long as the limit is fine


def test_the_overlap_is_not_counted_twice_against_the_limit(tmp_path):
    processed, manifest = _chunks(tmp_path, seconds=40, count=3, overlap_ms=4000)
    merged = merge_audio(
        processed,
        manifest,
        str(tmp_path / "merged.wav"),
        pairing="off",
        max_input_s=112,
        align=False,
        match_level=False,
    )
    assert len(merged) == 112_000


def test_merge_checks_the_limit_before_reading_anything(tmp_path):
    processed, manifest = _chunks(tmp_path, seconds=1, count=1)
    with pytest.raises(ValueError, match="max_duration_s"):
        merge_audio(processed, manifest, str(tmp_path / "m.wav"), pairing="off", max_input_s=-5)


# -- comparing chunks with the originals and the status of a job ----------------------


def test_a_file_that_is_compared_is_limited_like_any_other(ten_minutes, tmp_path):
    with pytest.raises(AudioTooLongError):
        signature(ten_minutes, 60)
    signature(ten_minutes)  # inside the default limit: read (silence has an empty signature)
    original = tmp_path / "original.wav"
    _silent(5).export(str(original), format="wav")
    (match,) = identify([ten_minutes], [str(original)], max_input_s=60)
    assert match.part is None and "SeamSplit reads at most 1 min" in (match.error or "")
    (match,) = identify([ten_minutes], [str(original)])
    assert match.error is None  # not an error with the default


def _spy_on_signatures(monkeypatch):
    seen = []
    real = pairing.signature

    def spy(path, max_input_s=DEFAULT_MAX_DURATION_S):
        seen.append(max_input_s)
        return real(path, max_input_s)

    monkeypatch.setattr(pairing, "signature", spy)
    return seen


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # silent chunks: not told apart
def test_the_merge_gives_its_limit_to_the_comparison_of_the_chunks(tmp_path, monkeypatch):
    processed, manifest = _chunks(tmp_path, seconds=3, count=2)
    for name in os.listdir(processed):  # the originals, where the manifest is
        (tmp_path / name).write_bytes((tmp_path / "processed" / name).read_bytes())
    seen = _spy_on_signatures(monkeypatch)
    merge_audio(
        processed,
        manifest,
        str(tmp_path / "merged.wav"),
        pairing="verify",
        originals_dir=str(tmp_path),
        max_input_s=500,
    )
    assert seen and set(seen) == {500.0}


def test_the_status_gives_its_limit_to_the_comparison_of_files_that_lost_their_names(
    tmp_path, monkeypatch
):
    processed, manifest = _chunks(tmp_path, seconds=3, count=2)
    for name in os.listdir(processed):  # the originals, where the manifest is
        (tmp_path / name).write_bytes((tmp_path / "processed" / name).read_bytes())
    for i, name in enumerate(sorted(os.listdir(processed))):  # no chunk name in them any more
        os.rename(os.path.join(processed, name), os.path.join(processed, f"result{i}.wav"))
    seen = _spy_on_signatures(monkeypatch)
    compute_status(str(tmp_path), processed, max_input_s=123)
    assert seen and set(seen) == {123}
    seen.clear()
    compute_status(str(tmp_path), processed)
    assert set(seen) == {DEFAULT_MAX_DURATION_S}


# -- the command line ----------------------------------------------------------------


def _cli(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_split_has_the_flag_and_names_it_in_the_error(ten_minutes, tmp_path, capsys):
    out = str(tmp_path / "chunks")
    code, _, err = _cli(
        capsys, "split", ten_minutes, "--out", out, "--max-input-duration", "60", "--target", "40"
    )
    assert code == 1 and "--max-input-duration" in err and "about 10 min long" in err
    assert not os.path.exists(out)


def test_zero_removes_the_limit_on_the_command_line(ten_minutes, tmp_path, capsys):
    out = str(tmp_path / "chunks")
    code, _, err = _cli(
        capsys, "split", ten_minutes, "--out", out, "--max-input-duration", "0", "--dry-run"
    )
    assert code == 0, err


@pytest.mark.parametrize("value", ["-3", "nan"])
def test_a_limit_that_is_not_above_zero_is_an_error(ten_minutes, tmp_path, capsys, value):
    code, _, err = _cli(
        capsys, "split", ten_minutes, "--out", str(tmp_path / "c"), "--max-input-duration", value
    )
    assert code == 1 and "must be above 0" in err


def test_merge_has_the_flag(tmp_path, capsys):
    processed, manifest = _chunks(tmp_path, seconds=40, count=3)
    out = str(tmp_path / "merged.wav")
    args = ("merge", processed, "--manifest", manifest, "--out", out, "--pairing", "off")
    code, _, err = _cli(capsys, *args, "--max-input-duration", "100")
    assert code == 1 and "the merged audio reaches 120 s" in err and not os.path.exists(out)
    code, _, err = _cli(capsys, *args, "--max-input-duration", "0")
    assert code == 0 and os.path.exists(out), err


def test_run_has_the_flag_too(ten_minutes, tmp_path, capsys):
    code, _, err = _cli(
        capsys,
        "run",
        ten_minutes,
        "--job",
        str(tmp_path / "job"),
        "--max-input-duration",
        "60",
        "--target",
        "40",
    )
    assert code == 1 and "--max-input-duration" in err
