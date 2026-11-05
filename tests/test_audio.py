"""``seamsplit.audio``: the AudioSegment the rest of the package stands on, and reading and
writing files through PyAV."""

import io
import tracemalloc
import wave

import av
import numpy as np
import pytest

from seamsplit.audio import (
    AudioSegment,
    CouldntDecodeError,
    CouldntEncodeError,
    db_to_float,
    duration_s,
    engine_info,
)
from seamsplit.audio_io import parse_bitrate


def _pcm(values, width=2):
    return np.array(values, dtype={1: np.int8, 2: np.int16, 4: np.int32}[width])


def _seg(values, width=2, rate=1000, channels=1):
    return AudioSegment(
        _pcm(values, width).tobytes(), sample_width=width, frame_rate=rate, channels=channels
    )


def _tone(seconds=1.0, rate=44100, channels=1, freq=440.0, width=2, amp=0.5):
    t = np.arange(int(rate * seconds)) / rate
    x = np.sin(2 * np.pi * freq * t) * amp * (2 ** (8 * width - 1) - 1)
    rows = np.stack([x * (1.0 - 0.3 * c) for c in range(channels)], axis=1)
    dtype = {2: np.int16, 4: np.int32}[width]
    return AudioSegment(
        rows.astype(dtype).tobytes(), sample_width=width, frame_rate=rate, channels=channels
    )


def _snr_db(reference, other):
    n = min(len(reference), len(other))
    a = reference[:n].astype(np.float64)
    b = other[:n].astype(np.float64)
    return 10 * np.log10((a**2).sum() / max(((a - b) ** 2).sum(), 1e-9))


def _aligned_snr_db(reference, other, max_lag=2100):
    """The SNR of ``other`` against ``reference`` at the shift (up to ``max_lag`` frames of
    encoder delay) where they agree best."""
    a = reference.astype(np.float64)
    b = other.astype(np.float64)
    n = min(len(a), len(b)) - max_lag
    best = max(range(max_lag + 1), key=lambda k: float(np.dot(a[:n], b[k : k + n])))
    return _snr_db(a[:n], b[best : best + n])


# --- the segment ---------------------------------------------------------------------


def test_a_segment_knows_its_format_and_its_length():
    seg = _seg(np.arange(44100) % 1000, rate=44100, channels=1)
    assert (seg.sample_width, seg.frame_rate, seg.channels, seg.frame_width) == (2, 44100, 1, 2)
    assert len(seg) == 1000 and seg.duration_seconds == 1.0
    assert seg.array_type == "h" and seg.dtype == np.dtype(np.int16)
    assert seg.frame_count() == 44100 and seg.frame_count(ms=500) == 22050
    assert np.array_equal(seg.samples, np.arange(44100) % 1000)
    assert seg.frames.shape == (44100, 1) and len(seg.raw_data) == 88200
    assert list(seg.get_array_of_samples()[:3]) == [0, 1, 2]
    assert seg.get_array_of_samples().typecode == "h"


def test_a_segment_refuses_data_that_is_not_whole_frames_or_a_format_that_is_not_one():
    with pytest.raises(ValueError, match="multiple"):
        AudioSegment(b"\0\0\0", sample_width=2, frame_rate=8000, channels=1)
    with pytest.raises(ValueError, match="sample_width"):
        AudioSegment(b"", sample_width=5, frame_rate=8000, channels=1)
    with pytest.raises(ValueError, match="frame_rate"):
        AudioSegment(b"", sample_width=2, frame_rate=0, channels=1)
    with pytest.raises(ValueError, match="channels"):
        AudioSegment(b"", sample_width=2, frame_rate=8000, channels=0)


def test_24_bit_samples_are_held_as_32_bit_scaled_up():
    raw = bytes([0x01, 0x02, 0x03, 0xFF, 0xFF, 0xFF])  # 0x030201 and -1, little endian
    seg = AudioSegment(raw, sample_width=3, frame_rate=8000, channels=1)
    assert seg.sample_width == 4
    values = seg.samples.astype(np.int64)
    assert values[0] >> 8 == 0x030201 and values[1] >> 8 == -1


def test_an_array_or_a_buffer_can_be_given_as_the_data():
    data = np.arange(10, dtype=np.int16)
    assert AudioSegment(data, sample_width=2, frame_rate=8000, channels=1).frame_count() == 10
    assert (
        AudioSegment(
            memoryview(data.tobytes()), sample_width=2, frame_rate=8000, channels=2
        ).frame_count()
        == 5
    )


def test_from_samples_takes_the_width_from_the_array():
    seg = AudioSegment.from_samples(np.zeros((10, 2), dtype=np.int32), 8000, channels=2)
    assert (seg.sample_width, seg.channels, seg.frame_count()) == (4, 2, 10)
    with pytest.raises(ValueError, match="int16"):
        AudioSegment.from_samples(np.zeros(4, dtype=np.float32), 8000)


def test_silence_is_mono_16_bit():
    seg = AudioSegment.silent(250, frame_rate=8000)
    assert (seg.channels, seg.sample_width, seg.frame_count()) == (1, 2, 2000)
    assert not np.any(seg.samples)


def test_slices_are_in_milliseconds_and_clamp_to_the_ends():
    seg = _seg(range(1000), rate=1000)  # one frame per millisecond
    assert list(seg[10:13].samples) == [10, 11, 12]
    assert list(seg[:3].samples) == [0, 1, 2]
    assert list(seg[-3:].samples) == [997, 998, 999]
    assert len(seg[900:5000]) == 100
    assert len(seg[500:500]) == 0
    assert list(seg[5].samples) == [5]
    with pytest.raises(ValueError, match="step"):
        seg[::2]


def test_a_length_that_rounds_up_is_filled_with_silence_not_cut():
    # 44.1 kHz: 100 frames are 2.27 ms, which len() calls 2 ms (88 frames asked for)
    seg = _seg(np.ones(44100), rate=44100)
    assert len(seg[:2]) == 2 and seg[:2].frame_count() == 88
    short = _seg(np.ones(20), rate=44100)  # 0.45 ms
    assert len(short) == 0


def test_len_rounds_to_the_nearest_millisecond():
    assert len(_seg(np.zeros(1499), rate=1000)) == 1499
    assert len(_seg(np.zeros(100), rate=44100)) == 2  # 2.267 ms
    assert len(_seg(np.zeros(110), rate=44100)) == 2  # 2.494 ms
    assert len(_seg(np.zeros(111), rate=44100)) == 3  # 2.517 ms


def test_segments_are_joined_in_the_widest_format_and_gain_is_added_with_a_number():
    a = _seg([1, 2, 3], width=2, rate=1000)
    b = _seg([4, 5], width=4, rate=1000)
    joined = a + b
    assert joined.sample_width == 4 and joined.frame_count() == 5
    assert list(joined.samples) == [1 << 16, 2 << 16, 3 << 16, 4, 5]  # a is widened to 32 bit
    assert list((a + 6.0206).samples) == [2, 4, 6]  # +6.02 dB is a factor of 2
    assert sum([a, a]).frame_count() == 6


def test_gain_rounds_down_and_stays_inside_the_sample_range():
    seg = _seg([-1, 1, 100, -100, 32767, -32768])
    louder = seg.apply_gain(3.0)
    # 10 ** (3 / 20) = 1.41254: -1.41 -> -2, 1.41 -> 1, 141.25 -> 141, -141.25 -> -142
    assert list(louder.samples[:4]) == [-2, 1, 141, -142]
    assert list(louder.samples[4:]) == [32767, -32768]  # clipped, not wrapped
    assert db_to_float(-20) == pytest.approx(0.1) and db_to_float(0) == 1.0
    assert list(seg.apply_gain(-120.0).samples) == [-1, 0, 0, -1, 0, -1]  # floor(-0.000001)


def test_a_long_fade_steps_the_gain_once_per_millisecond():
    seg = _seg(np.full(1000, 10000), rate=1000)
    faded = seg.fade_out(500)  # ms 500 .. 1000
    out = faded.samples
    assert np.all(out[:500] == 10000)
    assert out[500] == 10000 and out[999] < 100  # goes from full to nothing
    assert np.all(np.diff(out[500:]) <= 0)
    assert len(faded) == 1000


def test_a_short_fade_steps_the_gain_once_per_sample():
    seg = _seg(np.full(44100, 10000), rate=44100)
    faded = seg.fade_in(50)
    n = int(44.1 * 50)
    assert faded.samples[0] == 0 and faded.samples[n - 1] > 9900
    assert np.all(np.diff(faded.samples[:n].astype(int)) >= 0)  # smooth, no per-ms staircase
    assert len(np.unique(faded.samples[:n])) > 1000
    assert np.all(faded.samples[n:] == 10000)


def test_fades_need_two_of_start_end_and_duration_and_no_fade_is_a_no_op():
    seg = _seg(np.full(500, 100), rate=1000)
    with pytest.raises(TypeError):
        seg.fade(to_gain=-6, start=1, end=2, duration=3)
    with pytest.raises(ValueError):
        seg.fade(to_gain=-6, start=0, duration=-5)
    assert seg.fade() is seg


def test_a_fade_between_two_gains_in_the_middle():
    seg = _seg(np.full(1000, 8000), rate=1000)
    out = seg.fade(to_gain=-12.0, from_gain=-6.0, start=200, end=700).samples
    assert (
        out[0] == int(8000 * db_to_float(-6.0)) or abs(int(out[0]) - 8000 * db_to_float(-6.0)) <= 1
    )
    assert abs(int(out[-1]) - 8000 * db_to_float(-12.0)) <= 1
    assert out[200] > out[699]


def test_sample_width_changes_move_the_bits_and_keep_the_level():
    seg = _seg([0, 1, -1, 12345, -12345], width=2)
    wide = seg.set_sample_width(4)
    assert list(wide.samples) == [v << 16 for v in (0, 1, -1, 12345, -12345)]
    back = wide.set_sample_width(2)
    assert back.raw_data == seg.raw_data
    narrow = seg.set_sample_width(1)
    assert narrow.sample_width == 1 and list(narrow.samples) == [0, 0, -1, 48, -49]
    assert seg.set_sample_width(2) is seg
    with pytest.raises(ValueError):
        seg.set_sample_width(3)


def test_mono_to_stereo_copies_and_stereo_to_mono_averages_rounding_down():
    mono = _seg([5, -5, 7], rate=8000)
    stereo = mono.set_channels(2)
    assert stereo.channels == 2 and list(stereo.samples) == [5, 5, -5, -5, 7, 7]
    two = AudioSegment(
        _pcm([1, 2, -1, -2, 10, 30]).tobytes(), sample_width=2, frame_rate=8000, channels=2
    )
    assert list(two.set_channels(1).samples) == [1, -2, 20]  # 1.5 -> 1, -1.5 -> -2
    four = AudioSegment(
        _pcm([4, 8, 12, 16, -4, -8, -12, -17]).tobytes(),
        sample_width=2,
        frame_rate=8000,
        channels=4,
    )
    # four channels: each is floor-divided by 4 and the results are added
    assert list(four.set_channels(1).samples) == [10, -11]
    assert list(mono.set_channels(3).samples[:6]) == [5, 5, 5, -5, -5, -5]
    with pytest.raises(ValueError, match="mono"):
        four.set_channels(2)
    assert mono.set_channels(1) is mono


def test_mono_segments_become_the_channels_of_one():
    left, right = _seg([1, 2, 3], rate=8000), _seg([9, 8], rate=8000)
    both = AudioSegment.from_mono_audiosegments(left, right)
    assert both.channels == 2 and list(both.samples) == [1, 9, 2, 8, 3, 0]
    with pytest.raises(ValueError):
        AudioSegment.from_mono_audiosegments()
    with pytest.raises(ValueError, match="mono"):
        AudioSegment.from_mono_audiosegments(both)


@pytest.mark.parametrize(
    "source, target", [(44100, 48000), (48000, 44100), (44100, 22050), (22050, 44100)]
)
def test_resampling_keeps_a_tone_and_the_length(source, target):
    seg = _tone(1.0, rate=source, freq=1000.0)
    out = seg.set_frame_rate(target)
    assert out.frame_rate == target and out.frame_count() == target  # a second stays a second
    n = out.frame_count()
    ideal = np.sin(2 * np.pi * 1000.0 * np.arange(int(n)) / target) * 0.5 * 32767
    middle = slice(int(n * 0.1), int(n * 0.9))
    assert _snr_db(ideal[middle], out.samples[middle]) > 70  # dB; a linear resampler stays near 55


def test_resampling_several_channels_and_32_bit_and_nothing():
    stereo = _tone(0.5, rate=44100, channels=2, freq=500.0).set_frame_rate(16000)
    assert stereo.channels == 2 and stereo.frame_count() == 8000
    left, right = stereo.frames[:, 0].astype(float), stereo.frames[:, 1].astype(float)
    assert np.corrcoef(left, right)[0, 1] > 0.999
    wide = _tone(0.5, rate=44100, width=4).set_frame_rate(22050)
    assert wide.sample_width == 4 and wide.frame_count() == 11025
    empty = AudioSegment(b"", sample_width=2, frame_rate=44100, channels=1).set_frame_rate(8000)
    assert (empty.frame_rate, empty.frame_count()) == (8000, 0)
    same = _tone(0.1)
    assert same.set_frame_rate(44100) is same


# --- files ---------------------------------------------------------------------------


@pytest.mark.parametrize("width, channels", [(2, 1), (2, 2), (4, 2)])
def test_wav_round_trips_exactly(tmp_path, width, channels):
    seg = _tone(0.3, rate=22050, channels=channels, width=width)
    path = tmp_path / "a.wav"
    seg.export(path, format="wav")
    back = AudioSegment.from_file(path)
    assert back.raw_data == seg.raw_data
    assert (back.sample_width, back.frame_rate, back.channels) == (width, 22050, channels)


def test_a_wav_is_written_by_the_standard_library_with_a_plain_header(tmp_path):
    seg = _tone(0.1, rate=8000)
    path = tmp_path / "a.wav"
    seg.export(path, format="wav")
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:16] == b"WAVEfmt " and len(data) == 44 + len(seg.raw_data)
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 8000)


def test_flac_and_aiff_are_lossless_at_16_bit_and_flac_keeps_24_of_32_bits(tmp_path):
    seg = _tone(0.3, rate=44100, channels=2)
    for fmt in ("flac", "aiff", "aif"):
        path = tmp_path / f"a.{fmt}"
        seg.export(path, format=fmt)
        assert AudioSegment.from_file(path).raw_data == seg.raw_data, fmt
    wide = _tone(0.3, rate=44100, channels=2, width=4)
    wide.export(tmp_path / "w.aiff", format="aiff")
    assert AudioSegment.from_file(tmp_path / "w.aiff").raw_data == wide.raw_data
    wide.export(tmp_path / "w.flac", format="flac")
    back = AudioSegment.from_file(tmp_path / "w.flac")
    assert back.sample_width == 4
    assert np.array_equal(back.samples >> 8, wide.samples >> 8)  # FLAC stores 24 bits


@pytest.mark.parametrize(
    "fmt, bitrate, floor", [("mp3", "320k", 55), ("mp3", None, 20), ("m4a", "256k", 25)]
)
def test_lossy_formats_come_back_the_same_length_and_close_in_sound(tmp_path, fmt, bitrate, floor):
    seg = _tone(1.5, rate=44100, channels=2)
    path = tmp_path / f"a.{fmt}"
    seg.export(path, format=fmt, bitrate=bitrate)
    back = AudioSegment.from_file(path)
    assert (back.sample_width, back.frame_rate, back.channels) == (2, 44100, 2)
    if fmt == "mp3":  # the gapless tag: the padding is trimmed on reading
        assert back.frame_count() == seg.frame_count()
    else:  # AAC in MP4 is trimmed to the frame or not at all, depending on the FFmpeg version
        assert abs(back.frame_count() - seg.frame_count()) <= 2048
    assert _aligned_snr_db(seg.samples, back.samples) > floor


def test_the_bitrate_is_honoured_and_a_larger_one_makes_a_larger_file(tmp_path):
    seg = _tone(2.0, rate=44100, channels=2)
    sizes = {}
    for rate in ("64k", "320k"):
        seg.export(tmp_path / f"{rate}.mp3", format="mp3", bitrate=rate)
        sizes[rate] = (tmp_path / f"{rate}.mp3").stat().st_size
    assert sizes["64k"] == pytest.approx(64_000 / 8 * 2.0, rel=0.15)
    assert sizes["320k"] == pytest.approx(320_000 / 8 * 2.0, rel=0.15)


def test_a_sample_rate_an_encoder_cannot_take_is_converted_for_it(tmp_path):
    seg = _tone(1.0, rate=44100, channels=2)
    seg.export(tmp_path / "a.opus", format="opus", bitrate="128k")
    back = AudioSegment.from_file(tmp_path / "a.opus")
    assert back.frame_rate == 48000 and back.frame_count() == pytest.approx(48000, abs=1500)
    high = _tone(1.0, rate=96000, channels=2)
    high.export(tmp_path / "hi.mp3", format="mp3", bitrate="192k")
    assert AudioSegment.from_file(tmp_path / "hi.mp3").frame_rate == 48000


def test_ogg_is_written_and_mono_either_works_or_says_why_not(tmp_path):
    """FFmpeg's own Vorbis encoder does stereo only; builds with libvorbis do mono too."""
    _tone(1.0, channels=2).export(tmp_path / "a.ogg", format="ogg")
    assert AudioSegment.from_file(tmp_path / "a.ogg").channels == 2
    try:
        _tone(1.0, channels=1).export(tmp_path / "m.ogg", format="ogg")
    except CouldntEncodeError as exc:
        assert "stereo" in str(exc)
    else:
        assert AudioSegment.from_file(tmp_path / "m.ogg").channels == 1


def test_files_can_be_read_and_written_through_file_objects_and_paths(tmp_path):
    seg = _tone(0.4, rate=22050, channels=2)
    for fmt in ("wav", "flac", "mp3"):
        buffer = io.BytesIO()
        seg.export(buffer, format=fmt, bitrate="256k")
        buffer.seek(0)
        assert AudioSegment.from_file(buffer).frame_count() == seg.frame_count(), fmt
    (tmp_path / "x").mkdir()
    seg.export(tmp_path / "x" / "a.wav", format=".WAV")  # path objects, dotted and upper case
    assert AudioSegment.from_file(str(tmp_path / "x" / "a.wav")).raw_data == seg.raw_data


def test_raw_export_is_the_samples_alone(tmp_path):
    seg = _tone(0.1, rate=8000)
    seg.export(tmp_path / "a.raw", format="raw")
    assert (tmp_path / "a.raw").read_bytes() == seg.raw_data


def test_8_bit_audio_is_written_unsigned_and_read_back_signed(tmp_path):
    seg = _seg([-128, -1, 0, 1, 127], width=1, rate=8000)
    seg.export(tmp_path / "a.wav", format="wav")
    with wave.open(str(tmp_path / "a.wav")) as w:
        assert w.getsampwidth() == 1 and list(w.readframes(5)) == [0, 127, 128, 129, 255]
    back = AudioSegment.from_file(tmp_path / "a.wav")
    assert back.sample_width == 1 and list(back.samples) == [-128, -1, 0, 1, 127]
    seg.export(tmp_path / "a.flac", format="flac")  # encoders get 16 bit
    assert AudioSegment.from_file(tmp_path / "a.flac").sample_width == 2


def _av_file(path, codec, fmt_name, samples, rate=8000, layout="mono"):
    """A file written with PyAV directly, for formats SeamSplit itself does not write."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=rate)
        stream.layout = layout
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format=fmt_name, layout=layout)
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_float_pcm_is_read_as_32_bit_integers(tmp_path):
    x = np.array([0.0, 0.5, -0.5, 0.25, -1.0], dtype=np.float32)
    _av_file(tmp_path / "f.wav", "pcm_f32le", "flt", x)
    seg = AudioSegment.from_file(tmp_path / "f.wav")
    assert seg.sample_width == 4
    assert list(seg.samples) == [0, 2**30, -(2**30), 2**29, -(2**31)]


def test_24_bit_wav_is_read_as_32_bit_scaled_up(tmp_path):
    values = np.array([0, 1, -1, 4_000_000, -4_000_000], dtype=np.int64)
    packed = b"".join(int(v).to_bytes(3, "little", signed=True) for v in values)
    with wave.open(str(tmp_path / "w24.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(8000)
        w.writeframes(packed)
    seg = AudioSegment.from_file(tmp_path / "w24.wav")
    assert seg.sample_width == 4 and np.array_equal(seg.samples.astype(np.int64) >> 8, values)


def test_what_is_not_audio_or_not_there_is_reported_as_such(tmp_path):
    (tmp_path / "junk.mp3").write_bytes(b"this is not audio" * 100)
    with pytest.raises(CouldntDecodeError):
        AudioSegment.from_file(tmp_path / "junk.mp3")
    with pytest.raises(FileNotFoundError):
        AudioSegment.from_file(tmp_path / "missing.wav")
    with pytest.raises(CouldntDecodeError):
        AudioSegment.from_file(io.BytesIO(b"\0" * 64))


def test_a_file_with_no_samples_is_an_empty_segment_but_cannot_be_encoded_lossy(tmp_path):
    empty = AudioSegment(b"", sample_width=2, frame_rate=44100, channels=1)
    empty.export(tmp_path / "e.wav", format="wav")
    back = AudioSegment.from_file(tmp_path / "e.wav")
    assert back.frame_count() == 0 and back.frame_rate == 44100
    with pytest.raises(CouldntEncodeError, match="no audio"):
        empty.export(tmp_path / "e.mp3", format="mp3")


def test_an_unknown_format_is_refused_with_the_ones_that_work(tmp_path):
    with pytest.raises(CouldntEncodeError, match="flac"):
        _tone(0.1).export(tmp_path / "a.xyz", format="xyz")


def test_a_wav_that_cannot_be_written_is_a_couldnt_encode_error(tmp_path):
    with pytest.raises(CouldntEncodeError):
        _tone(0.1).export(tmp_path / "no-such-folder" / "a.wav", format="wav")


# --- lengths and what is used ------------------------------------------------------------


@pytest.mark.parametrize("rate, mp3_s", [(44100, 59.037), (22050, 59.063)])
def test_duration_counts_the_padding_a_lossy_encoder_adds(tmp_path, rate, mp3_s):
    seg = AudioSegment.silent(59_000, frame_rate=rate)
    for fmt in ("wav", "flac", "aiff"):
        seg.export(tmp_path / f"a.{fmt}", format=fmt)
        assert duration_s(tmp_path / f"a.{fmt}") == pytest.approx(59.0, abs=0.001), fmt
    seg.export(tmp_path / "a.mp3", format="mp3", bitrate="320k")
    assert duration_s(str(tmp_path / "a.mp3")) == pytest.approx(mp3_s, abs=0.002)


def test_duration_is_none_for_what_is_not_audio(tmp_path):
    (tmp_path / "t.txt").write_text("hello")
    assert duration_s(str(tmp_path / "t.txt")) is None
    assert duration_s(str(tmp_path / "missing.wav")) is None


def test_bitrates_are_read_the_way_ffmpeg_reads_them():
    assert parse_bitrate("320k") == 320_000
    assert parse_bitrate("192K") == 192_000
    assert parse_bitrate("1.5m") == 1_500_000
    assert parse_bitrate("128000") == 128_000
    assert parse_bitrate("64 kbps") == 64_000
    for bad in ("fast", "", "k320", "12x"):
        with pytest.raises(ValueError, match="bitrate"):
            parse_bitrate(bad)


def test_the_engine_is_named_for_the_record():
    info = engine_info()
    assert info["pyav"] == av.__version__ and info["ffmpeg"]


@pytest.mark.parametrize("channels", [2, 3])
def test_a_long_file_is_mixed_down_without_copies_of_all_of_it(channels):
    """The temporaries of a downmix are floats or 64-bit integers per sample, four to eight
    times the size of the file when made for all of it at once (it was 8x for 5 minutes of
    stereo); they are made a block at a time."""
    frames = 44100 * 300  # long enough for a per-file copy to dwarf the fixed cost of a block
    rng = np.random.default_rng(0)
    data = rng.integers(-20000, 20000, size=frames * channels).astype(np.int16)
    audio = AudioSegment.from_samples(data, frame_rate=44100, channels=channels)
    tracemalloc.start()
    try:
        mono = audio.set_channels(1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(mono.raw_data) == frames * 2
    assert peak < 2 * len(audio.raw_data)
