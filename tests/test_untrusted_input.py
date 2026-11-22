"""Files that come from someone else: what a name in a manifest can reach, what a file that
says it is audio can make FFmpeg open, and that a file that is replaced is replaced whole."""

import json
import ntpath
import os
import stat

import pytest
from seamsplit import audio_io
from seamsplit.audio import AudioSegment, CouldntDecodeError, CouldntEncodeError
from seamsplit.audio_io import atomic_write_text, duration_s, encode_audio, read_audio
from seamsplit.checks import file_name_problem
from seamsplit.gui import safe_name
from seamsplit.manifest import load_manifest

# -- a name in a manifest stays in its folder ------------------------------------------


def _manifest(tmp_path, *parts):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"parts": list(parts), "overlap_ms": 0}))
    return str(path)


def test_a_name_with_a_drive_leaves_the_folder_it_is_joined_to():
    """What this is about: on Windows the name replaces the folder."""
    assert ntpath.join(r"C:\work\chunks", "D:evil.flac") == "D:evil.flac"


@pytest.mark.parametrize("name", ["D:evil.flac", "d:evil", "C:song_part01.wav", "Z:"])
def test_a_manifest_naming_a_part_with_a_drive_is_refused_on_windows(tmp_path, monkeypatch, name):
    path = _manifest(tmp_path, "song_part01.wav", name)
    with monkeypatch.context() as patched:
        patched.setattr(os, "name", "nt")
        with pytest.raises(ValueError, match=r"parts\[1\]' must not begin with a drive"):
            load_manifest(path)
    # a plain file name where it is one: said for a system that is not Windows, because on
    # Windows (where this test also runs) it is not one
    with monkeypatch.context() as patched:
        patched.setattr(os, "name", "posix")
        assert load_manifest(path)["parts"][1] == name


@pytest.mark.parametrize("name", ["NUL.wav", "con", "COM1.flac", "lpt9", 'a"b.wav', "x<y.wav"])
def test_a_manifest_naming_a_part_that_windows_cannot_have_is_refused_there(
    tmp_path, monkeypatch, name
):
    path = _manifest(tmp_path, name)
    with monkeypatch.context() as patched:
        patched.setattr(os, "name", "nt")
        with pytest.raises(ValueError, match=r"parts\[0\]'"):
            load_manifest(path)


def test_a_song_whose_name_has_a_colon_still_splits_and_merges_where_that_is_a_name(
    tmp_path, monkeypatch
):
    """ "Artist: Title_part01.wav" is a fine file name on Linux and macOS."""
    path = _manifest(tmp_path, "Artist: Title_part01.wav", "Artist: Title_part02.wav")
    with monkeypatch.context() as patched:
        patched.setattr(os, "name", "posix")  # "where that is a name": not on a Windows host
        assert load_manifest(path)["parts"][0] == "Artist: Title_part01.wav"
    assert file_name_problem("Artist: Title_part01.wav", windows=False) is None
    assert file_name_problem("Artist: Title_part01.wav", windows=True)  # not on Windows


NAMES = [
    "a.wav",
    "song_part01.flac",
    "NUL.wav",
    "con",
    "a:b.wav",
    'q"r',
    "x<y",
    "COM1",
    "LPT\u00b9.t",
]


@pytest.mark.parametrize("name", NAMES)
def test_the_manifest_and_the_gui_agree_about_which_names_are_fine_on_windows(name):
    """One set of rules: the GUI makes a safe name of what it is sent, a manifest's names
    are checked against the same rules."""
    assert (file_name_problem(name, windows=True) is None) == (safe_name(name) == name)


# -- a file that says it is audio is not trusted to be only that --------------------------


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "sub").mkdir()
    tone = AudioSegment(bytes(8000 * 2 * 2), sample_width=2, frame_rate=8000, channels=1)
    tone.export(str(tmp_path / "other.wav"), format="wav")
    tone.export(str(tmp_path / "sub" / "inner.wav"), format="wav")
    return tmp_path


@pytest.mark.parametrize("target", ["other.wav", "sub/inner.wav"])
def test_a_text_file_called_mp3_does_not_decode_as_the_file_it_lists(folder, target):
    """FFmpeg takes it for an ffconcat playlist and decodes the file it names."""
    path = folder / "song.mp3"
    path.write_text(f"ffconcat version 1.0\nfile {target}\n")
    with pytest.raises(CouldntDecodeError, match="lists other files or addresses"):
        AudioSegment.from_file(str(path))
    assert duration_s(str(path)) is None


def test_a_playlist_is_refused_whatever_it_is_called(folder):
    for name in ("list.m3u8", "list.txt", "list.ffconcat", "list.wav", "list"):
        path = folder / name
        path.write_text("ffconcat version 1.0\nfile other.wav\n")
        with pytest.raises(CouldntDecodeError):
            read_audio(str(path))


def test_a_file_object_is_not_trusted_either(folder):
    import io

    playlist = io.BytesIO(b"ffconcat version 1.0\nfile other.wav\n")
    with pytest.raises(CouldntDecodeError):
        read_audio(playlist)


def test_the_formats_audio_comes_in_are_still_read(tmp_path):
    seg = AudioSegment(bytes(8000 * 2 * 2), sample_width=2, frame_rate=8000, channels=1)
    for fmt in ("wav", "flac", "mp3", "m4a", "aiff", "aac", "opus"):
        path = tmp_path / f"tone.{fmt}"
        seg.export(str(path), format=fmt)
        assert (
            abs(len(AudioSegment.from_file(str(path))) - 2000) < 300
        ), fmt  # lossy ones have padding
        assert duration_s(str(path)) is not None, fmt


def test_a_format_that_is_named_is_used_as_named(tmp_path):
    seg = AudioSegment(bytes(8000 * 2 * 2), sample_width=2, frame_rate=8000, channels=1)
    path = tmp_path / "audio.bin"
    seg.export(str(path), format="wav")
    assert len(AudioSegment.from_file(str(path), format="wav")) == 2000


def test_no_input_opens_a_network_protocol():
    with pytest.raises(Exception, match="whitelist"):
        audio_io._open_input("http://127.0.0.1:1/x.wav", None)
    with pytest.raises(Exception, match="whitelist"):
        audio_io._open_input("http://127.0.0.1:1/x.wav", "wav")  # a named format as well


def test_the_demuxers_that_follow_references_are_the_ones_left_out():
    allowed = audio_io._open_options(probing=True)["format_whitelist"].split(",")
    assert not set(allowed) & audio_io._LISTING_FORMATS
    assert {"wav", "mp3", "flac", "ogg", "mov", "matroska", "aiff", "aac"} <= set(allowed)
    assert "format_whitelist" not in audio_io._open_options(probing=False)


# -- a file that is replaced is replaced whole or not at all -------------------------------


def _tone(seconds=1, rate=8000):
    return AudioSegment(bytes(rate * 2 * seconds), sample_width=2, frame_rate=rate, channels=1)


def _leftovers(folder, keep):
    return sorted(set(os.listdir(folder)) - set(keep))


@pytest.mark.parametrize("fmt", ["wav", "flac", "mp3"])
def test_an_encoding_that_fails_leaves_the_old_file_as_it_was(tmp_path, monkeypatch, fmt):
    out = tmp_path / f"merged.{fmt}"
    out.write_bytes(b"the old result")
    if fmt == "wav":

        def breaks(f, *args):
            f.write(b"RIFF half of a new file")
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(audio_io, "_write_wav_to", breaks)
    else:

        def breaks(*args):
            raise ValueError("the encoder gave up")

        monkeypatch.setattr(audio_io, "_encoder_format", breaks)  # after the output is opened
    with pytest.raises(CouldntEncodeError):
        _tone().export(str(out), format=fmt)
    assert out.read_bytes() == b"the old result"
    assert _leftovers(tmp_path, [out.name]) == []  # and nothing of the new one


@pytest.mark.parametrize("fmt", ["flac", "mp3"])
def test_an_encoder_that_fails_after_it_has_written_part_of_the_file(tmp_path, monkeypatch, fmt):
    import numpy as np

    out = tmp_path / f"merged.{fmt}"
    out.write_bytes(b"the old result")
    calls = []
    real = np.ascontiguousarray

    def fails_on_the_second_block(array, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("the encoder gave up")
        return real(array, *args, **kwargs)

    monkeypatch.setattr(audio_io.np, "ascontiguousarray", fails_on_the_second_block)
    with pytest.raises(CouldntEncodeError):
        _tone(seconds=4, rate=44_100).export(str(out), format=fmt)  # three blocks of 65,536
    monkeypatch.undo()
    assert len(calls) == 2
    assert out.read_bytes() == b"the old result" and _leftovers(tmp_path, [out.name]) == []


def test_an_interrupt_leaves_the_old_file_as_it_was_too(tmp_path, monkeypatch):
    out = tmp_path / "merged.wav"
    out.write_bytes(b"the old result")

    def interrupted(f, *args):
        f.write(b"half")
        raise KeyboardInterrupt

    monkeypatch.setattr(audio_io, "_write_wav_to", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _tone().export(str(out), format="wav")
    assert out.read_bytes() == b"the old result" and _leftovers(tmp_path, [out.name]) == []


def test_the_new_file_is_written_next_to_the_old_one_under_a_hidden_name(tmp_path, monkeypatch):
    out = tmp_path / "merged.wav"
    seen = []
    real = audio_io._write_wav_to

    def spy(f, *args):
        seen.extend(os.listdir(tmp_path))
        real(f, *args)

    monkeypatch.setattr(audio_io, "_write_wav_to", spy)
    _tone().export(str(out), format="wav")
    (partial,) = seen
    assert partial.startswith(".") and partial.endswith(".part")  # hidden: no listing finds it
    assert os.listdir(tmp_path) == ["merged.wav"]


@pytest.mark.parametrize("fmt", ["wav", "flac", "raw"])
def test_a_file_that_is_written_properly_replaces_the_old_one(tmp_path, fmt):
    out = tmp_path / f"merged.{fmt}"
    out.write_bytes(b"old")
    _tone().export(str(out), format=fmt)
    assert out.stat().st_size > 100 and os.listdir(tmp_path) == [out.name]
    if fmt != "raw":
        assert len(AudioSegment.from_file(str(out))) == 1000


def test_the_new_file_has_the_permissions_a_file_made_by_open_has(tmp_path):
    plain = tmp_path / "plain"
    plain.write_bytes(b"x")
    out = tmp_path / "merged.wav"
    _tone().export(str(out), format="wav")
    assert stat.S_IMODE(out.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need rights on Windows")
def test_a_symbolic_link_is_written_through_and_stays_a_link(tmp_path):
    real = tmp_path / "real.wav"
    real.write_bytes(b"old")
    link = tmp_path / "link.wav"
    link.symlink_to(real)
    _tone().export(str(link), format="wav")
    assert link.is_symlink() and len(AudioSegment.from_file(str(real))) == 1000


@pytest.mark.skipif(not os.path.exists("/dev/null"), reason="no /dev/null")
def test_what_is_not_a_regular_file_is_written_to_as_it_is():
    _tone().export("/dev/null", format="wav")  # not replaced with a file in /dev


def test_a_file_object_is_written_to_as_before():
    import io

    buffer = io.BytesIO()
    _tone().export(buffer, format="wav")
    assert len(AudioSegment.from_file(io.BytesIO(buffer.getvalue()))) == 1000


def test_a_text_file_is_replaced_whole_too(tmp_path, monkeypatch):
    out = tmp_path / "report.md"
    out.write_text("old report")
    atomic_write_text(str(out), "new \u00e9 report")
    assert out.read_text(encoding="utf-8") == "new \u00e9 report" and os.listdir(tmp_path) == [
        "report.md"
    ]

    class Failing:
        def write(self, text):
            raise OSError(28, "No space left on device")

    real_open = open

    def opening(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if str(path).endswith(".part"):
            handle.close()
            return _Raising()
        return handle

    class _Raising:
        def __enter__(self):
            return Failing()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("builtins.open", opening)
    with pytest.raises(OSError):
        atomic_write_text(str(out), "half")
    monkeypatch.undo()
    assert out.read_text(encoding="utf-8") == "new \u00e9 report" and os.listdir(tmp_path) == [
        "report.md"
    ]


def test_encode_audio_to_a_path_in_a_folder_that_is_not_there_is_an_encoding_error(tmp_path):
    import numpy as np

    rows = np.zeros((100, 1), dtype=np.int16)
    with pytest.raises(CouldntEncodeError):
        encode_audio(str(tmp_path / "missing" / "x.flac"), "flac", rows, 2, 8000)
