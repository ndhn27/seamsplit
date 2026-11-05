"""``packaging/build_info.py`` reads the licence and the options of the FFmpeg that PyAV
carries; the release workflow attaches what it prints to every one-file program."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "packaging" / "build_info.py"


@pytest.fixture(scope="module")
def build_info():
    # loaded by path: "packaging" is also the name of a well-known third-party package
    spec = importlib.util.spec_from_file_location("seamsplit_build_info", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OPTIONS = (
    "--disable-static --enable-shared --prefix=/tmp/vendor --enable-version3 "
    "--enable-libx264 --enable-libx265"
)


def make_wheel_tree(tmp_path, libs_folder):
    av_dir = tmp_path / "av"
    folder = tmp_path / libs_folder if libs_folder == "av.libs" else av_dir / ".dylibs"
    folder.mkdir(parents=True)
    av_dir.mkdir(exist_ok=True)
    blob = b"junk\x00LGPL version 3 or later\x00" + OPTIONS.encode() + b"\x00more junk"
    (folder / "libavutil-1a2b3c.so.61").write_bytes(blob)
    (folder / "libx264-9f.so.165").write_bytes(b"x")
    return str(av_dir)


@pytest.mark.parametrize("libs_folder", ["av.libs", ".dylibs"])
def test_reads_the_licence_and_options_from_either_wheel_layout(build_info, tmp_path, libs_folder):
    av_dir = make_wheel_tree(tmp_path, libs_folder)
    licence, options = build_info.ffmpeg_licence_and_options(av_dir)
    assert licence == "LGPL version 3 or later"
    assert options == OPTIONS
    assert build_info.bundled_libraries(av_dir) == ["libavutil-1a2b3c.so.61", "libx264-9f.so.165"]


def test_a_pyav_without_bundled_libraries_is_reported_not_guessed(build_info, tmp_path):
    (tmp_path / "av").mkdir()
    assert build_info.ffmpeg_licence_and_options(str(tmp_path / "av")) == (None, None)
    assert build_info.bundled_libraries(str(tmp_path / "av")) == []


def test_describe_names_the_versions_of_the_installed_pyav(build_info):
    import av

    text = build_info.describe()
    assert f"PyAV {av.__version__}" in text
    assert f"FFmpeg {av.ffmpeg_version_info}" in text
    assert "FFmpeg licence, as its own build reports it:" in text
