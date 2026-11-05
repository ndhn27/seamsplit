"""Service presets: the built-in one, the user's own file, and how a preset and the
options given on the command line settle the length limit, format and overlap."""

import json
import os

import pytest

from fake_service import make_song, to_segment
from seamsplit.cli import main
from seamsplit.services import (
    BUILTIN_PRESETS,
    SERVICES_ENV,
    ServicePreset,
    apply_preset,
    available_presets,
    default_services_file,
    find_preset,
    load_user_presets,
)


@pytest.fixture(autouse=True)
def _no_user_presets(tmp_path, monkeypatch):
    """Whatever the machine running the tests has in its configuration folder."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv(SERVICES_ENV, raising=False)


def _write(tmp_path, data, name="services.json"):
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return str(path)


def test_xminus_is_built_in_with_the_limit_that_was_given():
    preset = find_preset("xminus")
    assert preset.limit_s == 59.0 and preset.builtin
    assert preset.format is None and preset.overlap_s is None
    assert "not checked" in preset.describe()  # it says it is a claim, not a measurement


def test_a_preset_is_found_whatever_the_case_and_an_unknown_one_lists_the_others(tmp_path):
    assert find_preset("  XMinus ").name == "xminus"
    with pytest.raises(ValueError, match=r"Unknown service 'nope' \(available: xminus\)"):
        find_preset("nope")
    path = _write(tmp_path, {"mine": {"limit_s": 30}})
    with pytest.raises(ValueError, match="available: mine, xminus"):
        find_preset("nope", path)


def test_the_users_file_adds_presets_and_may_replace_a_built_in_one(tmp_path):
    path = _write(
        tmp_path,
        {
            "Mine": {"limit_s": 120, "format": "WAV", "overlap_s": 5, "note": "my plan"},
            "xminus": {"limit_s": 30},
        },
    )
    presets = available_presets(path)
    mine = presets["mine"]
    assert (mine.limit_s, mine.format, mine.overlap_s, mine.note) == (120.0, "wav", 5.0, "my plan")
    assert presets["xminus"].limit_s == 30.0 and not presets["xminus"].builtin
    assert BUILTIN_PRESETS["xminus"].limit_s == 59.0  # the built-in one itself is untouched
    assert "limit 120s, format wav, overlap 5s — my plan" == mine.describe()
    assert ServicePreset("bare").describe() == "no settings"


def test_the_default_file_is_read_when_it_is_there_and_ignored_when_it_is_not(
    tmp_path, monkeypatch
):
    assert default_services_file() == str(tmp_path / "config" / "seamsplit" / "services.json")
    assert set(available_presets()) == {"xminus"}
    (tmp_path / "config" / "seamsplit").mkdir(parents=True)
    _write(tmp_path / "config" / "seamsplit", {"mine": {"limit_s": 45}})
    assert set(available_presets()) == {"xminus", "mine"}
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert default_services_file().replace(os.sep, "/").endswith(".config/seamsplit/services.json")


def test_a_file_that_was_named_must_exist(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        available_presets(str(tmp_path / "missing.json"))
    monkeypatch.setenv(SERVICES_ENV, str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        available_presets()
    path = _write(tmp_path, {"viaenv": {"limit_s": 20}})
    monkeypatch.setenv(SERVICES_ENV, path)
    assert "viaenv" in available_presets()


@pytest.mark.parametrize(
    "content, message",
    [
        ("{not json", "is not valid JSON"),
        ("[1, 2]", "must hold an object of presets, got array"),
        ({"a": 5}, r"service 'a' must be an object"),
        ({"a": {"limit": 5}}, "unknown keys limit"),
        ({"a": {"limit_s": "fast"}}, "'limit_s' must be a number"),
        ({"a": {"limit_s": -3}}, "'limit_s' must not be negative"),
        ({"a": {"limit_s": 0}}, "'limit_s' must be greater than 0"),
        ({"a": {"overlap_s": -1}}, "'overlap_s' must not be negative"),
        ({"a": {"format": 3}}, "'format' must be a file extension"),
        ({"a": {"format": ".wav!"}}, "'format' must be a file extension"),
        ({"a": {"note": 4}}, "'note' must be text"),
        ({"bad name": {}}, "is not a usable service name"),
    ],
)
def test_a_wrong_preset_file_says_what_is_wrong(tmp_path, content, message):
    with pytest.raises(ValueError, match=message):
        load_user_presets(_write(tmp_path, content))


def test_a_file_that_is_not_text_is_refused_clearly(tmp_path):
    path = tmp_path / "services.json"
    path.write_bytes(b"\xff\xfe\x00{")
    with pytest.raises(ValueError, match="is not UTF-8 text"):
        load_user_presets(str(path))


def test_what_is_given_wins_and_the_preset_fills_the_rest():
    preset = ServicePreset("p", limit_s=59.0, format="wav", overlap_s=3.0)
    filled = apply_preset(preset, target=None, limit=None, fmt=None, overlap=None)
    assert filled == {
        "limit": 59.0,
        "fmt": "wav",
        "overlap": 3.0,
        "notes": ["limit 59s", "format wav", "overlap 3s"],
    }
    given = apply_preset(preset, target=None, limit=40.0, fmt="flac", overlap=0.0)
    assert (given["limit"], given["fmt"], given["overlap"], given["notes"]) == (
        40.0,
        "flac",
        0.0,
        [],
    )


def test_without_a_preset_nothing_changes():
    out = apply_preset(None, target=50.0, limit=None, fmt=None, overlap=None)
    assert out == {"limit": None, "fmt": None, "overlap": None, "notes": []}


def test_a_preset_limit_cannot_be_combined_with_a_target_but_a_limit_can_replace_it():
    preset = ServicePreset("p", limit_s=59.0)
    with pytest.raises(ValueError, match="cannot be combined with --target"):
        apply_preset(preset, target=50.0, limit=None, fmt=None, overlap=None)
    # a preset with no limit has nothing to clash with
    free = apply_preset(ServicePreset("q"), target=50.0, limit=None, fmt=None, overlap=None)
    assert free["limit"] is None


# -- on the command line -----------------------------------------------------


@pytest.fixture
def song_file(tmp_path):
    path = tmp_path / "song.wav"
    to_segment(make_song(30.0)).export(str(path), format="wav")
    return str(path)


def _split(song_file, tmp_path, capsys, *extra, expect=0):
    rc = main(["split", song_file, "--dry-run", "--out", str(tmp_path / "out"), *extra])
    captured = capsys.readouterr()
    assert rc == expect, captured.err
    return captured


def test_service_sets_the_limit_and_the_target_follows_from_it(song_file, tmp_path, capsys):
    out = _split(song_file, tmp_path, capsys, "--service", "xminus").out
    assert "Service xminus: limit 59s." in out
    # 59 - after-slack 3 - overlap 4 - the 0.25 s an MP3 adds
    assert "Target chunk length: 51.75s (from --limit 59s" in out


def test_a_limit_given_beats_the_services_and_other_options_are_kept(song_file, tmp_path, capsys):
    out = _split(song_file, tmp_path, capsys, "--service", "xminus", "--limit", "40").out
    assert "Service xminus" not in out
    assert "Target chunk length: 32.75s (from --limit 40s" in out
    out = _split(song_file, tmp_path, capsys, "--service", "xminus", "--overlap", "2").out
    assert "overlap 2s" in out and "Target chunk length: 53.75s" in out


def test_a_service_with_a_format_and_overlap_fills_them_in(song_file, tmp_path, capsys):
    path = _write(tmp_path, {"mine": {"limit_s": 35, "format": "wav", "overlap_s": 1}})
    out = _split(song_file, tmp_path, capsys, "--service", "mine", "--services-file", path).out
    assert "Service mine: limit 35s, format wav, overlap 1s." in out
    assert "Target chunk length: 31s (from --limit 35s - after-slack 3s - overlap 1s)" in out


def test_service_and_target_do_not_go_together(song_file, tmp_path, capsys):
    err = _split(song_file, tmp_path, capsys, "--service", "xminus", "--target", "30", expect=1).err
    assert "cannot be combined with --target" in err


def test_an_unknown_service_is_an_error_that_lists_the_known_ones(song_file, tmp_path, capsys):
    err = _split(song_file, tmp_path, capsys, "--service", "nope", expect=1).err
    assert "Unknown service 'nope' (available: xminus)" in err


def test_without_a_service_the_defaults_are_what_they_were(song_file, tmp_path, capsys):
    out = _split(song_file, tmp_path, capsys).out
    assert "Service" not in out and "overlap=4.0s" in out


def test_the_services_command_lists_the_presets_and_where_yours_go(tmp_path, capsys):
    assert main(["services"]) == 0
    out = capsys.readouterr().out
    assert "xminus (built in): limit 59s" in out
    assert str(tmp_path / "config" / "seamsplit" / "services.json") in out
    path = _write(tmp_path, {"mine": {"limit_s": 25}})
    assert main(["services", "--services-file", path]) == 0
    out = capsys.readouterr().out
    assert "mine (yours): limit 25s" in out and path in out
