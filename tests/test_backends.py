"""Running a separator that is installed on this computer on a chunk: the command lines of
the presets, the backend of one's own, and what happens to what the program writes. The
programs are stand-ins (``fake_programs``), so what is tested is SeamSplit's side of it."""

import json
import os
import shutil
import subprocess

import pytest

from fake_service import make_song, to_segment
from seamsplit.backends import (
    Backend,
    BackendError,
    BackendUnavailable,
    CustomBackendNotStored,
    backend_from_dict,
    backend_from_options,
    backend_stem_names,
    backend_to_dict,
    check_available,
    classify,
    custom_backend,
    fingerprint,
    preset_backend,
    separate_chunk,
    split_command,
    upgrade_legacy_backend,
)


@pytest.fixture
def chunk(tmp_path):
    folder = tmp_path / "chunks"
    folder.mkdir()
    path = folder / "song_part01.wav"
    to_segment(make_song(6.0)).export(str(path), format="wav")
    return str(path)


# -- what a backend is ---------------------------------------------------------


def test_demucs_is_asked_for_two_stems_into_the_output_folder():
    backend = preset_backend("demucs")
    assert backend.command == ("demucs", "--two-stems=vocals", "-o", "{output}", "{input}")
    assert backend.stem_names == ["vocals", "instrumental"] and backend.program == "demucs"
    assert preset_backend("Demucs", "mdx_extra").command[-3:] == ("-n", "mdx_extra", "{input}")


def test_audio_separator_is_the_backend_for_uvr_too():
    backend = preset_backend("audio-separator", "some_model.onnx")
    assert backend.command[0] == "audio-separator" and backend.command[1] == "{input}"
    assert "--model_filename" in backend.command and "some_model.onnx" in backend.command
    assert "--model_filename" not in preset_backend("uvr").command
    assert preset_backend("uvr").name == "audio-separator"


def test_an_unknown_backend_lists_the_ones_there_are():
    with pytest.raises(ValueError, match="Unknown backend 'spleeter' .*demucs, audio-separator"):
        preset_backend("spleeter")


def test_a_backend_of_ones_own_needs_the_places_for_input_and_output_and_the_stems():
    backend = custom_backend("mytool --in {input} --out {output}", ["Vocals=^v", "rest=^r"])
    assert backend.command == ("mytool", "--in", "{input}", "--out", "{output}")
    assert backend.stems == (("vocals", "^v"), ("rest", "^r"))
    for command, stems, message in [
        ("  ", ["a=b"], "is empty"),
        ("tool {input}", ["a=b"], r"needs \{input\}.*\{output\}"),
        ("tool {output}", ["a=b"], r"needs \{input\}.*\{output\}"),
        ("tool {input} {output}", [], "needs --backend-stem"),
        ("tool {input} {output}", ["novalue"], "must be STEM=REGEX"),
        ("tool {input} {output}", ["=x"], "must be STEM=REGEX"),
        ("tool {input} {output}", ["a="], "must be STEM=REGEX"),
        ("tool {input} {output}", ["a b=x"], "not a usable stem name"),
        ("tool {input} {output}", ["a=("], "not a regular expression"),
        ("tool {input} {output}", ["a=x", "A=y"], "names a stem twice"),
    ]:
        with pytest.raises(ValueError, match=message):
            custom_backend(command, stems)


def test_the_options_of_the_command_line_make_at_most_one_backend():
    assert backend_from_options(None, None, None, []) is None
    assert backend_from_options("demucs", "m", None, []).name == "demucs"
    assert backend_from_options(None, None, "t {input} {output}", ["a=b"]).name == "custom"
    for args, message in [
        (("demucs", None, "t {input} {output}", ["a=b"]), "replaces --backend"),
        ((None, "m", "t {input} {output}", ["a=b"]), "replaces --backend"),
        ((None, "m", None, []), "--backend-model needs --backend"),
        (("demucs", None, None, ["a=b"]), "--backend-stem goes with --backend-cmd"),
    ]:
        with pytest.raises(ValueError, match=message):
            backend_from_options(*args)


# -- the command line is read as the system reads one ---------------------------------


@pytest.mark.parametrize(
    "command, expected",
    [
        (
            r"C:\Tools\demucs\demucs.exe {input} {output}",
            [r"C:\Tools\demucs\demucs.exe", "{input}", "{output}"],
        ),
        (
            r'"C:\Program Files\Demucs\demucs.exe" -o "{output}" {input}',
            [r"C:\Program Files\Demucs\demucs.exe", "-o", "{output}", "{input}"],
        ),
        (r'tool.exe "C:\out dir\\" x', ["tool.exe", "C:\\out dir\\", "x"]),  # \\" is \ and the end
        ("tool\t  a   b ", ["tool", "a", "b"]),
        ("'not quotes' here", ["'not", "quotes'", "here"]),  # single quotes mean nothing
        ('a "" b', ["a", "", "b"]),  # an empty argument
        ('"a ""quoted"" b"', ['a "quoted" b']),
        # the examples Microsoft gives of how a command line is parsed
        ('"abc" d e', ["abc", "d", "e"]),
        (r"a\\\b d" + '"e f"g h', [r"a\\\b", "de fg", "h"]),
        (r"a\\\"b c d", [r"a\"b", "c", "d"]),
        (r'a\\"b c" d e', [r"a\b c", "d", "e"]),
        ("", []),
    ],
)
def test_a_windows_command_line_keeps_its_backslashes(command, expected):
    assert split_command(command, windows=True) == expected


@pytest.mark.parametrize(
    "arguments",
    [
        [r"C:\Program Files\tool.exe", r"C:\out dir\x.wav", "--flag=a b"],
        ["plain", "", 'with "quotes"', "trailing\\", "two\\\\", 'q\\"x', "tab\there"],
        [r"\\server\share\my song.wav", "a" * 3 + "\\" * 5 + '"'],
    ],
)
def test_what_subprocess_would_write_is_read_back_as_the_same_arguments(arguments):
    """The command the backend runs is a list, which Windows is given as one line made by
    ``list2cmdline``: reading a line the way Windows does must undo that."""
    assert split_command(subprocess.list2cmdline(arguments), windows=True) == arguments


def test_an_unclosed_quote_is_an_error_on_both_systems():
    for windows in (True, False):
        with pytest.raises(ValueError, match="No closing quotation"):
            split_command('tool "unclosed {input}', windows=windows)


def test_on_other_systems_a_command_line_is_read_as_a_posix_shell_reads_it():
    command = r"""mytool --in {input} 'a b' "c d" e\ f"""
    assert split_command(command, windows=False) == [
        "mytool", "--in", "{input}", "a b", "c d", "e f",
    ]  # fmt: skip
    assert split_command(r"C:\Tools\demucs.exe", windows=False) == ["C:Toolsdemucs.exe"]


def test_the_system_the_program_runs_on_decides_how_a_command_line_is_read(monkeypatch):
    command = r'"C:\Program Files\Demucs\demucs.exe" -o {output} {input}'
    with monkeypatch.context() as patched:
        patched.setattr(os, "name", "nt")
        windows = split_command(command)
        backend = custom_backend(command, ["vocals=^v"])
    assert windows == [r"C:\Program Files\Demucs\demucs.exe", "-o", "{output}", "{input}"]
    assert backend.command == tuple(windows) and backend.program == windows[0]
    unquoted = r"C:\Tools\demucs\demucs.exe -o {output} {input}"
    with monkeypatch.context() as patched:
        patched.setattr(os, "name", "posix")
        assert split_command(unquoted)[0] == "C:Toolsdemucsdemucs.exe"  # a shell eats them
    with monkeypatch.context() as patched:
        patched.setattr(os, "name", "nt")
        assert split_command(unquoted)[0] == r"C:\Tools\demucs\demucs.exe"


# -- what a job file records of a backend: never a command ----------------------


def test_a_preset_survives_being_written_to_a_job_file():
    for backend in (
        preset_backend("demucs"),
        preset_backend("demucs", "m"),
        preset_backend("uvr"),
        preset_backend("audio-separator", "UVR-MDX-NET-Voc_FT.onnx"),
    ):
        assert backend_from_dict(backend_to_dict(backend)) == backend
    assert backend_to_dict(preset_backend("demucs", "m")) == {"name": "demucs", "model": "m"}
    assert backend_to_dict(preset_backend("uvr")) == {"name": "audio-separator", "model": None}


def test_a_backend_of_ones_own_is_recorded_as_a_hash_and_cannot_be_read_back():
    command = "mytool --token hunter2 {input} {output}"
    backend = custom_backend(command, ["vocals=^v", "rest=^r"])
    data = backend_to_dict(backend)
    assert data == {
        "name": "custom",
        "stems": ["vocals", "rest"],
        "fingerprint": fingerprint(backend),
    }
    assert "mytool" not in json.dumps(data) and "hunter2" not in json.dumps(data)
    assert backend_stem_names(data) == ["vocals", "rest"]
    with pytest.raises(CustomBackendNotStored, match="--backend-cmd"):
        backend_from_dict(data)
    # the hash follows the command and the patterns, and nothing else
    assert fingerprint(custom_backend(command, ["vocals=^v", "rest=^r"])) == fingerprint(backend)
    assert fingerprint(custom_backend(command + " -x", ["vocals=^v", "rest=^r"])) != fingerprint(
        backend
    )
    assert fingerprint(custom_backend(command, ["vocals=^v", "rest=^x"])) != fingerprint(backend)


def test_a_job_file_cannot_say_what_a_preset_is_to_run():
    hostile = {
        "name": "demucs",
        "model": None,
        "command": ["sh", "-c", "id > PWNED.txt"],
        "stems": [["vocals", "x"]],
        "install_hint": "run this",
    }
    assert backend_from_dict(hostile) == preset_backend("demucs")
    assert backend_stem_names(hostile) == ["vocals", "instrumental"]


@pytest.mark.parametrize(
    "model",
    [
        "htdemucs_ft",
        "UVR-MDX-NET-Voc_FT.onnx",
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "UVR (v2).onnx",
    ],
)
def test_the_names_of_real_models_are_read_back(model):
    assert backend_from_dict({"name": "audio-separator", "model": model}).model == model


@pytest.mark.parametrize(
    "model",
    [
        "--output_dir",
        "-n",
        "../evil.ckpt",
        "/etc/passwd",
        "a\\b",
        "x\ny",
        " lead",
        "",
        "a" * 201,
        5,
        ["m"],
    ],
)
def test_a_model_that_is_not_a_name_is_not_passed_on(model):
    with pytest.raises(ValueError, match="not a model name"):
        backend_from_dict({"name": "demucs", "model": model})


@pytest.mark.parametrize(
    "data",
    [
        None,
        "demucs",
        [],
        {},
        {"name": "rm"},
        {"name": ["demucs"]},
        {"name": "uvr2"},
        {"command": ["demucs"], "stems": []},
    ],
)
def test_a_backend_that_is_none_of_the_presets_is_refused(data):
    with pytest.raises(ValueError, match="not valid"):
        backend_from_dict(data)
    with pytest.raises(ValueError, match="not valid"):
        backend_stem_names(data)


@pytest.mark.parametrize("stems", [None, [], "vocals", ["a/b"], ["../x"], ["A"], ["a", "a"], [1]])
def test_the_stems_of_a_job_file_are_names_or_nothing(stems):
    with pytest.raises(ValueError, match="stems of the backend"):
        backend_stem_names({"name": "custom", "stems": stems, "fingerprint": "x"})


def _written_by_1_1_1(backend):
    """What a job file made before the command stopped being recorded held."""
    return {
        "name": backend.name,
        "command": list(backend.command),
        "stems": [list(item) for item in backend.stems],
        "install_hint": backend.install_hint,
    }


def test_a_job_file_of_an_older_version_keeps_working_for_a_preset():
    for backend in (preset_backend("demucs", "m"), preset_backend("uvr"), preset_backend("demucs")):
        assert upgrade_legacy_backend(_written_by_1_1_1(backend)) == backend_to_dict(backend)


def test_an_older_job_file_whose_command_is_not_the_presets_is_not_run():
    edited = _written_by_1_1_1(preset_backend("demucs"))
    edited["command"] = ["sh", "-c", "id > PWNED.txt"]
    upgraded = upgrade_legacy_backend(edited)
    assert upgraded["name"] == "custom" and upgraded["stems"] == ["vocals", "instrumental"]
    with pytest.raises(CustomBackendNotStored):
        backend_from_dict(upgraded)
    # a changed stem pattern is a changed backend too
    other = _written_by_1_1_1(preset_backend("demucs"))
    other["stems"][0][1] = "."
    assert upgrade_legacy_backend(other)["name"] == "custom"
    # and the command a user gave once is recognised when it is given again
    own = custom_backend("t {input} {output}", ["a=b"])
    assert upgrade_legacy_backend(_written_by_1_1_1(own)) == backend_to_dict(own)


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {"command": [], "stems": []},
        {"command": "sh", "stems": []},
        {"command": ["sh", 1], "stems": []},
        {"command": ["sh"], "stems": "ab"},
        {"command": ["sh"], "stems": [["a"]]},
        {"command": ["sh"], "stems": [["a", 1]]},
    ],
)
def test_an_older_job_file_that_is_not_a_backend_is_refused(data):
    with pytest.raises(ValueError, match="not valid"):
        upgrade_legacy_backend(data)


def test_an_older_job_file_with_a_stem_that_is_not_a_name_is_refused():
    with pytest.raises(ValueError, match="stems of the backend"):
        upgrade_legacy_backend({"name": "x", "command": ["t"], "stems": [["../a", "b"]]})


def test_a_program_that_is_not_installed_says_how_to_get_it(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(BackendUnavailable, match="'demucs' was not found.*pip install demucs"):
        check_available(preset_backend("demucs"))
    with pytest.raises(BackendUnavailable) as caught:
        check_available(custom_backend("nothere {input} {output}", ["a=b"]))
    assert "was not found on this computer" in str(caught.value) and " — " not in str(caught.value)


def test_which_file_is_which_stem_is_decided_by_the_name_alone():
    demucs = preset_backend("demucs")
    files = ["/o/m/t/vocals.wav", "/o/m/t/no_vocals.wav", "/o/m/t/readme.wav"]
    assert classify(demucs, files) == {
        "vocals": "/o/m/t/vocals.wav",
        "instrumental": "/o/m/t/no_vocals.wav",
    }
    separator = preset_backend("uvr")
    named = ["x_(Vocals)_m.wav", "x_(Instrumental)_m.wav"]
    assert classify(separator, named) == {"vocals": named[0], "instrumental": named[1]}
    assert classify(separator, ["x_(No Vocals)_m.wav", "x_(Vocals)_m.wav"])["instrumental"]


def test_a_stem_that_is_not_found_exactly_once_lists_what_was_written():
    demucs = preset_backend("demucs")
    with pytest.raises(
        BackendError, match="wrote bass.wav, drums.wav: expected exactly one.*'vocals'"
    ):
        classify(demucs, ["/x/bass.wav", "/x/drums.wav"])
    with pytest.raises(BackendError, match="wrote no audio file"):
        classify(demucs, [])
    with pytest.raises(BackendError, match="found 2"):
        classify(demucs, ["/a/vocals.wav", "/b/vocals.flac", "/a/no_vocals.wav"])


# -- running one ---------------------------------------------------------------


def _listing(folder):
    return sorted(os.listdir(folder))


def test_a_chunk_is_separated_and_the_stems_land_next_to_each_other(fake_programs, chunk, tmp_path):
    processed = str(tmp_path / "processed")
    written = separate_chunk(preset_backend("demucs"), chunk, processed)
    assert _listing(processed) == ["song_part01_instrumental.wav", "song_part01_vocals.wav"]
    assert sorted(os.path.basename(p) for p in written) == _listing(processed)


def test_the_same_goes_for_the_naming_of_audio_separator(fake_programs, chunk, tmp_path):
    processed = str(tmp_path / "processed")
    separate_chunk(preset_backend("audio-separator", "model.v1"), chunk, processed)
    assert _listing(processed) == ["song_part01_instrumental.wav", "song_part01_vocals.wav"]


def test_a_backend_of_ones_own_runs_through_the_same_path(fake_programs, chunk, tmp_path):
    backend = custom_backend(
        "audio-separator {input} --output_dir {output}",
        [r"voice=\(vocals\)", r"backing=\(instrumental\)"],
    )
    processed = str(tmp_path / "processed")
    separate_chunk(backend, chunk, processed)
    assert _listing(processed) == ["song_part01_backing.wav", "song_part01_voice.wav"]


def test_a_failing_program_leaves_nothing_behind_and_its_last_words_are_kept(
    fake_programs, chunk, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_FAIL_ON", "part01")
    processed = str(tmp_path / "processed")
    with pytest.raises(BackendError, match="demucs exited with status 1:\nCUDA out of memory"):
        separate_chunk(preset_backend("demucs"), chunk, processed)
    assert _listing(processed) == []  # not even the work folder


def test_output_that_is_not_the_stems_is_an_error_that_says_what_was_written(
    fake_programs, chunk, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_BAD_OUTPUT", "1")
    processed = str(tmp_path / "processed")
    with pytest.raises(BackendError, match="demucs wrote something_else.wav"):
        separate_chunk(preset_backend("demucs"), chunk, processed)
    assert _listing(processed) == []


def test_a_program_that_takes_too_long_is_given_up_on(fake_programs, chunk, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "30")
    with pytest.raises(BackendError, match="took more than 0.5s"):
        separate_chunk(preset_backend("demucs"), chunk, str(tmp_path / "p"), timeout_s=0.5)


def test_a_program_that_cannot_be_started_is_unavailable_not_a_failure_of_the_chunk(
    chunk, tmp_path, monkeypatch
):
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(BackendUnavailable, match="'demucs' could not be started"):
        separate_chunk(preset_backend("demucs"), chunk, str(tmp_path / "p"))
    # a file that is there but cannot be run
    script = tmp_path / "notexec"
    script.write_text("not a program")
    script.chmod(0o755)
    backend = Backend("odd", (str(script), "{input}", "{output}"), (("a", "x"),))
    with pytest.raises(BackendUnavailable, match="could not be started"):
        separate_chunk(backend, chunk, str(tmp_path / "q"))


def test_stems_that_were_there_are_replaced_by_a_new_run(fake_programs, chunk, tmp_path):
    processed = str(tmp_path / "processed")
    separate_chunk(preset_backend("demucs"), chunk, processed)
    target = os.path.join(processed, "song_part01_vocals.wav")
    with open(target, "wb") as f:
        f.write(b"stale")
    separate_chunk(preset_backend("demucs"), chunk, processed)
    assert os.path.getsize(target) > 1000
    assert shutil.which("demucs")  # the fixture's stand-in, still the one found
