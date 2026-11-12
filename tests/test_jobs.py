"""A job folder: splitting once, separating on this computer or waiting for results put there
by hand, resuming, merging; and the ``run`` and ``status`` commands on top of it."""

import json
import os

import numpy as np
import pytest
from seamsplit.audio import AudioSegment

from fake_service import SR, hand_back, make_song, to_array, to_segment
from seamsplit.backends import BackendUnavailable, preset_backend
from seamsplit.cli import main
from seamsplit.job import (
    JOB_FILE,
    Job,
    job_stems,
    load_job,
    main_and_other,
    run_job,
    save_job,
)
from seamsplit.splitter import SplitConfig


@pytest.fixture
def song(tmp_path):
    path = tmp_path / "My Song.wav"
    to_segment(make_song(50.0)).export(str(path), format="wav")
    return str(path)


def _config(chunks):
    config = SplitConfig(
        target_s=15,
        before_slack_s=8,
        after_slack_s=3,
        overlap_s=2,
        fade_ms=0,
        output_dir=chunks,
        export_format="wav",
    )
    return config, {"service": "test"}


def _run(folder, song, **options):
    messages = []
    options.setdefault("out_format", "wav")
    options.setdefault("merge_options", {"samplerate": SR})
    result = run_job(folder, song, _config, say=messages.append, **options)
    return result, messages


def _hand_back(folder, stems=("instrumental", "vocals")):
    chunks = os.path.join(folder, "chunks")
    with open(os.path.join(chunks, "manifest.json"), encoding="utf-8") as f:
        parts = json.load(f)["parts"]
    n = len(parts)
    hand_back(
        chunks,
        os.path.join(folder, "processed"),
        parts,
        [30, -50, 0][:n],
        [0.0] * n,
        np.random.default_rng(1),
        stems,
    )


# -- without a backend: split, wait, merge --------------------------------------


def test_a_new_job_splits_the_song_and_waits_for_the_results(tmp_path, song):
    folder = str(tmp_path / "job")
    result, messages = _run(folder, song, match="instrumental")
    assert result.split_now and not result.done and result.merged == []
    assert result.status.pending_parts() == [1, 2, 3]
    assert os.path.isfile(os.path.join(folder, "chunks", "manifest.json"))
    assert os.path.isdir(os.path.join(folder, "processed"))
    assert messages[0].startswith("Splitting ") and "3 chunk(s) written" in messages[1]
    job = load_job(folder)
    assert job.input_file == os.path.abspath(song) and job.backend is None
    assert job.split["service"] == "test" and job.split["target_s"] == 15
    assert job.split["overlap_s"] == 2 and job.split["format"] == "wav"


def test_when_the_results_are_there_the_job_merges_and_does_not_split_again(tmp_path, song):
    folder = str(tmp_path / "job")
    _run(folder, song, match="instrumental", other_match="vocals")
    chunks = os.path.join(folder, "chunks")
    stamp = os.path.getmtime(os.path.join(chunks, "manifest.json"))
    _hand_back(folder)
    result, messages = _run(folder, None, match="instrumental", other_match="vocals")
    assert not result.split_now and result.done
    names = sorted(os.path.basename(p) for p in result.merged)
    assert names == ["My_Song_instrumental.wav", "My_Song_vocals.wav"]
    assert all(os.path.isfile(p) for p in result.merged)
    assert os.path.dirname(result.merged[0]) == os.path.join(folder, "output")
    assert os.path.getmtime(os.path.join(chunks, "manifest.json")) == stamp
    assert any("merging into" in m for m in messages)
    assert result.report is not None
    merged = to_array(AudioSegment.from_file(result.merged[0]))
    assert abs(merged.shape[1] / SR - 50.0) < 0.2


def test_one_stem_only_goes_to_one_file_and_the_output_can_go_elsewhere(tmp_path, song):
    folder = str(tmp_path / "job")
    _run(folder, song, match="instrumental")
    _hand_back(folder, stems=("instrumental",))
    elsewhere = str(tmp_path / "elsewhere")
    result, _ = _run(folder, None, match="instrumental", out_dir=elsewhere, out_format="flac")
    assert [os.path.basename(p) for p in result.merged] == ["My_Song_instrumental.flac"]
    assert os.listdir(elsewhere) == ["My_Song_instrumental.flac"]


def test_with_no_stem_words_the_files_are_one_per_chunk(tmp_path, song):
    folder = str(tmp_path / "job")
    _run(folder, song)
    _hand_back(folder, stems=("instrumental",))
    for name in os.listdir(os.path.join(folder, "processed")):
        base = name.replace("_instrumental", "")
        os.rename(os.path.join(folder, "processed", name), os.path.join(folder, "processed", base))
    result, _ = _run(folder, None)
    assert [os.path.basename(p) for p in result.merged] == ["My_Song_merged.wav"]


def test_a_partly_filled_job_says_what_is_still_missing(tmp_path, song):
    folder = str(tmp_path / "job")
    _run(folder, song, match="instrumental")
    _hand_back(folder, stems=("instrumental",))
    os.remove(os.path.join(folder, "processed", "My Song_part02_instrumental.wav"))
    result, messages = _run(folder, None, match="instrumental")
    assert not result.done and result.status.pending_parts() == [2]
    assert not any("merging" in m for m in messages)


def test_a_folder_that_lost_its_job_file_is_not_taken_over(tmp_path, song):
    folder = str(tmp_path / "job")
    _run(folder, song, match="instrumental")
    os.remove(os.path.join(folder, JOB_FILE))
    with pytest.raises(ValueError, match="has no job in it"):
        _run(folder, None, match="instrumental")
    with pytest.raises(ValueError, match="already holds files but there is no job.json"):
        _run(folder, song, match="instrumental")


# -- what a job refuses ----------------------------------------------------------


def test_another_song_cannot_use_the_folder_of_a_job(tmp_path, song):
    folder = str(tmp_path / "job")
    _run(folder, song, match="instrumental")
    other = tmp_path / "other.wav"
    to_segment(make_song(20.0)).export(str(other), format="wav")
    with pytest.raises(ValueError, match="is the job of .*My Song.wav.*not of .*other.wav"):
        _run(folder, str(other), match="instrumental")


def test_a_job_needs_a_song_to_start_from(tmp_path):
    with pytest.raises(ValueError, match="has no job in it: name the audio file"):
        _run(str(tmp_path / "empty"), None)
    with pytest.raises(FileNotFoundError, match="No such file"):
        _run(str(tmp_path / "empty"), str(tmp_path / "missing.wav"))


class _Killed(BaseException):
    """What a kill -9 looks like from the files: the program is simply gone."""


@pytest.fixture
def killed_in_the_split(monkeypatch):
    """The next split writes some chunks and dies before ``manifest.json``."""

    def dying(input_file, config, **kwargs):
        os.makedirs(config.output_dir, exist_ok=True)
        for n in (1, 2, 3, 99):  # 99: a chunk a split with other settings would not make
            with open(os.path.join(config.output_dir, f"My Song_part{n:02d}.wav"), "wb") as f:
                f.write(b"half a chunk")
        raise _Killed

    import seamsplit.job as job_module

    monkeypatch.setattr(job_module, "split_audio", dying)
    return monkeypatch


def test_job_json_is_there_before_the_split_so_a_killed_split_can_be_resumed(
    tmp_path, song, killed_in_the_split
):
    folder = str(tmp_path / "job")
    with pytest.raises(_Killed):
        _run(folder, song)
    chunks = os.path.join(folder, "chunks")
    assert os.listdir(chunks) and not os.path.exists(os.path.join(chunks, "manifest.json"))
    job = load_job(folder)  # written before the split started
    assert job is not None and job.input_file == os.path.abspath(song)
    assert job.split["target_s"] == 15

    killed_in_the_split.undo()  # the next run is a normal one
    result, messages = _run(folder, None)  # no need to name the file again

    assert any("interrupted" in m for m in messages)
    assert result.split_now and not result.status.complete  # waiting for the results, as usual
    names = sorted(n for n in os.listdir(chunks) if n.endswith(".wav"))
    with open(os.path.join(chunks, "manifest.json"), encoding="utf-8") as f:
        assert names == sorted(json.load(f)["parts"])  # the dead chunks (part99) are gone
    assert not [n for n in os.listdir(chunks) if n.endswith(".tmp")]

    _hand_back(folder, stems=("instrumental",))  # and the rest of the job works
    assert _run(folder, song, match="instrumental")[0].done


def test_a_killed_split_can_be_started_over_with_another_song(tmp_path, song, killed_in_the_split):
    folder = str(tmp_path / "job")
    with pytest.raises(_Killed):
        _run(folder, song)
    other = tmp_path / "Other.wav"
    to_segment(make_song(40.0)).export(str(other), format="wav")

    killed_in_the_split.undo()
    result, _ = _run(folder, str(other))

    assert result.split_now
    assert load_job(folder).input_file == os.path.abspath(str(other))


def test_what_a_killed_split_did_not_make_is_not_deleted(tmp_path, song, killed_in_the_split):
    folder = str(tmp_path / "job")
    with pytest.raises(_Killed):
        _run(folder, song)
    notes = os.path.join(folder, "chunks", "notes.txt")
    with open(notes, "w", encoding="utf-8") as f:
        f.write("mine")

    killed_in_the_split.undo()
    with pytest.raises(ValueError, match="already holds files"):
        _run(folder, song)
    assert os.path.isfile(notes)


def test_status_of_a_job_whose_split_was_killed_says_how_to_go_on(
    tmp_path, song, killed_in_the_split, capsys
):
    folder = str(tmp_path / "job")
    with pytest.raises(_Killed):
        _run(folder, song)
    killed_in_the_split.undo()

    assert main(["status", folder]) != 0
    assert "interrupted" in capsys.readouterr().err


def test_a_finished_split_leaves_no_temporary_manifest(tmp_path, song):
    folder = str(tmp_path / "job")
    _run(folder, song)
    assert sorted(n for n in os.listdir(os.path.join(folder, "chunks")) if "manifest" in n) == [
        "manifest.json"
    ]


def test_a_job_whose_song_was_moved_cannot_be_split_but_can_be_merged(tmp_path, song):
    folder = str(tmp_path / "job")
    _run(folder, song, match="instrumental")
    _hand_back(folder, stems=("instrumental",))
    os.rename(song, song + ".moved")
    result, _ = _run(folder, None, match="instrumental")
    assert result.done  # the chunks are all it needs

    lost = str(tmp_path / "lost")
    _run(lost, song + ".moved", match="instrumental")
    for name in os.listdir(os.path.join(lost, "chunks")):
        os.remove(os.path.join(lost, "chunks", name))
    os.rename(song + ".moved", song)
    os.remove(song)
    with pytest.raises(FileNotFoundError, match="audio file is gone"):
        _run(lost, None, match="instrumental")


# -- with a backend --------------------------------------------------------------


def test_a_backend_separates_every_chunk_and_the_job_merges(fake_programs, tmp_path, song):
    folder = str(tmp_path / "job")
    result, messages = _run(folder, song, backend=preset_backend("demucs"))
    assert result.done and result.separated == 3 and result.failures == {}
    assert [os.path.basename(p) for p in result.merged] == [
        "My_Song_instrumental.wav",
        "My_Song_vocals.wav",
    ]
    assert sum("demucs on " in m for m in messages) == 3
    assert load_job(folder).backend["name"] == "demucs"
    assert result.status.complete


def test_a_run_that_was_stopped_carries_on_with_the_chunks_that_are_not_done(
    fake_programs, tmp_path, song, monkeypatch
):
    folder = str(tmp_path / "job")
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_FAIL_ON", "part02")
    result, messages = _run(folder, song, backend=preset_backend("demucs"))
    assert not result.done and result.separated == 2
    assert list(result.failures) == ["My Song_part02.wav"]
    assert "CUDA out of memory" not in messages[-1] or "failed" in messages[-1]
    assert result.status.pending_parts() == [2]
    assert not os.path.exists(os.path.join(folder, "output"))

    monkeypatch.delenv("FAKE_FAIL_ON")
    log.write_text("")
    result, _ = _run(folder, None)  # the backend is the job's own
    assert result.done and result.separated == 1
    assert log.read_text().split("\n")[:-1] == ["My Song_part02.wav"]


def test_a_stem_that_was_lost_is_made_again_and_only_that_chunk(
    fake_programs, tmp_path, song, monkeypatch
):
    folder = str(tmp_path / "job")
    _run(folder, song, backend=preset_backend("demucs"))
    os.remove(os.path.join(folder, "processed", "My Song_part03_vocals.wav"))
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LOG", str(log))
    result, _ = _run(folder, None)
    assert result.separated == 1 and result.done
    assert log.read_text().split("\n")[:-1] == ["My Song_part03.wav"]


def test_naming_the_same_backend_again_is_fine_and_another_is_not(fake_programs, tmp_path, song):
    folder = str(tmp_path / "job")
    _run(folder, song, backend=preset_backend("demucs"))
    _run(folder, song, backend=preset_backend("demucs"))
    with pytest.raises(ValueError, match="another backend or model"):
        _run(folder, song, backend=preset_backend("demucs", "other_model"))
    plain = str(tmp_path / "plain")
    _run(plain, song, match="instrumental")
    with pytest.raises(ValueError, match="started without a backend"):
        _run(plain, song, backend=preset_backend("demucs"))


def test_a_backend_that_is_not_installed_stops_the_job_before_the_split(
    tmp_path, song, monkeypatch
):
    monkeypatch.setenv("PATH", "/nonexistent")
    folder = str(tmp_path / "job")
    with pytest.raises(BackendUnavailable, match="'demucs' was not found"):
        _run(folder, song, backend=preset_backend("demucs"))
    assert not os.path.exists(os.path.join(folder, "chunks"))


# -- the job file and the stems ----------------------------------------------------


def test_the_job_file_is_read_back_as_it_was_written(tmp_path):
    job = Job("/a/song.wav", 123, {"service": "x"}, {"name": "demucs"})
    save_job(str(tmp_path), job)
    assert load_job(str(tmp_path)) == job
    assert not (tmp_path / "job.json.tmp").exists()
    assert load_job(str(tmp_path / "nowhere")) is None


@pytest.mark.parametrize(
    "content, message",
    [
        ("{nope", "is not valid JSON"),
        ("[]", "is not a seamsplit job file"),
        ('{"seamsplit_job": 99}', "is not a seamsplit job file"),
        ('{"seamsplit_job": 1, "input_file": 3}', "'input_file' must be a file name"),
        ('{"seamsplit_job": 1, "input_file": "a", "backend": 5}', "'backend' must be an object"),
    ],
)
def test_a_job_file_that_is_not_one_is_refused_clearly(tmp_path, content, message):
    (tmp_path / JOB_FILE).write_text(content)
    with pytest.raises(ValueError, match=message):
        load_job(str(tmp_path))


def test_a_job_file_with_missing_extras_still_loads(tmp_path):
    (tmp_path / JOB_FILE).write_text('{"seamsplit_job": 1, "input_file": "a.wav", "split": 4}')
    job = load_job(str(tmp_path))
    assert (job.input_file, job.input_size, job.split, job.backend) == ("a.wav", None, {}, None)


def test_the_stems_wanted_come_from_the_backend_or_the_words_given():
    plain = Job("a", None)
    assert job_stems(plain, None, None) == [None]
    assert job_stems(plain, "instrumental", None) == ["instrumental"]
    assert job_stems(plain, "instrumental", "vocals") == ["instrumental", "vocals"]
    from seamsplit.backends import backend_to_dict

    local = Job("a", None, backend=backend_to_dict(preset_backend("demucs")))
    assert job_stems(local, "ignored", None) == ["vocals", "instrumental"]


def test_the_main_stem_is_the_instrumental_when_there_is_one():
    assert main_and_other([None]) == (None, None)
    assert main_and_other(["drums"]) == ("drums", None)
    assert main_and_other(["vocals", "instrumental"]) == ("instrumental", "vocals")
    assert main_and_other(["bass", "drums"]) == ("bass", "drums")
    with pytest.raises(ValueError, match="3 stems are wanted"):
        main_and_other(["a", "b", "c"])


# -- the commands -------------------------------------------------------------------


def _cli(capsys, *args, expect=0):
    rc = main(list(args))
    captured = capsys.readouterr()
    assert rc == expect, captured.err + captured.out
    return captured


SPLIT = ("--target", "15", "--before-slack", "8", "--after-slack", "3", "--overlap", "2")


def test_run_with_a_backend_goes_from_the_song_to_the_merged_stems(
    fake_programs, tmp_path, song, capsys
):
    out = _cli(capsys, "run", song, "--backend", "demucs", *SPLIT, "--samplerate", str(SR)).out
    folder = os.path.splitext(song)[0] + ".job"
    assert "Splitting" in out and "[3/3] demucs on" in out and "Done. Merged:" in out
    assert sorted(os.listdir(os.path.join(folder, "output"))) == [
        "My_Song_instrumental.wav",
        "My_Song_vocals.wav",
    ]
    again = _cli(capsys, "run", "--job", folder, "--samplerate", str(SR)).out
    assert "Splitting" not in again and "demucs on" not in again and "Done. Merged:" in again


def test_run_without_a_backend_waits_with_status_3_and_says_what_to_do(tmp_path, song, capsys):
    folder = str(tmp_path / "j")
    first = _cli(
        capsys, "run", song, "--job", folder, "--match", "instrumental", *SPLIT, expect=3
    ).out
    assert "Not complete: 3 of 3 chunk(s) still lack a result." in first
    # the folder is written with the system's separator; compare both sides with "/"
    wanted = f"put what it returns in {folder}/processed".replace(os.sep, "/")
    assert wanted in first.replace(os.sep, "/")
    assert "seamsplit status" in first
    _hand_back(folder, stems=("instrumental",))
    done = _cli(
        capsys,
        "run",
        "--job",
        folder,
        "--match",
        "instrumental",
        "--samplerate",
        str(SR),
        "--report",
        str(tmp_path / "r.json"),
    )
    assert "Done. Merged:" in done.out and "Seam report:" in done.out
    assert json.loads((tmp_path / "r.json").read_text())["boundaries"]


def test_run_with_a_service_records_it_and_uses_its_limit(tmp_path, song, capsys):
    folder = str(tmp_path / "j")
    out = _cli(capsys, "run", song, "--job", folder, "--service", "xminus", expect=3).out
    assert "Service xminus: limit 59s." in out
    job = load_job(folder)
    assert job.split["service"] == "xminus" and job.split["limit_s"] == 59.0
    assert job.split["target_s"] == pytest.approx(52.0)  # flac by default: no encoder margin


def test_a_local_backend_cuts_long_chunks_in_wav_unless_told_otherwise(
    fake_programs, tmp_path, song, capsys
):
    folder = str(tmp_path / "j")
    _cli(capsys, "run", song, "--job", folder, "--backend", "demucs", "--samplerate", str(SR))
    job = load_job(folder)
    assert job.split["target_s"] == 120.0 and job.split["format"] == "wav"
    assert job.split["limit_s"] is None
    kept = str(tmp_path / "k")
    _cli(
        capsys, "run", song, "--job", kept, "--backend", "demucs", "--limit", "40",
        "--format", "flac", "--samplerate", str(SR),
    )  # fmt: skip
    assert load_job(kept).split["format"] == "flac"
    assert load_job(kept).split["limit_s"] == 40.0


def test_run_tells_which_chunks_failed_and_waits(
    fake_programs, tmp_path, song, capsys, monkeypatch
):
    monkeypatch.setenv("FAKE_FAIL_ON", "part01")
    folder = str(tmp_path / "j")
    captured = _cli(capsys, "run", song, "--job", folder, "--backend", "demucs", *SPLIT, expect=3)
    assert "1 chunk(s) failed:" in captured.err and "part01.wav: demucs exited" in captured.err
    assert "Run the command again to retry them." in captured.err
    assert "Send the files" not in captured.out  # a backend job does not wait for uploads


def test_run_with_a_command_of_ones_own(fake_programs, tmp_path, song, capsys):
    folder = str(tmp_path / "j")
    _cli(
        capsys, "run", song, "--job", folder, *SPLIT, "--samplerate", str(SR),
        "--backend-cmd", "audio-separator {input} --output_dir {output}",
        "--backend-stem", r"vocals=\(vocals\)", "--backend-stem", r"instrumental=\(instrumental\)",
    )  # fmt: skip
    assert len(os.listdir(os.path.join(folder, "output"))) == 2


@pytest.mark.parametrize(
    "extra, message",
    [
        (("--backend-timeout", "5"), "--backend-timeout goes with --backend"),
        (("--backend", "demucs", "--backend-cmd", "x {input} {output}"), "replaces --backend"),
        (("--backend", "nope"), "Unknown backend 'nope'"),
        (("--service", "nope"), "Unknown service 'nope'"),
    ],
)
def test_run_refuses_options_that_do_not_fit_together(tmp_path, song, capsys, extra, message):
    err = _cli(capsys, "run", song, "--job", str(tmp_path / "j"), *extra, expect=1).err
    assert message in err


def test_run_needs_a_job_folder_or_a_song(capsys):
    err = _cli(capsys, "run", expect=1).err
    assert "name the job folder with --job, or the audio file" in err


def test_status_of_a_job_lists_what_is_missing_and_exits_3(tmp_path, song, capsys):
    folder = str(tmp_path / "j")
    _cli(capsys, "run", song, "--job", folder, "--match", "instrumental", *SPLIT, expect=3)
    out = _cli(capsys, "status", folder, "--match", "instrumental", expect=3).out
    assert "0 of 3 chunk(s) are complete." in out and "MISSING" in out
    _hand_back(folder, stems=("instrumental",))
    complete = _cli(capsys, "status", folder, "--match", "instrumental").out
    assert "3 of 3 chunk(s) are complete." in complete


def test_status_can_be_read_by_a_program(tmp_path, song, capsys):
    folder = str(tmp_path / "j")
    _cli(capsys, "run", song, "--job", folder, "--match", "instrumental", *SPLIT, expect=3)
    data = json.loads(
        _cli(capsys, "status", folder, "--match", "instrumental", "--json", expect=3).out
    )
    assert data["complete"] is False and data["pending_parts"] == [1, 2, 3]


def test_status_of_a_job_with_a_backend_knows_its_stems(fake_programs, tmp_path, song, capsys):
    folder = str(tmp_path / "j")
    _cli(capsys, "run", song, "--job", folder, "--backend", "demucs", "--samplerate", str(SR))
    out = _cli(capsys, "status", folder).out
    assert "[vocals]" in out and "[instrumental]" in out


def test_status_of_a_split_folder_needs_to_be_told_where_the_results_are(service, capsys):
    err = _cli(capsys, "status", service.chunks_dir, expect=1).err
    assert "give the folder of the results with --processed" in err
    out = _cli(
        capsys, "status", service.chunks_dir, "--processed", service.processed_dir,
        "--match", "instrumental",
    ).out  # fmt: skip
    assert "3 of 3 chunk(s) are complete." in out


def test_status_of_something_that_is_not_a_job_or_a_split_says_so(tmp_path, capsys):
    err = _cli(capsys, "status", str(tmp_path), expect=1).err
    assert "neither a job folder (no job.json) nor a split folder (no manifest.json)" in err


def test_run_puts_results_with_lost_names_in_order_and_says_so(tmp_path, song, capsys):
    folder = str(tmp_path / "j")
    _cli(capsys, "run", song, "--job", folder, "--match", "instrumental", *SPLIT, expect=3)
    _hand_back(folder, stems=("instrumental",))
    processed = os.path.join(folder, "processed")
    names = sorted(os.listdir(processed))
    # the service renamed them and (it happens) gave them back in the wrong order
    for k, name in enumerate(reversed(names)):
        os.rename(os.path.join(processed, name), os.path.join(processed, f"tmp{k}"))
    for k in range(len(names)):
        os.rename(
            os.path.join(processed, f"tmp{k}"), os.path.join(processed, f"instrumental {k}.wav")
        )
    captured = _cli(
        capsys, "run", "--job", folder, "--match", "instrumental", "--samplerate", str(SR)
    )
    assert "processed file(s) were not in part order" in captured.err
    assert "Done. Merged:" in captured.out


def test_the_markdown_report_has_the_notes_too():
    from seamsplit.report import MergeReport

    report = MergeReport(notes=["all went well"], warnings=["something did not"])
    text = report.to_markdown()
    assert "## Warnings\n\n- something did not" in text and "## Notes\n\n- all went well" in text
