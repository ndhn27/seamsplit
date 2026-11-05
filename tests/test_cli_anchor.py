"""The command line side of the anchored merge, the listening kit and the answer summary."""

import json
import os

import pytest

from fake_service import SR
from seamsplit.audition import KIT_FILE, KIT_PAGE
from seamsplit.choices import CHOICES_FILE, new_choices, record_choice, save_choices
from seamsplit.cli import main


def _merge(service, tmp_path, capsys, *extra, expect=0):
    args = [
        "merge",
        service.processed_dir,
        "--manifest",
        service.manifest_path,
        "--out",
        str(tmp_path / "out.wav"),
        "--match",
        "instrumental",
        "--samplerate",
        str(SR),
        *extra,
    ]
    rc = main(args)
    captured = capsys.readouterr()
    assert rc == expect, captured.err
    return captured


def test_the_anchored_merge_says_what_it_found_and_where_the_second_stem_went(
    service, tmp_path, capsys
):
    out = _merge(service, tmp_path, capsys, "--anchor", "--other-match", "vocals").out
    assert "Anchored 3 of 3 chunk(s) to the originals in time; largest offset" in out
    assert "Took the level of 3 chunk(s) from the originals" in out
    assert "The two stems add up to the original to" in out and "dB at worst" in out
    assert f"Second stem: {tmp_path / 'out_vocals.wav'}" in out
    assert sorted(os.listdir(tmp_path)) == ["out.wav", "out_vocals.wav"]


def test_the_second_stem_can_be_sent_elsewhere(service, tmp_path, capsys):
    other = tmp_path / "the_voice.wav"
    out = _merge(
        service, tmp_path, capsys, "--anchor", "--other-match", "vocals", "--other-out", str(other)
    ).out
    assert f"Second stem: {other}" in out and other.exists()


def test_with_one_stem_only_the_timing_is_anchored_and_the_user_is_told_why(
    service, tmp_path, capsys
):
    out = _merge(service, tmp_path, capsys, "--anchor").out
    assert "to the originals in time" in out
    assert "The level was not taken from the originals: one stem alone is quieter" in out
    assert "Second stem" not in out


def test_originals_can_be_named_and_a_missing_set_is_an_error(service, tmp_path, capsys):
    out = _merge(service, tmp_path, capsys, "--anchor", "--originals", service.chunks_dir).out
    assert "to the originals in time" in out
    empty = tmp_path / "nowhere"
    empty.mkdir()
    captured = _merge(service, tmp_path, capsys, "--anchor", "--originals", str(empty), expect=1)
    assert "Error:" in captured.err and "3 of 3" in captured.err and "--originals" in captured.err


def test_recon_warn_db_flags_chunks_and_warns(make_service, tmp_path, capsys):
    odd = make_service(normalize_stems_separately=True)
    captured = _merge(odd, tmp_path, capsys, "--anchor", "--other-match", "vocals")
    assert "worth a listen" in captured.out and "worth a listen" in captured.err
    quiet = _merge(
        odd, tmp_path, capsys, "--anchor", "--other-match", "vocals", "--recon-warn-db", "-50"
    )
    assert "chunk(s) worth a listen" not in quiet.out


@pytest.mark.parametrize(
    "extra, message",
    [
        (
            ["--originals", "x", "--pairing", "off"],
            "--originals only means something with --anchor",
        ),
        (["--other-out", "x.wav"], "--other-out needs --other-match"),
        (["--audition-window", "3"], "--audition-window only mean something with --audition"),
        (
            ["--audition-seams", "all", "--audition-format", "wav", "--audition-max", "2"],
            "--audition-seams, --audition-format, --audition-max only mean something",
        ),
    ],
)
def test_options_that_need_another_option_say_so(service, tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as exc:
        _merge(service, tmp_path, capsys, *extra)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.wav").exists()


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--audition-seams", "x"], "not 'flagged', 'all' or a list"),
        (["--audition-window", "0"], "above 0"),
        (["--audition-max", "0"], "at least 1"),
    ],
)
def test_bad_listening_kit_options_are_errors_before_anything_is_read(
    service, tmp_path, capsys, extra, message
):
    kit = str(tmp_path / "kit")
    captured = _merge(service, tmp_path, capsys, "--audition", kit, *extra, expect=1)
    assert message in captured.err and not os.path.exists(kit)
    assert not (tmp_path / "out.wav").exists()


def test_the_listening_kit_format_is_one_of_the_three(service, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _merge(
            service, tmp_path, capsys, "--audition", str(tmp_path / "k"), "--audition-format", "ogg"
        )
    assert "invalid choice: 'ogg'" in capsys.readouterr().err


def test_the_listening_kit_is_made_and_pointed_to(service, tmp_path, capsys):
    kit = tmp_path / "kit"
    out = _merge(
        service,
        tmp_path,
        capsys,
        "--audition",
        str(kit),
        "--audition-seams",
        "1",
        "--audition-window",
        "2",
        "--audition-format",
        "wav",
    ).out
    assert f"Listening kit: seam(s) 1 in '{kit}'" in out and KIT_PAGE in out
    data = json.loads((kit / KIT_FILE).read_text())
    assert [s["seam"] for s in data["seams"]] == [1] and data["audio_format"] == "wav"
    assert data["window_s"] == 2.0


def test_no_flagged_seam_means_no_kit_and_a_hint(service, tmp_path, capsys):
    kit = tmp_path / "kit"
    out = _merge(service, tmp_path, capsys, "--audition", str(kit), "--seam-warn-db", "-50").out
    assert "No listening kit: no seam was selected (flagged)" in out
    assert "--audition-seams all" in out and not kit.exists()


def test_the_kit_can_be_made_of_an_anchored_merge_of_both_stems(service, tmp_path, capsys):
    kit = tmp_path / "kit"
    _merge(
        service,
        tmp_path,
        capsys,
        "--anchor",
        "--other-match",
        "vocals",
        "--audition",
        str(kit),
        "--audition-seams",
        "all",
    )
    data = json.loads((kit / KIT_FILE).read_text())
    assert data["settings"]["anchored"] and data["settings"]["both_stems"]
    assert len(data["seams"]) == 2


# --- audition-summary ---------------------------------------------------------------------


def _answers(folder, kinds, blind=True):
    from test_choices import _kit

    folder.mkdir(parents=True, exist_ok=True)
    kit = _kit(len(kinds))
    doc = new_choices(kit)
    ids = {"seam_search": "seam_search", "crossfade": "adaptive", "tie": "tie"}
    for number, kind in enumerate(kinds, start=1):
        record_choice(doc, kit, number, ids[kind], blind)
    return save_choices(doc, str(folder))


def test_audition_summary_pools_files_and_folders(tmp_path, capsys):
    first = _answers(tmp_path / "a", ["seam_search"] * 3)
    _answers(tmp_path / "b", ["seam_search"] * 3 + ["tie"])
    assert main(["audition-summary", first, str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    assert "7 answer(s), 7 of them blind" in out
    assert "Seam search against the crossfade: 6 to 0" in out and "seam search preferred" in out


def test_audition_summary_as_json(tmp_path, capsys):
    path = _answers(tmp_path / "a", ["crossfade", "tie"])
    assert main(["audition-summary", path, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["choices"] == 2 and data["counts"]["tie"] == 1
    assert data["seam_search_vs_crossfade"]["verdict"] == "too_few"


def test_audition_summary_says_what_is_wrong_with_a_file(tmp_path, capsys):
    bad = tmp_path / CHOICES_FILE
    bad.write_text("{nope")
    assert main(["audition-summary", str(bad)]) == 1
    assert "Error:" in capsys.readouterr().err
    assert main(["audition-summary", str(tmp_path / "missing.json")]) == 1
    assert "Error:" in capsys.readouterr().err
