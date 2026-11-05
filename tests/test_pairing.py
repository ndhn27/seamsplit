"""Telling which chunk a processed file is by comparing it with the original chunks, and
what ``merge`` does with that when the file names were lost."""

import json
import os
import shutil

import numpy as np
import pytest
from seamsplit.audio import AudioSegment

from fake_service import SR, snr_db, to_array, to_segment
from seamsplit.cli import main
from seamsplit.merger import merge_audio
from seamsplit.pairing import (
    MIN_PAIR_MARGIN,
    MIN_PAIR_NCC,
    SIGNATURE_RATE,
    Match,
    check_pairing,
    identify,
    resolve_order,
    score_matrix,
    signature,
)
from seamsplit.report import MergeReport


def _noise(seconds, seed):
    return np.random.default_rng(seed).standard_normal((1, int(SR * seconds))) * 0.2


def _wav(path, x):
    to_segment(x).export(str(path), format="wav")
    return str(path)


def _renamed(service, tmp_path, order, stems=("instrumental",), name="renamed"):
    """The processed files of ``stems`` as ``download (k)``, with the part at
    ``order[k]`` in the place ``k`` (the stem word is kept so --match can tell the stems)."""
    folder = tmp_path / name
    folder.mkdir()
    for k, part in enumerate(order):
        base = service.parts[part].rsplit(".", 1)[0]
        for stem in stems:
            source = os.path.join(service.processed_dir, f"{base}_{stem}.wav")
            shutil.copy(source, folder / f"download ({k + 1}) {stem}.wav")
    return str(folder)


def _originals(service):
    return [os.path.join(service.chunks_dir, name) for name in service.parts]


def _stem_files(service, stem):
    return [
        os.path.join(service.processed_dir, f"{name.rsplit('.', 1)[0]}_{stem}.wav")
        for name in service.parts
    ]


# -- the signature and the scores --------------------------------------------


def test_a_signature_has_the_same_rate_whatever_the_file_was(tmp_path):
    x = _noise(4.0, 1)
    plain = signature(_wav(tmp_path / "a.wav", x))
    assert abs(len(plain) - 4 * SIGNATURE_RATE) <= 2
    stereo = AudioSegment.from_file(_wav(tmp_path / "s.wav", x)).set_channels(2)
    stereo.export(str(tmp_path / "stereo.wav"), format="wav")
    fast = AudioSegment.from_file(str(tmp_path / "a.wav")).set_frame_rate(48000)
    fast.export(str(tmp_path / "fast.wav"), format="wav")
    for name in ("stereo.wav", "fast.wav"):
        other = signature(str(tmp_path / name))
        assert abs(len(other) - len(plain)) <= 2
        # the same recording read through another format is the same signature
        n = min(len(other), len(plain))
        assert float(np.corrcoef(other[:n], plain[:n])[0, 1]) > 0.9


def test_a_file_with_no_sound_has_an_empty_signature(tmp_path):
    assert len(signature(_wav(tmp_path / "silent.wav", np.zeros((1, SR))))) == 0
    assert len(signature(_wav(tmp_path / "tiny.wav", np.ones((1, 4)) * 0.1))) == 0
    empty = AudioSegment.silent(duration=0, frame_rate=SR)
    empty.export(str(tmp_path / "empty.wav"), format="wav")
    assert len(signature(str(tmp_path / "empty.wav"))) == 0


def test_a_file_that_is_not_audio_cannot_be_read(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(Exception):
        signature(str(bad))


def _sig(seed, seconds=10.0, shift=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(SIGNATURE_RATE * (seconds + 3)))
    return x[shift : shift + int(SIGNATURE_RATE * seconds)]


def test_scores_say_how_alike_signals_are_within_the_lag_that_is_allowed():
    a = _sig(1)
    shifted = _sig(1, shift=500)  # the same signal, 250 ms later: within +-1 s
    far = _sig(1, shift=2500)  # 1.25 s: outside it
    other = _sig(2)
    scores = score_matrix([a, shifted, far, other], [a], max_lag_ms=1000)
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert scores[1, 0] > 0.9
    assert scores[2, 0] < 0.1 and scores[3, 0] < 0.1
    wide = score_matrix([far], [a], max_lag_ms=2000)
    assert wide[0, 0] > 0.8  # the part of it that still overlaps
    assert score_matrix([far], [a], max_lag_ms=0)[0, 0] < 0.1


def test_a_signal_with_nothing_in_it_scores_zero_and_nothing_at_all_is_fine():
    a = _sig(1)
    scores = score_matrix([np.zeros(0), a], [a, np.zeros(0)])
    assert scores.tolist() == [[0.0, 0.0], [pytest.approx(1.0, abs=1e-6), 0.0]]
    assert score_matrix([np.zeros(0)], [np.zeros(0)]).tolist() == [[0.0]]


def test_the_thresholds_are_the_documented_ones():
    assert (MIN_PAIR_NCC, MIN_PAIR_MARGIN) == (0.15, 0.08)


# -- identifying files --------------------------------------------------------


@pytest.mark.parametrize("stem", ["instrumental", "vocals"])
def test_each_stem_of_a_chunk_is_identified_as_its_chunk(service, stem):
    matches = identify(_stem_files(service, stem), _originals(service))
    assert [m.part for m in matches] == [0, 1, 2]
    assert all(m.confident and m.score > m.runner_up + MIN_PAIR_MARGIN for m in matches)
    assert all(m.error is None for m in matches)


def test_files_in_any_order_are_found_and_both_stems_may_name_the_same_part(service):
    files = _stem_files(service, "vocals")[::-1] + _stem_files(service, "instrumental")
    matches = identify(files, _originals(service))
    assert [m.part for m in matches] == [2, 1, 0, 0, 1, 2]


def test_a_file_that_is_nothing_like_any_chunk_is_not_sure(service, tmp_path):
    stranger = _wav(tmp_path / "other.wav", _noise(15.0, 99))
    (match,) = identify([stranger], _originals(service))
    assert not match.confident and match.score < MIN_PAIR_NCC


def test_chunks_that_repeat_exactly_cannot_be_told_apart_and_say_so(tmp_path):
    loop, other = _noise(8.0, 1), _noise(8.0, 2)
    originals = [_wav(tmp_path / f"o{i}.wav", x) for i, x in enumerate((loop, loop, other))]
    matches = identify(originals, originals)
    assert [m.confident for m in matches] == [False, False, True]
    assert matches[2].part == 2


def test_an_unreadable_or_silent_file_has_no_part_and_a_reason(service, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    silent = _wav(tmp_path / "silent.wav", np.zeros((1, SR * 3)))
    broken, quiet = identify([str(bad), silent], _originals(service))
    assert broken.part is None and not broken.confident and broken.error
    assert quiet.part is None and quiet.error is None


def test_an_original_that_cannot_be_read_is_an_error(service, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(OSError, match="could not read the original chunk"):
        identify(_stem_files(service, "vocals")[:1], [str(bad)])


def test_progress_is_reported_for_every_file_read(service):
    seen = []
    identify(
        _stem_files(service, "vocals"),
        _originals(service),
        progress=lambda i, n, name: seen.append((i, n)),
    )
    assert seen == [(i, 6) for i in range(1, 7)]


# -- putting files in order ---------------------------------------------------


def _match(name, part, sure=True, score=0.6, runner_up=0.1):
    return Match(name, part, score, runner_up, sure)


NAMES = ["p1.wav", "p2.wav", "p3.wav", "p4.wav"]


def test_files_that_are_where_they_belong_stay():
    matches = [_match(f"f{k}", k) for k in range(4)]
    result = resolve_order(matches, NAMES)
    assert result.problem is None and result.order == [0, 1, 2, 3]
    assert result.moved == [] and result.unsure == []


def test_files_that_clearly_belong_elsewhere_are_moved():
    matches = [_match("f0", 1), _match("f1", 0), _match("f2", 2), _match("f3", 3)]
    result = resolve_order(matches, NAMES)
    assert result.problem is None and result.order == [1, 0, 2, 3]
    assert result.moved == [0, 1]


def test_a_file_nothing_could_be_said_about_stays_where_it_was_put():
    matches = [_match("f0", 1), _match("f1", 0), _match("f2", None, False, 0, 0), _match("f3", 3)]
    result = resolve_order(matches, NAMES)
    assert result.problem is None and result.order == [1, 0, 2, 3] and result.unsure == [2]


def test_files_that_do_not_fit_the_parts_are_a_problem_that_names_them():
    matches = [_match("f0", 1), _match("f1", 1), _match("f2", 2), _match("f3", 3)]
    result = resolve_order(matches, NAMES)
    assert result.problem is not None
    assert "'f0', 'f1' all belong to part 2 ('p2.wav')" in result.problem
    assert "nothing was found for part 1" in result.problem
    # a sure file may not take the place of one that cannot be moved
    taken = resolve_order(
        [_match("f0", 2), _match("f1", 1, False), _match("f2", 2, False)], NAMES[:3]
    )
    assert taken.problem and "'f0', 'f2' all belong to part 3" in taken.problem


# -- check_pairing ------------------------------------------------------------


def test_files_in_the_right_order_are_confirmed_in_a_note(service):
    report = MergeReport()
    files = _stem_files(service, "instrumental")
    out = check_pairing(files, _originals(service), service.parts, report)
    assert out == files and report.warnings == []
    assert len(report.notes) == 1 and "all are that chunk (lowest similarity" in report.notes[0]


def test_files_out_of_order_are_put_right_and_the_report_says_which(service):
    report = MergeReport()
    files = _stem_files(service, "instrumental")
    shuffled = [files[2], files[0], files[1]]
    out = check_pairing(shuffled, _originals(service), service.parts, report, "instrumental")
    assert out == files
    (warning,) = report.warnings
    assert "3 processed file(s) (instrumental) were not in part order" in warning
    assert "'song_part03_instrumental.wav' → part 3" in warning and "put in order" in warning
    assert report.notes == []


def test_files_that_cannot_be_told_apart_stay_and_the_report_lists_them(service, tmp_path):
    report = MergeReport()
    files = _stem_files(service, "instrumental")
    stranger = _wav(tmp_path / "stranger.wav", _noise(15.0, 5))
    out = check_pairing([files[0], stranger, files[2]], _originals(service), service.parts, report)
    assert out == [files[0], stranger, files[2]]
    (warning,) = report.warnings
    assert "1 processed file(s) could not be told apart" in warning and "'stranger.wav'" in warning
    assert report.notes == []


def test_many_unsure_files_are_listed_briefly(tmp_path):
    originals = [_wav(tmp_path / f"o{i}.wav", _noise(6.0, i)) for i in range(8)]
    strangers = [_wav(tmp_path / f"s{i}.wav", _noise(6.0, 100 + i)) for i in range(8)]
    report = MergeReport()
    check_pairing(strangers, originals, [os.path.basename(p) for p in originals], report)
    (warning,) = report.warnings
    assert (
        warning.startswith("8 processed file(s) could not be told apart")
        and "and 2 more" in warning
    )


def test_many_moved_files_are_listed_briefly(tmp_path):
    xs = [_noise(6.0, i) for i in range(8)]
    originals = [_wav(tmp_path / f"o{i}.wav", x) for i, x in enumerate(xs)]
    names = [os.path.basename(p) for p in originals]
    moved = [_wav(tmp_path / f"m{i}.wav", xs[(i + 1) % 8]) for i in range(8)]
    report = MergeReport()
    out = check_pairing(moved, originals, names, report)
    assert [os.path.basename(p) for p in out] == [f"m{(i - 1) % 8}.wav" for i in range(8)]
    (warning,) = report.warnings
    assert (
        warning.startswith("8 processed file(s) were not in part order") and "and 2 more" in warning
    )


def test_files_that_do_not_fit_the_parts_are_refused_with_a_way_out(service):
    files = _stem_files(service, "instrumental")
    with pytest.raises(ValueError, match="do not fit the parts.*--pairing off"):
        check_pairing([files[0], files[0], files[2]], _originals(service), service.parts)


def test_without_a_report_what_was_found_is_a_runtime_warning(service):
    files = _stem_files(service, "instrumental")
    with pytest.warns(RuntimeWarning, match="were not in part order"):
        check_pairing(files[::-1], _originals(service), service.parts)
    # nothing to say, nothing is said
    check_pairing(files, _originals(service), service.parts)


# -- merge --------------------------------------------------------------------


def _merge(service, folder, tmp_path, name="out.wav", **options):
    report = MergeReport()
    options.setdefault("match", "instrumental")
    merged = merge_audio(
        folder, service.manifest_path, str(tmp_path / name), report=report, samplerate=SR, **options
    )
    return merged, report


def test_renamed_files_in_the_wrong_order_are_merged_as_if_they_had_their_names(
    make_service, tmp_path
):
    service = make_service(seconds=80)
    assert len(service.parts) == 5
    named, _ = _merge(service, service.processed_dir, tmp_path, "named.wav")
    folder = _renamed(service, tmp_path, [2, 3, 4, 0, 1])
    merged, report = _merge(service, folder, tmp_path, "renamed.wav")
    assert to_array(merged).tolist() == to_array(named).tolist()
    assert (
        len(report.warnings) == 1
        and "5 processed file(s) were not in part order" in report.warnings[0]
    )
    assert not any(b.flagged for b in report.boundaries)
    # what pairing by order alone would have made of them
    blind, blind_report = _merge(service, folder, tmp_path, "blind.wav", pairing="off")
    assert any(b.flagged for b in blind_report.boundaries) or len(blind_report.warnings) > 1
    assert "paired with the parts by order" in blind_report.warnings[0]
    assert "check that by comparing" not in blind_report.warnings[0]


def test_renamed_files_in_the_right_order_are_confirmed(service, tmp_path):
    folder = _renamed(service, tmp_path, [0, 1, 2])
    _, report = _merge(service, folder, tmp_path)
    assert report.warnings == []
    assert len(report.notes) == 1 and "all are that chunk" in report.notes[0]


def test_without_the_original_chunks_the_guess_stays_a_guess_and_says_how_to_improve_it(
    service, tmp_path
):
    folder = _renamed(service, tmp_path, [0, 1, 2])
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    shutil.copy(service.manifest_path, lonely / "manifest.json")
    report = MergeReport()
    merge_audio(
        folder,
        str(lonely / "manifest.json"),
        str(tmp_path / "lonely.wav"),
        match="instrumental",
        report=report,
    )
    assert "paired with the parts by order" in report.warnings[0]
    assert "check that by comparing them with the files" in report.warnings[0]
    assert report.notes == []


def test_originals_can_be_somewhere_else(service, tmp_path):
    folder = _renamed(service, tmp_path, [1, 0, 2])
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    shutil.copy(service.manifest_path, lonely / "manifest.json")
    report = MergeReport()
    merge_audio(
        folder,
        str(lonely / "manifest.json"),
        str(tmp_path / "out.wav"),
        match="instrumental",
        report=report,
        originals_dir=service.chunks_dir,
        pairing="verify",
    )
    assert "2 processed file(s) were not in part order" in report.warnings[0]


def test_files_found_by_name_are_not_compared_unless_asked(service, tmp_path):
    _, report = _merge(service, service.processed_dir, tmp_path)
    assert report.warnings == [] and report.notes == []
    _, report = _merge(service, service.processed_dir, tmp_path, pairing="verify")
    assert report.warnings == [] and "lowest similarity" in report.notes[0]


def test_verify_catches_files_whose_names_say_the_wrong_part(make_service, tmp_path):
    service = make_service(seconds=80)
    named, _ = _merge(service, service.processed_dir, tmp_path, "named.wav")
    swapped = tmp_path / "swapped"
    swapped.mkdir()
    base = [n.rsplit(".", 1)[0] for n in service.parts]
    for part, content in enumerate([0, 2, 1, 3, 4]):  # parts 2 and 3 carry each other's names
        shutil.copy(
            os.path.join(service.processed_dir, f"{base[content]}_instrumental.wav"),
            swapped / f"{base[part]}_instrumental.wav",
        )
    merged, report = _merge(service, str(swapped), tmp_path, "fixed.wav", pairing="verify")
    assert to_array(merged).tolist() == to_array(named).tolist()
    assert "2 processed file(s) were not in part order" in report.warnings[0]


def test_verify_needs_the_originals(service, tmp_path):
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    shutil.copy(service.manifest_path, lonely / "manifest.json")
    with pytest.raises(FileNotFoundError, match="original chunks are needed"):
        merge_audio(
            service.processed_dir,
            str(lonely / "manifest.json"),
            str(tmp_path / "out.wav"),
            match="instrumental",
            pairing="verify",
        )


def test_both_stems_are_checked_each_on_its_own(make_service, tmp_path):
    service = make_service(seconds=80)
    named, _ = _merge(service, service.processed_dir, tmp_path, "named.wav", other_match="vocals")
    folder = _renamed(service, tmp_path, [1, 0, 2, 4, 3], stems=("instrumental", "vocals"))
    merged, report = _merge(service, folder, tmp_path, "both.wav", other_match="vocals")
    assert to_array(merged).tolist() == to_array(named).tolist()
    assert len(report.warnings) == 2
    assert any("(instrumental)" in w for w in report.warnings)
    assert any("(vocals)" in w for w in report.warnings)
    second = to_array(AudioSegment.from_file(str(tmp_path / "both_vocals.wav")))
    reference = to_array(AudioSegment.from_file(str(tmp_path / "named_vocals.wav")))
    assert second.tolist() == reference.tolist()
    assert snr_db(reference, second) > 90


def test_the_pairing_options_are_checked(service, tmp_path):
    with pytest.raises(ValueError, match="Unknown pairing: 'sometimes'"):
        _merge(service, service.processed_dir, tmp_path, pairing="sometimes")
    with pytest.raises(ValueError, match="only used with anchor=True or when the pairing is"):
        _merge(
            service,
            service.processed_dir,
            tmp_path,
            originals_dir=service.chunks_dir,
            pairing="off",
        )
    _merge(service, service.processed_dir, tmp_path, originals_dir=service.chunks_dir)


def test_files_that_do_not_fit_the_parts_stop_the_merge_before_anything_is_written(
    service, tmp_path
):
    folder = _renamed(service, tmp_path, [0, 0, 2])
    with pytest.raises(ValueError, match="do not fit the parts"):
        _merge(service, folder, tmp_path)
    assert not (tmp_path / "out.wav").exists()


# -- the command line ---------------------------------------------------------


def _cli_merge(service, folder, tmp_path, capsys, *extra, expect=0):
    rc = main(
        [
            "merge",
            folder,
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
    )
    captured = capsys.readouterr()
    assert rc == expect, captured.err
    return captured


def test_the_command_line_tells_what_was_confirmed_and_what_was_put_right(
    service, tmp_path, capsys
):
    ok = _cli_merge(service, _renamed(service, tmp_path, [0, 1, 2]), tmp_path, capsys)
    assert "Note: the 3 processed files were each compared with the original chunk" in ok.out
    wrong = _cli_merge(
        service, _renamed(service, tmp_path, [2, 0, 1], name="wrong"), tmp_path, capsys
    )
    assert "Warning: 3 processed file(s) were not in part order" in wrong.err


def test_pairing_off_and_verify_are_options_of_merge(service, tmp_path, capsys):
    folder = _renamed(service, tmp_path, [0, 1, 2])
    off = _cli_merge(service, folder, tmp_path, capsys, "--pairing", "off")
    assert "Note:" not in off.out and "paired with the parts by order" in off.err
    verified = _cli_merge(service, service.processed_dir, tmp_path, capsys, "--pairing", "verify")
    assert "Note: the 3 processed files were each compared" in verified.out


def test_originals_without_anchor_need_a_pairing_that_checks(service, tmp_path, capsys):
    args = ["--originals", service.chunks_dir]
    with pytest.raises(SystemExit):
        _cli_merge(service, service.processed_dir, tmp_path, capsys, *args, "--pairing", "off")
    assert "--originals only means something with --anchor, or when --pairing checks" in (
        capsys.readouterr().err
    )
    _cli_merge(service, service.processed_dir, tmp_path, capsys, *args)
    _cli_merge(service, service.processed_dir, tmp_path, capsys, *args, "--pairing", "verify")


def test_the_notes_are_in_the_report_file_when_there_are_any(service, tmp_path, capsys):
    folder = _renamed(service, tmp_path, [0, 1, 2])
    report_json = tmp_path / "report.json"
    _cli_merge(service, folder, tmp_path, capsys, "--report", str(report_json))
    assert "notes" in json.loads(report_json.read_text())
    plain = tmp_path / "plain.json"
    _cli_merge(service, service.processed_dir, tmp_path, capsys, "--report", str(plain))
    assert "notes" not in json.loads(plain.read_text())
