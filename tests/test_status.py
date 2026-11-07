"""``status``: which chunks have a result and which are missing, including results whose
names were lost."""

import json
import os
import shutil

import numpy as np
import pytest
from seamsplit.audio import AudioSegment

from fake_service import SR, to_segment
from seamsplit.status import (
    AMBIGUOUS,
    DONE,
    MISSING,
    compute_status,
    format_status,
)

STEMS = ("vocals", "instrumental")


def _copy_results(service, tmp_path, name="processed"):
    folder = tmp_path / name
    shutil.copytree(service.processed_dir, folder)
    return str(folder)


def _status(service, processed, stems=STEMS, **options):
    return compute_status(service.chunks_dir, processed, stems, **options)


def test_a_job_that_is_all_there_is_complete(service):
    status = _status(service, service.processed_dir)
    assert status.complete and status.missing == [] and status.pending_parts() == []
    assert status.parts == 3 and len(status.slots) == 6
    assert all(s.state == DONE and s.how == "name" for s in status.slots)
    assert status.unmatched == [] and status.notes == []
    assert "3 of 3 chunk(s) are complete." in format_status(status)
    assert "Still to come" not in format_status(status)


def test_a_missing_stem_makes_its_chunk_pending_and_is_listed(service, tmp_path):
    processed = _copy_results(service, tmp_path)
    os.remove(os.path.join(processed, "song_part02_vocals.wav"))
    status = _status(service, processed)
    assert not status.complete and status.pending_parts() == [2]
    (slot,) = status.missing
    assert (slot.part, slot.stem, slot.state, slot.files) == (2, "vocals", MISSING, [])
    text = format_status(status)
    assert "2 of 3 chunk(s) are complete." in text
    assert "part   2 song_part02.wav [vocals]: MISSING" in text
    assert "Still to come back: chunk(s) 2." in text


def test_with_no_results_folder_yet_everything_is_missing(service, tmp_path):
    status = _status(service, str(tmp_path / "not-yet"))
    assert status.pending_parts() == [1, 2, 3] and not status.complete
    assert "0 of 3 chunk(s) are complete." in format_status(status)


def test_one_file_per_chunk_when_no_stem_words_are_given(service, tmp_path):
    processed = tmp_path / "one"
    processed.mkdir()
    for name in service.parts[:2]:
        base = name.rsplit(".", 1)[0]
        shutil.copy(
            os.path.join(service.processed_dir, f"{base}_instrumental.wav"),
            processed / f"{base}.wav",
        )
    status = compute_status(service.chunks_dir, str(processed))
    assert status.pending_parts() == [3]
    assert "[" not in format_status(status).split("MISSING")[0]  # no stem labels to show


def test_two_files_for_one_chunk_are_ambiguous_and_the_text_says_what_to_do(service, tmp_path):
    processed = _copy_results(service, tmp_path)
    shutil.copy(
        os.path.join(processed, "song_part01_vocals.wav"),
        os.path.join(processed, "song_part01_vocals (1).wav"),
    )
    status = _status(service, processed)
    slot = next(s for s in status.slots if s.part == 1 and s.stem == "vocals")
    assert slot.state == AMBIGUOUS and len(slot.files) == 2
    assert status.pending_parts() == [1]
    text = format_status(status)
    assert "2 files (song_part01_vocals (1).wav, song_part01_vocals.wav); narrow them" in text


def test_without_stem_words_both_stems_of_a_chunk_are_ambiguous(service):
    status = compute_status(service.chunks_dir, service.processed_dir)
    assert {s.state for s in status.slots} == {AMBIGUOUS}


def _renamed(service, tmp_path, order, stem="instrumental"):
    folder = tmp_path / "renamed"
    folder.mkdir()
    for k, part in enumerate(order):
        base = service.parts[part].rsplit(".", 1)[0]
        shutil.copy(
            os.path.join(service.processed_dir, f"{base}_{stem}.wav"),
            folder / f"{stem} download ({k + 1}).wav",
        )
    return str(folder)


def test_files_that_lost_their_names_are_told_apart_by_how_they_sound(service, tmp_path):
    folder = _renamed(service, tmp_path, [2, 0])  # chunk 2 never came back
    status = compute_status(service.chunks_dir, folder, ("instrumental",))
    assert status.pending_parts() == [2]
    sound = [s for s in status.slots if s.how == "sound"]
    assert {s.part for s in sound} == {1, 3}
    assert all(s.similarity is not None and s.similarity > 0.15 for s in sound)
    text = format_status(status)
    assert "named differently; it sounds like this chunk, similarity" in text
    assert "Still to come back: chunk(s) 2." in text


def test_renamed_files_cannot_be_identified_without_the_original_chunks(service, tmp_path):
    folder = _renamed(service, tmp_path, [0, 1, 2])
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    shutil.copy(service.manifest_path, lonely / "manifest.json")
    status = compute_status(str(lonely), folder, ("instrumental",))
    assert status.pending_parts() == [1, 2, 3]
    assert "original chunks are not in" in status.notes[0]
    assert "Note: some files carry no chunk name" in format_status(status)


def test_a_renamed_file_that_is_no_chunk_is_listed_as_not_one(service, tmp_path):
    folder = _renamed(service, tmp_path, [0, 1])
    stranger = np.random.default_rng(4).standard_normal((1, SR * 15)) * 0.2
    to_segment(stranger).export(os.path.join(folder, "instrumental other.wav"), format="wav")
    with open(os.path.join(folder, "instrumental notes.txt"), "w") as f:
        f.write("not audio")
    status = compute_status(service.chunks_dir, folder, ("instrumental",))
    assert status.pending_parts() == [3]
    assert [os.path.basename(p) for p in status.unmatched] == ["instrumental other.wav"]
    assert "1 audio file(s) are none of the chunks:\n  instrumental other.wav" in format_status(
        status
    )


def test_files_that_do_not_carry_a_stem_word_are_not_counted_against_the_job(service, tmp_path):
    processed = _copy_results(service, tmp_path)
    shutil.copy(os.path.join(processed, "song_part01_vocals.wav"), os.path.join(processed, "x.wav"))
    assert _status(service, processed).unmatched == []


def test_a_file_of_the_wrong_length_is_pointed_out(service, tmp_path):
    processed = _copy_results(service, tmp_path)
    target = os.path.join(processed, "song_part03_instrumental.wav")
    AudioSegment.from_file(target)[:4000].export(target, format="wav")
    status = _status(service, processed)
    off = [s for s in status.slots if s.length_off]
    assert [(s.part, s.stem) for s in off] == [(3, "instrumental")]
    assert off[0].length_ms == pytest.approx(4000, abs=60)
    assert "differs from the chunk's" in format_status(status)
    # lengths are not measured when that is not wanted
    assert not any(s.length_off for s in _status(service, processed, check_lengths=False).slots)


def test_a_manifest_without_segments_has_no_expected_lengths(service, tmp_path):
    folder = tmp_path / "chunks"
    shutil.copytree(service.chunks_dir, folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    del manifest["segments"]
    (folder / "manifest.json").write_text(json.dumps(manifest))
    status = compute_status(str(folder), service.processed_dir, STEMS)
    assert status.complete and all(s.expected_ms is None and not s.length_off for s in status.slots)


def test_the_status_as_data(service, tmp_path):
    processed = _copy_results(service, tmp_path)
    os.remove(os.path.join(processed, "song_part02_vocals.wav"))
    data = _status(service, processed).to_dict()
    assert data["complete"] is False and data["chunks"] == 3 and data["pending_parts"] == [2]
    assert len(data["slots"]) == 6 and data["unmatched"] == [] and data["notes"] == []
    json.dumps(data)  # it is plain data


def test_a_long_list_of_pending_chunks_is_cut_short(tmp_path):
    from fake_service import build_service

    service = build_service(str(tmp_path / "long"), seconds=300, target_s=12, overlap_s=1)
    n = len(service.parts)
    assert n > 20
    text = format_status(compute_status(service.chunks_dir, str(tmp_path / "none")))
    listed = ", ".join(str(i) for i in range(1, 21))
    assert f"chunk(s) {listed} and {n - 20} more." in text
    assert f"0 of {n} chunk(s) are complete." in text
