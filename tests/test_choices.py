"""What is done with the answers of a listening kit."""

import json

import pytest

from seamsplit.audition import variants_for
from seamsplit.choices import (
    ALPHA,
    CHOICES_FILE,
    MIN_DECIDED,
    TIE,
    binomial_two_sided,
    describe_summary,
    load_choices,
    new_choices,
    record_choice,
    save_choices,
    summarize_choices,
)


def _kit(n=3, identical=()):
    variants = variants_for("adaptive")
    seams = []
    for number in range(1, n + 1):
        clips = {v["id"]: {"file": f"s{number}_{v['id']}.flac"} for v in variants}
        clips["seam_search"]["identical_to"] = "adaptive" if number in identical else None
        seams.append(
            {"seam": number, "clips": clips, "metrics": {"overlap_match_db": 10.0 + number}}
        )
    return {"source_file": "song.wav", "settings": {"method": "adaptive"}, "window_s": 5.0,
            "variants": variants, "seams": seams}  # fmt: skip


def _doc(kinds, blind=True, agreement=None):
    """An answer sheet with one answer per kind, for seams 1, 2, ..."""
    kit = _kit(len(kinds))
    ids = {"straight": "straight", "crossfade": "adaptive", "seam_search": "seam_search", TIE: TIE}
    doc = new_choices(kit)
    for number, kind in enumerate(kinds, start=1):
        record_choice(doc, kit, number, ids[kind], blind)
        if agreement is not None:
            doc["choices"][-1]["metrics"] = {"overlap_match_db": agreement[number - 1]}
    return doc


# --- recording ---------------------------------------------------------------------------


def test_an_answer_keeps_what_was_measured_at_its_seam():
    kit = _kit(2, identical=(2,))
    doc = new_choices(kit)
    assert doc["choices"] == [] and doc["source_file"] == "song.wav" and doc["format"] == 1
    record_choice(doc, kit, 2, "seam_search", blind=True)
    (entry,) = doc["choices"]
    assert entry["seam"] == 2 and entry["choice"] == "seam_search" == entry["kind"]
    assert entry["blind"] is True and entry["metrics"] == {"overlap_match_db": 12.0}
    assert entry["seam_search_identical"] is True


def test_an_answer_replaces_the_earlier_one_for_the_same_seam_and_the_sheet_stays_sorted():
    kit = _kit(3)
    doc = new_choices(kit)
    for seam, choice in ((3, "straight"), (1, "adaptive"), (3, TIE)):
        record_choice(doc, kit, seam, choice, False)
    assert [(c["seam"], c["kind"]) for c in doc["choices"]] == [(1, "crossfade"), (3, TIE)]
    assert doc["choices"][0]["blind"] is False


@pytest.mark.parametrize(
    "seam, choice, message",
    [(9, "straight", "no seam 9"), (1, "loudest", "choice must be one of")],
)
def test_an_answer_for_something_the_kit_does_not_have_is_refused(seam, choice, message):
    kit = _kit(2)
    doc = new_choices(kit)
    with pytest.raises(ValueError, match=message):
        record_choice(doc, kit, seam, choice, True)
    assert doc["choices"] == []


# --- the sign test -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "k, n, p",
    [(0, 6, 2 * 0.5**6), (6, 6, 2 * 0.5**6), (3, 6, 1.0), (5, 6, 0.21875), (2, 10, 0.109375)],
)
def test_the_exact_two_sided_binomial_p_value(k, n, p):
    assert binomial_two_sided(k, n) == pytest.approx(p)


def test_p_value_edge_cases():
    assert binomial_two_sided(0, 0) == 1.0
    assert binomial_two_sided(10, 20) == 1.0
    assert 0 < binomial_two_sided(0, 200) < 1e-50


def test_six_decided_seams_is_the_least_that_can_ever_reach_significance():
    assert binomial_two_sided(0, MIN_DECIDED) < ALPHA <= binomial_two_sided(0, MIN_DECIDED - 1)


def test_summary_of_nothing():
    summary = summarize_choices([])
    assert summary["choices"] == 0 and summary["blind"] == 0
    assert summary["seam_search_vs_crossfade"]["verdict"] == "too_few"
    assert describe_summary(summary) == ["No answers yet."]


def test_a_clear_win_for_the_seam_search_is_found_only_with_enough_blind_answers():
    won = summarize_choices([_doc(["seam_search"] * 6)])["seam_search_vs_crossfade"]
    assert (won["n"], won["seam_search"], won["crossfade"]) == (6, 6, 0)
    assert won["verdict"] == "seam_search" and won["p_value"] < ALPHA

    few = summarize_choices([_doc(["seam_search"] * 5)])["seam_search_vs_crossfade"]
    assert few["verdict"] == "too_few"

    split = summarize_choices([_doc(["seam_search"] * 4 + ["crossfade"] * 3)])
    assert split["seam_search_vs_crossfade"]["verdict"] == "no_clear_preference"

    lost = summarize_choices([_doc(["crossfade"] * 8 + ["seam_search"])])
    assert lost["seam_search_vs_crossfade"]["verdict"] == "crossfade"


def test_only_blind_answers_count_for_the_tests():
    seen = _doc(["seam_search"] * 8, blind=False)
    summary = summarize_choices([seen])
    assert summary["choices"] == 8 and summary["blind"] == 0
    assert summary["counts"]["seam_search"] == 8  # they are counted, not tested
    assert summary["seam_search_vs_crossfade"]["n"] == 0
    mixed = summarize_choices([seen, _doc(["seam_search"] * 6)])
    assert mixed["seam_search_vs_crossfade"]["n"] == 6 and mixed["choices"] == 14


def test_a_seam_where_the_search_made_the_same_audio_is_not_a_preference():
    kit = _kit(8, identical=(1, 2, 3, 4, 5, 6, 7, 8))
    doc = new_choices(kit)
    for seam in range(1, 9):
        record_choice(doc, kit, seam, "seam_search", True)
    summary = summarize_choices([doc])["seam_search_vs_crossfade"]
    assert summary["n"] == 0 and summary["verdict"] == "too_few"


def test_ties_are_counted_but_decide_nothing():
    summary = summarize_choices([_doc([TIE] * 5 + ["seam_search"] * 6)])
    assert summary["counts"][TIE] == 5
    assert summary["seam_search_vs_crossfade"]["n"] == 6
    assert summary["straight_vs_processed"]["n"] == 6  # the ties are not among the decided


def test_the_straight_join_against_the_two_that_process_the_seam():
    worse = summarize_choices([_doc(["crossfade"] * 4 + ["seam_search"] * 4)])
    assert worse["straight_vs_processed"]["verdict"] == "processed"
    better = summarize_choices([_doc(["straight"] * 8)])
    assert better["straight_vs_processed"]["verdict"] == "straight"
    assert better["straight_vs_processed"]["straight"] == 8
    # 7 to 1 is not yet beyond what chance does: p = 0.07
    close = summarize_choices([_doc(["straight"] * 7 + ["crossfade"])])
    assert close["straight_vs_processed"]["verdict"] == "no_clear_preference"


def test_the_answers_of_several_kits_are_pooled():
    docs = [_doc(["seam_search"] * 3), _doc(["seam_search"] * 3)]
    assert summarize_choices(docs)["seam_search_vs_crossfade"]["verdict"] == "seam_search"


def test_agreement_is_compared_between_seams_that_sounded_the_same_and_those_that_did_not():
    doc = _doc([TIE, TIE, TIE, "crossfade", "straight"], agreement=[24.0, 28.0, 20.0, 9.0, 3.0])
    spread = summarize_choices([doc])["agreement_db"]
    assert spread["no_difference"] == {"n": 3, "min": 20.0, "median": 24.0, "max": 28.0}
    assert spread["heard_difference"] == {"n": 2, "min": 3.0, "median": 6.0, "max": 9.0}
    none = summarize_choices([_doc(["crossfade"], agreement=[None])])["agreement_db"]
    assert none == {"no_difference": None, "heard_difference": None}


def test_the_summary_in_words():
    doc = _doc(["seam_search"] * 6 + [TIE], agreement=[10.0] * 6 + [25.0])
    lines = describe_summary(summarize_choices([doc]))
    text = "\n".join(lines)
    assert "7 answer(s), 7 of them blind" in lines[0] and "seam search 6" in lines[0]
    assert "6 to 0" in text and "seam search preferred" in text
    assert "the straight join" in text.lower() or "Straight join against" in text
    assert "Agreement at the seams with nothing audible: 25 to 25 dB" in text
    assert "a difference: 10 to 10 dB" in text
    few = "\n".join(describe_summary(summarize_choices([_doc(["crossfade"])])))
    assert "too few decided seams (at least 6 are needed to tell)" in few
    flat = "\n".join(
        describe_summary(summarize_choices([_doc(["crossfade"] * 4 + ["seam_search"] * 3)]))
    )
    assert "no clear preference" in flat


# --- files ------------------------------------------------------------------------------


def test_an_answer_sheet_is_saved_whole_and_read_back(tmp_path):
    doc = _doc(["seam_search", TIE])
    path = save_choices(doc, str(tmp_path))
    assert path.endswith(CHOICES_FILE) and [p.name for p in tmp_path.iterdir()] == [CHOICES_FILE]
    assert load_choices(path) == doc
    again = save_choices(_doc(["crossfade"]), str(tmp_path))
    assert len(load_choices(again)["choices"]) == 1


def test_a_file_that_is_not_an_answer_sheet_is_refused_with_its_name(tmp_path):
    def attempt(content):
        path = tmp_path / "x.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="x.json"):
            load_choices(str(path))

    attempt("{not json")
    attempt("[1, 2]")
    attempt(json.dumps({"choices": "many"}))
    attempt(json.dumps({"choices": [{"seam": 1}]}))
    attempt(json.dumps({"choices": [3]}))
