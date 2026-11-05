"""The listening kit: which seams, what is written, and that it is what the merge did."""

import json
import math
import os

import numpy as np
import pytest
from seamsplit.audio import AudioSegment

from fake_service import SR, to_segment
from seamsplit.audition import (
    AuditionOptions,
    KIT_FILE,
    KIT_PAGE,
    export_auditions,
    parse_seams,
    player_script,
    select_seams,
    variants_for,
    write_kit_page,
)
from seamsplit.choices import load_choices, record_choice, new_choices, summarize_choices
from seamsplit.dsp import anchor_fit
from seamsplit.merger import BoundaryInfo, MergeReport, merge_audio
from seamsplit.plan import MergePlan


def _merge(service, tmp_path, audition=None, **kw):
    report = MergeReport()
    kw.setdefault("samplerate", SR)
    merge_audio(
        service.processed_dir,
        service.manifest_path,
        str(tmp_path / "out.wav"),
        match="instrumental",
        report=report,
        audition=audition,
        **kw,
    )
    return report


def _kit(folder):
    with open(os.path.join(folder, KIT_FILE), encoding="utf-8") as f:
        return json.load(f)


def _boundary(number, agree=None, flagged=False):
    return BoundaryInfo(number, 0.0, 0.9, True, overlap_match_db=agree, flagged=flagged)


# --- options ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("flagged", "flagged"),
        (" ALL ", "all"),
        ("3", [3]),
        ("3, 7,9", [3, 7, 9]),
    ],
)
def test_parse_seams(text, expected):
    assert parse_seams(text) == expected


@pytest.mark.parametrize("text", ["", "x", "1,,2", "0", "-2,3", "1.5"])
def test_parse_seams_rejects_what_is_not_a_list_of_seams(text):
    with pytest.raises(ValueError, match="seam"):
        parse_seams(text)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"window_s": 0}, "above 0"),
        ({"window_s": -1}, "above 0"),
        ({"window_s": math.inf}, "above 0"),
        ({"window_s": math.nan}, "above 0"),
        ({"window_s": "5"}, "above 0"),
        ({"audio_format": "ogg"}, "format must be one of"),
        ({"max_seams": 0}, "at least 1"),
        ({"max_seams": 2.5}, "at least 1"),
        ({"seams": "some"}, "'flagged', 'all' or a list"),
        ({"seams": []}, "'flagged', 'all' or a list"),
        ({"seams": [1, 0]}, "'flagged', 'all' or a list"),
    ],
)
def test_options_are_checked_when_made(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AuditionOptions("kit", **kwargs)


def test_the_format_is_written_the_way_the_splitter_does():
    assert AuditionOptions("kit", audio_format=".FLAC").audio_format == "flac"


def test_the_variants_are_the_three_ways_of_joining_in_order():
    straight, crossfade, search = variants_for("equal_power")
    assert [v["kind"] for v in (straight, crossfade, search)] == [
        "straight",
        "crossfade",
        "seam_search",
    ]
    assert crossfade["id"] == "equal_power" and "equal_power" in crossfade["label"]
    assert len({v["id"] for v in (straight, crossfade, search)}) == 3


# --- which seams ------------------------------------------------------------------------


def test_the_flagged_seams_are_chosen_worst_first_and_listed_in_track_order():
    report = MergeReport(
        boundaries=[
            _boundary(1, 20.0),
            _boundary(2, 4.0, flagged=True),
            _boundary(3, None, flagged=True),  # overlaps that did not correspond: the worst
            _boundary(4, 9.0, flagged=True),
            _boundary(5, 12.0, flagged=True),
        ]
    )
    options = AuditionOptions("kit", max_seams=3)
    assert [b.boundary for b in select_seams(report, options)] == [2, 3, 4]
    assert [b.boundary for b in select_seams(report, AuditionOptions("kit"))] == [2, 3, 4, 5]
    everything = AuditionOptions("kit", seams="all", max_seams=4)
    assert [b.boundary for b in select_seams(report, everything)] == [2, 3, 4, 5]


def test_listed_seams_are_taken_as_they_are_and_unknown_ones_dropped():
    report = MergeReport(boundaries=[_boundary(n, 10.0) for n in (1, 2, 3)])
    options = AuditionOptions("kit", seams=[3, 9, 1, 3], max_seams=1)
    assert [b.boundary for b in select_seams(report, options)] == [1, 3]


# --- a kit of a merge ---------------------------------------------------------------------


def test_a_kit_has_every_variant_of_every_seam_and_a_page_that_plays_them(service, tmp_path):
    kit_dir = tmp_path / "kit"
    report = _merge(service, tmp_path, AuditionOptions(str(kit_dir), seams="all"))
    kit = _kit(kit_dir)
    assert kit["format"] == 1 and kit["audio_format"] == "flac" and kit["window_s"] == 5.0
    assert [v["kind"] for v in kit["variants"]] == ["straight", "crossfade", "seam_search"]
    assert [s["seam"] for s in kit["seams"]] == [b.boundary for b in report.boundaries]
    assert kit["settings"]["anchored"] is False and kit["settings"]["method"] == "adaptive"

    names = sorted(os.listdir(kit_dir))
    assert names[:2] == [KIT_FILE, KIT_PAGE]
    for entry in kit["seams"]:
        assert entry["of"] == len(report.boundaries)
        lengths = []
        for variant in kit["variants"]:
            clip = entry["clips"][variant["id"]]
            assert clip["file"] == f"seam{entry['seam']:02d}_{variant['id']}.flac"
            decoded = AudioSegment.from_file(kit_dir / clip["file"])
            assert len(decoded) == pytest.approx(clip["length_ms"], abs=2)
            lengths.append(len(decoded))
        assert max(lengths) - min(lengths) <= 2  # the clips can be switched between mid-play
        assert 0 < entry["centre_ms"] < lengths[0]
        assert entry["metrics"]["overlap_match_db"] is not None


def test_the_window_decides_the_length_of_a_clip(service, tmp_path):
    lengths = {}
    for window in (1.0, 5.0):
        folder = tmp_path / f"kit{window}"
        _merge(service, tmp_path, AuditionOptions(str(folder), seams=[1], window_s=window))
        clip = _kit(folder)["seams"][0]["clips"]["straight"]
        lengths[window] = clip["length_ms"]
    # at least half the overlap and a little more either side, whatever the window
    assert lengths[1.0] == pytest.approx(2 * (service.manifest["overlap_ms"] / 2 + 500), abs=5)
    assert lengths[5.0] == pytest.approx(10000, abs=5)


@pytest.mark.parametrize("fmt", ["wav", "mp3"])
def test_other_audio_formats(service, tmp_path, fmt):
    folder = tmp_path / "kit"
    _merge(service, tmp_path, AuditionOptions(str(folder), seams=[1], audio_format=fmt))
    kit = _kit(folder)
    assert kit["audio_format"] == fmt
    for clip in kit["seams"][0]["clips"].values():
        assert clip["file"].endswith("." + fmt) and os.path.getsize(folder / clip["file"]) > 0


def test_a_clip_is_the_merge_as_heard_at_that_seam(service, tmp_path):
    folder = tmp_path / "kit"
    _merge(service, tmp_path, AuditionOptions(str(folder), seams="all"), anchor=True)
    merged = AudioSegment.from_file(tmp_path / "out.wav")
    kit = _kit(folder)
    assert kit["settings"]["anchored"] is True
    for entry in kit["seams"]:
        clip = AudioSegment.from_file(folder / entry["clips"]["adaptive"]["file"])
        start = entry["at_ms"] - entry["centre_ms"]
        around = merged[int(start) : int(start) + len(clip)]
        mono = lambda s: np.array(s.get_array_of_samples(), dtype=float)  # noqa: E731
        fit = anchor_fit(mono(around), mono(clip), SR // 20)
        assert fit.correlation > 0.999 and fit.recon_db > 25  # the same audio, to within a ms


def test_the_three_ways_really_differ(service, tmp_path):
    folder = tmp_path / "kit"
    _merge(service, tmp_path, AuditionOptions(str(folder), seams=[1]))
    entry = _kit(folder)["seams"][0]
    raw = {
        v: AudioSegment.from_file(folder / c["file"]).raw_data for v, c in entry["clips"].items()
    }
    assert raw["straight"] != raw["adaptive"]
    assert entry["clips"]["straight"]["fade_ms"] == 0
    assert entry["clips"]["adaptive"]["fade_ms"] > entry["clips"]["seam_search"]["fade_ms"] > 0
    assert entry["clips"]["seam_search"]["match_db"] is not None
    if raw["seam_search"] == raw["adaptive"]:
        assert entry["clips"]["seam_search"]["identical_to"] == "adaptive"
    else:
        assert entry["clips"]["seam_search"]["identical_to"] is None


def test_both_stems_and_the_original_show_in_the_kit(service, tmp_path):
    folder = tmp_path / "kit"
    _merge(
        service,
        tmp_path,
        AuditionOptions(str(folder), seams=[1]),
        anchor=True,
        other_match="vocals",
    )
    kit = _kit(folder)
    assert kit["settings"]["both_stems"] is True
    first, second = kit["seams"][0]["metrics"]["recon_db"]
    assert first > 20 and second > 20
    # the kit plays the stem that was asked for, as a mix of what the player will hear
    assert AudioSegment.from_file(folder / "seam01_adaptive.flac").channels == 1


def test_a_seam_search_that_changes_nothing_is_said_to_be_identical(tmp_path):
    seg = to_segment(np.zeros((1, SR * 10)))
    names = []
    for k in (1, 2):
        names.append(str(tmp_path / f"c{k}.wav"))
        seg.export(names[-1], format="wav")
    plan = MergePlan({"parts": ["c1.wav", "c2.wav"]}, 2000.0, names)
    report = MergeReport(boundaries=[_boundary(1, 1.0, flagged=True)])
    kit = export_auditions(plan, report, AuditionOptions(str(tmp_path / "kit")), "silence.wav")
    assert kit["source_file"] == "silence.wav"
    assert kit["seams"][0]["clips"]["seam_search"]["identical_to"] == "adaptive"


def test_no_seam_to_listen_to_means_no_kit(service, tmp_path):
    folder = tmp_path / "kit"
    report = MergeReport(boundaries=[_boundary(1, 40.0)])  # fine: nothing flagged
    plan = MergePlan(service.manifest, 2000.0, [])
    assert export_auditions(plan, report, AuditionOptions(str(folder))) is None
    assert not folder.exists()


def test_a_merge_without_overlap_has_no_seams_to_listen_to(make_service, tmp_path):
    flat = make_service(overlap_s=0)
    folder = tmp_path / "kit"
    report = _merge(flat, tmp_path, AuditionOptions(str(folder), seams="all"))
    assert not folder.exists() and not report.boundaries


def test_progress_is_reported_per_seam_and_can_stop_the_kit(service, tmp_path):
    calls = []
    options = AuditionOptions(
        str(tmp_path / "kit"), seams="all", progress=lambda i, n, name: calls.append((i, n, name))
    )
    _merge(service, tmp_path, options)
    assert calls == [(1, 2, "seam 1"), (2, 2, "seam 2")]

    class Stop(Exception):
        pass

    def stop(i, n, name):
        raise Stop

    with pytest.raises(Stop):  # a cancel is not swallowed like a failure of the kit is
        _merge(service, tmp_path, AuditionOptions(str(tmp_path / "k2"), seams="all", progress=stop))


def test_a_kit_that_cannot_be_written_does_not_cost_the_merged_file(service, tmp_path):
    blocked = tmp_path / "in_the_way"
    blocked.write_text("a file where the folder should be")
    report = _merge(service, tmp_path, AuditionOptions(str(blocked), seams="all"))
    assert (tmp_path / "out.wav").stat().st_size > 0
    assert any("could not prepare the listening kit" in w for w in report.warnings)


# --- the page -----------------------------------------------------------------------------


def test_the_page_carries_the_kit_and_the_player_and_needs_nothing_else(tmp_path):
    kit = {
        "format": 1,
        "created": "2026-01-01T00:00:00+00:00",
        "source_file": "a</script><b>.wav",  # must not end the script block it sits in
        "variants": variants_for("adaptive"),
        "seams": [],
    }
    path = write_kit_page(str(tmp_path), kit)
    page = open(path, encoding="utf-8").read()
    assert os.path.basename(path) == KIT_PAGE
    assert "/*PLAYER*/" not in page and "/*KIT*/" not in page
    assert page.count("</script>") == 2  # the player's and the page's own
    assert "SeamAudition" in page and "a<\\/script><b>.wav" in page
    assert 'src="' not in page and "http://" not in page.replace("http://www.w3.org", "")


def test_the_player_is_shipped_with_the_package():
    script = player_script()
    assert "SeamAudition" in script and "TEXT" in script
    import seamsplit

    root = os.path.dirname(seamsplit.__file__)
    assert os.path.isfile(os.path.join(root, "static", "audition.js"))
    assert os.path.isfile(os.path.join(root, "static", "audition_page.html"))


def test_the_player_and_the_gui_page_do_not_share_text_keys_by_accident():
    """The GUI page adds the player's text to its own (Object.assign), so a key both
    define would silently take the player's wording, and one language could lack a key."""
    import shutil
    import subprocess

    import seamsplit

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to read the texts out of the page")
    root = os.path.join(os.path.dirname(seamsplit.__file__), "static")
    program = r"""
const fs = require("fs");
const player = {};
new Function("window", fs.readFileSync(process.argv[1], "utf8"))(player);
const page = fs.readFileSync(process.argv[2], "utf8").match(/const TEXT = \{[\s\S]*?\n\};/)[0];
const host = new Function(page + "; return TEXT;")();
const keys = (t) => ({ en: Object.keys(t.en), vi: Object.keys(t.vi) });
console.log(JSON.stringify({ player: keys(player.SeamAudition.TEXT), host: keys(host) }));
"""
    out = subprocess.run(
        [
            node,
            "-e",
            program,
            os.path.join(root, "audition.js"),
            os.path.join(root, "index.html"),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    keys = json.loads(out)
    for language in ("en", "vi"):
        assert len(keys["player"][language]) > 20  # it did read them
        assert not set(keys["player"][language]) & set(keys["host"][language]), language
    assert set(keys["player"]["en"]) == set(keys["player"]["vi"])
    assert set(keys["host"]["en"]) == set(keys["host"]["vi"])


def _browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    playwright = sync_api.sync_playwright().start()
    try:
        return playwright, playwright.chromium.launch()
    except Exception as exc:  # no browser installed here
        playwright.stop()
        pytest.skip(f"no browser to run the page in: {exc}")


def test_the_page_works_in_a_browser_and_its_answers_load_back(service, tmp_path):
    folder = tmp_path / "kit"
    _merge(service, tmp_path, AuditionOptions(str(folder), seams="all"))
    playwright, browser = _browser()
    try:
        page = browser.new_page()
        problems = []
        page.on("pageerror", lambda e: problems.append(str(e)))
        page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
        page.goto((folder / KIT_PAGE).as_uri())
        page.wait_for_selector(".sa-seam")
        assert page.locator(".sa-seam").count() == 2
        for k in range(2):  # answer both, one of them "no audible difference"
            row = page.locator(".sa-seam").nth(k).locator(".sa-row").nth(1)
            row.locator("button").nth(0 if k == 0 else 3).click()
        with page.expect_download() as download:
            page.locator(".sa-summary button").click()
        saved = tmp_path / "choices.json"
        download.value.save_as(str(saved))
        assert problems == []
    finally:
        browser.close()
        playwright.stop()
    doc = load_choices(str(saved))
    assert [c["seam"] for c in doc["choices"]] == [1, 2]
    assert doc["choices"][1]["kind"] == "tie" and all(c["blind"] for c in doc["choices"])
    assert summarize_choices([doc])["choices"] == 2


def test_an_answer_sheet_filled_by_the_library_matches_the_one_the_page_writes(service, tmp_path):
    folder = tmp_path / "kit"
    _merge(service, tmp_path, AuditionOptions(str(folder), seams="all"))
    kit = _kit(folder)
    doc = record_choice(new_choices(kit), kit, 1, "adaptive", True)
    entry = doc["choices"][0]
    assert entry["kind"] == "crossfade" and entry["blind"] is True
    assert set(entry) == {
        "seam",
        "choice",
        "kind",
        "blind",
        "at",
        "metrics",
        "seam_search_identical",
    }


def test_chunks_of_different_formats_are_still_glued(tmp_path):
    from seamsplit.audition import _straight_join

    first = to_segment(np.zeros((1, SR * 6)))
    second = to_segment(np.zeros((2, SR * 6))).set_frame_rate(44100)
    names = [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]
    first.export(names[0], format="wav")
    second.export(names[1], format="wav")
    plan = MergePlan({"parts": ["a.wav", "b.wav"]}, 2000.0, names)
    glued, cut = _straight_join(plan, 0, 3.0)
    assert (glued.frame_rate, glued.channels) == (SR, 1)
    assert cut == 5000 and len(glued) == pytest.approx(10000, abs=2)  # 6 s + 6 s - the 2 s overlap
