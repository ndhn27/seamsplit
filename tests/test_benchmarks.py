"""The benchmark harness (benchmarks/): its measures, its chunk sets and its experiments, on
short synthetic songs and a stand-in for the separator, so that it is known to work before
anyone spends half an hour separating real music with it."""

import json
import os
import sys

import numpy as np
import pytest

BENCH_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")
sys.path.insert(0, BENCH_ROOT)

from bench import (  # noqa: E402
    chunks,
    corpus,
    cuts,
    degraded,
    metrics,
    pairing,
    report,
    seams,
    workset,
)
from bench.fake import FakeSeparator  # noqa: E402
from seamsplit.audio import AudioSegment  # noqa: E402

RATE = 44100


def synthetic_song(seconds=20, seed=0):
    """Notes that decay, with a gap of near-silence every few seconds, in stereo."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    x = np.zeros(n)
    for start in np.arange(0.0, seconds - 0.5, 0.25):
        if (start % 5.0) > 4.2:  # a rest
            continue
        f = 110.0 * 2 ** (rng.integers(0, 24) / 12)
        i = int(start * RATE)
        m = min(n - i, int(0.6 * RATE))
        env = np.exp(-6 * np.arange(m) / RATE)
        x[i : i + m] += 0.2 * env * np.sin(2 * np.pi * f * t[:m]) + 0.05 * env * np.sin(
            2 * np.pi * 6 * f * t[:m]
        )
    x += 0.002 * rng.standard_normal(n)
    left = np.clip(x, -1, 1)
    right = np.clip(0.8 * np.roll(x, 40), -1, 1)
    data = (np.stack([left, right], axis=1) * 32767).astype(np.int16)
    return AudioSegment.from_samples(data.reshape(-1), frame_rate=RATE, channels=2)


@pytest.fixture(scope="module")
def songs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("songs")
    out = []
    for k, seconds in enumerate((20, 24)):
        mix = synthetic_song(seconds, seed=k)
        path = str(folder / f"song{k}.wav")
        mix.export(path, format="wav")
        out.append(
            corpus.Song(
                id=f"song{k}",
                title=f"Synthetic {k}",
                path=path,
                mix=mix,
                vocals=None,
                instrumental=None,
                redistributable=True,
            )
        )
    return out


@pytest.fixture(scope="module")
def works(songs, tmp_path_factory):
    work_dir = str(tmp_path_factory.mktemp("work"))
    prepared = workset.prepare(songs, work_dir)
    workset.separate_all(prepared, FakeSeparator(str(tmp_path_factory.mktemp("stems"))))
    return prepared, work_dir


@pytest.fixture(scope="module")
def seams_result(works, tmp_path_factory):
    prepared, work_dir = works
    return seams.run(prepared, work_dir, str(tmp_path_factory.mktemp("merged")))


@pytest.fixture(scope="module")
def pairing_result(works):
    prepared, work_dir = works
    return pairing.run(prepared, work_dir, trials=2, seed=3)


@pytest.fixture(scope="module")
def degraded_result(works, tmp_path_factory):
    prepared, work_dir = works
    return degraded.run(prepared, work_dir, str(tmp_path_factory.mktemp("damaged")))


# the corpus -------------------------------------------------------------------------------


def test_every_corpus_file_is_pinned_by_commit_and_hash():
    data = corpus.load_corpus()
    assert data["songs"]
    for song in data["songs"]:
        assert "mix" in song["files"]
        for entry in song["files"].values():
            assert len(entry["commit"]) == 40
            assert len(entry["sha256"]) == 64
            assert entry["path"].endswith(".mp3")
        url = corpus.raw_url(song["files"]["mix"])
        assert url.startswith("https://raw.githubusercontent.com/")
        assert " " not in url


def test_songs_with_ground_truth_have_both_stems():
    for song in corpus.load_corpus()["songs"]:
        roles = set(song["files"])
        assert roles in ({"mix"}, {"mix", "vocals", "instrumental"})


def test_a_damaged_download_is_refused(tmp_path, monkeypatch):
    entry = {"repo": "a/b", "commit": "0" * 40, "path": "x.mp3", "sha256": "1" * 64}
    data = {"songs": [{"id": "s", "files": {"mix": entry}}]}

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"not the file"

    monkeypatch.setattr(corpus.urllib.request, "urlopen", lambda *a, **k: Reply())
    with pytest.raises(ValueError, match="sha256"):
        corpus.fetch(data, str(tmp_path), say=lambda *_: None)
    assert not os.path.exists(corpus.local_path(data["songs"][0], "mix", str(tmp_path)))


# the measures -----------------------------------------------------------------------------


def test_ratio_is_the_plain_energy_ratio():
    a = np.ones(1000)
    assert metrics.ratio_db(a, a) == 99.0
    assert metrics.ratio_db(a, a * 0.9) == pytest.approx(20.0, abs=1e-6)
    assert metrics.ratio_db(a, np.zeros(1000)) == pytest.approx(0.0, abs=1e-6)


def test_windows_are_the_cut_point_plus_and_minus_half():
    assert metrics.window_slices([1000], 500, 1000, 5000) == [(500, 1500)]
    assert metrics.window_slices([100], 500, 1000, 5000) == [(0, 600)]  # held inside the file
    assert metrics.window_slices([9000], 500, 1000, 5000) == []


def test_error_is_counted_in_the_window_it_is_in():
    song = synthetic_song(10)
    frames = song.frames.copy()
    frames[5 * RATE : 5 * RATE + 2000] = 0  # damage near 5 s
    broken = AudioSegment.from_samples(frames.reshape(-1), frame_rate=RATE, channels=2)
    near = metrics.snr_pooled_windows_db(song, broken, [5000], 500)
    far = metrics.snr_pooled_windows_db(song, broken, [2000], 500)
    away = metrics.snr_outside_windows_db(song, broken, [5000], 500)
    assert near < 10 and far == 99.0 and away == 99.0


def test_a_level_that_is_off_as_a_whole_is_not_an_error_once_fitted():
    song = synthetic_song(8)
    quiet = song.apply_gain(-4)
    plain = metrics.snr_db(song, quiet)
    fitted, _, shift = metrics.snr_fitted(song, quiet)
    assert plain < 15 and fitted > 40 and shift == 0.0


def test_a_position_that_is_off_as_a_whole_is_not_an_error_once_fitted():
    song = synthetic_song(8)
    late = degraded.shift(song, 20)
    assert len(late) == len(song)
    plain = metrics.snr_db(song, late)
    fitted, _, shift = metrics.snr_fitted(song, late)
    assert plain < 5 and shift == pytest.approx(-20.0, abs=0.1) and fitted > 15  # negative: late


def test_a_click_stands_out_and_a_smooth_join_does_not():
    smooth = synthetic_song(6)
    frames = smooth.frames.copy().astype(np.int32)
    at = int(3.125 * RATE)  # in the middle of a note, not on an onset
    frames[at:] += 4000  # a step
    stepped = AudioSegment.from_samples(
        np.clip(frames, -32768, 32767).astype(np.int16).reshape(-1), frame_rate=RATE, channels=2
    )
    assert metrics.click_db(stepped, 3125) > metrics.click_db(smooth, 3125) + 10


def test_stems_that_add_up_to_the_mix_say_so():
    song = synthetic_song(6)
    half = song.apply_gain(-6.0206)
    assert metrics.reconstruction_db(song, half, half) > 30
    assert metrics.reconstruction_db(song, half, half.apply_gain(-12)) < 20


# the chunk sets ---------------------------------------------------------------------------


def test_chunk_sets_are_cut_as_they_say(songs, tmp_path):
    for name, options in chunks.CHUNK_SETS.items():
        manifest = chunks.make_chunk_set(songs[0].path, name, str(tmp_path / name))
        assert manifest["set"] == name
        assert manifest["overlap_ms"] == int(options["overlap_s"] * 1000)
        assert all(os.path.isfile(p) for p in manifest["paths"])
        assert len(manifest["parts"]) >= 2
    fixed = chunks.make_chunk_set(songs[0].path, "fixed2", str(tmp_path / "again"))
    ranges = [s["end_ms"] - s["start_ms"] for s in fixed["segments"]]
    assert ranges[0] == 9000  # 8 s plus the half of the overlap that goes into the next


def test_every_millisecond_of_the_song_belongs_to_one_chunk(songs, tmp_path):
    for name in chunks.CHUNK_SETS:
        manifest = chunks.make_chunk_set(songs[1].path, name, str(tmp_path / name))
        spans = chunks.chunk_ranges_ms(manifest)
        assert spans[0]["from_ms"] == 0
        assert spans[-1]["to_ms"] == manifest["source_length_ms"]
        for a, b in zip(spans, spans[1:]):
            assert a["to_ms"] == b["from_ms"]
        for span in spans:  # what a chunk owns lies inside what it holds
            assert span["start_ms"] <= span["from_ms"] < span["to_ms"] <= span["end_ms"]
        assert len(chunks.seam_centres_ms(manifest)) == len(spans) - 1


def test_an_unknown_chunk_set_is_named():
    with pytest.raises(ValueError, match="nope"):
        chunks.only_sets(["quiet2", "nope"])


# the experiments --------------------------------------------------------------------------


def test_the_fake_separator_says_it_is_fake_and_adds_up(songs, tmp_path):
    fake = FakeSeparator(str(tmp_path))
    assert "FAKE" in fake.name
    stems = fake.stems_for([songs[0].path])[0]
    low = AudioSegment.from_file(stems["instrumental"])
    high = AudioSegment.from_file(stems["vocals"])
    assert metrics.reconstruction_db(songs[0].mix, low, high) > 60


def test_hard_cut_gives_back_the_song_when_the_stems_are_the_song(works):
    prepared, work_dir = works
    chunk_set = prepared[0].sets["fixed2"]
    joined = seams.hard_cut(chunk_set, "instrumental")
    whole = AudioSegment.from_file(prepared[0].reference["instrumental"])
    assert abs(len(joined) - len(whole)) <= 1
    assert metrics.snr_db(whole, joined) > 30  # the fake sees only the chunk, not the song


def test_seams_experiment_covers_every_way_of_joining(seams_result):
    result = seams_result
    song = result["song0"]
    assert set(song["sets"]["quiet2"]["variants"]) == set(seams.VARIANTS)
    # nothing to merge when there is no overlap: only the plain join
    assert set(song["sets"]["fixed0"]["variants"]) == {"hardcut"}
    entry = song["sets"]["quiet2"]["variants"]["adaptive"]
    assert set(entry["stems"]) == {"vocals", "instrumental"}
    assert len(entry["stems"]["vocals"]["click_db"]) == song["sets"]["quiet2"]["chunks"] - 1
    summary = seams.summarise(result)
    adaptive = summary["quiet2"]["adaptive"]
    expected = sum((r["sets"]["quiet2"]["chunks"] - 1) * 2 for r in result.values())  # two stems
    assert adaptive["seams_compared"] == expected
    assert "seams_better_than_hardcut" not in summary["quiet2"]["hardcut"]
    json.dumps(result)  # it is written as JSON


def test_cuts_experiment_measures_every_cut(works, seams_result):
    prepared, _ = works
    found = cuts.run(prepared, seams_result)
    summary = cuts.summarise(found)
    assert set(summary) == set(chunks.CHUNK_SETS)
    for name, row in summary.items():
        assert row["cuts"] > 0 and 0 <= row["percent_of_song_quieter_mean"] <= 100
    # cut at the quietest place of a window, a cut is in a quieter place than at fixed times
    assert (
        summary["quiet2"]["percent_of_song_quieter_mean"]
        <= summary["fixed2"]["percent_of_song_quieter_mean"] + 1e-9
    )


def test_pairing_experiment_puts_renamed_files_back(pairing_result):
    summary = pairing.summarise(pairing_result)
    for stem, row in summary.items():
        assert row["shuffled_trials"] > 0
        assert row["pairing_auto"] == {"identical": row["shuffled_trials"]}
        assert "identical" not in row["pairing_off"]  # the check is what puts them right


def test_damage_experiment_shows_what_merging_undoes(degraded_result):
    found = degraded_result
    summary = degraded.summarise(found)
    assert set(summary) == set(degraded.DAMAGE)
    # a level that differs from chunk to chunk is put right by the level matching
    assert summary["gain"]["default"]["vs_clean_db"] > summary["gain"]["off"]["vs_clean_db"] + 10
    # and a position that does by the alignment
    assert summary["offset"]["default"]["vs_clean_db"] > summary["offset"]["off"]["vs_clean_db"] + 5
    applied = next(iter(found.values()))["damage"]["all"]["applied"]
    assert any(a["gain_db"] != 0 for a in applied) and any(a["offset_ms"] != 0 for a in applied)


def test_the_same_damage_is_made_every_time(works, tmp_path):
    prepared, _ = works
    chunk_set = prepared[0].sets["quiet2"]
    a = degraded.damage_files(chunk_set, str(tmp_path / "a"), "all", "seed")
    b = degraded.damage_files(chunk_set, str(tmp_path / "b"), "all", "seed")
    c = degraded.damage_files(chunk_set, str(tmp_path / "c"), "all", "other")
    assert a == b and a != c


def test_the_report_is_written_from_the_results(
    works, seams_result, pairing_result, degraded_result, tmp_path
):
    prepared, _ = works
    environment = {
        "date": "2026-01-01",
        "cpu": "a cpu",
        "cpu_threads": 2,
        "memory_gb": 7.0,
        "platform": "linux",
        "python": "3.13",
        "numpy": "2",
        "audio_engine": {"pyav": "14.0.0", "ffmpeg": "7.1"},
        "separator": {"name": "FAKE", "model_sha256": "ab" * 32, "versions": {}},
        "songs": {"song0": {"title": "Synthetic 0", "seconds": 20.0, "has_truth": False}},
    }
    perf = {
        "input_seconds": 300.0,
        "repeats": 1,
        "current": {
            "decode_wav": {"seconds_median": 0.2, "peak_rss_mb": 200.0},
            "merge": {"seconds_median": 1.5, "peak_rss_mb": 230.0},
        },
        "legacy": {
            "decode_wav": {"seconds_median": 0.1, "peak_rss_mb": 118.0},
            "merge": {"not_available": True},
        },
        "executable": {
            "path": "seamsplit",
            "size_mb": 61.0,
            "start_seconds_median": 1.4,
            "start_seconds_first": 6.0,
            "split_with_empty_path": {"seconds": 1.5, "exit_code": 0, "chunks_written": 5},
        },
    }
    for name, data in (
        ("environment", environment),
        ("seams", {"songs": seams_result, "summary": seams.summarise(seams_result)}),
        ("cuts", {"songs": {}, "summary": cuts.summarise(cuts.run(prepared, seams_result))}),
        ("pairing", {"summary": pairing.summarise(pairing_result)}),
        ("degraded", {"summary": degraded.summarise(degraded_result)}),
        ("perf", perf),
        ("backend", {"skipped": "needs a real separator"}),
    ):
        with open(os.path.join(str(tmp_path), name + ".json"), "w") as f:
            json.dump(data, f, default=float)
    path = report.write(str(tmp_path))
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for heading in (
        "## What it was measured on",
        "## 1. Seams",
        "## 2. Cut points",
        "## 3. Pairing",
        "## 4. What services do",
        "## 5. Speed and memory",
        "## 6. The whole program",
    ):
        assert heading in text
    assert "`adaptive`" in text and "`hardcut`" in text and "not in it" in text
    assert "needs a real separator" in text
