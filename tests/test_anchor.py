"""Merging against the original chunks, and both stems of a service as one."""

import json
import math
import os

import numpy as np
import pytest
from seamsplit.audio import AudioSegment

from fake_service import SR, snr_db, to_array, to_segment
from seamsplit.anchor import (
    ANCHOR_MAX_GAIN_DB,
    AnchorOptions,
    _pack,
    _shift,
    other_output_path,
    prepare_chunk,
    split_stems,
)
from seamsplit.checks import _check_threshold
from seamsplit.dsp import anchor_fit
from seamsplit.matching import find_original_chunks
from seamsplit.merger import ChunkInfo, MergeReport, merge_audio


def _noise(n=SR * 6, seed=0):
    return np.random.default_rng(seed).standard_normal(n) * 0.2


def _seg(x, channels=1):
    return to_segment(np.atleast_2d(x) if channels == 1 else x)


# --- the measurement ----------------------------------------------------------------


@pytest.mark.parametrize("lag", [-300, 0, 123, 400])
def test_anchor_fit_recovers_lag_and_gain(lag):
    original = _noise()
    processed = (
        original[lag:] * 0.5 if lag > 0 else np.concatenate([np.zeros(-lag), original]) * 0.5
    )
    fit = anchor_fit(original, processed, 600)
    assert fit.lag == lag  # processed[k] ~ original[k + lag]
    assert fit.gain == pytest.approx(2.0, rel=1e-3)
    assert fit.correlation > 0.999
    assert fit.recon_db > 60


def test_anchor_fit_on_a_window_still_measures_the_whole():
    original = _noise(SR * 30)
    fit = anchor_fit(original, original[200:] * 1.5, 600, window=SR * 5)
    assert fit.lag == 200
    assert fit.gain == pytest.approx(1 / 1.5, rel=1e-3)


def test_anchor_fit_gain_ignores_noise_the_original_does_not_have():
    original = _noise()
    processed = original + np.random.default_rng(5).standard_normal(len(original)) * 0.1
    fit = anchor_fit(original, processed, 50)
    assert fit.gain < 1.0  # least squares, not a ratio of levels
    assert 5 < fit.recon_db < 40


def test_anchor_fit_of_silence_has_nothing_to_say():
    fit = anchor_fit(np.zeros(1000), _noise(1000), 10)
    assert (fit.correlation, fit.gain, fit.recon_db) == (0.0, 1.0, 0.0)


def test_shift_puts_silence_in_front_or_drops_it():
    x = np.arange(1, 6, dtype=float).reshape(-1, 1)
    assert _shift(x, 2)[:, 0].tolist() == [0, 0, 1, 2, 3, 4, 5]
    assert _shift(x, -2)[:, 0].tolist() == [3, 4, 5]
    assert _shift(x, 0) is x


@pytest.mark.parametrize(
    "output, word, expected",
    [
        ("out/song.mp3", "vocals", "out/song_vocals.mp3"),
        ("song.wav", " Lead Vocals! ", "song_Lead_Vocals.wav"),
        ("song", "???", "song_other"),
    ],
)
def test_other_output_path_puts_the_word_in_the_name(output, word, expected):
    assert other_output_path(output, word) == expected


def test_pack_counts_what_it_clips_and_split_stems_undoes_the_stacking():
    loud = np.array([[40000.0, -40000.0, 100.0], [1.0, 2.0, 3.0]]).T  # two channels of int16
    like = AudioSegment(b"\0\0" * 6, sample_width=2, frame_rate=SR, channels=2)
    packed, clipped = _pack([loud, loud * 0.0], like)
    assert clipped == 2 and packed.channels == 4
    first, second = split_stems(packed, 2)
    assert first.channels == second.channels == 2
    assert np.array(first.get_array_of_samples()).tolist() == [32767, 1, -32768, 2, 100, 3]
    assert not any(second.get_array_of_samples())


def test_check_threshold_takes_a_number_or_none():
    _check_threshold("recon_warn_db", None)
    _check_threshold("recon_warn_db", -3)
    for bad in (float("nan"), "15", True):
        with pytest.raises(ValueError, match="recon_warn_db"):
            _check_threshold("recon_warn_db", bad)


# --- one chunk ------------------------------------------------------------------------


def _stems(original, lag=0, gain=1.0, noise=0.002, seed=2):
    """A processed chunk of a mono ``original`` as [instrumental, vocals] stems."""
    rng = np.random.default_rng(seed)
    voice = np.convolve(original, np.ones(5) / 5, mode="same")
    parts = [original - voice, voice]
    out = []
    for p in parts:
        p = p + rng.standard_normal(len(p)) * noise
        p = p[lag:] if lag > 0 else np.concatenate([np.zeros(-lag), p])
        out.append(_seg(p * gain))
    return out


def test_prepare_chunk_with_nothing_to_do_hands_back_the_chunk():
    chunk = _seg(_noise())
    out, clipped = prepare_chunk(1, chunk, None, None, AnchorOptions())
    assert out is chunk and clipped == 0


def test_prepare_chunk_stacks_the_other_stem_and_pads_the_shorter_one():
    main, other = _seg(_noise()), _seg(_noise(SR * 6 - 3000))
    report = MergeReport()
    out, _ = prepare_chunk(2, main, other, None, AnchorOptions(), report, 60.0)
    assert out.channels == 2 and len(out.raw_data) == len(main.raw_data) * 2
    assert any("part 2" in w and "differ in length" in w for w in report.warnings)
    assert report.chunks == []  # nothing was compared with an original


def test_prepare_chunk_cancels_the_offset_and_the_level_of_both_stems():
    original = _noise()
    inst, voc = _stems(original, lag=250, gain=0.5)
    report = MergeReport()
    out, _ = prepare_chunk(1, inst, voc, _seg(original), AnchorOptions(), report)
    (info,) = report.chunks
    assert info.part == 1 and not info.flagged
    assert info.lag_ms == pytest.approx(1000 * 250 / SR, abs=0.1)
    assert info.gain_db == pytest.approx(6.02, abs=0.1)
    assert info.recon_db > 30
    a, b = split_stems(out, 1)
    summed = to_array(a)[0] + to_array(b)[0]
    # in place and at level (what the service trimmed off the start is gone, so skip it)
    assert snr_db(original, summed, skip=300) > 30
    assert abs(len(a) - len(_seg(original))) <= 1


def test_prepare_chunk_with_one_stem_anchors_the_time_and_leaves_the_level():
    original = _noise()
    inst, _ = _stems(original, lag=-200, gain=0.5)
    report = MergeReport()
    out, _ = prepare_chunk(1, inst, None, _seg(original), AnchorOptions(), report)
    (info,) = report.chunks
    assert info.lag_ms == pytest.approx(-1000 * 200 / SR, abs=0.1)
    assert info.gain_db is None and info.recon_db is None  # one stem says nothing of the mix
    assert out.channels == 1
    x = to_array(out)[0]
    assert np.abs(x).max() < np.abs(original).max()  # still quieter: level untouched


def test_prepare_chunk_respects_no_align_and_no_match_level():
    original = _noise()
    inst, voc = _stems(original, gain=0.5)  # no offset: nothing for the search to find
    report = MergeReport()
    options = AnchorOptions(max_lag_ms=0.0, level=False)
    out, _ = prepare_chunk(1, inst, voc, _seg(original), options, report)
    (info,) = report.chunks
    assert info.lag_ms is None and info.gain_db is None and info.recon_db > 30
    assert len(out) == len(inst)  # nothing shifted
    assert np.abs(to_array(out)).max() < 0.6 * np.abs(original).max()  # nor scaled
    # with --no-align a chunk that really is offset is not forced into place
    late, voc = _stems(original, lag=250)
    quiet = MergeReport()
    prepare_chunk(1, late, voc, _seg(original), options, quiet)
    assert quiet.chunks[0].lag_ms is None


def test_prepare_chunk_adds_extra_db():
    inst, voc = _stems(_noise() * 0.4)
    plain, _ = prepare_chunk(1, inst, voc, None, AnchorOptions(), None)
    louder, _ = prepare_chunk(1, inst, voc, None, AnchorOptions(), None, extra_db=6.0)
    assert np.abs(to_array(louder)).max() == pytest.approx(
        10 ** (6 / 20) * np.abs(to_array(plain)).max(), rel=0.01
    )


def test_prepare_chunk_says_when_a_chunk_does_not_match_its_original():
    report = MergeReport()
    out, _ = prepare_chunk(
        3, _seg(_noise(seed=1)), None, _seg(_noise(seed=2)), AnchorOptions(), report
    )
    assert any("part 3" in w and "barely correlates" in w for w in report.warnings)
    assert report.chunks[0].lag_ms is None and out.channels == 1


def test_prepare_chunk_says_when_the_offset_reaches_the_search_limit():
    # a signal that still resembles itself a few samples off, so that an offset just
    # beyond the search is seen as one at its edge (white noise would show nothing)
    smooth = np.convolve(_noise(), np.ones(50) / 50, mode="same")
    original = smooth / smooth.std() * 0.2
    inst, voc = _stems(original, lag=446)  # 20.2 ms, just beyond a 20 ms search (441 samples)
    report = MergeReport()
    prepare_chunk(1, inst, voc, _seg(original), AnchorOptions(max_lag_ms=20.0), report)
    assert any("search limit" in w and "--max-lag-ms" in w for w in report.warnings)


def test_prepare_chunk_flags_stems_that_do_not_add_up_and_does_not_take_their_level():
    original = _noise()
    inst, voc = _stems(original, noise=0.05)  # a fair amount of what is not in the original
    report = MergeReport()
    prepare_chunk(1, inst, voc, _seg(original), AnchorOptions(recon_warn_db=15), report)
    (info,) = report.chunks
    assert info.flagged and info.recon_db < 15
    assert any("worth a listen" in w for w in report.warnings)

    quiet = MergeReport()
    prepare_chunk(1, inst, voc, _seg(original), AnchorOptions(recon_warn_db=None), quiet)
    assert not quiet.chunks[0].flagged and not any("worth a listen" in w for w in quiet.warnings)


def test_prepare_chunk_does_not_take_a_level_it_cannot_trust():
    original = _noise()
    inst, voc = _stems(original, gain=10 ** (-(ANCHOR_MAX_GAIN_DB + 6) / 20))
    report = MergeReport()
    prepare_chunk(1, inst, voc, _seg(original), AnchorOptions(), report)
    assert report.chunks[0].gain_db is None and report.chunks[0].flagged
    assert any("level was not taken" in w and "limit" in w for w in report.warnings)

    # a chunk that only half agrees with the original: no gain either
    mixed = _seg(0.6 * original + 0.8 * _noise(seed=9))
    voice = _seg(np.zeros(len(original)))
    report = MergeReport()
    prepare_chunk(1, mixed, voice, _seg(original), AnchorOptions(), report)
    assert report.chunks[0].gain_db is None
    assert any("do not add up" in w for w in report.warnings)


def test_prepare_chunk_of_silence_measures_nothing():
    silence = _seg(np.zeros(SR))
    report = MergeReport()
    out, _ = prepare_chunk(1, silence, silence, silence, AnchorOptions(), report)
    assert report.chunks[0].lag_ms is None and not any(out.get_array_of_samples())


def test_prepare_chunk_counts_the_samples_a_boost_clips():
    inst, voc = _stems(_noise() / 0.2 * 0.5)
    _, clipped = prepare_chunk(1, inst, voc, None, AnchorOptions(), None)
    assert clipped == 0
    boosted, clipped = prepare_chunk(1, inst, voc, None, AnchorOptions(), None, extra_db=20.0)
    assert clipped > 0 and boosted.channels == 2


def test_a_plan_says_once_how_many_samples_were_clipped(service):
    from seamsplit.matching import find_processed_files
    from seamsplit.plan import MergePlan

    paths = find_processed_files(service.processed_dir, service.parts, "instrumental")
    plan = MergePlan(service.manifest, 2000.0, paths)
    report = MergeReport()
    list(plan.stream(range(len(paths)), report, extra_db=30.0))
    clipping = [w for w in report.warnings if "full scale" in w]
    assert len(clipping) == 1 and "clipped" in clipping[0]
    quiet = MergeReport()
    list(plan.stream(range(len(paths)), quiet))
    assert not quiet.warnings


# --- matching the originals ---------------------------------------------------------


def test_find_original_chunks_names_every_missing_one(tmp_path):
    names = ["a_part01.wav", "a_part02.wav", "a_part03.wav"]
    (tmp_path / names[1]).write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="2 of 3") as exc:
        find_original_chunks(str(tmp_path), names)
    assert "a_part01.wav" in str(exc.value) and "a_part03.wav" in str(exc.value)
    assert "--originals" in str(exc.value)
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    assert find_original_chunks(str(tmp_path), names) == [str(tmp_path / n) for n in names]


# --- the whole merge, against what the service did ------------------------------------


def _merge(service, tmp_path, name="out.wav", **kw):
    report = MergeReport()
    kw.setdefault("samplerate", SR)
    merge_audio(
        service.processed_dir,
        service.manifest_path,
        str(tmp_path / name),
        match="instrumental",
        report=report,
        **kw,
    )
    return report


def _load(path):
    return to_array(AudioSegment.from_file(path))


def test_the_anchored_merge_finds_every_offset_and_gain_the_service_applied(service, tmp_path):
    report = _merge(service, tmp_path, anchor=True, other_match="vocals")
    assert [c.part for c in report.chunks] == [1, 2, 3]
    for info, lag_ms, gain_db in zip(report.chunks, service.lag_ms, service.gains_db):
        assert info.lag_ms == pytest.approx(lag_ms, abs=0.1)
        assert info.gain_db == pytest.approx(-gain_db, abs=0.5)  # what undoes the service's
        assert info.recon_db > 25 and not info.flagged
    assert report.to_dict()["chunks"][0]["part"] == 1
    assert "Chunks against the original chunks" in report.to_markdown()


def test_both_stems_are_written_and_add_up_to_the_source(service, tmp_path):
    _merge(service, tmp_path, anchor=True, other_match="vocals")
    assert sorted(os.listdir(tmp_path)) == ["out.wav", "out_vocals.wav"]
    inst, voc = _load(tmp_path / "out.wav"), _load(tmp_path / "out_vocals.wav")
    assert abs(inst.shape[1] - service.song.shape[1]) <= SR // 100
    assert snr_db(service.song, inst + voc, skip=SR) > 20
    # without the originals the merge can only be as good as the chunks agree with each other
    _merge(service, tmp_path, "plain.wav", other_match="vocals")
    plain = _load(tmp_path / "plain.wav") + _load(tmp_path / "plain_vocals.wav")
    assert snr_db(service.song, inst + voc, skip=SR) > snr_db(service.song, plain, skip=SR)


def test_the_second_stem_goes_where_it_is_told_and_merges_like_the_first(service, tmp_path):
    _merge(
        service, tmp_path, anchor=True, other_match="vocals", other_output=str(tmp_path / "v.wav")
    )
    assert (tmp_path / "v.wav").exists() and not (tmp_path / "out_vocals.wav").exists()


def test_anchoring_with_one_stem_fixes_the_timing_only(service, tmp_path):
    report = _merge(service, tmp_path, anchor=True)
    assert [c.recon_db for c in report.chunks] == [None] * 3
    assert [c.gain_db for c in report.chunks] == [None] * 3
    for info, lag_ms in zip(report.chunks, service.lag_ms):
        assert info.lag_ms == pytest.approx(lag_ms, abs=0.1)
    assert os.listdir(tmp_path) == ["out.wav"]


def test_originals_can_be_somewhere_else(service, tmp_path):
    import shutil

    moved = tmp_path / "originals"
    shutil.copytree(service.chunks_dir, moved)
    shutil.copy(service.manifest_path, tmp_path / "manifest.json")
    report = MergeReport()
    merge_audio(
        service.processed_dir,
        str(tmp_path / "manifest.json"),
        str(tmp_path / "out.wav"),
        match="instrumental",
        samplerate=SR,
        anchor=True,
        originals_dir=str(moved),
        report=report,
    )
    assert len(report.chunks) == 3
    with pytest.raises(FileNotFoundError, match="original chunks"):
        merge_audio(  # next to a manifest that has no chunks beside it
            service.processed_dir,
            str(tmp_path / "manifest.json"),
            str(tmp_path / "again.wav"),
            match="instrumental",
            anchor=True,
        )


def test_a_service_that_normalizes_each_stem_on_its_own_is_flagged_not_trusted(
    make_service, tmp_path
):
    odd = make_service(normalize_stems_separately=True)
    report = _merge(odd, tmp_path, anchor=True, other_match="vocals")
    assert all(c.flagged and c.recon_db < 15 for c in report.chunks)
    assert any("worth a listen" in w for w in report.warnings)


def test_a_service_that_gives_back_exactly_what_it_got_needs_no_correction(make_service, tmp_path):
    exact = make_service(lags=[0, 0, 0], gains_db=[0.0, 0.0, 0.0])
    report = _merge(exact, tmp_path, anchor=True, other_match="vocals")
    for info in report.chunks:
        assert info.lag_ms == pytest.approx(0, abs=0.05)
        assert info.gain_db == pytest.approx(0, abs=0.3)


def test_stereo_chunks_keep_their_channels(make_service, tmp_path):
    stereo = make_service(channels=2)
    _merge(stereo, tmp_path, anchor=True, other_match="vocals")
    for name in ("out.wav", "out_vocals.wav"):
        assert AudioSegment.from_file(tmp_path / name).channels == 2
    total = _load(tmp_path / "out.wav") + _load(tmp_path / "out_vocals.wav")
    assert snr_db(stereo.song, total, skip=SR) > 20


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"other_output": "x.wav"}, "other_output needs other_match"),
        ({"originals_dir": "somewhere", "pairing": "off"}, "only used with anchor"),
        ({"other_match": "  "}, "must not be empty"),
        ({"other_match": "vocals", "other_output": "out.wav"}, "same file"),
        ({"other_match": "instrumental"}, "pick out the same files"),
        ({"recon_warn_db": float("nan")}, "recon_warn_db"),
        ({"audition": object()}, "needs a MergeReport"),
    ],
)
def test_merge_audio_checks_the_new_options_before_reading_anything(
    service, tmp_path, kwargs, message
):
    kwargs = dict(kwargs)
    if kwargs.get("other_output") == "out.wav":
        kwargs["other_output"] = str(tmp_path / "out.wav")
    with pytest.raises(ValueError, match=message):
        merge_audio(
            service.processed_dir,
            service.manifest_path,
            str(tmp_path / "out.wav"),
            match="instrumental",
            **kwargs,
        )
    assert not any(tmp_path.iterdir())


def test_the_report_json_keeps_its_old_shape_without_anchoring(service, tmp_path):
    report = _merge(service, tmp_path)
    assert set(report.to_dict()) == {"boundaries", "warnings"}
    assert "Chunks against" not in report.to_markdown()
    json.dumps(report.to_dict())
    assert isinstance(ChunkInfo(1), ChunkInfo) and math.isfinite(ChunkInfo(1).correlation)


# --- chunks that did not come back in the format they went in -------------------------------


def test_a_service_that_changes_the_format_is_still_compared_with_the_original():
    original = _noise()
    inst, voc = _stems(original, lag=250, gain=0.5)
    # the service returns 44.1 kHz stereo, 24-bit-ish: a different rate, width and channels
    inst = inst.set_frame_rate(44100).set_channels(2)
    voc = voc.set_frame_rate(44100).set_sample_width(4).set_channels(2)
    report = MergeReport()
    out, _ = prepare_chunk(1, inst, voc, _seg(original), AnchorOptions(), report)
    (info,) = report.chunks
    assert info.lag_ms == pytest.approx(1000 * 250 / SR, abs=0.2)
    assert info.gain_db == pytest.approx(6.0, abs=0.3) and info.recon_db > 20
    assert out.frame_rate == 44100 and out.channels == 4

    # ... and the original is the one that is converted, not the stems
    plain = MergeReport()
    mono, _ = prepare_chunk(
        1, inst, None, _seg(original).set_sample_width(1), AnchorOptions(), plain
    )
    assert plain.chunks[0].lag_ms == pytest.approx(1000 * 250 / SR, abs=0.2)
    assert mono.channels == 2
