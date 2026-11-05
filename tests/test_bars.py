"""Cutting on a bar line: where the bars start, and a cut that goes there."""

import json

import numpy as np
import pytest
from seamsplit.audio import AudioSegment

from seamsplit.cli import main
from seamsplit.cutpoints import (
    DEFAULT_BEATS_PER_BAR,
    MIN_BAR_CONFIDENCE,
    CutFeatures,
    analyze_audio,
    bar_phase,
    beat_grid,
    distance_to_beat_ms,
)
from seamsplit.splitter import (
    BAR_REACH_MS,
    SplitConfig,
    choose_cut_points,
    find_quietest_point,
    plan_cuts,
    split_audio,
)

SR = 22050


def _audio(x, channels=2):
    pcm = (np.clip(x, -1, 1) * 30000).astype(np.int16)
    return AudioSegment(pcm.reshape(-1).tobytes(), sample_width=2, frame_rate=SR, channels=channels)


def bar_mix(
    bpm=100, seconds=60.0, bar=4, accent=1.0, rest=(30.0, 34.0), seed=1, first=0.5, pad=0.04
):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    beat = 60.0 / bpm
    kick = np.zeros(n)
    bass = np.zeros(n)
    for k in range(int((seconds - first) / beat)):
        pos = k % bar
        i, m = int((first + k * beat) * SR), int(0.12 * SR)
        e = np.arange(m) / SR
        loud = 1.0 + accent * (pos == 0)
        kick[i : i + m] += (
            0.25 * loud * np.sin(2 * np.pi * (50 + 80 * np.exp(-e * 35)) * e) * np.exp(-e * 22)
        )
        if pos == 0:
            m2 = min(int(0.5 * SR), n - i)
            e2 = np.arange(m2) / SR
            f = 55.0 * (1.0 + 0.26 * ((k // bar) % 2))
            bass[i : i + m2] += (
                accent
                * 0.3
                * np.sin(2 * np.pi * f * e2)
                * np.minimum(e2 / 0.01, 1)
                * np.exp(-e2 * 4)
            )

    def pad_channel():
        sp = np.fft.rfft(rng.standard_normal(n))
        fr = np.fft.rfftfreq(n, 1 / SR)
        sp[(fr < 250) | (fr > 2500)] = 0
        x = np.fft.irfft(sp, n)
        return pad * x / x.std()

    phase = 2 * np.pi * np.cumsum(220 * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))) / SR
    voice = sum(np.sin(h * phase) / h for h in range(1, 9)) * 0.16
    voice *= 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 1.7 + seed))
    voice *= ((t < rest[0]) | (t > rest[1])).astype(float)
    centre = kick + bass + voice
    st = np.stack(
        [centre + pad_channel(), centre + pad_channel()], axis=1
    ) + 0.002 * rng.standard_normal((n, 2))
    return _audio(st / np.abs(st).max() * 0.66), dict(
        beat_ms=1000 * beat, bar_ms=1000 * beat * bar, first_ms=1000 * first
    )


def _true_bars(info, seconds=60.0):
    return info["first_ms"] + info["bar_ms"] * np.arange(int(seconds * 1000 / info["bar_ms"]))


REST = (30_000, 34_000)
CFG = dict(target_s=32, before_slack_s=8, after_slack_s=3, overlap_s=2, min_final_s=5)


# ---- where the bars start ---------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("accent", [1.0, 0.5, 0.25])
def test_the_bar_lines_are_found_when_the_first_beat_is_struck_harder(accent, seed):
    audio, info = bar_mix(accent=accent, seed=seed)
    grid = beat_grid(analyze_audio(audio), 4)
    assert grid.tempo_bpm == pytest.approx(100, rel=0.02)
    assert grid.bar_confidence >= MIN_BAR_CONFIDENCE and len(grid.bars_ms) >= 20
    assert np.mean(distance_to_beat_ms(grid.bars_ms, _true_bars(info)) <= 60) >= 0.9
    # one in four beats: they are beats of the tracker's own grid
    assert np.all(distance_to_beat_ms(grid.bars_ms, grid.beats_ms) < 1e-6)
    assert len(grid.beats_ms) / len(grid.bars_ms) == pytest.approx(4, rel=0.1)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_bar_line_is_claimed_when_every_beat_is_struck_alike(seed):
    audio, _ = bar_mix(accent=0.0, seed=seed)
    grid = beat_grid(analyze_audio(audio), 4)
    assert len(grid.beats_ms) > 50  # the beat is there ...
    assert grid.bar_confidence < MIN_BAR_CONFIDENCE and len(grid.bars_ms) == 0  # ... the bars not


def test_a_waltz_is_found_with_three_beats_to_the_bar_and_not_with_four():
    audio, info = bar_mix(bpm=110, bar=3, accent=1.0, seed=2)
    features = analyze_audio(audio)
    three = beat_grid(features, 3)
    assert three.bar_confidence >= MIN_BAR_CONFIDENCE and len(three.bars_ms) > 15
    assert np.mean(distance_to_beat_ms(three.bars_ms, _true_bars(info)) <= 60) >= 0.9
    four = beat_grid(features, 4)
    assert four.bar_confidence < three.bar_confidence


def test_too_few_beats_say_nothing_about_bars():
    audio, _ = bar_mix(seconds=12.0)  # about 20 beats: under six bars of four
    grid = beat_grid(analyze_audio(audio), 4)
    assert 10 < len(grid.beats_ms) < 24
    assert grid.bar_confidence is None and len(grid.bars_ms) == 0


def test_no_beat_no_bars():
    rng = np.random.default_rng(2)
    grid = beat_grid(analyze_audio(_audio(0.2 * rng.standard_normal(SR * 40), 1)), 4)
    assert len(grid.beats_ms) == 0 and len(grid.bars_ms) == 0
    assert grid.tempo_bpm is None and grid.bar_confidence is None


def test_bar_phase_needs_the_low_onset_curve_and_a_sensible_bar():
    audio, _ = bar_mix(seconds=30.0)
    features = analyze_audio(audio)
    frames = np.arange(40) * 50
    assert bar_phase(frames, features, 1) is None
    bare = CutFeatures(features.times_ms, features.voice_power, features.onset, 30.0, 10.0)
    assert bar_phase(frames, bare, 4) is None
    flat = CutFeatures(
        features.times_ms,
        features.voice_power,
        np.ones_like(features.onset),
        30.0,
        10.0,
        np.ones_like(features.onset),
    )
    assert bar_phase(frames, flat, 4) is None  # nothing differs from one beat to the next


def test_a_kick_is_in_the_low_onset_and_a_click_is_not():
    t = np.arange(SR * 2) / SR
    thump = np.zeros_like(t)
    click = np.zeros_like(t)
    thump[SR : SR + 2000] = 0.5 * np.sin(2 * np.pi * 70 * t[:2000]) * np.exp(-t[:2000] * 30)
    click[SR : SR + 200] = 0.5 * np.sin(2 * np.pi * 4000 * t[:200])
    low = analyze_audio(_audio(thump, 1)).low_onset
    high = analyze_audio(_audio(click, 1)).low_onset
    assert low.max() > 20 * high.max() and len(low) == len(analyze_audio(_audio(click, 1)).onset)


# ---- a cut on a bar line ----------------------------------------------------------------


def test_the_cut_goes_onto_a_bar_line_inside_the_rest():
    audio, info = bar_mix(accent=1.0)
    bars = _true_bars(info)
    plan = plan_cuts(audio, SplitConfig(**CFG, cut_by="voice", snap_beat=True, snap_bar=True))
    (cut,) = plan.cuts_ms[1:-1]
    assert REST[0] <= cut <= REST[1]
    assert distance_to_beat_ms(np.array([cut]), bars)[0] <= BAR_REACH_MS
    assert plan.snap_bar and plan.bar_confidence >= MIN_BAR_CONFIDENCE
    assert plan.bar_offsets_ms == [pytest.approx(0, abs=BAR_REACH_MS)]
    assert plan.beat_offsets_ms is not None and plan.tempo_bpm == pytest.approx(100, rel=0.02)
    # the same track with beats only: on a beat of the rest, not necessarily the bar line
    beats_only = plan_cuts(audio, SplitConfig(**CFG, cut_by="voice", snap_beat=True))
    assert REST[0] <= beats_only.cuts_ms[1] <= REST[1] and beats_only.bar_offsets_ms is None


def test_snap_bar_alone_still_uses_the_beats_and_the_voice_curve():
    audio, info = bar_mix(accent=1.0)
    plan = plan_cuts(audio, SplitConfig(**CFG, cut_by="voice", snap_bar=True))
    assert REST[0] <= plan.cuts_ms[1] <= REST[1] and plan.beat_offsets_ms[0] <= 30


def test_without_a_marked_bar_it_is_a_beat_snap_and_says_why():
    audio, _ = bar_mix(accent=0.0)
    barred = plan_cuts(audio, SplitConfig(**CFG, cut_by="voice", snap_bar=True))
    beaten = plan_cuts(audio, SplitConfig(**CFG, cut_by="voice", snap_beat=True))
    assert barred.cuts_ms == beaten.cuts_ms
    assert barred.bar_offsets_ms is None and barred.bar_confidence < MIN_BAR_CONFIDENCE
    assert barred.beat_offsets_ms == beaten.beat_offsets_ms


def test_a_track_without_a_beat_is_cut_as_before():
    rng = np.random.default_rng(2)
    audio = _audio(0.2 * rng.standard_normal(SR * 40), 1)
    cfg = dict(target_s=24, before_slack_s=8, after_slack_s=3, overlap_s=2, min_final_s=5)
    plan = plan_cuts(audio, SplitConfig(**cfg, snap_bar=True))
    assert plan.tempo_bpm is None and plan.bar_confidence is None and plan.bar_offsets_ms is None
    assert plan.cuts_ms == plan_cuts(audio, SplitConfig(**cfg)).cuts_ms


def test_the_default_plan_does_not_look_for_beats_or_bars(monkeypatch):
    import seamsplit.splitter as splitter

    def boom(*args, **kwargs):
        raise AssertionError("analysed what was not asked for")

    monkeypatch.setattr(splitter, "analyze_audio", boom)
    audio, _ = bar_mix(seconds=40.0)
    assert plan_cuts(audio, SplitConfig(**CFG)).bar_confidence is None


def test_the_search_prefers_a_bar_line_to_a_beat_but_only_when_one_is_in_reach():
    times = np.arange(0.0, 10_000.0, 10.0)
    rest = np.full_like(times, 1.0)  # flat: every point is as quiet as the next
    beats = np.arange(95.0, 10_000.0, 500.0)
    bars = np.array([3_095.0, 5_095.0])
    pick = find_quietest_point(times, rest, 2_000, 4_000, beats, 6.0, bars)
    assert abs(pick - 3_095) <= 10
    # no bar line within the window: the beat rule decides, as without bars
    far = np.array([7_095.0])
    beat_pick = find_quietest_point(times, rest, 2_000, 4_000, beats, 6.0)
    assert find_quietest_point(times, rest, 2_000, 4_000, beats, 6.0, far) == beat_pick
    # a bar line in a loud stretch is not worth it: only the quiet points count
    loud = rest.copy()
    loud[(times > 3_000) & (times < 3_200)] = 100.0
    assert abs(find_quietest_point(times, loud, 2_000, 4_000, beats, 6.0, bars) - 3_095) > 100
    # bar lines with no beats at all still count
    assert abs(find_quietest_point(times, rest, 2_000, 4_000, None, 6.0, bars) - 3_095) <= 10
    assert find_quietest_point(times, rest, 2_000, 4_000, None, 6.0, far) == 2_000.0


def test_choose_cut_points_passes_the_bar_lines_to_every_cut():
    times = np.arange(0.0, 90_000.0, 10.0)
    curve = np.full_like(times, 1.0)
    bars = np.arange(95.0, 90_000.0, 2_400.0)
    beats = np.arange(95.0, 90_000.0, 600.0)
    cuts = choose_cut_points(times, curve, 90_000, 25_000, 8_000, 3_000, 5_000, beats, 6.0, bars)
    assert len(cuts) > 3
    assert all(distance_to_beat_ms(np.array([c]), bars)[0] <= BAR_REACH_MS for c in cuts[1:-1])


# ---- options, manifest, command line ----------------------------------------------------------


@pytest.mark.parametrize("value", [1, 17, 0, -4])
def test_bar_beats_is_a_count_in_range(value):
    with pytest.raises(ValueError, match="between 2 and 16"):
        SplitConfig(snap_bar=True, bar_beats=value)


@pytest.mark.parametrize("value", [4.0, "4", True, None])
def test_bar_beats_must_be_a_whole_number(value):
    with pytest.raises(ValueError, match="whole number"):
        SplitConfig(bar_beats=value)


def test_the_manifest_records_the_bars(tmp_path):
    audio, _ = bar_mix(accent=1.0)
    source = tmp_path / "song.wav"
    audio.export(str(source), format="wav")
    plain = split_audio(
        str(source), SplitConfig(**CFG, output_dir=str(tmp_path / "a"), export_format="wav")
    )
    assert (plain["snap_bar"], plain["downbeat_confidence"], plain["cut_bar_offsets_ms"]) == (
        False,
        None,
        None,
    )
    assert plain["bar_beats"] == DEFAULT_BEATS_PER_BAR
    cfg = SplitConfig(
        **CFG, cut_by="voice", snap_bar=True, output_dir=str(tmp_path / "b"), export_format="wav"
    )
    barred = split_audio(str(source), cfg)
    assert barred["snap_bar"] is True and barred["downbeat_confidence"] >= MIN_BAR_CONFIDENCE
    assert len(barred["cut_bar_offsets_ms"]) == len(barred["parts"]) - 1
    saved = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert saved["cut_bar_offsets_ms"] == barred["cut_bar_offsets_ms"]


def _song(tmp_path, **kw):
    audio, _ = bar_mix(**kw)
    path = tmp_path / "song.wav"
    audio.export(str(path), format="wav")
    return str(path)


def test_the_command_says_how_well_the_cuts_sit_on_the_bar(tmp_path, capsys):
    song = _song(tmp_path, accent=1.0)
    args = ["split", song, "--target", "32", "--before-slack", "8", "--after-slack", "3"]
    args += ["--overlap", "2", "--cut-by", "voice", "--snap-bar", "--dry-run", "--format", "wav"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Bars of 4 beats found (confidence" in out
    assert "1 of 1 cuts are on a bar line, within 50 ms" in out
    assert "of 1 cuts are within 50 ms of a beat" in out  # the beat line is still there


def test_the_command_says_when_the_bars_cannot_be_told(tmp_path, capsys):
    song = _song(tmp_path, accent=0.0)
    args = ["split", song, "--target", "32", "--before-slack", "8", "--after-slack", "3"]
    args += ["--overlap", "2", "--snap-bar", "--dry-run", "--format", "wav"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "does not stand out enough" in out and "moved onto beats only" in out

    short = _song(tmp_path, seconds=12.0)
    args = ["split", short, "--target", "6", "--before-slack", "2", "--after-slack", "1"]
    args += ["--overlap", "1", "--min-final", "2", "--snap-bar", "--dry-run", "--format", "wav"]
    assert main(args) == 0
    assert "too few beats to tell where the bars start" in capsys.readouterr().out


def test_the_command_checks_bar_beats(tmp_path, capsys):
    song = _song(tmp_path, seconds=20.0)
    assert main(["split", song, "--snap-bar", "--bar-beats", "40", "--dry-run"]) == 1
    assert "between 2 and 16" in capsys.readouterr().err
