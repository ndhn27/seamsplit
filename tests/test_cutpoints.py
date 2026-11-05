"""Cut points that know more than loudness: the voice band, and the beat."""

import json

import numpy as np
import pytest
from seamsplit.audio import AudioSegment

from seamsplit.cli import main
from seamsplit.cutpoints import (
    CUT_BY,
    analyze_audio,
    beat_times_ms,
    distance_to_beat_ms,
    track_beats,
    voice_amplitude,
)
from seamsplit.splitter import (
    SplitConfig,
    choose_cut_points,
    compute_rms_curve,
    find_quietest_point,
    plan_cuts,
    split_audio,
)

SR = 22050


def _audio(samples, channels=1, sr=SR):
    """float samples in [-1, 1] (one column per channel) -> a 16-bit AudioSegment"""
    pcm = (np.clip(np.asarray(samples, dtype=np.float64), -1, 1) * 30000).astype(np.int16)
    return AudioSegment(pcm.reshape(-1).tobytes(), sample_width=2, frame_rate=sr, channels=channels)


def _tone(freq, seconds=1.0, amp=0.5, sr=SR, phase=0.0):
    t = np.arange(int(sr * seconds)) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


def _voice_level(audio):
    return float(np.median(voice_amplitude(analyze_audio(audio).voice_power))) / 30000


# ---- the analysis frames are the RMS curve's frames -------------------------------


@pytest.mark.parametrize("sr, channels", [(SR, 1), (SR, 2), (44100, 2), (16000, 1)])
def test_analysis_frames_are_those_of_the_rms_curve(sr, channels):
    rng = np.random.default_rng(0)
    audio = _audio(0.2 * rng.standard_normal((sr * 3, channels)), channels, sr)
    features = analyze_audio(audio)
    times, rms = compute_rms_curve(audio)
    assert np.array_equal(features.times_ms, times)
    assert len(features.voice_power) == len(features.onset) == len(rms)


def test_audio_shorter_than_one_window_has_no_frames():
    features = analyze_audio(_audio(_tone(440, 0.01)))
    assert len(features.times_ms) == len(features.voice_power) == len(features.onset) == 0


# ---- what counts as voice -----------------------------------------------------------


def test_a_tone_in_the_voice_band_is_measured_at_its_rms():
    mono = _voice_level(_audio(_tone(1000, amp=0.5)))
    assert mono == pytest.approx(0.5 / np.sqrt(2), rel=0.05)
    centre = np.stack([_tone(1000, amp=0.5)] * 2, axis=1)
    assert _voice_level(_audio(centre, 2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)


@pytest.mark.parametrize("freq", [60, 120, 6000, 9000])
def test_tones_outside_the_voice_band_do_not_count(freq):
    assert _voice_level(_audio(_tone(freq, amp=0.8))) < 0.01


def test_in_stereo_only_what_the_channels_share_counts():
    tone = _tone(1000, amp=0.5)
    quarter_turn = _tone(1000, amp=0.5, phase=np.pi / 2)
    assert _voice_level(_audio(np.stack([tone, -tone], axis=1), 2)) == 0.0  # out of phase
    assert _voice_level(_audio(np.stack([tone, quarter_turn], axis=1), 2)) < 0.01
    assert _voice_level(_audio(np.stack([tone, np.zeros_like(tone)], axis=1), 2)) < 0.01  # one side


def test_unrelated_noise_in_two_channels_averages_to_nothing():
    rng = np.random.default_rng(3)
    wide = 0.3 * rng.standard_normal((SR * 4, 2))
    centre = _tone(1000, seconds=4, amp=0.08)
    plain = _voice_level(_audio(wide, 2))
    with_voice = _voice_level(_audio(wide + centre[:, None], 2))
    assert plain < 0.02  # 0.3 RMS of noise in the band, and (almost) none of it counted
    assert with_voice > 2 * plain


# ---- beats ------------------------------------------------------------------------


def _clicks(bpm_at, seconds=40.0, gap=None, sr=SR, seed=0):
    """Low thumps on a beat that may speed up; returns (audio, true beat times in ms)."""
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(int(sr * seconds))
    times, t = [], 0.4
    while t < seconds - 0.3:
        times.append(t * 1000)
        if not (gap and gap[0] <= t < gap[1]):
            i, m = int(t * sr), int(0.06 * sr)
            k = np.arange(m)
            x[i : i + m] += np.exp(-k / (0.012 * sr)) * (
                0.8 * np.sin(2 * np.pi * 90 * k / sr) + 0.3 * rng.standard_normal(m)
            )
        t += 60.0 / bpm_at(t)
    return _audio(x), np.array(times)


@pytest.mark.parametrize("bpm", [90, 120, 150])
def test_a_steady_beat_is_found_with_its_tempo(bpm):
    audio, truth = _clicks(lambda t: bpm)
    beats, tempo = beat_times_ms(analyze_audio(audio))
    assert tempo == pytest.approx(bpm, rel=0.02)
    assert np.mean(distance_to_beat_ms(truth, beats) <= 30) >= 0.95


def _interior_misses(audio, truth, within_ms):
    """Share of the true beats, but the first and the last of the track (the least
    certain ones), that have a tracked beat within ``within_ms``."""
    beats, _ = beat_times_ms(analyze_audio(audio))
    return float(np.mean(distance_to_beat_ms(truth[1:-1], beats) <= within_ms))


def test_the_beat_is_followed_through_a_rest():
    audio, truth = _clicks(lambda t: 120, gap=(14.0, 18.0))  # four seconds with no click at all
    assert _interior_misses(audio, truth, 30) >= 0.97


def test_the_beat_follows_a_drift_of_tempo():
    audio, truth = _clicks(lambda t: 100 + 12 * t / 40)
    assert _interior_misses(audio, truth, 30) >= 0.97


def test_in_a_rest_while_the_tempo_drifts_the_beat_is_only_extrapolated():
    # no onsets to follow for four seconds, so the spacing is the one it had: the error
    # grows to a few tens of milliseconds, not to a beat
    audio, truth = _clicks(lambda t: 100 + 12 * t / 40, gap=(14.0, 18.0))
    assert _interior_misses(audio, truth, 30) >= 0.85
    assert _interior_misses(audio, truth, 80) == 1.0


def test_no_beat_in_silence_noise_a_steady_tone_or_a_very_short_track():
    rng = np.random.default_rng(1)
    quiet = [
        _audio(np.zeros(SR * 20)),
        _audio(_tone(440, 30)),
        _audio(_tone(440, 30, 0.3) + _tone(660, 30, 0.3)),
    ]
    quiet += [_audio(0.2 * rng.standard_normal(SR * 30)) for _ in range(6)]  # nothing repeats
    for walk in range(3):
        brown = np.cumsum(rng.standard_normal(SR * 30))
        quiet.append(_audio(brown / np.abs(brown).max() / 1.2))
    for audio in quiet:
        beats, tempo = beat_times_ms(analyze_audio(audio))
        assert len(beats) == 0 and tempo is None
    assert track_beats(np.ones(10), 100.0)[1] is None


def test_distance_to_the_nearest_beat():
    beats = np.array([100.0, 200.0, 300.0])
    got = distance_to_beat_ms(np.array([0.0, 140.0, 160.0, 300.0, 500.0]), beats)
    assert got.tolist() == [100.0, 40.0, 40.0, 0.0, 200.0]
    assert np.all(np.isinf(distance_to_beat_ms(np.array([1.0]), np.empty(0))))


# ---- choosing among the quiet points ---------------------------------------------------


def _flat_rest():
    times = np.arange(0, 2000, 10.0)
    curve = np.full(len(times), 100.0)
    curve[(times >= 500) & (times <= 1500)] = 1.0  # a rest, flat all the way
    return times, curve


def test_a_cut_in_a_rest_moves_onto_the_beat():
    times, curve = _flat_rest()
    beats = np.array([450.0, 870.0, 1290.0, 1710.0])
    assert find_quietest_point(times, curve, 0, 2000) == 500.0  # the first of equals
    assert find_quietest_point(times, curve, 0, 2000, beats) == 870.0


def test_a_beat_is_not_worth_a_cut_in_something_louder_than_the_tolerance():
    times, curve = _flat_rest()
    curve[(times >= 860) & (times <= 880)] = 3.0  # the beat at 870 is 9.5 dB over the rest
    beats = np.array([870.0])
    # too loud to cut in at 6 dB: the point of the rest nearest to the beat instead
    assert find_quietest_point(times, curve, 0, 2000, beats, tolerance_db=6) == 850.0
    assert find_quietest_point(times, curve, 0, 2000, beats, tolerance_db=0) == 850.0
    assert find_quietest_point(times, curve, 0, 2000, beats, tolerance_db=12) == 870.0
    curve[(times >= 1280) & (times <= 1300)] = 1.6  # a beat at 1290, only 4 dB over the rest
    assert find_quietest_point(times, curve, 0, 2000, np.array([1290.0]), tolerance_db=6) == 1290.0


def test_a_digital_silence_rest_can_still_be_snapped():
    times = np.arange(0, 2000, 10.0)
    curve = np.where((times >= 500) & (times <= 1500), 0.0, 50.0)
    assert find_quietest_point(times, curve, 0, 2000, np.array([1000.0])) == 1000.0


def test_without_beats_the_cuts_are_what_they_always_were():
    rng = np.random.default_rng(4)
    times = np.arange(0, 200_000, 10.0)
    curve = rng.uniform(1, 100, len(times))
    args = (times, curve, 200_000.0, 50_000.0, 15_000.0, 3_000.0, 5_000.0)
    assert choose_cut_points(*args) == choose_cut_points(*args, beats_ms=None)
    assert choose_cut_points(*args) == choose_cut_points(*args, beats_ms=np.empty(0))


# ---- a mix: bass, kick, wide pad, centred voice -----------------------------------------------


def _mix(bpm=100, seconds=40, pad=0.04, seed=1, rest=(22.0, 25.0), dip=(18.0, 18.9)):
    """Loud bass with a dip, a kick on every beat, a wide pad, and a centred voice
    that rests for three seconds. Where it is quietest overall is not where the voice rests."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    beat, first = 60.0 / bpm, 0.5
    bass = 0.55 * np.sin(2 * np.pi * 55 * t) * (0.85 + 0.15 * np.sin(2 * np.pi * t / beat))
    bass *= 1 - np.clip(np.minimum((t - dip[0]) / 0.05, (dip[1] - t) / 0.05), 0, 1)
    kick = np.zeros(n)
    for k in range(int((seconds - first) / beat)):
        i, m = int((first + k * beat) * SR), int(0.12 * SR)
        e = np.arange(m) / SR
        kick[i : i + m] += (
            0.5 * np.sin(2 * np.pi * (50 + 80 * np.exp(-e * 35)) * e) * np.exp(-e * 22)
        )

    def pad_channel():
        spectrum = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1 / SR)
        spectrum[(freqs < 250) | (freqs > 2500)] = 0
        x = np.fft.irfft(spectrum, n)
        return pad * x / x.std()

    phase = 2 * np.pi * np.cumsum(220 * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))) / SR
    voice = sum(np.sin(h * phase) / h for h in range(1, 9)) * 0.16
    voice *= 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 1.7 + seed))
    voice *= ((t < rest[0]) | (t > rest[1])).astype(float)
    centre = bass + kick + voice
    stereo = np.stack([centre + pad_channel(), centre + pad_channel()], axis=1)
    stereo += 0.002 * rng.standard_normal(stereo.shape)
    return _audio(stereo / np.abs(stereo).max() * 0.66, 2), {
        "beat_ms": 1000 * beat,
        "first_ms": 1000 * first,
    }


MIX_CFG = dict(target_s=24, before_slack_s=8, after_slack_s=3, overlap_s=2, min_final_s=5)
REST, DIP = (22_000, 25_000), (18_000, 18_900)


def test_loudness_cuts_where_the_bass_dips_and_voice_activity_where_the_voice_rests():
    audio, _ = _mix()
    by_loudness = plan_cuts(audio, SplitConfig(**MIX_CFG)).cuts_ms[1]
    by_voice = plan_cuts(audio, SplitConfig(**MIX_CFG, cut_by="voice")).cuts_ms[1]
    assert DIP[0] <= by_loudness <= DIP[1]  # quietest overall, mid-phrase
    assert REST[0] <= by_voice <= REST[1]  # the singer is not singing


def test_snapping_puts_the_cut_on_a_beat_inside_the_rest():
    audio, info = _mix()
    plan = plan_cuts(audio, SplitConfig(**MIX_CFG, cut_by="voice", snap_beat=True))
    assert REST[0] <= plan.cuts_ms[1] <= REST[1]
    assert plan.tempo_bpm == pytest.approx(100, rel=0.2) or plan.tempo_bpm == pytest.approx(
        200, rel=0.2
    )
    assert plan.beat_offsets_ms is not None and plan.beat_offsets_ms[0] <= 30
    # a beat of the true grid, give or take the tracker
    true = info["first_ms"] + info["beat_ms"] * np.round(
        (plan.cuts_ms[1] - info["first_ms"]) / info["beat_ms"]
    )
    assert abs(plan.cuts_ms[1] - true) <= 60


def test_a_track_without_a_beat_is_not_snapped_and_says_so():
    rng = np.random.default_rng(2)
    audio = _audio(0.2 * rng.standard_normal(SR * 40))
    plan = plan_cuts(audio, SplitConfig(**MIX_CFG, snap_beat=True))
    assert plan.tempo_bpm is None and plan.beat_offsets_ms is None
    assert plan.cuts_ms == plan_cuts(audio, SplitConfig(**MIX_CFG)).cuts_ms  # the loudness cuts


def test_the_default_plan_is_the_loudness_plan_and_computes_nothing_else(monkeypatch):
    import seamsplit.splitter as splitter

    monkeypatch.setattr(splitter, "analyze_audio", lambda *a, **k: pytest.fail("analysed"))
    audio, _ = _mix(seconds=30)
    plan = plan_cuts(audio, SplitConfig(**MIX_CFG))
    assert (plan.cut_by, plan.snap_beat, plan.tempo_bpm, plan.beat_offsets_ms) == (
        "rms",
        False,
        None,
        None,
    )


# ---- the options ---------------------------------------------------------------------------


def test_split_config_checks_the_new_options():
    assert CUT_BY == ("rms", "voice")
    with pytest.raises(ValueError, match="cut_by must be one of"):
        SplitConfig(cut_by="vocals")
    with pytest.raises(ValueError, match="beat tolerance"):
        SplitConfig(beat_tolerance_db=-1)


def test_the_manifest_records_how_the_cuts_were_chosen(tmp_path):
    audio, _ = _mix(seconds=40)
    source = tmp_path / "mix.wav"
    audio.export(str(source), format="wav")
    plain = split_audio(
        str(source), SplitConfig(**MIX_CFG, output_dir=str(tmp_path / "a"), export_format="wav")
    )
    assert (
        plain["cut_by"],
        plain["snap_beat"],
        plain["tempo_bpm"],
        plain["cut_beat_offsets_ms"],
    ) == (
        "rms",
        False,
        None,
        None,
    )
    cfg = SplitConfig(
        **MIX_CFG,
        cut_by="voice",
        snap_beat=True,
        output_dir=str(tmp_path / "b"),
        export_format="wav",
    )
    snapped = split_audio(str(source), cfg)
    assert snapped["cut_by"] == "voice" and snapped["snap_beat"] is True
    assert (
        snapped["tempo_bpm"] > 0
        and len(snapped["cut_beat_offsets_ms"]) == len(snapped["parts"]) - 1
    )
    assert json.loads((tmp_path / "b" / "manifest.json").read_text())["cut_by"] == "voice"


def test_the_cli_shows_how_well_the_cuts_sit_on_the_beat(tmp_path, capsys):
    audio, _ = _mix(seconds=40)
    source = tmp_path / "mix.wav"
    audio.export(str(source), format="wav")
    args = [
        "split", str(source), "--target", "24", "--before-slack", "8", "--overlap", "2",
        "--cut-by", "voice", "--snap-beat", "--dry-run",
    ]  # fmt: skip
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "voice-band activity" in out and "BPM" in out and "within 50 ms of a beat" in out


def test_the_cli_says_when_it_found_no_beat(tmp_path, capsys):
    rng = np.random.default_rng(2)
    source = tmp_path / "noise.wav"
    _audio(0.2 * rng.standard_normal(SR * 40)).export(str(source), format="wav")
    args = ["split", str(source), "--target", "24", "--before-slack", "8", "--overlap", "2"]
    assert main([*args, "--snap-beat", "--dry-run"]) == 0
    assert "no steady beat was found" in capsys.readouterr().out


def test_the_cli_rejects_an_unknown_cut_by(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["split", "x.wav", "--cut-by", "vocals"])
    assert exc.value.code == 2


def test_voice_mode_smooths_longer_by_default_but_obeys_an_explicit_value():
    assert SplitConfig().effective_smooth_ms == 250.0
    assert SplitConfig(cut_by="voice").effective_smooth_ms == 1000.0
    assert SplitConfig(cut_by="voice", smooth_ms=250).effective_smooth_ms == 250.0
    assert SplitConfig(cut_by="voice", smooth_ms=0).effective_smooth_ms == 0.0


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_the_voice_cut_finds_the_rest_with_a_loud_wide_pad_around_it(seed):
    # the wide pad is about as loud in the voice band as the voice is
    audio, _ = _mix(pad=0.12, seed=seed)
    cut = plan_cuts(audio, SplitConfig(**MIX_CFG, cut_by="voice")).cuts_ms[1]
    assert REST[0] <= cut <= REST[1]
