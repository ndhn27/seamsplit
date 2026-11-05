"""End-to-end: split -> (identity "AI") -> merge must give back the source."""

import json
import shutil

import numpy as np
import pytest
from seamsplit.audio import AudioSegment

from seamsplit.cli import main
from seamsplit.merger import load_manifest, merge_audio
from seamsplit.splitter import (
    SplitConfig,
    default_margin_s,
    probe_duration_s,
    split_audio,
)

SR = 22050


def _song(channels: int, seconds: int = 50) -> AudioSegment:
    """Tones + noise, with a quiet gap that the splitter should pick as a cut."""
    rng = np.random.default_rng(1)
    t = np.arange(SR * seconds) / SR
    left = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))
    right = 0.3 * np.sin(2 * np.pi * 331 * t) + 0.05 * rng.standard_normal(len(t))
    for gap in (18, 34):
        for ch in (left, right):
            ch[int(gap * SR) : int((gap + 0.4) * SR)] *= 0.01

    def to_seg(x):
        pcm = (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16)
        return AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=SR, channels=1)

    if channels == 1:
        return to_seg(left)
    return AudioSegment.from_mono_audiosegments(to_seg(left), to_seg(right))


def _snr_db(ref: np.ndarray, out: np.ndarray) -> float:
    err = out - ref
    return 10 * np.log10(np.mean(ref**2) / max(np.mean(err**2), 1e-12))


@pytest.mark.parametrize("channels", [1, 2])
def test_split_then_merge_reproduces_the_source(tmp_path, channels):
    source = _song(channels)
    src = tmp_path / "song.wav"
    source.export(str(src), format="wav")

    cfg = SplitConfig(
        target_s=15,
        before_slack_s=8,
        after_slack_s=3,
        overlap_s=2,
        fade_ms=0,  # a fade at the track edges would (correctly) change the audio
        output_dir=str(tmp_path / "chunks"),
        export_format="wav",
    )
    manifest = split_audio(str(src), cfg)

    assert len(manifest["parts"]) >= 3
    assert manifest["max_export_ms"] <= (15 + 3 + 2) * 1000  # target + after + overlap

    # pretend an AI service returned each chunk with a suffix appended
    processed = tmp_path / "processed"
    processed.mkdir()
    for name in manifest["parts"]:
        shutil.copy(
            tmp_path / "chunks" / name, processed / name.replace(".wav", "_instrumental.wav")
        )

    merged = merge_audio(
        str(processed),
        str(tmp_path / "chunks" / "manifest.json"),
        str(tmp_path / "out.wav"),
        samplerate=SR,
    )

    ref = np.array(source.get_array_of_samples()).astype(float)
    out = np.array(merged.get_array_of_samples()).astype(float)
    assert merged.channels == channels
    assert len(out) == len(ref)
    assert _snr_db(ref, out) > 60


def test_manifest_overlap_is_exact_in_whole_milliseconds(tmp_path):
    source = _song(1)
    src = tmp_path / "song.wav"
    source.export(str(src), format="wav")
    cfg = SplitConfig(
        target_s=15,
        before_slack_s=8,
        after_slack_s=3,
        overlap_s=2.001,
        output_dir=str(tmp_path / "chunks"),
        export_format="wav",
    )
    manifest = split_audio(str(src), cfg)
    segs = manifest["segments"]

    assert manifest["overlap_ms"] == 2001
    for prev, nxt in zip(segs, segs[1:]):
        assert prev["end_ms"] - nxt["start_ms"] == 2001
    on_disk = json.loads((tmp_path / "chunks" / "manifest.json").read_text())
    assert on_disk["overlap_ms"] == 2001


@pytest.mark.parametrize(
    "extra", [["--before-slack", "52"], ["--before-slack", "60"], ["--target", "0"]]
)
def test_cli_reports_bad_split_settings_instead_of_hanging(tmp_path, capsys, extra):
    rc = main(["split", str(tmp_path / "missing.wav"), "--out", str(tmp_path / "c"), *extra])
    assert rc == 1
    assert "Error:" in capsys.readouterr().err


def _make_song(tmp_path, seconds=50):
    src = tmp_path / "song.wav"
    _song(1, seconds).export(str(src), format="wav")
    return src


def test_cli_split_and_merge_end_to_end(tmp_path, capsys):
    src = _make_song(tmp_path)
    chunks = tmp_path / "chunks"

    rc = main(
        [
            "split",
            str(src),
            "--out",
            str(chunks),
            "--format",
            "wav",
            "--target",
            "15",
            "--before-slack",
            "8",
            "--after-slack",
            "3",
            "--overlap",
            "2",
            "--fade-ms",
            "0",
        ]
    )
    assert rc == 0
    assert (chunks / "manifest.json").exists()
    assert "Longest exported chunk" in capsys.readouterr().out

    rc = main(
        [
            "merge",
            str(chunks),
            "--manifest",
            str(chunks / "manifest.json"),
            "--out",
            str(tmp_path / "merged.wav"),
            "--samplerate",
            str(SR),
        ]
    )
    assert rc == 0
    ref = np.array(AudioSegment.from_file(str(src)).get_array_of_samples()).astype(float)
    out = np.array(AudioSegment.from_file(str(tmp_path / "merged.wav")).get_array_of_samples())
    assert _snr_db(ref, out.astype(float)) > 60


def test_cli_merge_matches_the_level_of_differently_normalized_chunks(tmp_path, capsys):
    src = _make_song(tmp_path)
    chunks = tmp_path / "chunks"
    split_args = [
        "split", str(src), "--out", str(chunks), "--format", "wav", "--target", "15",
        "--before-slack", "8", "--after-slack", "3", "--overlap", "2", "--fade-ms", "0",
    ]  # fmt: skip
    assert main(split_args) == 0
    manifest = json.loads((chunks / "manifest.json").read_text())

    # a service that normalizes every chunk to a different loudness (the first
    # stays as it was so the result can be compared with the source)
    processed = tmp_path / "processed"
    processed.mkdir()
    gains_db = [0.0, -3.0, 2.0, -1.5, 2.5]
    for name, gain in zip(manifest["parts"], gains_db):
        seg = AudioSegment.from_file(str(chunks / name)).apply_gain(gain)
        seg.export(str(processed / name), format="wav")
    capsys.readouterr()

    def merge(*extra):
        out = tmp_path / f"merged{len(extra)}.wav"
        rc = main(
            ["merge", str(processed), "--manifest", str(chunks / "manifest.json")]
            + ["--out", str(out), "--samplerate", str(SR), *extra]
        )
        assert rc == 0
        samples = AudioSegment.from_file(str(out)).get_array_of_samples()
        return np.array(samples).astype(float), capsys.readouterr().out

    ref = np.array(AudioSegment.from_file(str(src)).get_array_of_samples()).astype(float)

    matched, text = merge()
    assert "Matched level on" in text
    assert _snr_db(ref, matched) > 40

    plain, text = merge("--no-match-level")
    assert "Matched level" not in text
    assert _snr_db(ref, plain) < 25


def _split_for_merge(tmp_path, capsys):
    src = _make_song(tmp_path)
    chunks = tmp_path / "chunks"
    split_args = [
        "split", str(src), "--out", str(chunks), "--format", "wav", "--target", "15",
        "--before-slack", "8", "--after-slack", "3", "--overlap", "2", "--fade-ms", "0",
    ]  # fmt: skip
    assert main(split_args) == 0
    capsys.readouterr()
    return chunks


def _merge(tmp_path, capsys, chunks, *extra):
    out = tmp_path / f"merged{abs(hash(extra))}.wav"
    rc = main(
        ["merge", str(chunks), "--manifest", str(chunks / "manifest.json"), "--out", str(out)]
        + ["--samplerate", str(SR), *extra]
    )
    captured = capsys.readouterr()
    assert rc == 0
    return captured


def test_cli_merge_prints_a_seam_summary_and_writes_reports(tmp_path, capsys):
    chunks = _split_for_merge(tmp_path, capsys)
    md, js = tmp_path / "seams.md", tmp_path / "seams.json"

    out = _merge(tmp_path, capsys, chunks, "--report", str(md)).out
    _merge(tmp_path, capsys, chunks, "--report", str(js))

    n = len(json.loads((chunks / "manifest.json").read_text())["parts"]) - 1  # seams
    assert n >= 2
    # seam search is opt-in, so the default is a fade across the whole 2 s overlap
    assert f"Seams: {n} 2000 ms crossfade(s) across the whole overlap" in out
    assert "agree to 99.0 dB at worst" in out
    assert "worth a listen" not in out
    assert f"Seam report: {md}" in out
    assert md.read_text().startswith("# Seam report") and "| 1/2 |" in md.read_text()
    data = json.loads(js.read_text())
    assert [b["boundary"] for b in data["boundaries"]] == list(range(1, n + 1))
    assert all(not b["flagged"] for b in data["boundaries"])


def test_cli_merge_flags_seams_where_the_chunks_disagree(tmp_path, capsys):
    chunks = _split_for_merge(tmp_path, capsys)
    rng = np.random.default_rng(0)
    for f in sorted(chunks.glob("*.wav")):  # every chunk gets its own independent error
        seg = AudioSegment.from_file(str(f))
        x = np.array(seg.get_array_of_samples()).astype(float)
        x += rng.standard_normal(len(x)) * 0.5 * np.sqrt(np.mean(x**2))
        pcm = np.clip(x, -32768, 32767).astype(np.int16)
        seg._spawn(pcm.tobytes()).export(str(f), format="wav")

    captured = _merge(tmp_path, capsys, chunks)

    assert "seam(s) worth a listen" in captured.out
    assert "Seams: " in captured.out  # these still correspond, so they are rated
    assert "the chunks only agree to" in captured.err and "worth a listen" in captured.err


def test_cli_merge_flags_seams_that_do_not_correspond_at_all(tmp_path, capsys):
    chunks = _split_for_merge(tmp_path, capsys)
    rng = np.random.default_rng(0)
    for f in sorted(chunks.glob("*.wav")):  # each chunk replaced by unrelated noise
        seg = AudioSegment.from_file(str(f))
        pcm = (rng.standard_normal(len(seg.get_array_of_samples())) * 4000).astype(np.int16)
        seg._spawn(pcm.tobytes()).export(str(f), format="wav")

    captured = _merge(tmp_path, capsys, chunks)

    assert "seam(s) worth a listen" in captured.out  # no rating possible, still reported
    assert "Seams: " not in captured.out
    assert "barely correlate" in captured.err


def test_cli_merge_seam_options(tmp_path, capsys):
    chunks = _split_for_merge(tmp_path, capsys)

    default = _merge(tmp_path, capsys, chunks).out
    whole = _merge(tmp_path, capsys, chunks, "--no-seam-search").out
    short = _merge(tmp_path, capsys, chunks, "--seam-search", "--seam-fade-ms", "100").out
    long = _merge(tmp_path, capsys, chunks, "--seam-search").out

    assert "100 ms crossfade(s) at the best-matching point" in short
    assert "250 ms crossfade(s) at the best-matching point" in long
    # without --seam-search (the default, same as --no-seam-search) the fade spans the overlap
    for out in (default, whole):
        assert "crossfade(s) across the whole overlap" in out
        assert "2000 ms crossfade" in out  # the overlap is 2 s

    with pytest.raises(SystemExit):
        main(["merge", str(chunks), "--seam-warn-db", "loud"])


def test_cli_merge_accepts_every_crossfade_method(tmp_path, capsys):
    src = _make_song(tmp_path)
    chunks = tmp_path / "chunks"
    split_args = ["split", str(src), "--out", str(chunks), "--format", "wav", "--target", "15"]
    assert main(split_args + ["--before-slack", "8", "--overlap", "2"]) == 0
    capsys.readouterr()

    for method in ("adaptive", "linear", "equalpower"):
        rc = main(
            ["merge", str(chunks), "--manifest", str(chunks / "manifest.json")]
            + ["--out", str(tmp_path / f"{method}.wav"), "--method", method]
            + ["--max-gain-db", "3", "--samplerate", str(SR)]
        )
        assert rc == 0
        assert f"Crossfade method: {method}" in capsys.readouterr().out


def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    src = _make_song(tmp_path)
    chunks = tmp_path / "chunks"

    rc = main(
        [
            "split",
            str(src),
            "--out",
            str(chunks),
            "--format",
            "wav",
            "--target",
            "15",
            "--before-slack",
            "8",
            "--overlap",
            "2",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert not chunks.exists()
    assert "Dry run" in out and "song_part01" in out


def test_cli_max_duration_sets_the_target(tmp_path, capsys):
    src = _make_song(tmp_path)
    chunks = tmp_path / "chunks"

    rc = main(
        [
            "split",
            str(src),
            "--out",
            str(chunks),
            "--format",
            "wav",
            "--max-duration",
            "20",
            "--before-slack",
            "8",
            "--after-slack",
            "3",
            "--overlap",
            "2",
        ]
    )

    assert rc == 0
    manifest = json.loads((chunks / "manifest.json").read_text())
    assert manifest["max_export_ms"] <= 20_000


def test_cli_max_duration_conflicts_with_target(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["split", str(tmp_path / "x.wav"), "--target", "30", "--max-duration", "60"])
    assert exc.value.code == 2


def test_cli_max_duration_too_small_is_an_error(tmp_path, capsys):
    rc = main(
        [
            "split",
            str(tmp_path / "x.wav"),
            "--max-duration",
            "5",
            "--after-slack",
            "3",
            "--overlap",
            "4",
        ]
    )
    assert rc == 1
    assert "--limit is too small" in capsys.readouterr().err


def _dry_run(tmp_path, capsys, *extra):
    src = _make_song(tmp_path)
    rc = main(["split", str(src), "--out", str(tmp_path / "c"), "--dry-run", *extra])
    out = capsys.readouterr().out
    assert rc == 0
    return out


def test_cli_limit_derives_the_target_from_the_service_limit(tmp_path, capsys):
    out = _dry_run(tmp_path, capsys, "--limit", "59", "--format", "wav")
    # default after-slack 3 and overlap 4: 59 - 3 - 4 = 52, nothing reserved for a lossless file
    assert "Target chunk length: 52s (from --limit 59s" in out
    assert "margin" not in out.splitlines()[0]


def test_cli_limit_leaves_room_for_the_padding_of_a_lossy_encoder(tmp_path, capsys):
    out = _dry_run(tmp_path, capsys, "--limit", "59")  # mp3 is the default format
    assert "Target chunk length: 51.75s (from --limit 59s" in out
    assert "margin 0.25s" in out


def test_cli_margin_overrides_the_default(tmp_path, capsys):
    assert "Target chunk length: 52s" in _dry_run(
        tmp_path, capsys, "--limit", "59", "--margin", "0"
    )
    assert "Target chunk length: 51s" in _dry_run(
        tmp_path, capsys, "--limit", "59", "--margin", "1"
    )


def test_cli_negative_margin_is_an_error(tmp_path, capsys):
    rc = main(["split", str(tmp_path / "x.wav"), "--limit", "59", "--margin", "-1"])
    assert rc == 1
    assert "--margin must not be negative" in capsys.readouterr().err


def test_cli_max_duration_is_still_accepted_as_the_old_name(tmp_path, capsys):
    out = _dry_run(tmp_path, capsys, "--max-duration", "59", "--format", "wav")
    assert "Target chunk length: 52s (from --limit 59s" in out


def test_cli_limit_conflicts_with_target(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["split", str(tmp_path / "x.wav"), "--target", "30", "--limit", "60"])
    assert exc.value.code == 2


def test_default_margin_by_format():
    assert default_margin_s("wav") == default_margin_s("FLAC") == default_margin_s(".wav") == 0.0
    assert default_margin_s("mp3") == default_margin_s("ogg") == 0.25


def test_probe_duration_reads_the_container_and_copes_with_junk(tmp_path):
    seg = AudioSegment.silent(3000, frame_rate=SR)
    seg.export(str(tmp_path / "a.wav"), format="wav")
    assert probe_duration_s(str(tmp_path / "a.wav")) == pytest.approx(3.0, abs=0.001)
    assert probe_duration_s(str(tmp_path / "missing.mp3")) is None
    (tmp_path / "junk.mp3").write_bytes(b"not audio")
    assert probe_duration_s(str(tmp_path / "junk.mp3")) is None


def test_a_full_length_mp3_reads_longer_than_its_audio(tmp_path):
    """The reason a limit needs a margin: the encoder pads the stream. FFmpeg 7 and later
    read the gapless tag and report the length of the audio as the container's duration; a
    player or service that does not reads what the frames hold, which is what the size of
    this constant bit rate file says and what ``probe_duration_s`` reports."""
    seg = AudioSegment.silent(10_000, frame_rate=44100)
    seg.export(str(tmp_path / "a.mp3"), format="mp3", bitrate="320k")
    seg.export(str(tmp_path / "a.flac"), format="flac")

    assert 10.0 < (tmp_path / "a.mp3").stat().st_size * 8 / 320_000 < 10.25
    assert 10.0 < probe_duration_s(str(tmp_path / "a.mp3")) < 10.25
    assert probe_duration_s(str(tmp_path / "a.flac")) == pytest.approx(10.0, abs=0.001)


def _sawtooth_song(seconds, period=55.0):
    """Loudness falls steadily within every ``period`` seconds, so the quietest
    point of each cut window is its very last moment and the chunks come out as
    long as the settings allow (here 58.995 s of audio for a 59 s limit)."""
    rng = np.random.default_rng(3)
    t = np.arange(int(SR * seconds)) / SR
    env = 1.0 - 0.99 * ((t % period) / period)
    x = env * (0.5 * np.sin(2 * np.pi * 220 * t) + 0.1 * rng.standard_normal(len(t)))
    pcm = (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16)
    return AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=SR, channels=1)


def _split_with_limit(tmp_path, capsys, limit, *extra, seconds=175, fmt="mp3", song=None):
    src = tmp_path / "long.wav"
    (song or _sawtooth_song(seconds)).export(str(src), format="wav")
    chunks = tmp_path / "chunks"
    args = ["split", str(src), "--out", str(chunks), "--format", fmt, "--limit", str(limit)]
    rc = main(args + list(extra))
    captured = capsys.readouterr()
    assert rc == 0
    return chunks, captured


def test_cli_limit_keeps_every_exported_mp3_under_the_limit(tmp_path, capsys):
    chunks, captured = _split_with_limit(tmp_path, capsys, 59)

    files = sorted(chunks.glob("*.mp3"))
    assert len(files) >= 4
    assert all(probe_duration_s(str(f)) <= 59.0 for f in files)
    assert "Longest chunk as exported:" in captured.out
    assert "Warning" not in captured.err


def test_without_a_margin_full_length_mp3_chunks_read_over_the_limit(tmp_path, capsys):
    """The case --margin exists for: the audio fits the limit (58.995 s), the files
    do not (the encoder's padding makes them read 59.06 s)."""
    chunks, captured = _split_with_limit(tmp_path, capsys, 59, "--margin", "0")

    manifest = json.loads((chunks / "manifest.json").read_text())
    assert manifest["max_export_ms"] <= 59_000  # a check on the audio alone passes
    assert "file(s) exceed --limit 59s as exported" in captured.err
    assert max(probe_duration_s(str(f)) for f in chunks.glob("*.mp3")) > 59.0


def test_cli_warns_about_exported_files_over_the_limit(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("seamsplit.cli.probe_duration_s", lambda path: 59.4)

    _, captured = _split_with_limit(tmp_path, capsys, 59, fmt="wav", seconds=60)

    assert "file(s) exceed --limit 59s as exported" in captured.err
    assert "59.40s" in captured.err and "Raise --margin" in captured.err


def test_cli_says_so_when_the_exported_files_cannot_be_measured(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("seamsplit.cli.probe_duration_s", lambda path: None)

    _, captured = _split_with_limit(tmp_path, capsys, 59, fmt="wav", seconds=60)

    assert "could not measure the exported files" in captured.err


def test_cli_without_a_limit_points_at_the_option(tmp_path, capsys):
    src = _make_song(tmp_path)
    assert main(["split", str(src), "--out", str(tmp_path / "c"), "--format", "wav"]) == 0
    assert "Pass --limit SECONDS" in capsys.readouterr().out


def test_cli_merge_reports_a_bad_manifest(tmp_path, capsys):
    bad = tmp_path / "manifest.json"
    bad.write_text("nope")
    rc = main(["merge", str(tmp_path), "--manifest", str(bad), "--out", str(tmp_path / "o.wav")])
    assert rc == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_cli_merge_reports_a_manifest_with_the_wrong_types(tmp_path, capsys):
    """A manifest with the wrong types is reported as an error, not as a TypeError traceback."""
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"parts": [1], "overlap_ms": 0}))
    rc = main(["merge", str(tmp_path), "--manifest", str(bad), "--out", str(tmp_path / "o.wav")])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("Error:") and "'parts[0]' must be a file name" in err
    assert not (tmp_path / "o.wav").exists()


def test_load_manifest_accepts_the_manifest_split_wrote(tmp_path):
    src = tmp_path / "song.wav"
    _song(1).export(str(src), format="wav")
    cfg = SplitConfig(
        target_s=15,
        before_slack_s=8,
        after_slack_s=3,
        overlap_s=2,
        output_dir=str(tmp_path / "chunks"),
        export_format="wav",
    )
    written = split_audio(str(src), cfg)
    assert len(written["parts"]) > 1  # several parts and segments get checked

    loaded = load_manifest(str(tmp_path / "chunks" / "manifest.json"))
    assert loaded == json.loads(json.dumps(written))


def test_cli_merge_reports_a_missing_manifest(tmp_path, capsys):
    rc = main(
        [
            "merge",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "nope.json"),
            "--out",
            str(tmp_path / "o.wav"),
        ]
    )
    assert rc == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_explains_a_file_that_is_not_audio_and_a_format_it_cannot_write():
    from seamsplit.audio import CouldntDecodeError, CouldntEncodeError
    from seamsplit.cli import _explain

    assert "could not read the audio" in _explain(CouldntDecodeError("no audio stream"))
    msg = _explain(CouldntEncodeError("'xyz' is not a format SeamSplit can write"))
    assert "could not write the audio" in msg and "xyz" in msg


# --- lossless chunks ----------------------------------------------------------


@pytest.mark.parametrize("fmt", ["wav", "flac", ".FLAC"])
def test_lossless_chunks_have_exact_lengths_and_round_trip(tmp_path, fmt):
    source = _song(2)
    src = tmp_path / "song.wav"
    source.export(str(src), format="wav")
    cfg = SplitConfig(
        target_s=15,
        before_slack_s=8,
        after_slack_s=3,
        overlap_s=2,
        fade_ms=0,
        output_dir=str(tmp_path / "chunks"),
        export_format=fmt,
        bitrate="64k",  # must be ignored, not passed to ffmpeg
    )
    manifest = split_audio(str(src), cfg)
    ext = fmt.lstrip(".").lower()

    assert all(name.endswith(f"_part{i:02d}.{ext}") for i, name in enumerate(manifest["parts"], 1))
    assert not any(".." in name for name in manifest["parts"])
    for name, seg in zip(manifest["parts"], manifest["segments"]):
        written = probe_duration_s(str(tmp_path / "chunks" / name))
        # no encoder padding: the file lasts exactly as long as the audio in it
        planned_s = (seg["end_ms"] - seg["start_ms"]) / 1000
        assert written == pytest.approx(planned_s, abs=0.002)

    out = tmp_path / f"out.{ext}"
    merged = merge_audio(
        str(tmp_path / "chunks"),
        str(tmp_path / "chunks" / "manifest.json"),
        str(out),
        samplerate=SR,
    )
    ref = np.array(source.get_array_of_samples()).astype(float)
    got = np.array(AudioSegment.from_file(str(out)).get_array_of_samples()).astype(float)
    assert len(merged) == len(source)
    assert len(got) == len(ref)
    assert _snr_db(ref, got) > 60


def test_cli_split_and_merge_with_flac(tmp_path, capsys):
    src = _make_song(tmp_path)
    chunks = tmp_path / "chunks"

    # --limit 20 - after-slack 3 - overlap 2 - margin 0 (lossless) = a 15 s target
    rc = main(
        ["split", str(src), "--out", str(chunks), "--format", "flac", "--bitrate", "320k"]
        + ["--limit", "20", "--before-slack", "8", "--after-slack", "3", "--overlap", "2"]
        + ["--fade-ms", "0"]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "Target chunk length: 15s" in captured.out
    assert "exceed" not in captured.err  # no encoder padding to leave room for

    parts = json.loads((chunks / "manifest.json").read_text())["parts"]
    assert parts and all(name.endswith(".flac") for name in parts)
    assert sorted(p.name for p in chunks.glob("*.flac")) == sorted(parts)

    rc = main(
        ["merge", str(chunks), "--manifest", str(chunks / "manifest.json")]
        + ["--out", str(tmp_path / "merged.flac"), "--samplerate", str(SR)]
    )
    assert rc == 0
    ref = np.array(AudioSegment.from_file(str(src)).get_array_of_samples()).astype(float)
    out = np.array(AudioSegment.from_file(str(tmp_path / "merged.flac")).get_array_of_samples())
    assert _snr_db(ref, out.astype(float)) > 60


# --- more than 99 chunks --------------------------------------------------------


def test_split_then_merge_with_more_than_99_chunks(tmp_path):
    """``song_part10`` must not be mistaken for ``song_part100`` (or the other way
    round) once the part numbers reach three digits."""
    sr = 8000
    rng = np.random.default_rng(3)
    t = np.arange(sr * 130) / sr
    x = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * rng.standard_normal(len(t))
    pcm = (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16)
    source = AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=sr, channels=1)
    src = tmp_path / "song.wav"
    source.export(str(src), format="wav")

    cfg = SplitConfig(
        target_s=1.0,
        before_slack_s=0.5,
        after_slack_s=0.2,
        overlap_s=0.2,
        min_final_s=0.3,
        fade_ms=0,
        output_dir=str(tmp_path / "chunks"),
        export_format="wav",
    )
    manifest = split_audio(str(src), cfg)
    names = manifest["parts"]
    assert len(names) > 100
    assert "song_part10.wav" in names and "song_part100.wav" in names

    processed = tmp_path / "processed"
    processed.mkdir()
    for name in names:
        shutil.copy(tmp_path / "chunks" / name, processed / name.replace(".wav", "_vocals.wav"))

    merged = merge_audio(
        str(processed),
        str(tmp_path / "chunks" / "manifest.json"),
        str(tmp_path / "out.wav"),
        samplerate=sr,
    )
    ref = np.array(source.get_array_of_samples()).astype(float)
    out = np.array(merged.get_array_of_samples()).astype(float)
    assert len(out) == len(ref)
    assert _snr_db(ref, out) > 40
