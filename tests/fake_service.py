"""A pretend stem-separation service, to test the anchored merge against ground truth.

A song is split into chunks, and every chunk is handed back as two stems the way a real
service would give them, with the things that make a real one hard to merge:

* the stems are not exact: a spectral mask picks out the "voice" (250-2500 Hz) of that
  chunk, the instrumental is what is left, and a little noise that differs per chunk is
  added to both;
* the start of the chunk is trimmed (``lag > 0``) or silence is put in front of it
  (``lag < 0``), by a different number of samples each time;
* the level of every chunk is different (``gain_db``).

The true lags and gains are kept, so a test can say how close the merge got to them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from seamsplit.audio import AudioSegment

from seamsplit.splitter import SplitConfig, split_audio

SR = 22050
STEMS = ("instrumental", "vocals")


def make_song(seconds: float = 50.0, channels: int = 1) -> np.ndarray:
    """A song of tones, drum-like bursts and a melody with rests: float, ``(channels, n)``."""
    rng = np.random.default_rng(7)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    x = np.zeros((channels, n))
    for c in range(channels):
        for f, a in ((110, 0.25), (165, 0.12), (220, 0.10), (330, 0.06)):
            x[c] += a * np.sin(2 * np.pi * f * (1 + 0.003 * c) * t + rng.uniform(0, 6))
        for k in range(int(seconds * 2)):
            i = int(k * 0.5 * SR)
            m = min(2000, n - i)
            x[c, i : i + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / 400) * 0.25
    note = 0
    while note * 1.7 < seconds:  # a voice: harmonics under a hanning envelope, with rests
        i = int(note * 1.7 * SR)
        m = min(int(1.2 * SR), n - i)
        f0 = 300 * 2 ** (rng.integers(0, 8) / 12)
        k = np.arange(m) / SR
        voice = sum(np.sin(2 * np.pi * f0 * h * k + h) / h for h in range(1, 6))
        x[:, i : i + m] += 0.35 * np.hanning(m) * voice
        note += 1
    # leave quiet gaps for the splitter to cut in
    for gap in (18.0, 34.0):
        x[:, int(gap * SR) : int((gap + 0.4) * SR)] *= 0.01
    return x / np.abs(x).max() * 0.7


def to_segment(x: np.ndarray) -> AudioSegment:
    """``(channels, n)`` floats as 16-bit audio."""
    pcm = np.clip(np.rint(x.T * 32767), -32768, 32767).astype("<i2")
    return AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=SR, channels=x.shape[0])


def to_array(seg: AudioSegment) -> np.ndarray:
    """16-bit audio as ``(channels, n)`` floats."""
    raw = np.array(seg.get_array_of_samples()).reshape(-1, seg.channels).T
    return raw.astype(np.float64) / 32767.0


def separate(chunk: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    """``chunk`` as ``[instrumental, vocals]`` with the errors of a mask and some noise."""
    n = chunk.shape[1]
    spectrum = np.fft.rfft(chunk, axis=1)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    mid = np.abs(np.fft.rfft(chunk.mean(axis=0)))
    smooth = np.convolve(mid, np.ones(31) / 31, mode="same")
    band = ((freqs > 250) & (freqs < 2500)).astype(float)
    mask = np.clip(band * smooth / (smooth + 0.02 * smooth.max() + 1e-9), 0, 1)
    vocals = np.fft.irfft(spectrum * mask, n, axis=1)
    instrumental = chunk - vocals
    noise = 0.002
    return [
        instrumental + rng.normal(0, noise, chunk.shape),
        vocals + rng.normal(0, noise, chunk.shape),
    ]


def _offset(x: np.ndarray, lag: int) -> np.ndarray:
    """What a service does to the start: ``out[k] = x[k + lag]``."""
    if lag > 0:
        return x[:, lag:]
    if lag < 0:
        return np.concatenate([np.zeros((x.shape[0], -lag)), x], axis=1)
    return x


@dataclass
class Service:
    """A song, split, and handed back by the pretend service."""

    root: str
    song: np.ndarray
    manifest: dict
    lags: List[int]  # samples; processed[k] = original[k + lag]
    gains_db: List[float]
    stems: Sequence[str] = field(default=STEMS)

    @property
    def chunks_dir(self) -> str:
        return os.path.join(self.root, "chunks")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.chunks_dir, "manifest.json")

    @property
    def parts(self) -> List[str]:
        return list(self.manifest["parts"])

    @property
    def lag_ms(self) -> List[float]:
        return [1000.0 * lag / SR for lag in self.lags]

    def source_file(self) -> str:
        return os.path.join(self.root, "song.wav")


def build_service(
    root: str,
    *,
    seconds: float = 50.0,
    channels: int = 1,
    target_s: float = 15.0,
    overlap_s: float = 2.0,
    lags: Optional[Sequence[int]] = None,
    gains_db: Optional[Sequence[float]] = None,
    stems: Sequence[str] = STEMS,
    normalize_stems_separately: bool = False,
    seed: int = 3,
) -> Service:
    """Split a song into WAV chunks under ``root/chunks`` and write what the service
    gives back for each, one file per stem, under ``root/processed``."""
    os.makedirs(root, exist_ok=True)
    song = make_song(seconds, channels)
    source = os.path.join(root, "song.wav")
    to_segment(song).export(source, format="wav")
    config = SplitConfig(
        target_s=target_s,
        before_slack_s=8,
        after_slack_s=3,
        overlap_s=overlap_s,
        fade_ms=0,
        output_dir=os.path.join(root, "chunks"),
        export_format="wav",
    )
    manifest = split_audio(source, config)
    n = len(manifest["parts"])
    rng = np.random.default_rng(seed)
    lags = list(lags) if lags is not None else [int(v) for v in rng.integers(-400, 401, n)]
    gains = list(gains_db) if gains_db is not None else [float(v) for v in rng.uniform(-4, 4, n)]
    assert len(lags) == n and len(gains) == n
    service = Service(root, song, manifest, lags, gains, stems)
    hand_back(
        service.chunks_dir,
        service.processed_dir,
        service.parts,
        lags,
        gains,
        rng,
        stems,
        normalize_stems_separately,
    )
    return service


def hand_back(
    chunks_dir: str,
    out_dir: str,
    parts: Sequence[str],
    lags: Sequence[int],
    gains_db: Sequence[float],
    rng: np.random.Generator,
    stems: Sequence[str] = STEMS,
    normalize_stems_separately: bool = False,
) -> None:
    """What the service gives back for the chunks in ``chunks_dir``: one file per stem
    and chunk in ``out_dir``, named like the chunk with the stem added."""
    os.makedirs(out_dir, exist_ok=True)
    for name, lag, gain_db in zip(parts, lags, gains_db):
        chunk = to_array(AudioSegment.from_file(os.path.join(chunks_dir, name)))
        both = dict(zip(STEMS, separate(chunk, rng)))
        base = name.rsplit(".", 1)[0]
        for stem in stems:
            x = _offset(both[stem], lag) * 10 ** (gain_db / 20)
            if normalize_stems_separately:
                x = x / np.abs(x).max() * 0.5
            to_segment(x).export(os.path.join(out_dir, f"{base}_{stem}.wav"), format="wav")


def snr_db(reference: np.ndarray, other: np.ndarray, skip: int = 0) -> float:
    """How closely ``other`` reproduces ``reference`` (same length is not required: the
    shorter one decides), the first and last ``skip`` samples left out."""
    n = min(reference.shape[-1], other.shape[-1])
    a, b = reference[..., skip : n - skip], other[..., skip : n - skip]
    err = float(np.mean((a - b) ** 2))
    return 10 * np.log10(float(np.mean(a**2)) / max(err, 1e-18))
