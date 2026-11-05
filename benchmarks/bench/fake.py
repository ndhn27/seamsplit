"""A stand-in for the separator, for tests and for trying the benchmarks out in seconds.

It "separates" with a brick-wall low-pass at 1 kHz (the low part is called the instrumental,
the rest the vocals), which is nothing like a model: **numbers made with it mean nothing
about SeamSplit** and are written with the name of the separator so that they cannot be
mistaken for the real ones.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from seamsplit.audio import AudioSegment

from .corpus import sha256_of

CUTOFF_HZ = 1000.0


class FakeSeparator:
    name = "FAKE (low-pass at 1 kHz; tests and trial runs only)"
    model = "fake"
    model_dir = ""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="seamsplit-fake-")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.loaded_s = 0.0
        self.calls = 0

    def stems_for(self, sources: Sequence[str]) -> List[Dict[str, Any]]:
        return [self._one(source) for source in sources]

    def _one(self, source: str) -> Dict[str, Any]:
        key = sha256_of(source)[:24]
        paths = {
            "vocals": os.path.join(self.cache_dir, key + "_vocals.flac"),
            "instrumental": os.path.join(self.cache_dir, key + "_instrumental.flac"),
        }
        if not all(os.path.isfile(p) for p in paths.values()):
            self.calls += 1
            audio = AudioSegment.from_file(source).set_sample_width(2)
            x = audio.frames.astype(np.float64)
            spectrum = np.fft.rfft(x, axis=0)
            freqs = np.fft.rfftfreq(len(x), 1.0 / audio.frame_rate)
            low = np.fft.irfft(spectrum * (freqs <= CUTOFF_HZ)[:, None], n=len(x), axis=0)
            high = x - low
            for stem, data in (("instrumental", low), ("vocals", high)):
                ints = np.clip(np.round(data), -32768, 32767).astype(np.int16)
                AudioSegment.from_samples(
                    ints.reshape(-1), frame_rate=audio.frame_rate, channels=audio.channels
                ).export(paths[stem], format="flac")
        return {**paths, "seconds": 0.0}
