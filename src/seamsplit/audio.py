"""Audio in memory, and on disk through PyAV.

:class:`AudioSegment` holds a clip and the operations the rest of SeamSplit needs (slicing in
milliseconds, joining, gain, fades, mixing down, resampling), on top of two things that come
with ``pip install``:

* :mod:`numpy` for the samples, and
* `PyAV <https://pyav.basswood-io.com/>`_, whose wheels carry the FFmpeg libraries, to decode
  and encode any format ffmpeg knows. There is no program to find and nothing is written
  to a temporary file.

An :class:`AudioSegment` is immutable: every operation returns a new one. The samples are
integers of 1, 2 or 4 bytes (24 bit audio is held as 32 bit, scaled up by 8 bits) with the
channels interleaved; ``raw_data`` holds them as bytes and ``samples`` as a numpy array.

Sample rate conversion (:meth:`AudioSegment.set_frame_rate`) is done by FFmpeg's resampler,
a band-limited filter. WAV files are written by the standard library; anything else is
written by FFmpeg's encoders.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np

from .audio_io import (  # noqa: F401  (re-exported: part of this module's interface)
    SAMPLE_TYPES,
    CouldntDecodeError,
    CouldntEncodeError,
    PathOrFile,
    duration_s,
    encode_audio,
    engine_info,
    read_audio,
    resample_rows,
    write_bytes,
    write_wav,
)

_DTYPES = SAMPLE_TYPES
_ARRAY_TYPES = {1: "b", 2: "h", 4: "i"}

# Blocks of this many values are processed at a time by operations that go through float64,
# so a long song does not need eight bytes for every sample at once.
_BLOCK = 1 << 20


def db_to_float(db: float) -> float:
    """The amplitude factor for a change of ``db`` decibels."""
    return 10 ** (float(db) / 20)


_DOWNMIX_BLOCK = 1 << 20  # frames at a time: the temporaries stay a few MB whatever the length


def _downmix(rows: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Two or more channels to one: two channels are ``floor(0.5 l + 0.5 r)``
    in floating point, more are the sum of ``floor(x / n)`` of each. Done in blocks, because
    the temporaries of a whole file (float64 or int64 per sample) are several times its size."""
    n, channels = rows.shape
    out = np.empty(n, dtype=dtype)
    for start in range(0, n, _DOWNMIX_BLOCK):
        block = rows[start : start + _DOWNMIX_BLOCK]
        if channels == 2:
            both = block.astype(np.float64)
            out[start : start + len(block)] = np.floor(both[:, 0] * 0.5 + both[:, 1] * 0.5)
        else:
            out[start : start + len(block)] = (block.astype(np.int64) // channels).sum(axis=1)
    return out


def _scale(samples: np.ndarray, factor: Any, width: int) -> np.ndarray:
    """``samples`` times ``factor`` (a number, or one factor per row): in floating
    point, rounded down, and kept inside the range of the sample type. Rows are frames;
    works in blocks so memory stays small."""
    dtype = _DTYPES[width]
    top = float(np.iinfo(dtype).max)
    bottom = float(np.iinfo(dtype).min)
    out = np.empty(samples.shape, dtype=dtype)
    step = max(1, _BLOCK // (samples.shape[1] if samples.ndim > 1 else 1))
    per_row = isinstance(factor, np.ndarray)
    for lo in range(0, len(samples), step):
        part = samples[lo : lo + step].astype(np.float64)
        part *= factor[lo : lo + step, None] if per_row else factor
        np.floor(part, out=part)
        part[part > top] = top
        part[part < bottom + 1] = bottom
        out[lo : lo + step] = part
    return out


class AudioSegment:
    """A piece of audio: interleaved integer samples and the format they are in.

    ``AudioSegment(data, sample_width=2, frame_rate=44100, channels=2)`` makes one from raw
    bytes (a multiple of ``sample_width * channels`` long); ``AudioSegment.from_file`` reads
    a file. ``len()`` is the length in milliseconds and slicing takes milliseconds."""

    __slots__ = ("_data", "sample_width", "frame_rate", "channels", "frame_width")

    def __init__(
        self,
        data: Any = b"",
        *,
        sample_width: int,
        frame_rate: int,
        channels: int,
    ) -> None:
        if hasattr(data, "tobytes"):
            data = data.tobytes()
        data = bytes(data)
        if sample_width not in (1, 2, 3, 4):
            raise ValueError(f"sample_width must be 1, 2, 3 or 4 bytes, not {sample_width}")
        if not frame_rate or frame_rate < 0:
            raise ValueError(f"frame_rate must be positive, not {frame_rate}")
        if not channels or channels < 1:
            raise ValueError(f"channels must be at least 1, not {channels}")
        if len(data) % (sample_width * channels) != 0:
            raise ValueError("data length must be a multiple of '(sample_width * channels)'")
        if sample_width == 3:
            # 24 bit -> 32 bit: the three bytes become the upper ones
            triples = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
            wide = np.zeros((len(triples), 4), dtype=np.uint8)
            wide[:, 0] = np.where(triples[:, 2] > 0x7F, 0xFF, 0x00)
            wide[:, 1:] = triples
            data = wide.tobytes()
            sample_width = 4
        self._data = data
        self.sample_width = int(sample_width)
        self.frame_rate = int(frame_rate)
        self.channels = int(channels)
        self.frame_width = self.sample_width * self.channels

    # -- constructors ---------------------------------------------------------------

    @classmethod
    def from_samples(
        cls, samples: np.ndarray, frame_rate: int, channels: int = 1
    ) -> "AudioSegment":
        """A segment from an integer array (int8, int16 or int32; any shape, read in
        interleaved order)."""
        samples = np.asarray(samples)
        if samples.dtype not in _DTYPES.values():
            raise ValueError(f"samples must be int8, int16 or int32, not {samples.dtype}")
        return cls(
            np.ascontiguousarray(samples).tobytes(),
            sample_width=samples.dtype.itemsize,
            frame_rate=frame_rate,
            channels=channels,
        )

    def _spawn(self, data: Any, **overrides: int) -> "AudioSegment":
        params = {
            "sample_width": self.sample_width,
            "frame_rate": self.frame_rate,
            "channels": self.channels,
        }
        params.update(overrides)
        return AudioSegment(data, **params)

    @classmethod
    def silent(cls, duration: float = 1000, frame_rate: int = 11025) -> "AudioSegment":
        """``duration`` milliseconds of mono 16 bit silence."""
        frames = int(frame_rate * (duration / 1000.0))
        return cls(b"\0\0" * frames, sample_width=2, frame_rate=frame_rate, channels=1)

    @classmethod
    def from_mono_audiosegments(cls, *mono: "AudioSegment") -> "AudioSegment":
        """Mono segments side by side as the channels of one (shorter ones padded with
        silence; the format is the widest of them, as in ``_sync``)."""
        if not mono:
            raise ValueError("At least one AudioSegment instance is required")
        segs = cls._sync(*mono)
        if segs[0].channels != 1:
            raise ValueError(
                "AudioSegment.from_mono_audiosegments requires all arguments are mono "
                "AudioSegment instances"
            )
        rows = max(len(s._data) // s.frame_width for s in segs)
        out = np.zeros((rows, len(segs)), dtype=_DTYPES[segs[0].sample_width])
        for i, seg in enumerate(segs):
            column = seg.samples
            out[: len(column), i] = column
        return segs[0]._spawn(out.tobytes(), channels=len(segs))

    # -- reading --------------------------------------------------------------------

    @property
    def raw_data(self) -> bytes:
        """The samples as bytes, channels interleaved."""
        return self._data

    @property
    def array_type(self) -> str:
        """The :mod:`array` type code of a sample (``b``, ``h`` or ``i``)."""
        return _ARRAY_TYPES[self.sample_width]

    @property
    def dtype(self) -> np.dtype:
        """The numpy type of a sample."""
        return _DTYPES[self.sample_width]

    @property
    def samples(self) -> np.ndarray:
        """The samples as a flat, read-only numpy array, channels interleaved (no copy)."""
        return np.frombuffer(self._data, dtype=_DTYPES[self.sample_width])

    @property
    def frames(self) -> np.ndarray:
        """The samples as one row per frame, one column per channel (no copy)."""
        return self.samples.reshape(-1, self.channels)

    def get_array_of_samples(self) -> "Any":
        """The samples as an :class:`array.array` (``samples`` is the same without the
        copy)."""
        import array

        return array.array(self.array_type, self._data)

    def frame_count(self, ms: Optional[float] = None) -> float:
        """The number of frames in ``ms`` milliseconds, or in the whole segment."""
        if ms is not None:
            return ms * (self.frame_rate / 1000.0)
        return float(len(self._data) // self.frame_width)

    def __len__(self) -> int:
        return round(1000 * (self.frame_count() / self.frame_rate))

    @property
    def duration_seconds(self) -> float:
        return self.frame_count() / self.frame_rate

    def _parse_position(self, value: float) -> int:
        if value < 0:
            value = len(self) - abs(value)
        frames = (
            self.frame_count(ms=len(self)) if value == float("inf") else self.frame_count(ms=value)
        )
        return int(frames)

    def __getitem__(self, ms: Any) -> "AudioSegment":
        if isinstance(ms, slice):
            if ms.step:
                raise ValueError("an AudioSegment cannot be sliced with a step")
            start = ms.start if ms.start is not None else 0
            end = ms.stop if ms.stop is not None else len(self)
            start = min(start, len(self))
            end = min(end, len(self))
        else:
            start, end = ms, ms + 1
        first = self._parse_position(start) * self.frame_width
        last = self._parse_position(end) * self.frame_width
        data = self._data[first:last]
        # as long as was asked for: a rounded length can ask for one frame more than there is
        missing = ((last - first) - len(data)) // self.frame_width
        if missing:
            if missing > self.frame_count(ms=2):
                raise ValueError(f"cannot fill in {missing} missing frames with silence")
            data += bytes(self.frame_width * missing) if data else b""
        return self._spawn(data)

    # -- combining ------------------------------------------------------------------

    @classmethod
    def _sync(cls, *segs: "AudioSegment") -> Tuple["AudioSegment", ...]:
        channels = max(s.channels for s in segs)
        frame_rate = max(s.frame_rate for s in segs)
        width = max(s.sample_width for s in segs)
        return tuple(
            s.set_channels(channels).set_frame_rate(frame_rate).set_sample_width(width)
            for s in segs
        )

    def append(self, other: "AudioSegment") -> "AudioSegment":
        """``other`` after this one, in the widest of the two formats."""
        first, second = AudioSegment._sync(self, other)
        return first._spawn(first._data + second._data)

    def __add__(self, other: Any) -> "AudioSegment":
        if isinstance(other, AudioSegment):
            return self.append(other)
        return self.apply_gain(other)

    def __radd__(self, other: Any) -> "AudioSegment":
        if other == 0:  # so that sum() of segments works
            return self
        return NotImplemented

    # -- changing -------------------------------------------------------------------

    def apply_gain(self, volume_change: float) -> "AudioSegment":
        """Louder (or, if negative, quieter) by ``volume_change`` dB."""
        scaled = _scale(self.frames, db_to_float(volume_change), self.sample_width)
        return self._spawn(scaled.tobytes())

    def fade(
        self,
        to_gain: float = 0,
        from_gain: float = 0,
        start: Optional[float] = None,
        end: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> "AudioSegment":
        """Change the volume over time: from ``from_gain`` dB to ``to_gain`` dB between
        ``start`` and ``end`` (ms; give two of start, end, duration). Before the fade the
        audio is at ``from_gain``, after it at ``to_gain``. Fades longer than 100 ms change
        the gain once per millisecond, shorter ones once per sample."""
        if None not in [duration, end, start]:
            raise TypeError(
                'Only two of the three arguments, "start", "end", and "duration" may be specified'
            )
        if to_gain == 0 and from_gain == 0:
            return self
        length = len(self)
        start = min(length, start) if start is not None else None
        end = min(length, end) if end is not None else None
        if start is not None and start < 0:
            start += length
        if end is not None and end < 0:
            end += length
        if duration is not None and duration < 0:
            raise ValueError("duration must be a positive integer")
        if duration:
            if start is not None:
                end = start + duration
            elif end is not None:
                start = end - duration
        else:
            duration = end - start  # type: ignore[operator]
        assert start is not None  # (start and end were both None: the line above raised)

        from_power = db_to_float(from_gain)
        to_power = db_to_float(to_gain)
        gain_delta = to_power - from_power
        width = self.sample_width
        rows = self.frames
        pieces: List[np.ndarray] = []

        before = self[:start].frames
        pieces.append(_scale(before, from_power, width) if from_gain != 0 else before)

        if duration > 100:
            step = gain_delta / duration
            n = int(duration)
            # one gain per millisecond, over the frames of that millisecond
            edges = np.array(
                [self._parse_position(start + i) for i in range(n + 1)], dtype=np.int64
            )
            factors = from_power + step * np.arange(n)
            per_row = np.repeat(factors, np.diff(edges))
            span = rows[edges[0] : edges[-1]]
            if len(span) < len(per_row):  # the last millisecond may reach past the data
                span = np.vstack(
                    [span, np.zeros((len(per_row) - len(span), rows.shape[1]), rows.dtype)]
                )
            pieces.append(_scale(span, per_row, width))
        else:
            first = self.frame_count(ms=start)
            frames = self.frame_count(ms=end) - first
            step = gain_delta / frames
            n = int(frames)
            index = (first + np.arange(n)).astype(np.int64)
            factors = from_power + step * np.arange(n)
            keep = index < len(rows)  # a frame beyond the data is no frame
            pieces.append(_scale(rows[index[keep]], factors[keep], width))

        after = self[end:].frames
        pieces.append(_scale(after, to_power, width) if to_gain != 0 else after)
        return self._spawn(np.concatenate(pieces).tobytes())

    def fade_in(self, duration: float) -> "AudioSegment":
        return self.fade(from_gain=-120, duration=duration, start=0)

    def fade_out(self, duration: float) -> "AudioSegment":
        return self.fade(to_gain=-120, duration=duration, end=float("inf"))

    def set_sample_width(self, sample_width: int) -> "AudioSegment":
        """The same audio in samples of ``sample_width`` bytes (narrower ones lose the
        low bits, wider ones gain empty ones)."""
        if sample_width == self.sample_width:
            return self
        if sample_width not in _DTYPES:
            raise ValueError(f"sample_width must be 1, 2 or 4 bytes, not {sample_width}")
        shift = 8 * (sample_width - self.sample_width)
        wide = self.samples.astype(np.int64)
        converted = (wide << shift) if shift > 0 else (wide >> -shift)
        return self._spawn(
            converted.astype(_DTYPES[sample_width]).tobytes(), sample_width=sample_width
        )

    def set_channels(self, channels: int) -> "AudioSegment":
        """Mono to any number of channels (copies), two or more channels to mono (the
        average of them)."""
        if channels == self.channels:
            return self
        rows = self.frames
        if channels == 2 and self.channels == 1:
            out = np.repeat(rows, 2, axis=1)
        elif channels == 1 and self.channels >= 2:
            out = _downmix(rows, self.dtype)
        elif self.channels == 1:
            out = np.repeat(rows, channels, axis=1)
        else:
            raise ValueError(
                "AudioSegment.set_channels only supports mono-to-multi channel and "
                "multi-to-mono channel conversion"
            )
        return self._spawn(np.ascontiguousarray(out).tobytes(), channels=channels)

    def set_frame_rate(self, frame_rate: int) -> "AudioSegment":
        """The same audio at another sample rate, resampled by FFmpeg."""
        frame_rate = int(frame_rate)
        if frame_rate == self.frame_rate:
            return self
        if not self._data:
            return self._spawn(self._data, frame_rate=frame_rate)
        return self._spawn(
            resample_rows(self.frames, self.frame_rate, frame_rate, self.sample_width).tobytes(),
            frame_rate=frame_rate,
        )

    # -- files ----------------------------------------------------------------------

    @classmethod
    def from_file(cls, file: PathOrFile, format: Optional[str] = None) -> "AudioSegment":
        """Read an audio file (a path or a binary file object) in any format FFmpeg knows.

        The samples are 16 bit unless the file holds integers of another size (8, 32 bit,
        24 bit as 32) or 32 bit float PCM (read as 32 bit). Lossy formats are decoded to 16
        bit."""
        data, width, rate, channels = read_audio(file, format)
        return cls(data, sample_width=width, frame_rate=rate, channels=channels)

    def export(
        self,
        out_f: PathOrFile,
        format: str = "mp3",
        bitrate: Optional[str] = None,
    ) -> None:
        """Write the audio to a path or a binary file object. ``format`` is ``wav``,
        ``flac``, ``mp3``, ``ogg``, ``opus``, ``aac`` (ADTS), ``m4a``/``mp4``, ``aiff``;
        ``bitrate`` (``"320k"``) is for the lossy ones."""
        if isinstance(out_f, os.PathLike):
            out_f = os.fspath(out_f)
        fmt = format.lower().lstrip(".")
        fmt = {"aif": "aiff", "wave": "wav"}.get(fmt, fmt)
        if fmt == "wav":
            write_wav(out_f, self._data, self.sample_width, self.frame_rate, self.channels)
        elif fmt == "raw":
            write_bytes(out_f, self._data)
        else:
            seg = self.set_sample_width(2) if self.sample_width == 1 else self
            encode_audio(out_f, fmt, seg.frames, seg.sample_width, seg.frame_rate, bitrate)
