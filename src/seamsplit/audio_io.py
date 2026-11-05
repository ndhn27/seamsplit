"""Reading and writing audio files, and changing sample rate, with PyAV.

PyAV's wheels carry the FFmpeg libraries, so there is no ``ffmpeg`` or ``ffprobe`` program
to find and nothing is written to a temporary file. This module deals in plain arrays and
bytes; :mod:`seamsplit.audio` wraps them in :class:`~seamsplit.audio.AudioSegment`.

Reading gives the sample width the file's own samples call for: 16 bit for lossy formats
and for 16 bit PCM, 8 bit for 8 bit PCM, 32 bit for 24 and 32 bit PCM and for float PCM.
Writing: WAV by the standard library (so the bytes are what they always were), anything
else by FFmpeg's encoders (``mp3``, ``flac``, ``ogg``, ``opus``, ``aac``, ``m4a``/``mp4``,
``aiff``).
"""

from __future__ import annotations

import os
import re
import wave
from typing import Any, BinaryIO, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import numpy as np

try:  # PyAV is the one dependency this module has that can be missing
    import av
    import av.error

    av.logging.set_level(av.logging.ERROR)  # mp3 "estimating duration" and such is not news
except ImportError as exc:  # pragma: no cover - the package depends on it
    raise ImportError("SeamSplit needs PyAV to read and write audio: pip install av") from exc

PathOrFile = Union[str, "os.PathLike[str]", BinaryIO]

SAMPLE_TYPES: Dict[int, "np.dtype[Any]"] = {
    1: np.dtype(np.int8),
    2: np.dtype(np.int16),
    4: np.dtype(np.int32),
}


class CouldntDecodeError(Exception):
    """A file could not be read as audio."""


class CouldntEncodeError(Exception):
    """A file could not be written in the format asked for."""


class Decoded(NamedTuple):
    """What a file held: interleaved samples as bytes, and their format."""

    data: bytes
    sample_width: int
    frame_rate: int
    channels: int


# -- reading ------------------------------------------------------------------------

_DEMUXER_ALIASES = {"m4a": "mp4", "wave": "wav", "aif": "aiff"}


def _demuxer(format: Optional[str]) -> Optional[str]:
    if not format:
        return None
    name = format.lower().lstrip(".")
    return _DEMUXER_ALIASES.get(name, name)


def _decode_width(codec: str, sample_format: str) -> Tuple[int, str]:
    """The sample width to decode to, and the FFmpeg sample format for it. Integer
    formats keep their size (24 bit counts as 32), PCM float is read as 32 bit integers,
    and what lossy codecs decode to (float) is brought to 16 bit."""
    name = sample_format[:-1] if sample_format.endswith("p") else sample_format
    if name == "u8":
        return 1, "u8"
    if name == "s16":
        return 2, "s16"
    if name in ("s32", "s64"):
        return 4, "s32"
    if codec.startswith("pcm_f"):
        return 4, "s32"
    return 2, "s16"


def _layout_channels(layout: Any) -> int:
    return len(layout.channels)


def _decode(container: Any) -> Decoded:
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        raise CouldntDecodeError("Decoding failed: the file has no audio stream")
    resampler = None
    blocks: List[np.ndarray] = []
    width, channels, rate = 2, 0, 0

    def take(frames: Iterable[Any]) -> None:
        for out in frames:
            array = out.to_ndarray().reshape(-1)
            if width == 1:  # u8 -> signed
                array = (array.astype(np.int16) - 128).astype(np.int8)
            blocks.append(array)

    for frame in container.decode(stream):
        if resampler is None:
            width, target = _decode_width(stream.codec_context.name, frame.format.name)
            channels = _layout_channels(frame.layout)
            rate = frame.sample_rate
            resampler = av.AudioResampler(format=target, layout=frame.layout, rate=rate)
        take(resampler.resample(frame))
    if resampler is None:
        context = stream.codec_context
        if not context.sample_rate or not context.layout.channels:
            raise CouldntDecodeError("Decoding failed: the file holds no audio")
        return Decoded(b"", 2, context.sample_rate, _layout_channels(context.layout))
    take(resampler.resample(None))
    data = np.concatenate(blocks) if blocks else np.zeros(0, dtype=SAMPLE_TYPES[width])
    return Decoded(data.astype(SAMPLE_TYPES[width], copy=False).tobytes(), width, rate, channels)


def read_audio(file: PathOrFile, format: Optional[str] = None) -> Decoded:
    """Decode an audio file (a path or a binary file object) in any format FFmpeg knows.
    Raises :class:`CouldntDecodeError` for what is not audio, ``FileNotFoundError`` for a
    path that is not there."""
    if isinstance(file, os.PathLike):
        file = os.fspath(file)
    try:
        container = av.open(file, mode="r", format=_demuxer(format))
    except FileNotFoundError:
        raise
    except (av.error.FFmpegError, ValueError, OSError) as exc:
        raise CouldntDecodeError(f"Decoding failed: {exc}") from exc
    try:
        return _decode(container)
    except (av.error.FFmpegError, ValueError) as exc:
        raise CouldntDecodeError(f"Decoding failed: {exc}") from exc
    finally:
        container.close()


def engine_info() -> Dict[str, str]:
    """What reads and writes the audio: the PyAV and FFmpeg versions (for the record in
    a report or a benchmark)."""
    return {"pyav": av.__version__, "ffmpeg": str(av.ffmpeg_version_info)}


def duration_s(path: PathOrFile) -> Optional[float]:
    """How long ``path`` reads, in seconds, to a decoder that does not trim the padding a
    lossy encoder adds (the frames' own lengths added up, with no decoding). That is what
    a service with an older or simpler decoder will see, which is the length that matters
    for a limit; for WAV, FLAC and AIFF it is the length of the audio. ``None`` if the file
    cannot be read as audio. (FFmpeg 7 and later report the trimmed length as the
    container's ``duration``, which would hide the padding.)"""
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    try:
        container = av.open(path, mode="r")
    except (av.error.FFmpegError, ValueError, OSError):
        return None
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None or not stream.time_base:
            return None
        try:
            ticks = sum(p.duration or 0 for p in container.demux(stream))
        except (av.error.FFmpegError, ValueError):
            return None
        if ticks > 0:
            return float(ticks * stream.time_base)
        if container.duration is None:
            return None
        return float(container.duration) / av.time_base
    finally:
        container.close()


# -- resampling ---------------------------------------------------------------------


def resample_rows(rows: np.ndarray, source: int, target: int, width: int) -> np.ndarray:
    """``rows`` (frames x channels, integers) at ``target`` Hz. Each channel is resampled
    on its own, in double precision, and rounded back at the end."""
    dtype = SAMPLE_TYPES[width]
    limit = float(np.iinfo(dtype).max)
    low = float(np.iinfo(dtype).min)
    columns = []
    for c in range(rows.shape[1]):
        resampler = av.AudioResampler(format="dbl", layout="mono", rate=target)
        pieces: List[np.ndarray] = []
        signal = rows[:, c].astype(np.float64)
        for lo in range(0, len(signal), 1 << 16):
            frame = av.AudioFrame.from_ndarray(
                signal[lo : lo + (1 << 16)].reshape(1, -1), format="dbl", layout="mono"
            )
            frame.sample_rate = source
            pieces += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        pieces += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
        columns.append(np.concatenate(pieces))
    length = min(len(c) for c in columns)
    out = np.stack([c[:length] for c in columns], axis=1)
    return np.clip(np.rint(out), low, limit).astype(dtype)


# -- writing files ------------------------------------------------------------------


def write_bytes(out_f: PathOrFile, data: bytes) -> None:
    """``data`` to a path or a binary file object."""
    if isinstance(out_f, (str, os.PathLike)):
        with open(out_f, "wb") as f:
            f.write(data)
    else:
        out_f.write(data)


def write_wav(
    out_f: PathOrFile, data: bytes, sample_width: int, frame_rate: int, channels: int
) -> None:
    """A WAV file of PCM ``data`` (signed, except that WAV holds 8 bit audio unsigned)."""
    if sample_width == 1:
        pcm = (np.frombuffer(data, dtype=np.int8).astype(np.int16) + 128).astype(np.uint8)
        data = pcm.tobytes()
    try:
        target = os.fspath(out_f) if isinstance(out_f, os.PathLike) else out_f
        with wave.open(target, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(sample_width)
            w.setframerate(frame_rate)
            w.setnframes(len(data) // (sample_width * channels))
            w.writeframesraw(data)
    except (OSError, wave.Error) as exc:
        raise CouldntEncodeError(f"Encoding failed: {exc}") from exc


# format name -> (FFmpeg muxer, encoders to try in turn, default bitrate in bit/s or None)
ENCODERS = {
    "mp3": ("mp3", ("libmp3lame",), 128_000),
    "flac": ("flac", ("flac",), None),
    "ogg": ("ogg", ("libvorbis", "vorbis"), 128_000),
    "opus": ("opus", ("libopus",), 128_000),
    "aac": ("adts", ("aac",), 128_000),
    "m4a": ("ipod", ("aac",), 128_000),
    "mp4": ("mp4", ("aac",), 128_000),
    "aiff": ("aiff", ("pcm_s16be", "pcm_s32be"), None),
}

_BITRATE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*(?:b(?:it)?(?:/s|ps)?)?\s*$")


def parse_bitrate(text: str) -> int:
    """``"320k"`` -> 320000 (bit/s). A number with no suffix is bit/s, as for ffmpeg."""
    match = _BITRATE.match(str(text))
    if not match:
        raise ValueError(f"'{text}' is not a bitrate (try 320k)")
    value = float(match.group(1))
    unit = match.group(2).lower()
    return int(value * {"": 1, "k": 1_000, "m": 1_000_000}[unit])


def _encoder_rate(name: str, rate: int) -> int:
    """``rate``, or the closest rate the encoder ``name`` takes (Opus, for one, has no
    44.1 kHz); the audio is resampled to it as it is encoded."""
    allowed = list(av.codec.Codec(name, "w").audio_rates or ())
    if not allowed or rate in allowed:
        return rate
    above = [r for r in allowed if r > rate]
    return min(above) if above else max(allowed)


def _encoder_format(codec: Any, width: int) -> str:
    """The sample format an encoder takes that is closest to samples of ``width`` bytes
    (frames in the segment's own format are converted to it as they are encoded)."""
    offered = [f.name for f in (codec.audio_formats or ())]
    wanted = ("s32", "s32p", "fltp", "flt", "s16", "s16p")
    if width != 4:
        wanted = ("s16", "s16p", "fltp", "flt", "s32", "s32p")
    for name in wanted:
        if name in offered:
            return name
    return offered[0] if offered else "s16"


def encode_audio(
    out_f: PathOrFile,
    fmt: str,
    rows: np.ndarray,
    sample_width: int,
    frame_rate: int,
    bitrate: Optional[str] = None,
) -> None:
    """Write ``rows`` (frames x channels, integers of ``sample_width`` bytes) to ``out_f``
    as ``fmt``; ``fmt`` is one of :data:`ENCODERS`. Sample rates an encoder does not take
    are brought to the nearest one it does."""
    if fmt not in ENCODERS:
        raise CouldntEncodeError(
            f"Encoding failed: '{fmt}' is not a format SeamSplit can write "
            f"(wav, {', '.join(sorted(ENCODERS))})"
        )
    muxer, encoders, default_bitrate = ENCODERS[fmt]
    channels = rows.shape[1]
    if fmt == "aiff":  # big endian PCM, the width of the audio
        encoders = ("pcm_s32be",) if sample_width == 4 else ("pcm_s16be",)
    if not len(rows):
        raise CouldntEncodeError(f"Encoding failed: there is no audio to write as {fmt}")
    try:
        container = av.open(out_f, mode="w", format=muxer)
    except (av.error.FFmpegError, ValueError, OSError) as exc:
        raise CouldntEncodeError(f"Encoding failed: {exc}") from exc
    try:
        stream = None
        last: Optional[Exception] = None
        for name in encoders:
            try:
                stream = container.add_stream(name, rate=_encoder_rate(name, frame_rate))
                break
            except (av.error.FFmpegError, ValueError) as exc:  # this encoder is not built in
                last = exc
        if stream is None:
            raise CouldntEncodeError(f"Encoding failed: no encoder for {fmt}: {last}")
        if stream.codec.name == "vorbis" and channels != 2:
            raise CouldntEncodeError(
                "Encoding failed: the Vorbis encoder that comes with FFmpeg writes stereo "
                f"only (this audio has {channels} channel(s)); use flac, mp3 or wav"
            )
        layout = {1: "mono", 2: "stereo"}.get(channels, channels)
        context = stream.codec_context
        context.layout = layout
        context.format = _encoder_format(stream.codec, sample_width)
        if default_bitrate is not None:
            context.bit_rate = parse_bitrate(bitrate) if bitrate else default_bitrate
        if stream.codec.name == "vorbis":  # FFmpeg's own Vorbis encoder is marked experimental
            context.options = {"strict": "experimental"}
        sample_format = "s32" if sample_width == 4 else "s16"
        step = 1 << 16
        for lo in range(0, len(rows), step):
            block = np.ascontiguousarray(rows[lo : lo + step])
            frame = av.AudioFrame.from_ndarray(
                block.reshape(1, -1), format=sample_format, layout=layout
            )
            frame.sample_rate = frame_rate
            frame.pts = lo
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    except CouldntEncodeError:
        raise
    except (av.error.FFmpegError, ValueError, OSError) as exc:
        raise CouldntEncodeError(f"Encoding failed: {exc}") from exc
    finally:
        container.close()
