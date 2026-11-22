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

import contextlib
import io
import math
import os
import re
import wave
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union

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


class AudioTooLongError(CouldntDecodeError):
    """A file holds more audio than the limit allows (see :data:`DEFAULT_MAX_DURATION_S`):
    it was refused before it was decoded, or its decoding was stopped when it passed the
    limit. ``reason`` is the message without the advice on how to raise the limit (which
    is worded for the command line; a page words its own), ``limit_text`` the limit in words."""

    reason: str = ""
    limit_text: str = ""


# -- how much audio is read ----------------------------------------------------------
#
# Audio is held in memory as integers, all of it at once, so what a file takes is its
# length times its sample rate, channels and sample width, not its size on disk: a few
# kilobytes of FLAC silence decode to hundreds of megabytes (two hours of silence at 48 kHz
# in stereo, to 1.4 GB). A file is therefore read only if it holds less than a limit, in
# seconds, and the limit is checked twice: against the length the file says it has before
# anything is decoded, and again as the audio comes out of the decoder, which stops at the
# limit whatever the file says (a header can say anything).
DEFAULT_MAX_DURATION_S = 7200.0
# The limit is also a limit in bytes, so a file cannot get round it with a rate or a channel
# count that is not ordinary: it is the number of seconds at 48 kHz, stereo, 32 bit.
_BUDGET_BYTES_PER_S = 48_000 * 2 * 4


def check_max_duration(value: Optional[float]) -> Optional[float]:
    """``value`` (seconds, or ``None`` for no limit) if it can be a limit."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(f"max_duration_s must be a number of seconds above 0, not {value!r}")
    if math.isinf(value):
        return None
    return float(value)


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"
    if minutes:
        return f"{minutes} min {secs} s" if secs else f"{minutes} min"
    return f"{secs} s"


def too_long(name: str, limit: float, what: str) -> AudioTooLongError:
    """The error for audio (``name``) that ``what`` says is past the ``limit``."""
    reason = f"{name} {what}; SeamSplit reads at most {_hms(limit)} of audio at a time"
    error = AudioTooLongError(
        f"{reason}, because the memory it takes grows with its length. To read it anyway, "
        "raise the limit with --max-input-duration SECONDS (0 removes it; in code, "
        "max_input_s or max_duration_s=None)"
    )
    error.reason, error.limit_text = reason, _hms(limit)
    return error


class Decoded(NamedTuple):
    """What a file held: interleaved samples as bytes, and their format."""

    data: bytes
    sample_width: int
    frame_rate: int
    channels: int


# -- reading ------------------------------------------------------------------------

_DEMUXER_ALIASES = {"m4a": "mp4", "wave": "wav", "aif": "aiff"}


# -- what a file may make FFmpeg open ------------------------------------------------
#
# A file that claims to be audio is not trusted to be only that. FFmpeg also reads files
# that are lists of other things: a text file called ``song.mp3`` that says ``file other.wav``
# is detected as an ffconcat playlist and decodes as the audio of ``other.wav`` (a file next
# to it, or in a folder under it), a playlist can name addresses on a network. Those demuxers
# are not allowed when the format of a file is found by looking at it, and no input may
# open a network protocol. (FFmpeg refuses some of this itself, such as absolute paths in a
# playlist; which is refused depends on its version, so it is not left to it.)
_LISTING_FORMATS = frozenset(
    {
        "concat",  # ffconcat playlists
        "hls",  # HLS playlists
        "applehttp",
        "dash",  # DASH manifests
        "sdp",  # RTP sessions
        "rtsp",
        "rtp",
        "image2",  # numbered image sequences
        "image2pipe",
        "lavfi",  # filter graphs
        "ffmetadata",
    }
)
_PROTOCOLS = "file,pipe,fd"


def _open_options(probing: bool) -> Dict[str, str]:
    options = {"protocol_whitelist": _PROTOCOLS}
    if probing:
        # every demuxer there is but the ones above (FFmpeg only has a list of those allowed)
        options["format_whitelist"] = ",".join(
            sorted(name for name in av.formats_available if name not in _LISTING_FORMATS)
        )
    return options


def _open_input(file: PathOrFile, format: Optional[str]) -> Any:
    """``av.open`` for reading audio from ``file`` (see above). With a ``format`` that is
    named there is nothing to detect, and no demuxer of the list can be picked."""
    demuxer = _demuxer(format)
    return av.open(file, mode="r", format=demuxer, options=_open_options(demuxer is None))


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


def _declared_duration_s(container: Any, stream: Any) -> Optional[float]:
    """The length the file says its audio has, in seconds (``None`` if it says nothing).
    It is what the header says and can be wrong; see :func:`_decode`."""
    try:
        if stream.duration is not None and stream.time_base:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration) / av.time_base
    except (av.error.FFmpegError, ValueError, TypeError, ZeroDivisionError):
        pass
    return None


def _decode(container: Any, max_duration_s: Optional[float], name: str = "the file") -> Decoded:
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        raise CouldntDecodeError("Decoding failed: the file has no audio stream")
    if max_duration_s is not None:
        declared = _declared_duration_s(container, stream)
        if declared is not None and declared > max_duration_s:
            raise too_long(name, max_duration_s, f"is about {_hms(declared)} long")
    max_bytes = None if max_duration_s is None else int(max_duration_s * _BUDGET_BYTES_PER_S)
    resampler = None
    # The samples go straight into a BytesIO, which hands its buffer over as the bytes
    # object ``getvalue()`` returns (CPython does not copy it), so the audio is in memory
    # about once. Collecting the blocks, joining them and converting the result (as this
    # did) held it three times at the peak. The audio is not in memory twice for want of this.
    buffer = io.BytesIO()
    width, channels, rate = 2, 0, 0
    total_bytes = 0

    def take(frames: Iterable[Any]) -> None:
        nonlocal total_bytes
        for out in frames:
            array = out.to_ndarray().reshape(-1)
            if width == 1:  # u8 -> signed
                array = (array.astype(np.int16) - 128).astype(np.int8)
            array = np.ascontiguousarray(array.astype(SAMPLE_TYPES[width], copy=False))
            total_bytes += array.nbytes
            if max_duration_s is not None and max_bytes is not None:
                if (
                    total_bytes > max_bytes
                    or total_bytes > max_duration_s * width * channels * rate
                ):
                    raise too_long(name, max_duration_s, "holds more audio than that")
            # .data: numpy's stubs declare the buffer protocol only on Python 3.12+
            buffer.write(array.data)

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
    return Decoded(buffer.getvalue(), width, rate, channels)


def read_audio(
    file: PathOrFile,
    format: Optional[str] = None,
    max_duration_s: Optional[float] = DEFAULT_MAX_DURATION_S,
) -> Decoded:
    """Decode an audio file (a path or a binary file object) in any format FFmpeg knows.
    Raises :class:`CouldntDecodeError` for what is not audio, ``FileNotFoundError`` for a
    path that is not there, and :class:`AudioTooLongError` (a ``CouldntDecodeError``) for a
    file that holds more than ``max_duration_s`` seconds of audio (``None``: no limit): by
    the length it says it has, before it is decoded, or when decoding passes the limit."""
    max_duration_s = check_max_duration(max_duration_s)
    if isinstance(file, os.PathLike):
        file = os.fspath(file)
    name = f"'{file}'" if isinstance(file, str) else "the audio"
    try:
        container = _open_input(file, format)
    except FileNotFoundError:
        raise
    except av.error.ArgumentError as exc:  # what the list of demuxers (above) refuses
        raise CouldntDecodeError(  # (the message of ``exc`` is the whole list of demuxers)
            f"Decoding failed: {name} is not a kind of file SeamSplit reads as audio: "
            "it lists other files or addresses (a playlist), which are not followed"
        ) from exc
    except (av.error.FFmpegError, ValueError, OSError) as exc:
        raise CouldntDecodeError(f"Decoding failed: {exc}") from exc
    try:
        return _decode(container, max_duration_s, name)
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
        container = _open_input(path, None)
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


@contextlib.contextmanager
def atomic_output(path: Union[str, "os.PathLike[str]"]) -> Iterator[str]:
    """The path to write to in place of ``path``, so that a file that is replaced is replaced
    all at once or not at all. It is a file next to it (hidden, and gone again whatever
    happens), which takes the place of the old one only when the block has ended without an
    error: an encoder that fails half-way, a full disk or a kill leave the old file as it was,
    not a truncated new one. What is not a regular file (``/dev/null``, a pipe) is written to
    as it is, and a symbolic link is written through."""
    target = os.path.realpath(os.fspath(path))
    if os.path.exists(target) and not os.path.isfile(target):
        yield target
        return
    folder, name = os.path.split(target)
    partial = os.path.join(folder, f".{name}.{os.getpid()}.part")
    try:
        yield partial
        os.replace(partial, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(partial)
        raise


def atomic_write_text(path: str, text: str) -> None:
    """``text`` as UTF-8 in the file ``path``, all at once (see :func:`atomic_output`)."""
    with atomic_output(path) as target:
        with open(target, "w", encoding="utf-8") as f:
            f.write(text)


def write_bytes(out_f: PathOrFile, data: bytes) -> None:
    """``data`` to a path or a binary file object."""
    if isinstance(out_f, (str, os.PathLike)):
        with atomic_output(out_f) as target, open(target, "wb") as f:
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
        if isinstance(out_f, (str, os.PathLike)):
            # opened here, not by wave.open(path): when the open fails inside wave, the
            # half-built Wave_write raises again from __del__ (an "unraisable" warning)
            with atomic_output(out_f) as target, open(target, "wb") as f:
                _write_wav_to(f, data, sample_width, frame_rate, channels)
        else:
            _write_wav_to(out_f, data, sample_width, frame_rate, channels)
    except (OSError, wave.Error) as exc:
        raise CouldntEncodeError(f"Encoding failed: {exc}") from exc


def _write_wav_to(f: Any, data: bytes, sample_width: int, frame_rate: int, channels: int) -> None:
    with wave.open(f, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(frame_rate)
        w.setnframes(len(data) // (sample_width * channels))
        w.writeframesraw(data)


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
    are brought to the nearest one it does. A file that is replaced is replaced only when
    all of it has been written (see :func:`atomic_output`)."""
    if not isinstance(out_f, (str, os.PathLike)):
        _encode_to(out_f, fmt, rows, sample_width, frame_rate, bitrate)
        return
    try:
        with atomic_output(out_f) as target:
            _encode_to(target, fmt, rows, sample_width, frame_rate, bitrate)
    except OSError as exc:  # taking the new file's place: the encoding itself raises its own
        raise CouldntEncodeError(f"Encoding failed: {exc}") from exc


def _encode_to(
    out_f: PathOrFile,
    fmt: str,
    rows: np.ndarray,
    sample_width: int,
    frame_rate: int,
    bitrate: Optional[str],
) -> None:
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
