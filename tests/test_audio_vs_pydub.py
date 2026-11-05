"""``seamsplit.audio`` against pydub, the library it replaces: the same operations on the
same samples must give the same bytes.

pydub is not a dependency any more; this file runs only where it is installed (``pip
install pydub``, and ``audioop-lts`` on Python 3.13 and later) and is skipped otherwise. It
is the evidence that the replacement changed nothing a merge depends on: slicing, gain,
fades, sample width and channel changes all come out byte for byte the same. (Sample rate
conversion is the deliberate exception: FFmpeg's resampler replaces ``audioop.ratecv``,
which aliases, and is checked on its own in ``test_audio.py``.)
"""

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pydub = pytest.importorskip("pydub")

from seamsplit.audio import AudioSegment  # noqa: E402

_DTYPES = {1: np.int8, 2: np.int16, 4: np.int32}


def _pair(width, rate, channels, frames, seed=7):
    rng = np.random.default_rng(seed)
    info = np.iinfo(_DTYPES[width])
    x = rng.integers(
        info.min, info.max, size=frames * channels, dtype=_DTYPES[width], endpoint=True
    )
    x[:4] = [info.min, info.max, 0, -1]  # the extremes, and a zero next to a minus one
    raw = x.tobytes()
    params = dict(sample_width=width, frame_rate=rate, channels=channels)
    return pydub.AudioSegment(raw, **params), AudioSegment(raw, **params)


def _same(theirs, ours):
    assert (theirs.sample_width, theirs.frame_rate, theirs.channels) == (
        ours.sample_width,
        ours.frame_rate,
        ours.channels,
    )
    assert len(theirs) == len(ours)
    assert theirs.raw_data == ours.raw_data


CASES = [(w, sr, ch) for w in (1, 2, 4) for sr in (8000, 22050, 44100, 48000) for ch in (1, 2, 4)]


@pytest.fixture(params=CASES, ids=lambda c: "w{}-{}Hz-{}ch".format(*c))
def pair(request):
    width, rate, channels = request.param
    return _pair(width, rate, channels, int(rate * 0.43) + 7)  # a length that is not whole ms


@pytest.mark.parametrize("gain", [-60.0, -6.02, -0.3, 0.0, 3.0, 12.0])
def test_gain_is_the_same(pair, gain):
    theirs, ours = pair
    _same(theirs.apply_gain(gain), ours.apply_gain(gain))
    _same(theirs + gain, ours + gain)


@pytest.mark.parametrize(
    "bounds", [(0, 100), (50, 333), (-200, None), (None, -100), (10, 5000), (100, 100)]
)
def test_slices_are_the_same(pair, bounds):
    theirs, ours = pair
    _same(theirs[slice(*bounds)], ours[slice(*bounds)])


@pytest.mark.parametrize("duration", [5, 20, 99, 100, 101, 250, 400])
def test_fades_are_the_same(pair, duration):
    theirs, ours = pair
    _same(theirs.fade_in(duration), ours.fade_in(duration))
    _same(theirs.fade_out(duration), ours.fade_out(duration))


def test_fades_between_two_gains_are_the_same(pair):
    theirs, ours = pair
    a = dict(to_gain=-12, from_gain=-3, start=50, end=300)
    b = dict(to_gain=-20, start=10, duration=250)
    _same(theirs.fade(**a), ours.fade(**a))
    _same(theirs.fade(**b), ours.fade(**b))


def test_sample_width_and_channel_changes_are_the_same(pair):
    theirs, ours = pair
    for width in (1, 2, 4):
        _same(theirs.set_sample_width(width), ours.set_sample_width(width))
    for channels in (1, 2, 4):
        if channels != ours.channels and 1 in (channels, ours.channels):
            _same(theirs.set_channels(channels), ours.set_channels(channels))


def test_joining_is_the_same(pair):
    theirs, ours = pair
    _same(theirs + theirs[:100], ours + ours[:100])
    pad_theirs = pydub.AudioSegment.silent(30, frame_rate=theirs.frame_rate)
    pad_ours = AudioSegment.silent(30, frame_rate=ours.frame_rate)
    _same(pad_theirs + theirs, pad_ours + ours)  # silence is mono 16 bit: the sum is widened
    if ours.channels <= 2:  # pydub (and so this) cannot turn stereo into four channels
        _same(pad_theirs.set_channels(2) + theirs, pad_ours.set_channels(2) + ours)
