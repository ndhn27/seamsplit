"""The merge is a loop over small stages (``crossfade._measure``, ``_place_fade``, ``_blend``,
``_Settled``; ``merger._check_numbers``, ``_pair_stems``, ...). These pin what the split must
not change: where the warnings point, and what the pieces keep and count."""

import os
import warnings

import numpy as np
import pytest
from seamsplit.audio import AudioSegment
from seamsplit.crossfade import _crossfade_merge, _Settings, _Settled
from seamsplit.merger import merge_audio

SR = 8000


def _tone(seconds, gain=1.0):
    t = np.arange(int(SR * seconds)) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 330 * t + 1)
    pcm = np.clip(x * gain * 32767, -32768, 32767).astype("<i2")
    return AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=SR, channels=1)


def _settings(**kw):
    values = dict(
        method="adaptive",
        sr=SR,
        ch=1,
        overlap_frames=400,
        max_lag_frames=0,
        max_gain_db=6.0,
        max_ratio=2.0,
        fade_frames=2000,
        match_level=False,
        seam_search=False,
        seam_warn_db=15.0,
    )
    values.update(kw)
    return _Settings(**values)


# --- the pieces ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, analyse",
    [
        (dict(method="adaptive"), True),  # the curve is shaped by the measurement
        (dict(method="linear"), False),  # nothing depends on one
        (dict(method="linear", max_lag_frames=10), True),
        (dict(method="equalpower", match_level=True), True),
        (dict(method="linear", seam_search=True), True),
    ],
)
def test_the_overlaps_are_measured_only_when_something_depends_on_it(kw, analyse):
    assert _settings(**kw).analyse is analyse


def test_what_a_boundary_can_still_change_is_the_overlap_plus_the_lag_on_every_channel():
    assert _settings(overlap_frames=400, max_lag_frames=50, ch=2).keep_samples == 900


def test_settled_keeps_the_end_back_rounds_the_rest_and_counts_what_it_clips():
    settled = _Settled(np.dtype(np.int16), channels=1, keep=2)
    block = np.array([40000.0, -40000.0, 1.4, 2.6, 3.0])
    kept = settled.settle(block)
    assert kept.tolist() == [2.6, 3.0] and kept.base is None  # the end, as a copy
    assert settled.frames == 3 and settled.clipped == 2
    assert settled.settle(kept, 0).size == 0
    assert np.frombuffer(settled.take(), dtype="<i2").tolist() == [32767, -32768, 1, 3, 3]
    assert settled.take() == b""  # taken once


# --- where a warning without a report points ---------------------------------------------


def _warned_from(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [(str(w.message), os.path.basename(w.filename)) for w in caught]


def test_a_merge_without_a_report_points_its_warnings_at_its_caller():
    """``_emit`` points a warning at the caller of the function that calls it, so the stages
    return their messages and ``_crossfade_merge`` emits them itself (see its docstring)."""
    quiet, loud = _tone(2.0), _tone(2.0, gain=10.0)
    seen = _warned_from(
        lambda: _crossfade_merge([quiet, loud], 500, "adaptive", 50, None, True, max_gain_db=6)
    )
    assert any("correction limited" in message for message, _ in seen)
    assert {where for _, where in seen} == {os.path.basename(__file__)}


def test_merge_audio_without_a_report_points_its_warnings_at_its_caller(service, tmp_path):
    seen = _warned_from(
        lambda: merge_audio(
            service.processed_dir,
            service.manifest_path,
            str(tmp_path / "out.wav"),
            match="instrumental",
            overlap_s=1.0,  # the chunks were cut with another one: a warning
        )
    )
    # (a chunk of another length is also warned about, from where the chunks are read)
    assert [where for message, where in seen if "overlap override" in message] == [
        os.path.basename(__file__)
    ]
