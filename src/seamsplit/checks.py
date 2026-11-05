"""Checks on the numbers a caller passes in, made before any file is read, so a
bad value fails at once with a message that names it instead of somewhere deep
in the signal processing."""

from __future__ import annotations

import math
import numbers
from typing import Any, Optional

_JSON_TYPE_NAMES = (
    (bool, "boolean"),  # before int: bool is a subclass of it
    (int, "number"),
    (float, "number"),
    (str, "string"),
    (list, "array"),
    (dict, "object"),
)


def _describe(value: Any) -> str:
    """A value as its JSON type plus a short preview, for error messages
    (``string '100'``, ``number nan``, ``array``, ``null``)."""
    if value is None:
        return "null"
    kind = next((name for cls, name in _JSON_TYPE_NAMES if isinstance(value, cls)), "value")
    if isinstance(value, (list, dict)):
        return kind
    preview = repr(value)
    return f"{kind} {preview if len(preview) <= 40 else preview[:37] + '...'}"


def _number_problem(value: Any) -> Optional[str]:
    """Why ``value`` is not a finite, non-negative number (``None`` if it is
    one). A bool is not accepted (``True`` would otherwise count as 1), and
    neither are NaN / ±Infinity, which ``json`` parses by default."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"must be a number, got {_describe(value)}"
    try:
        finite = math.isfinite(value)
    except OverflowError:  # an int too large to convert to a float
        finite = False
    if not finite:
        return f"must be a finite number, got {_describe(value)}"
    if value < 0:
        return f"must not be negative, got {value!r}"
    return None


def _check_number(name: str, value: Any, *, positive: bool = False) -> float:
    """Validate a numeric argument of the public API and return it as a float.

    Must be a real number (a ``bool`` or a string is not accepted) and finite:
    NaN compares false with everything, so it would slip past a ``< 0`` check
    and silently turn a limit or a warning off. It must also not be negative
    (or, with ``positive=True``, must be greater than 0). The error names the
    argument, instead of leaving the caller to decode what a later step made
    of a bad value (a "low correlation" warning, an error from the audio
    library after the whole merge has run, ...).
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:  # an int too large to convert to a float
        number = math.inf
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    if number < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return number


def _check_samplerate(value: Any) -> int:
    """``samplerate`` must be a whole number of Hz greater than 0. It is only
    used for the final resample, so a bad one is rejected up front instead of surfacing
    (as an error from the audio library) after every chunk has been loaded and merged."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"samplerate must be a positive whole number (Hz), got {value!r}")
    return int(value)


def _check_merge_options(
    max_lag_ms: float,
    max_gain_db: float,
    seam_fade_ms: float,
    seam_warn_db: Optional[float],
) -> None:
    """The numeric options shared by every merge entry point."""
    _check_number("max_lag_ms", max_lag_ms)
    _check_number("max_gain_db", max_gain_db)
    _check_number("seam_fade_ms", seam_fade_ms, positive=True)
    if seam_warn_db is not None:
        # Any dB value is meaningful (even a negative one), but NaN never
        # compares true, so it would turn the seam warnings off unnoticed.
        if isinstance(seam_warn_db, bool) or not isinstance(seam_warn_db, numbers.Real):
            raise ValueError(f"seam_warn_db must be a number or None, got {seam_warn_db!r}")
        if math.isnan(seam_warn_db):
            raise ValueError("seam_warn_db must not be NaN (use None to switch the warnings off)")


def _check_threshold(name: str, value: Any) -> None:
    """A warning threshold in dB: any number (even a negative one is meaningful) or
    ``None`` for "never warn". NaN never compares true, so it would turn the warning
    off unnoticed."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number or None, got {value!r}")
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN (use None to switch the warnings off)")
