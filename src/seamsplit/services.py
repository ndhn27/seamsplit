"""Presets for the services chunks are sent to: the limit on the length of a file, and
the format and overlap to cut with, under a name (``--service xminus``).

A preset only fills in what the command line leaves unsaid; an option that is given
always wins (see :func:`apply_preset`). Besides the built-in presets there is a file of
the user's own, ``services.json`` in the configuration folder (or the path of
``--services-file``)::

    {"myservice": {"limit_s": 120, "format": "wav", "overlap_s": 5, "note": "my plan"}}

A name in that file replaces a built-in preset of the same name.

What a preset says about a service is a claim about a web site that can change at any
time, so nothing here is checked against the service: ``split --limit`` also measures
the files it wrote, and that is what tells whether a limit still holds.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .checks import _describe, _number_problem

SERVICES_ENV = "SEAMSPLIT_SERVICES"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORMAT_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")
_KEYS = ("limit_s", "format", "overlap_s", "note")


@dataclass(frozen=True)
class ServicePreset:
    """What is known about one service. ``None`` means "no opinion": the usual default
    of the option is kept."""

    name: str
    limit_s: Optional[float] = None
    format: Optional[str] = None
    overlap_s: Optional[float] = None
    note: str = ""
    builtin: bool = False

    def describe(self) -> str:
        parts = []
        if self.limit_s is not None:
            parts.append(f"limit {self.limit_s:g}s")
        if self.format is not None:
            parts.append(f"format {self.format}")
        if self.overlap_s is not None:
            parts.append(f"overlap {self.overlap_s:g}s")
        text = ", ".join(parts) or "no settings"
        return f"{text} — {self.note}" if self.note else text


BUILTIN_PRESETS: Dict[str, ServicePreset] = {
    "xminus": ServicePreset(
        "xminus",
        limit_s=59.0,
        note="the limit as it was reported to this project; not checked against the service",
        builtin=True,
    ),
}


def default_services_file() -> str:
    """Where the user's own presets are looked for when no path is given."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "seamsplit", "services.json")


def _preset_from(name: str, data: Any, source: str) -> ServicePreset:
    def invalid(problem: str) -> ValueError:
        return ValueError(f"'{source}': service '{name}' {problem}")

    if not _NAME_RE.match(name):
        raise ValueError(
            f"'{source}': '{name}' is not a usable service name (letters, digits, _ . -)"
        )
    if not isinstance(data, dict):
        raise invalid(f"must be an object, got {_describe(data)}")
    unknown = sorted(set(data) - set(_KEYS))
    if unknown:
        raise invalid(f"has unknown keys {', '.join(unknown)} (known: {', '.join(_KEYS)})")
    numbers: Dict[str, Optional[float]] = {}
    for key in ("limit_s", "overlap_s"):
        value = data.get(key)
        if value is not None:
            problem = _number_problem(value)
            if problem:
                raise invalid(f"'{key}' {problem}")
        numbers[key] = None if value is None else float(value)
    if numbers["limit_s"] is not None and not numbers["limit_s"] > 0:
        raise invalid("'limit_s' must be greater than 0")
    fmt = data.get("format")
    if fmt is not None:
        if not isinstance(fmt, str) or not _FORMAT_RE.match(fmt):
            raise invalid(f"'format' must be a file extension such as 'wav', got {_describe(fmt)}")
        fmt = fmt.lower()
    note = data.get("note", "")
    if not isinstance(note, str):
        raise invalid(f"'note' must be text, got {_describe(note)}")
    return ServicePreset(
        name.lower(),
        limit_s=numbers["limit_s"],
        format=fmt,
        overlap_s=numbers["overlap_s"],
        note=note,
    )


def load_user_presets(path: str) -> Dict[str, ServicePreset]:
    """The presets in a user's ``services.json``. Raises :class:`ValueError` naming what
    is wrong with the file, and :class:`FileNotFoundError` if it does not exist."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"'{path}' is not valid JSON: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"'{path}' is not UTF-8 text: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"'{path}' must hold an object of presets, got {_describe(data)}")
    presets = {}
    for name, body in data.items():
        preset = _preset_from(str(name), body, path)
        presets[preset.name] = preset
    return presets


def available_presets(services_file: Optional[str] = None) -> Dict[str, ServicePreset]:
    """The built-in presets plus the user's. A file named with ``services_file`` (or the
    ``SEAMSPLIT_SERVICES`` environment variable) must exist; the default one is read if
    it is there."""
    presets = dict(BUILTIN_PRESETS)
    explicit = services_file or os.environ.get(SERVICES_ENV)
    path = explicit or default_services_file()
    if explicit or os.path.isfile(path):
        presets.update(load_user_presets(path))
    return presets


def find_preset(name: str, services_file: Optional[str] = None) -> ServicePreset:
    """The preset called ``name`` (case does not matter); :class:`ValueError` lists the
    ones there are if it does not exist."""
    presets = available_presets(services_file)
    try:
        return presets[name.strip().lower()]
    except KeyError:
        known = ", ".join(sorted(presets)) or "none"
        raise ValueError(f"Unknown service '{name}' (available: {known})") from None


def apply_preset(
    preset: Optional[ServicePreset],
    *,
    target: Optional[float],
    limit: Optional[float],
    fmt: Optional[str],
    overlap: Optional[float],
) -> Dict[str, Any]:
    """Resolve ``limit``, ``fmt`` and ``overlap`` (``None`` where the caller said
    nothing) against a preset (``None``: no preset, nothing changes): what was given
    wins, the preset fills the rest.

    The preset's limit is a ``--limit``, so it cannot be combined with ``target`` (the
    caller who sets the chunk length by hand has taken the arithmetic over): that is a
    :class:`ValueError`. Returns ``{"limit", "fmt", "overlap", "notes"}`` where ``notes``
    lists in words what the preset decided."""
    notes: List[str] = []
    if preset is None:
        return {"limit": limit, "fmt": fmt, "overlap": overlap, "notes": notes}
    if target is not None and preset.limit_s is not None and limit is None:
        raise ValueError(
            f"--service {preset.name} sets a limit of {preset.limit_s:g}s, which cannot be "
            "combined with --target; pass --limit to use another limit"
        )
    if limit is None and preset.limit_s is not None:
        limit = preset.limit_s
        notes.append(f"limit {limit:g}s")
    if fmt is None and preset.format is not None:
        fmt = preset.format
        notes.append(f"format {fmt}")
    if overlap is None and preset.overlap_s is not None:
        overlap = preset.overlap_s
        notes.append(f"overlap {overlap:g}s")
    return {"limit": limit, "fmt": fmt, "overlap": overlap, "notes": notes}
