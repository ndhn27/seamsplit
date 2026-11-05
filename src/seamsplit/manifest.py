"""Reading and checking ``manifest.json``, the description of a split that
``seamsplit split`` writes and ``merge`` needs: the overlap, the part file names
and, for the integrity checks, where each part came from in the source."""

from __future__ import annotations

import json
from typing import Any, Optional

from .checks import _describe, _number_problem


def _part_name_problem(name: Any) -> Optional[str]:
    """Why ``name`` cannot be one of the chunk file names in ``parts`` (``None``
    if it can). ``split`` only ever writes bare names, and a processed chunk
    is looked up by that name inside one folder."""
    if not isinstance(name, str):
        return f"must be a file name (a string), got {_describe(name)}"
    if not name.strip() or name in (".", ".."):
        return f"is not a file name ({name!r})"
    if "\0" in name:
        return "must not contain a null character"
    if "/" in name or "\\" in name:
        return f"must be a bare file name without folders, got {name!r}"
    return None


def load_manifest(path: str) -> dict:
    """Read a ``manifest.json`` written by ``split`` and check its schema.

    ``parts`` must be a non-empty list of bare file names and ``overlap_ms``
    a finite, non-negative number. The optional fields are checked when
    present (``null`` counts as absent): ``source_file`` is a string,
    ``source_length_ms`` and ``max_export_ms`` are finite non-negative numbers,
    and ``segments`` has one object per part with finite, non-negative
    ``start_ms`` <= ``end_ms``. Anything else raises :class:`ValueError`
    naming the field, so callers never see a ``TypeError`` or ``OverflowError``
    from deep inside a merge."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"'{path}' is not valid JSON: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"'{path}' is not UTF-8 text: {exc}") from exc
        except RecursionError as exc:  # absurdly nested arrays/objects
            raise ValueError(f"'{path}' is nested too deeply to be a seamsplit manifest") from exc

    def invalid(problem: str) -> ValueError:
        return ValueError(f"'{path}' is not a valid seamsplit manifest: {problem}")

    def check_number(label: str, value: Any) -> None:
        problem = _number_problem(value)
        if problem:
            raise invalid(f"'{label}' {problem}")

    if not isinstance(manifest, dict):
        raise invalid(f"expected a JSON object, got {_describe(manifest)}")

    if "parts" not in manifest:
        raise invalid("'parts' is missing")
    parts = manifest["parts"]
    if not isinstance(parts, list):
        raise invalid(f"'parts' must be a list of file names, got {_describe(parts)}")
    if not parts:
        raise invalid("'parts' must not be empty")
    for i, name in enumerate(parts):
        problem = _part_name_problem(name)
        if problem:
            raise invalid(f"'parts[{i}]' {problem}")

    if "overlap_ms" not in manifest:
        raise invalid("'overlap_ms' is missing")
    check_number("overlap_ms", manifest["overlap_ms"])

    source_file = manifest.get("source_file")
    if source_file is not None and not isinstance(source_file, str):
        raise invalid(f"'source_file' must be a string, got {_describe(source_file)}")
    for key in ("source_length_ms", "max_export_ms"):
        if manifest.get(key) is not None:
            check_number(key, manifest[key])

    segments = manifest.get("segments")
    if segments is not None:
        if not isinstance(segments, list):
            raise invalid(f"'segments' must be a list, got {_describe(segments)}")
        if len(segments) != len(parts):
            raise invalid(f"'segments' has {len(segments)} entries but 'parts' has {len(parts)}")
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                raise invalid(f"'segments[{i}]' must be an object, got {_describe(seg)}")
            for key in ("start_ms", "end_ms"):
                if key not in seg:
                    raise invalid(f"'segments[{i}].{key}' is missing")
                check_number(f"segments[{i}].{key}", seg[key])
            if seg["end_ms"] < seg["start_ms"]:
                raise invalid(
                    f"'segments[{i}]' ends before it starts "
                    f"(start_ms {seg['start_ms']!r}, end_ms {seg['end_ms']!r})"
                )
    return manifest
