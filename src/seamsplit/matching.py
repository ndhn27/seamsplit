"""Pairing the manifest's parts with the processed files a service handed back:
by name when the names survived (a suffix such as ``_instrumental`` is fine), or,
when the files were renamed, by order."""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .report import MergeReport, _emit

# Extensions counted as audio when the processed files have been renamed and can
# only be paired with the parts by order (so a stray manifest.json or .txt is not
# mistaken for a chunk).
AUDIO_EXTENSIONS = frozenset(
    {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".oga",
        ".opus",
        ".m4a",
        ".aac",
        ".aif",
        ".aiff",
        ".aifc",
        ".wma",
    }
)


def _list_files(processed_dir: str) -> List[str]:
    return sorted(
        path
        for path in glob.glob(os.path.join(processed_dir, "**", "*"), recursive=True)
        if os.path.isfile(path)
    )


def _stem_matches(
    files: List[str], part_name: str, match_substr: Optional[str]
) -> Tuple[str, List[str]]:
    """The files whose name contains the stem of ``part_name`` (not followed by
    another digit, so ``song_part10`` is not found inside ``song_part100``),
    narrowed by ``match_substr``."""
    stem = os.path.splitext(part_name)[0]
    stem_re = re.compile(re.escape(stem) + r"(?!\d)")
    found = []
    for path in files:
        name = os.path.basename(path)
        if stem_re.search(name):
            if match_substr and match_substr.lower() not in name.lower():
                continue
            found.append(path)
    return stem, found


def _ambiguous(stem: str, found: List[str]) -> ValueError:
    listing = "\n".join(f"  - {c}" for c in sorted(found))
    return ValueError(
        f"{len(found)} files match '{stem}'; use --match to narrow it down:\n{listing}"
    )


def _filtered_by(match_substr: Optional[str]) -> str:
    return f" (filtered by '{match_substr}')" if match_substr else ""


def find_processed_file(
    processed_dir: str, original_part_name: str, match_substr: Optional[str] = None
) -> str:
    """Find the processed file in ``processed_dir`` corresponding to one
    original part (matched by filename stem, since an AI service may
    change the extension or append a suffix such as ``_instrumental``).
    ``song_part10`` does not match ``song_part100``.

    Raises ``FileNotFoundError`` if nothing matches, or ``ValueError`` if
    more than one file matches and ``match_substr`` doesn't disambiguate.
    """
    stem, found = _stem_matches(_list_files(processed_dir), original_part_name, match_substr)
    if not found:
        raise FileNotFoundError(
            f"No file in '{processed_dir}' matches '{stem}'" + _filtered_by(match_substr)
        )
    if len(found) > 1:
        raise _ambiguous(stem, found)
    return found[0]


def _natural_key(path: str) -> List[Any]:
    """Sort key that puts ``Song - 2`` before ``Song - 10``."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path)]


def _named_list(names: List[str], limit: int = 8) -> str:
    shown = ", ".join(f"'{n}'" for n in names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


@dataclass
class PairedFiles:
    """The processed files in part order, and how they were found: ``by_order`` is True
    when no name told which file is which part and they were paired by order (a guess),
    in which case ``order_note`` says what was assumed."""

    paths: List[str]
    by_order: bool = False
    order_note: str = ""


def pair_processed_files(
    processed_dir: str, part_names: List[str], match_substr: Optional[str] = None
) -> PairedFiles:
    """Find the processed file of every part, in part order.

    A part is looked up by its name, as in :func:`find_processed_file`. When
    *no* file carries any part name (the files were renamed, as an AI service or
    the user may do), the audio files are paired with the parts by order
    instead — natural sort, so ``Song - 2`` comes before ``Song - 10`` — which
    needs exactly one audio file per part (``match_substr`` narrows them); the result
    says so (``by_order``). When only some parts are found by name, a file is
    missing, and pairing by order would be a guess.

    Raises ``FileNotFoundError`` naming **every** part that has no file (or the
    file count that does not fit), or ``ValueError`` for a part with several
    files."""
    files = _list_files(processed_dir)
    matches = [_stem_matches(files, name, match_substr) for name in part_names]
    missing = [stem for stem, found in matches if not found]

    if not missing:
        for stem, found in matches:
            if len(found) > 1:
                raise _ambiguous(stem, found)
        return PairedFiles([found[0] for _, found in matches])

    if len(missing) < len(matches):
        if len(missing) == 1:
            what = f"matches '{missing[0]}'"
        else:
            what = f"matches {len(missing)} of {len(matches)} parts: {_named_list(missing)}"
        raise FileNotFoundError(f"No file in '{processed_dir}' {what}" + _filtered_by(match_substr))

    # Nothing carries a part name: the files were renamed. Pair them by order.
    audio = [
        path
        for path in files
        if os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS
        and (not match_substr or match_substr.lower() in os.path.basename(path).lower())
    ]
    audio.sort(key=lambda path: _natural_key(os.path.relpath(path, processed_dir)))
    if len(audio) != len(part_names):
        hint = ""
        if len(audio) > len(part_names) and not match_substr:
            hint = "; if the service returned several files per chunk, narrow them with --match"
        elif len(audio) < len(part_names):
            hint = "; some chunks may be missing ('seamsplit status' lists which)"
        raise FileNotFoundError(
            f"No file in '{processed_dir}' carries a part name (such as '{missing[0]}'), and "
            "files that were renamed can only be paired with the parts by order, which needs "
            f"one audio file per part: found {len(audio)}{_filtered_by(match_substr)} for "
            f"{len(part_names)} parts{hint}"
        )

    shown = os.path.relpath(audio[0], processed_dir), os.path.relpath(audio[-1], processed_dir)
    note = (
        f"no file name contains a part name (such as '{missing[0]}'), so the {len(audio)} audio "
        f"files were paired with the parts by order: '{shown[0]}' is part 1 … '{shown[1]}' is "
        f"part {len(audio)}. If that is not the order the parts were split in, rename them so "
        "they sort that way"
    )
    return PairedFiles(audio, True, note)


def find_processed_files(
    processed_dir: str,
    part_names: List[str],
    match_substr: Optional[str] = None,
    report: Optional[MergeReport] = None,
) -> List[str]:
    """:func:`pair_processed_files`, giving just the paths; a pairing by order is
    reported to ``report`` (or warned about) as the guess it is."""
    paired = pair_processed_files(processed_dir, part_names, match_substr)
    if paired.by_order:
        _emit(report, paired.order_note)
    return paired.paths


def find_original_chunks(originals_dir: str, part_names: List[str]) -> List[str]:
    """The original chunk files of ``part_names`` (the manifest's ``parts``) in
    ``originals_dir``, in part order: the files ``split`` wrote, which are the ones
    the service was given. Their names are exact (a service's renaming does not
    matter here, they never left this computer).

    Raises ``FileNotFoundError`` naming **every** chunk that is missing."""
    paths = [os.path.join(originals_dir, name) for name in part_names]
    missing = [name for name, path in zip(part_names, paths) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"The original chunks are needed to anchor the merge, but {len(missing)} of "
            f"{len(part_names)} are not in '{originals_dir}': {_named_list(missing)}. Point "
            "--originals at the folder 'split' wrote them to"
        )
    return paths
