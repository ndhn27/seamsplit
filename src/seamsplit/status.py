"""How far a job has got: for every chunk (and every stem wanted of it), is the processed
file there, and are there files that belong to no chunk?

Files are found the way ``merge`` finds them, by the chunk name in the file name. A file
whose name was lost is not simply "unmatched": when the original chunks are at hand it
is compared with them (:mod:`seamsplit.pairing`) and, if it is clearly one of the chunks
that has no file yet, it counts for that chunk. So ``status`` is also the answer to "which
of the chunks have I not got back?" when the service has renamed everything.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .manifest import load_manifest
from .matching import AUDIO_EXTENSIONS, _list_files, _stem_matches
from .pairing import identify
from .splitter import probe_duration_s

MANIFEST_FILE = "manifest.json"
# A file's length as ffprobe gives it includes what a lossy encoder adds, so this is
# looser than the check ``merge`` makes of the decoded audio.
LENGTH_TOLERANCE_MS = 300.0

DONE, MISSING, AMBIGUOUS = "done", "missing", "ambiguous"


@dataclass
class SlotStatus:
    """One thing wanted of a chunk: the processed file of ``stem`` (``None``: "the" file).

    ``state`` is ``done``, ``missing`` or ``ambiguous`` (several files for it). ``how`` says
    what tied the file to the chunk: ``name`` (its name carries the chunk's) or ``sound``
    (it was compared with the original chunk; ``similarity`` is the score). ``length_off``
    is True when the file's length is not that of the chunk by more than
    :data:`LENGTH_TOLERANCE_MS`."""

    part: int
    name: str
    stem: Optional[str]
    state: str
    files: List[str] = field(default_factory=list)
    how: Optional[str] = None
    similarity: Optional[float] = None
    length_ms: Optional[float] = None
    expected_ms: Optional[float] = None
    length_off: bool = False


@dataclass
class JobStatus:
    chunks_dir: str
    processed_dir: str
    slots: List[SlotStatus]
    # audio files that are not any chunk (or that could not be told apart)
    unmatched: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def missing(self) -> List[SlotStatus]:
        return [s for s in self.slots if s.state != DONE]

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def parts(self) -> int:
        return len({s.part for s in self.slots})

    def pending_parts(self) -> List[int]:
        """The 1-based numbers of the chunks that still lack something."""
        return sorted({s.part for s in self.missing})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complete": self.complete,
            "chunks": self.parts,
            "pending_parts": self.pending_parts(),
            "slots": [asdict(s) for s in self.slots],
            "unmatched": list(self.unmatched),
            "notes": list(self.notes),
        }


def _expected_ms(manifest: dict, index: int) -> Optional[float]:
    segments = manifest.get("segments")
    if not segments:
        return None
    seg = segments[index]
    return float(seg["end_ms"]) - float(seg["start_ms"])


def compute_status(
    chunks_dir: str,
    processed_dir: str,
    stems: Sequence[Optional[str]] = (None,),
    check_lengths: bool = True,
) -> JobStatus:
    """The status of the job whose split is in ``chunks_dir`` (it holds ``manifest.json``)
    and whose results are in ``processed_dir`` (which need not exist yet).

    ``stems`` are the words that tell the files wanted of every chunk apart (``("vocals",
    "instrumental")``); ``(None,)`` means one file per chunk, whatever it is called."""
    manifest = load_manifest(os.path.join(chunks_dir, MANIFEST_FILE))
    names: List[str] = manifest["parts"]
    originals = [os.path.join(chunks_dir, n) for n in names]
    files = _list_files(processed_dir) if os.path.isdir(processed_dir) else []
    audio = [p for p in files if os.path.splitext(p)[1].lower() in AUDIO_EXTENSIONS]
    status = JobStatus(chunks_dir, processed_dir, [])
    originals_here = all(os.path.isfile(p) for p in originals)
    explained = set()

    for stem in stems:
        pool = [p for p in audio if not stem or stem.lower() in os.path.basename(p).lower()]
        found = [_stem_matches(pool, name, None)[1] for name in names]
        named = {p for group in found for p in group}
        leftovers = [p for p in pool if p not in named]
        sound: Dict[int, List[Any]] = {}
        gaps = [i for i, group in enumerate(found) if not group]
        if leftovers and gaps:
            if originals_here:
                for match in identify(leftovers, originals):
                    if match.confident and match.part in gaps:
                        sound.setdefault(match.part, []).append(match)
            else:
                status.notes.append(
                    "some files carry no chunk name, and the original chunks are not in "
                    f"'{chunks_dir}' to tell which they are by comparison"
                )
        for i, name in enumerate(names):
            slot = SlotStatus(i + 1, name, stem, MISSING, expected_ms=_expected_ms(manifest, i))
            if found[i]:
                slot.files, slot.how = list(found[i]), "name"
            elif i in sound:
                slot.files = [m.file for m in sound[i]]
                slot.how = "sound"
                slot.similarity = max(m.score for m in sound[i])
            if slot.files:
                slot.state = DONE if len(slot.files) == 1 else AMBIGUOUS
                explained.update(slot.files)
                if check_lengths and slot.state == DONE:
                    seconds = probe_duration_s(slot.files[0])
                    if seconds is not None:
                        slot.length_ms = seconds * 1000.0
                        if slot.expected_ms is not None:
                            slot.length_off = abs(slot.length_ms - slot.expected_ms) > (
                                LENGTH_TOLERANCE_MS
                            )
            status.slots.append(slot)

    status.unmatched = sorted(p for p in audio if p not in explained and _wanted(p, stems))
    return status


def _wanted(path: str, stems: Sequence[Optional[str]]) -> bool:
    """Whether a stray file is one the job asked for (it carries one of the stem words)."""
    name = os.path.basename(path).lower()
    return any(not stem or stem.lower() in name for stem in stems)


def format_status(status: JobStatus) -> str:
    """The status as text: a line per chunk, then what is missing and what is left over."""
    lines = []
    stems = [s for s in dict.fromkeys(slot.stem for slot in status.slots)]
    by_part: Dict[int, List[SlotStatus]] = {}
    for slot in status.slots:
        by_part.setdefault(slot.part, []).append(slot)
    done = sum(1 for slots in by_part.values() if all(s.state == DONE for s in slots))
    lines.append(f"{done} of {len(by_part)} chunk(s) are complete.")
    for part, slots in sorted(by_part.items()):
        for slot in slots:
            label = f"  part {part:>3} {slot.name}"
            if slot.stem and len(stems) > 1:
                label += f" [{slot.stem}]"
            if slot.state == MISSING:
                lines.append(f"{label}: MISSING")
            elif slot.state == AMBIGUOUS:
                listing = ", ".join(os.path.basename(f) for f in slot.files)
                lines.append(
                    f"{label}: {len(slot.files)} files ({listing}); narrow them with --match"
                )
            else:
                where = os.path.basename(slot.files[0])
                if slot.how == "sound":
                    where += (
                        " (named differently; it sounds like this chunk, similarity "
                        f"{slot.similarity:.2f})"
                    )
                extra = ""
                if slot.length_off and slot.length_ms is not None and slot.expected_ms is not None:
                    extra = (
                        f"; its length ({slot.length_ms / 1000:.2f}s) differs from the "
                        f"chunk's ({slot.expected_ms / 1000:.2f}s)"
                    )
                lines.append(f"{label}: {where}{extra}")
    if status.unmatched:
        lines.append(f"\n{len(status.unmatched)} audio file(s) are none of the chunks:")
        lines += [f"  {os.path.relpath(p, status.processed_dir)}" for p in status.unmatched]
    for note in status.notes:
        lines.append(f"\nNote: {note}")
    pending = status.pending_parts()
    if pending:
        shown = ", ".join(str(p) for p in pending[:20])
        if len(pending) > 20:
            shown += f" and {len(pending) - 20} more"
        lines.append(f"\nStill to come back: chunk(s) {shown}.")
    return "\n".join(lines)
