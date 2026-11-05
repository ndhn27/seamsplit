"""Separating a chunk on this computer, with a program that is already installed (Demucs,
the models of UVR through ``audio-separator``, or any command line tool), in place of
sending the chunk to a web service.

A backend is a command and a description of what it writes. SeamSplit runs the command
on one chunk, in a work folder of its own, looks in what came out for each stem by file
name, and moves the stems next to the other results as ``<chunk>_<stem>.<ext>`` — the
names ``merge`` finds. A chunk is done only when every stem has been moved, so a run
that is stopped can be started again and carries on with the chunks that are not done.

Nothing here uploads anything, and nothing here is a separator: the quality is the
program's. The command lines and output names of the presets are those the programs'
documentation gives; they have not been run against the real programs from here, and a
run that cannot find the stems says which files the program wrote, which is what to
adjust (``--backend-cmd`` and ``--backend-stem`` take over completely).
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .matching import AUDIO_EXTENSIONS

# the stems every preset yields, in the order the main one comes first
VOCALS, INSTRUMENTAL = "vocals", "instrumental"


class BackendUnavailable(OSError):
    """The program is not installed (or cannot be started): every chunk would fail."""


class BackendError(ValueError):
    """The program ran and failed on this chunk, or wrote something that is not the stems."""


@dataclass(frozen=True)
class Backend:
    """``command`` is the program and its arguments, with ``{input}`` (the chunk) and
    ``{output}`` (a folder for the results) in them; ``stems`` maps each stem to a regular
    expression (case does not matter) its file name matches."""

    name: str
    command: Tuple[str, ...]
    stems: Tuple[Tuple[str, str], ...]
    install_hint: str = ""

    @property
    def stem_names(self) -> List[str]:
        return [stem for stem, _ in self.stems]

    @property
    def program(self) -> str:
        return self.command[0]


# Demucs: `demucs --two-stems=vocals -o OUT FILE` writes vocals.* and no_vocals.* under OUT.
_DEMUCS_STEMS = (
    (VOCALS, r"^vocals\.[a-z0-9]+$"),
    (INSTRUMENTAL, r"^(no_vocals|other)\.[a-z0-9]+$"),
)
# audio-separator writes <input>_(<Stem>)_<model>.<ext>.
_SEPARATOR_STEMS = (
    (VOCALS, r"\(vocals\)"),
    (INSTRUMENTAL, r"\((instrumental|no[ _]?vocals|other)\)"),
)

PRESETS = ("demucs", "audio-separator", "uvr")


def preset_backend(name: str, model: Optional[str] = None) -> Backend:
    """A built-in backend: ``demucs`` (two stems, vocals and the rest) or
    ``audio-separator`` (``uvr`` is the same: it is the command line of the UVR models).
    ``model`` is passed as the program's model option; without it the program's own default
    is used."""
    key = name.strip().lower()
    if key == "demucs":
        command = ["demucs", "--two-stems=vocals", "-o", "{output}"]
        if model:
            command += ["-n", model]
        return Backend(
            "demucs",
            tuple(command + ["{input}"]),
            _DEMUCS_STEMS,
            "pip install demucs (see its README for the model it downloads)",
        )
    if key in ("audio-separator", "uvr"):
        command = ["audio-separator", "{input}", "--output_dir", "{output}"]
        if model:
            command += ["--model_filename", model]
        return Backend(
            "audio-separator",
            tuple(command + ["--output_format", "WAV"]),
            _SEPARATOR_STEMS,
            "pip install audio-separator (its README says which extra to take for your "
            "hardware); the models are downloaded on first use",
        )
    raise ValueError(
        f"Unknown backend '{name}' (available: {', '.join(PRESETS)}, or --backend-cmd)"
    )


def custom_backend(command: str, stem_patterns: Sequence[str]) -> Backend:
    """A backend of the user's own: ``command`` is a command line with ``{input}`` and
    ``{output}`` in it, and ``stem_patterns`` are ``STEM=REGEX`` (the regular expression is
    matched against the file names the command writes)."""
    parts = shlex.split(command)
    if not parts:
        raise ValueError("--backend-cmd is empty")
    if not any("{input}" in p for p in parts) or not any("{output}" in p for p in parts):
        raise ValueError("--backend-cmd needs {input} (the chunk) and {output} (a folder) in it")
    if not stem_patterns:
        raise ValueError("--backend-cmd needs --backend-stem STEM=REGEX for each stem it writes")
    stems = []
    for item in stem_patterns:
        stem, sep, pattern = item.partition("=")
        stem = stem.strip().lower()
        if not sep or not stem or not pattern:
            raise ValueError(f"--backend-stem must be STEM=REGEX, got '{item}'")
        if not re.fullmatch(r"[a-z0-9_-]+", stem):
            raise ValueError(f"'{stem}' is not a usable stem name (letters, digits, _ -)")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"--backend-stem '{item}' is not a regular expression: {exc}") from exc
        stems.append((stem, pattern))
    if len({s for s, _ in stems}) != len(stems):
        raise ValueError("--backend-stem names a stem twice")
    return Backend("custom", tuple(parts), tuple(stems), "")


def backend_from_options(
    name: Optional[str], model: Optional[str], command: Optional[str], stems: Sequence[str]
) -> Optional[Backend]:
    """The backend the command line asks for (``None`` when it asks for none)."""
    if command is not None:
        if name is not None or model is not None:
            raise ValueError("--backend-cmd replaces --backend and --backend-model: give one")
        return custom_backend(command, stems)
    if stems:
        raise ValueError("--backend-stem goes with --backend-cmd")
    if name is None:
        if model is not None:
            raise ValueError("--backend-model needs --backend")
        return None
    return preset_backend(name, model)


def backend_to_dict(backend: Backend) -> Dict[str, object]:
    return {
        "name": backend.name,
        "command": list(backend.command),
        "stems": [list(item) for item in backend.stems],
        "install_hint": backend.install_hint,
    }


def backend_from_dict(data: Dict[str, object]) -> Backend:
    command = data["command"]
    stems = data["stems"]
    if not isinstance(command, list) or not isinstance(stems, list) or not command:
        raise ValueError("the backend in job.json is not valid")
    return Backend(
        str(data.get("name", "custom")),
        tuple(str(c) for c in command),
        tuple((str(s[0]), str(s[1])) for s in stems),
        str(data.get("install_hint", "")),
    )


def check_available(backend: Backend) -> None:
    """Raise :class:`BackendUnavailable` (with how to install it) if the program is not
    on the PATH."""
    if shutil.which(backend.program) is None:
        hint = f" — {backend.install_hint}" if backend.install_hint else ""
        raise BackendUnavailable(f"'{backend.program}' was not found on this computer{hint}")


def _written_audio(folder: str) -> List[str]:
    found = []
    for root, _, names in os.walk(folder):
        for name in names:
            if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                found.append(os.path.join(root, name))
    return sorted(found)


def classify(backend: Backend, written: Sequence[str]) -> Dict[str, str]:
    """Which of the files ``written`` is which stem. :class:`BackendError` lists what was
    written when a stem is not found exactly once."""
    chosen: Dict[str, str] = {}
    for stem, pattern in backend.stems:
        hits = [p for p in written if re.search(pattern, os.path.basename(p), re.IGNORECASE)]
        if len(hits) != 1:
            seen = ", ".join(os.path.basename(p) for p in written) or "no audio file"
            raise BackendError(
                f"{backend.name} wrote {seen}: expected exactly one file for '{stem}' "
                f"(pattern {pattern!r}), found {len(hits)}"
            )
        chosen[stem] = hits[0]
    return chosen


def separate_chunk(
    backend: Backend, chunk_path: str, processed_dir: str, timeout_s: Optional[float] = None
) -> List[str]:
    """Run ``backend`` on one chunk and put its stems in ``processed_dir`` as
    ``<chunk>_<stem>.<ext>``; returns the files written. Nothing is left in ``processed_dir``
    unless every stem was found. Raises :class:`BackendUnavailable` when the program cannot
    be started and :class:`BackendError` when it fails or writes something unexpected."""
    os.makedirs(processed_dir, exist_ok=True)
    stem_of_chunk = os.path.splitext(os.path.basename(chunk_path))[0]
    work = tempfile.mkdtemp(prefix=".work-", dir=processed_dir)
    try:
        command = [
            part.replace("{input}", os.path.abspath(chunk_path)).replace("{output}", work)
            for part in backend.command
        ]
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except FileNotFoundError as exc:
            raise BackendUnavailable(f"'{backend.program}' could not be started: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise BackendError(
                f"{backend.name} took more than {timeout_s:g}s on this chunk"
            ) from exc
        except OSError as exc:
            raise BackendUnavailable(f"'{backend.program}' could not be started: {exc}") from exc
        if done.returncode != 0:
            tail = "\n".join((done.stderr or done.stdout or "").strip().splitlines()[-6:])
            raise BackendError(
                f"{backend.name} exited with status {done.returncode}"
                + (f":\n{tail}" if tail else "")
            )
        chosen = classify(backend, _written_audio(work))
        written = []
        for stem, source in chosen.items():
            ext = os.path.splitext(source)[1].lower()
            target = os.path.join(processed_dir, f"{stem_of_chunk}_{stem}{ext}")
            os.replace(source, target)
            written.append(target)
        return written
    finally:
        shutil.rmtree(work, ignore_errors=True)
