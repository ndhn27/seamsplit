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
documentation gives. The ``audio-separator`` preset (``uvr`` is the same) has been run
against the real program: the benchmark's experiment 6 runs ``seamsplit run`` with
audio-separator 0.48.0 and the ``UVR-MDX-NET-Voc_FT.onnx`` model, from the audio file to
the merged stems (``benchmarks/results/RESULTS.md``). The ``demucs`` preset has only been
tested against the stand-in the tests use (``tests/fake_programs.py``), not the real
Demucs. A run that cannot find the stems says which files the program wrote, which is what
to adjust (``--backend-cmd`` and ``--backend-stem`` take over completely).
"""

from __future__ import annotations

import hashlib
import json
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


class CustomBackendNotStored(ValueError):
    """A job separates with a command of the user's own, and ``job.json`` does not keep the
    command: it has to be given again (``--backend-cmd``)."""


@dataclass(frozen=True)
class Backend:
    """``command`` is the program and its arguments, with ``{input}`` (the chunk) and
    ``{output}`` (a folder for the results) in them; ``stems`` maps each stem to a regular
    expression (case does not matter) its file name matches."""

    name: str
    command: Tuple[str, ...]
    stems: Tuple[Tuple[str, str], ...]
    install_hint: str = ""
    model: Optional[str] = None  # a preset's model; what a job file records of it

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
# The name of a backend made with --backend-cmd. No preset has it, so a job file that says
# "custom" can never be taken for a preset.
CUSTOM = "custom"
# The option each preset passes its model with: what is read back from a job file made before
# job files stopped recording the command.
_MODEL_OPTION = {"demucs": "-n", "audio-separator": "--model_filename"}
_STEM_NAME = re.compile(r"[a-z0-9_-]+")
# What a model read from a job file may look like: a name, not an option and not a path.
_MODEL_NAME = re.compile(r"[^\s/\\-][^/\\\x00-\x1f\x7f]{0,199}")


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
            model or None,
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
            model or None,
        )
    raise ValueError(
        f"Unknown backend '{name}' (available: {', '.join(PRESETS)}, or --backend-cmd)"
    )


def _split_windows(command: str) -> List[str]:
    """A command line read the way Windows reads one (the rules of the C runtime and of
    ``CommandLineToArgvW``): arguments are separated by spaces and tabs, double quotes group
    (and are not part of the argument), and a backslash is only a backslash unless it comes
    before a double quote, where each pair of them is one, and an odd one out makes the quote
    a plain character. Single quotes mean nothing. ``C:\\Tools\\demucs.exe`` stays as it is."""
    args: List[str] = []
    current: List[str] = []
    quoted = started = False
    i, n = 0, len(command)
    while i < n:
        char = command[i]
        if char == "\\":
            j = i
            while j < n and command[j] == "\\":
                j += 1
            run = j - i
            if j < n and command[j] == '"':
                current.append("\\" * (run // 2))
                if run % 2:  # the quote is escaped: a quote that is part of the argument
                    current.append('"')
                    j += 1
            else:
                current.append("\\" * run)
            started, i = True, j
            continue
        if char == '"':
            if quoted and command[i + 1 : i + 2] == '"':  # "" inside quotes: a quote
                current.append('"')
                i += 1
            else:
                quoted = not quoted
            started = True
        elif char in " \t" and not quoted:
            if started:
                args.append("".join(current))
                current, started = [], False
        else:
            current.append(char)
            started = True
        i += 1
    if quoted:
        raise ValueError("No closing quotation")
    if started:
        args.append("".join(current))
    return args


def split_command(command: str, windows: Optional[bool] = None) -> List[str]:
    """``command`` as the list of arguments it stands for: on Windows as Windows reads a
    command line (a backslash in ``C:\\Tools\\demucs.exe`` is a backslash), elsewhere as a
    POSIX shell does (quotes and backslash escapes). ``windows`` says which, by default
    the system the program runs on."""
    if windows is None:
        windows = os.name == "nt"
    return _split_windows(command) if windows else shlex.split(command)


def custom_backend(command: str, stem_patterns: Sequence[str]) -> Backend:
    """A backend of the user's own: ``command`` is a command line with ``{input}`` and
    ``{output}`` in it, and ``stem_patterns`` are ``STEM=REGEX`` (the regular expression is
    matched against the file names the command writes). The command line is read as the
    system reads one (see :func:`split_command`)."""
    parts = split_command(command)
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
    return Backend(CUSTOM, tuple(parts), tuple(stems), "")


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


def fingerprint(backend: Backend) -> str:
    """A hash of the command and the stem patterns of a backend: what a job file records of
    a backend of the user's own, to tell later that it is given the same one again."""
    blob = json.dumps(
        {"command": list(backend.command), "stems": [list(item) for item in backend.stems]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def backend_to_dict(backend: Backend) -> Dict[str, object]:
    """What ``job.json`` records of a backend. Never a command: a preset is its name and
    model (the command is made from them by this code), a backend of one's own is the
    names of its stems and the hash of its command. A job folder is made to be carried to
    another computer and opened there, so nothing in it may say what is to be run."""
    if backend.name == CUSTOM:
        return {"name": CUSTOM, "stems": backend.stem_names, "fingerprint": fingerprint(backend)}
    return {"name": backend.name, "model": backend.model}


def _check_stem_names(names: object) -> List[str]:
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and _STEM_NAME.fullmatch(n) for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("the stems of the backend in job.json are not valid")
    return list(names)


def backend_stem_names(data: object) -> List[str]:
    """The stems a job's backend writes, from what ``job.json`` records of it. Nothing is
    built that could be run."""
    if not isinstance(data, dict):
        raise ValueError("the backend in job.json is not valid")
    name = data.get("name")
    if name == CUSTOM:
        return _check_stem_names(data.get("stems"))
    if not isinstance(name, str) or name not in _MODEL_OPTION:
        raise ValueError("the backend in job.json is not valid")
    return preset_backend(name).stem_names


def backend_from_dict(data: object) -> Backend:
    """The backend a job file records, for a preset only: its command is made here, from
    the name and the model, whatever else the file says. A backend of one's own is not
    kept (see :func:`backend_to_dict`) and raises :class:`CustomBackendNotStored`."""
    if not isinstance(data, dict):
        raise ValueError("the backend in job.json is not valid")
    name = data.get("name")
    if name == CUSTOM:
        raise CustomBackendNotStored(
            "this job separates with a command of your own, and job.json does not keep the "
            "command (a job folder can come from someone else, and a command found in it "
            "would run without you having written it): give --backend-cmd and --backend-stem "
            "again, the same as when the job was started"
        )
    if not isinstance(name, str) or name not in _MODEL_OPTION:
        raise ValueError("the backend in job.json is not valid")
    model = data.get("model")
    if model is not None and not (isinstance(model, str) and _MODEL_NAME.fullmatch(model)):
        raise ValueError(
            f"the model in job.json ({model!r}) is not a model name: give --backend {name} "
            "--backend-model MODEL again"
        )
    return preset_backend(name, model)


def upgrade_legacy_backend(data: object) -> Dict[str, object]:
    """What a job file made by an earlier version recorded as the backend (the whole command), as
    :func:`backend_to_dict` records it now. A command that is the one this code makes for
    the preset it names becomes that preset; any other (a command of one's own, or one
    that was edited) becomes a backend of one's own, which is never run unless the same
    command is given again."""
    if not isinstance(data, dict):
        raise ValueError("the backend in job.json is not valid")
    command, stems = data.get("command"), data.get("stems")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(c, str) for c in command)
        or not isinstance(stems, list)
        or not all(
            isinstance(i, (list, tuple)) and len(i) == 2 and all(isinstance(x, str) for x in i)
            for i in stems
        )
    ):
        raise ValueError("the backend in job.json is not valid")
    pairs = tuple((str(i[0]), str(i[1])) for i in stems)
    name = str(data.get("name", "")).strip().lower()
    option = _MODEL_OPTION.get(name)
    if option is not None:
        at = command.index(option) if option in command else -1
        model = command[at + 1] if 0 <= at < len(command) - 1 else None
        try:
            made = preset_backend(name, model)
        except ValueError:  # pragma: no cover - a name in _MODEL_OPTION is always a preset
            made = None
        if made is not None and list(made.command) == command and made.stems == pairs:
            return backend_to_dict(made)
    old = Backend(CUSTOM, tuple(command), pairs, "")
    _check_stem_names(old.stem_names)
    return backend_to_dict(old)


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
