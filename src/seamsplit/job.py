"""A job: one song taken from the audio file to the merged stems, in a folder that can be
left and picked up again.

::

    JOB/job.json      what the job is (the file, how it was split, the backend)
    JOB/chunks/       what ``split`` writes: the chunks and manifest.json
    JOB/processed/    what came back for them: from a backend, or put there by hand
    JOB/output/       the merged files

:func:`run_job` does what is still to be done and stops where it has to wait. It splits
once; it has a backend separate every chunk that has no result yet (so a run that was
stopped, or that failed on some chunks, carries on with those); with no backend it waits
for the results of the web service to be put in ``processed`` by hand, and tells which
chunks are still missing; and when every chunk has what it should, it merges. Nothing in it
sends anything anywhere.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import __version__
from .backends import (
    INSTRUMENTAL,
    Backend,
    BackendError,
    backend_from_dict,
    backend_to_dict,
    check_available,
    separate_chunk,
)
from .merger import merge_audio
from .report import MergeReport
from .splitter import SplitConfig, split_audio
from .status import MANIFEST_FILE, JobStatus, compute_status

JOB_FILE = "job.json"
CHUNKS_DIR = "chunks"
PROCESSED_DIR = "processed"
OUTPUT_DIR = "output"
JOB_FORMAT = 1


def chunks_dir(folder: str) -> str:
    return os.path.join(folder, CHUNKS_DIR)


def processed_dir(folder: str) -> str:
    return os.path.join(folder, PROCESSED_DIR)


@dataclass
class Job:
    """What ``job.json`` holds. ``split`` is what was decided when the file was split (the
    service, the limit, the chunk length ...), for the record; ``backend`` is set when the
    job separates on this computer."""

    input_file: str
    input_size: Optional[int]
    split: Dict[str, Any] = field(default_factory=dict)
    backend: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seamsplit_job": JOB_FORMAT,
            "seamsplit_version": __version__,
            "input_file": self.input_file,
            "input_size": self.input_size,
            "split": self.split,
            "backend": self.backend,
        }


def load_job(folder: str) -> Optional[Job]:
    """The job in ``folder``, or ``None`` if there is no ``job.json`` there. Raises
    :class:`ValueError` for one that cannot be read."""
    path = os.path.join(folder, JOB_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"'{path}' is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("seamsplit_job") != JOB_FORMAT:
        raise ValueError(f"'{path}' is not a seamsplit job file (version {JOB_FORMAT})")
    if not isinstance(data.get("input_file"), str):
        raise ValueError(f"'{path}': 'input_file' must be a file name")
    size = data.get("input_size")
    backend = data.get("backend")
    if backend is not None and not isinstance(backend, dict):
        raise ValueError(f"'{path}': 'backend' must be an object")
    split: Dict[str, Any] = data["split"] if isinstance(data.get("split"), dict) else {}
    return Job(data["input_file"], size if isinstance(size, int) else None, split, backend)


def save_job(folder: str, job: Job) -> None:
    """Write ``job.json`` all at once, so an interrupted write cannot leave half of one."""
    path = os.path.join(folder, JOB_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def job_stems(job: Job, match: Optional[str], other_match: Optional[str]) -> List[Optional[str]]:
    """The words that tell the files wanted of each chunk apart: the backend's stems, or
    ``match`` and ``other_match`` (``[None]``: one file per chunk)."""
    if job.backend is not None:
        return list(backend_from_dict(job.backend).stem_names)
    words = [w for w in (match, other_match) if w]
    return list(words) or [None]


def main_and_other(stems: List[Optional[str]]) -> "tuple[Optional[str], Optional[str]]":
    """Of the stems wanted, the one merged as the main output and the other (the main
    one is ``instrumental`` when there is one)."""
    words = [s for s in stems if s]
    if len(words) > 2:
        raise ValueError(
            f"{len(words)} stems are wanted ({', '.join(words)}); merge handles one or two "
            "at a time"
        )
    if not words:
        return None, None
    if len(words) == 1:
        return words[0], None
    main = INSTRUMENTAL if INSTRUMENTAL in words else words[0]
    return main, next(w for w in words if w != main)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "song"


@dataclass
class RunResult:
    status: JobStatus
    merged: List[str] = field(default_factory=list)
    failures: Dict[str, str] = field(default_factory=dict)
    split_now: bool = False
    separated: int = 0
    report: Optional[MergeReport] = None

    @property
    def done(self) -> bool:
        return bool(self.merged)


def run_job(
    folder: str,
    input_file: Optional[str],
    make_config: Callable[[str], Tuple[SplitConfig, Dict[str, Any]]],
    backend: Optional[Backend] = None,
    match: Optional[str] = None,
    other_match: Optional[str] = None,
    out_dir: Optional[str] = None,
    out_format: str = "wav",
    merge_options: Optional[Dict[str, Any]] = None,
    timeout_s: Optional[float] = None,
    say: Callable[[str], None] = lambda text: None,
) -> RunResult:
    """Do what is left of the job in ``folder`` (see the module docstring).

    ``input_file`` is needed for a new job and may be left out afterwards (it must be the
    same one if given); ``make_config(chunks_dir)`` makes the :class:`SplitConfig` when the
    file has to be split, with what is to be recorded about it (the service ...). ``backend``
    separates the chunks here; for a job that has one, leaving it out carries on with it
    and naming another is an error. ``merge_options`` go to :func:`merge_audio`. ``say``
    receives progress lines."""
    job = load_job(folder)
    chunks = chunks_dir(folder)
    processed = processed_dir(folder)
    result_split = False

    if job is not None:
        if input_file is not None and os.path.abspath(input_file) != job.input_file:
            raise ValueError(
                f"'{folder}' is the job of '{job.input_file}', not of '{input_file}': "
                "use another --job folder"
            )
        if backend is not None:
            if job.backend is None:
                raise ValueError(
                    f"'{folder}' was started without a backend; use another --job folder to "
                    "separate this song here"
                )
            if backend_to_dict(backend) != job.backend:
                raise ValueError(
                    f"'{folder}' was started with another backend or model; use another "
                    "--job folder to change it"
                )
        elif job.backend is not None:
            backend = backend_from_dict(job.backend)
    else:
        if input_file is None:
            raise ValueError(f"'{folder}' has no job in it: name the audio file to start one")
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"No such file: '{input_file}'")
        if os.path.isdir(chunks) and os.listdir(chunks):
            raise ValueError(
                f"'{chunks}' already holds files but there is no {JOB_FILE}: use a new --job "
                "folder (or remove that one)"
            )
        os.makedirs(folder, exist_ok=True)
        job = Job(
            os.path.abspath(input_file),
            os.path.getsize(input_file),
            {},
            backend_to_dict(backend) if backend is not None else None,
        )

    if backend is not None:
        check_available(backend)

    if not os.path.isfile(os.path.join(chunks, MANIFEST_FILE)):
        if not os.path.isfile(job.input_file):
            raise FileNotFoundError(f"The job's audio file is gone: '{job.input_file}'")
        say(f"Splitting {job.input_file} ...")
        config, info = make_config(chunks)
        job.split = {
            **info,
            "target_s": config.target_s,
            "overlap_s": config.overlap_s,
            "format": config.export_format,
        }
        manifest = split_audio(job.input_file, config)
        say(f"  {len(manifest['parts'])} chunk(s) written to {chunks}")
        result_split = True
        save_job(folder, job)
    os.makedirs(processed, exist_ok=True)

    stems = job_stems(job, match, other_match)
    main_word, other_word = main_and_other(stems)
    result = RunResult(compute_status(chunks, processed, stems, check_lengths=False))
    result.split_now = result_split

    if backend is not None:
        pending = result.status.pending_parts()
        names = {s.part: s.name for s in result.status.slots}
        for n, part in enumerate(pending, 1):
            name = names[part]
            say(f"[{n}/{len(pending)}] {backend.name} on {name} ...")
            try:
                separate_chunk(backend, os.path.join(chunks, name), processed, timeout_s)
                result.separated += 1
            except BackendError as exc:
                result.failures[name] = str(exc)
                say(f"  failed: {str(exc).splitlines()[0]}")
        result.status = compute_status(chunks, processed, stems, check_lengths=False)

    if not result.status.complete:
        return result

    base = _slug(os.path.splitext(os.path.basename(job.input_file))[0])
    target = out_dir or os.path.join(folder, OUTPUT_DIR)
    os.makedirs(target, exist_ok=True)

    def path_for(word: Optional[str]) -> str:
        return os.path.join(target, f"{base}_{word or 'merged'}.{out_format}")

    result.report = MergeReport()
    out_main = path_for(main_word)
    options = dict(merge_options or {})
    say(
        f"All chunks are here: merging into {out_main}"
        + (f" and {path_for(other_word)}" if other_word else "")
    )
    merge_audio(
        processed,
        os.path.join(chunks, MANIFEST_FILE),
        out_main,
        match=main_word,
        other_match=other_word,
        other_output=path_for(other_word) if other_word else None,
        report=result.report,
        **options,
    )
    result.merged = [out_main] + ([path_for(other_word)] if other_word else [])
    return result
