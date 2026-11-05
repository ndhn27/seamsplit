"""
A drag-and-drop interface for ``split`` and ``merge`` that runs on your own
computer (``seamsplit gui``).

It is a small web server from the standard library (no extra dependency)
that serves one page and does the work with the same functions the command
line uses. Nothing leaves the machine: the server listens on ``127.0.0.1``
only, and every request has to carry a random token that is part of the URL
printed at start-up, so another website open in your browser cannot use it to
read or write your files (it can't guess the token, and a request whose
``Host`` or ``Origin`` is not this server is refused).

Files live in a work folder (``~/SeamSplit`` by default; use a folder that is
only for this) with one sub-folder per step::

    input/      the track you dropped on "Split"
    chunks/     the chunks and ``manifest.json`` written by split
    manifest/   a ``manifest.json`` you dropped on "Merge", if you have one
    processed/  the chunks as returned by the AI service
    merged/     the result of merge (both stems, when two were asked for), and its seam
                report (.md and .json)
    audition/   the listening kit of the merge: clips of the seams worth a listen, joined
                in each way, and the answers recorded for them (see :mod:`seamsplit.audition`)

Audio is read and written by PyAV, which carries its own copy of FFmpeg (see
:mod:`seamsplit.audio`): there is nothing else to install. A running split or merge
can be cancelled; it stops at the next chunk.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlsplit

from . import __version__
from .anchor import other_output_path
from .audio import CouldntDecodeError, CouldntEncodeError, engine_info
from .audition import KIT_FILE, AuditionOptions, parse_seams, player_script
from .choices import CHOICES_FILE, load_choices, new_choices, record_choice, save_choices
from .choices import summarize_choices
from .matching import find_original_chunks
from .merger import (
    DEFAULT_METHOD,
    DEFAULT_SEAM_FADE_MS,
    METHODS,
    MergeReport,
    format_time,
    load_manifest,
    merge_audio,
)
from .services import BUILTIN_PRESETS, available_presets
from .splitter import SplitConfig, default_margin_s, normalize_export_format, split_audio

AREAS = ("input", "chunks", "manifest", "processed", "merged", "audition")
MEDIA_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}
UPLOAD_AREAS = ("input", "manifest", "processed")
GUI_FORMATS = ("mp3", "flac", "wav")
BITRATES = ("128k", "192k", "256k", "320k")
MAX_UPLOAD_BYTES = 8 * 1024**3
DEFAULT_WORKDIR = Path.home() / "SeamSplit"

_BAD_NAME_CHARS = re.compile(r'[\x00-\x1f<>:"|?*]')
# Windows opens these as devices, whatever comes after the first dot: ``NUL.wav`` is ``NUL``
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{device}{digit}" for device in ("COM", "LPT") for digit in "0123456789\u00b9\u00b2\u00b3"}
)


def safe_name(name: str) -> str:
    """A file name that cannot leave its folder: no directories, no characters
    Windows refuses, no name Windows takes for a device (``CON``, ``NUL``, ``COM1``...,
    which get a ``_`` in front), no dot or space at the end, never empty or ``.`` / ``..``."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = _BAD_NAME_CHARS.sub("_", base).strip()[:200].strip(" .")
    if not base:
        raise ValueError("The file has no usable name")
    if base.split(".", 1)[0].rstrip(" ").upper() in _DEVICE_NAMES:
        base = "_" + base
    return base


class Workspace:
    """The work folder: one sub-folder per step, only ever holding plain files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        for area in AREAS:
            (root / area).mkdir(parents=True, exist_ok=True)

    def path(self, area: str, name: Optional[str] = None) -> Path:
        folder = self.root / area
        return folder if name is None else folder / safe_name(name)

    def clear(self, area: str, keep: Optional[Path] = None) -> None:
        """Delete every file in ``area`` except ``keep`` (compared by identity, so a
        case-insensitive file system can't make us delete the file we meant to keep)."""
        for item in self.path(area).iterdir():
            if item.is_file() and not (keep is not None and item.samefile(keep)):
                item.unlink()

    def files(self, area: str) -> List[Dict[str, Any]]:
        folder = self.path(area)
        found = [p for p in folder.iterdir() if p.is_file() and not p.name.endswith(".part")]
        return [{"name": p.name, "size": p.stat().st_size} for p in sorted(found)]


class JobCancelled(Exception):
    """Raised inside a job, at a point where it can stop cleanly, once the user
    has asked to cancel it."""


class Job:
    """One split or merge running in a thread; the page polls it.

    ``state`` is ``running``, then ``done``, ``error`` or ``cancelled``. A cancel
    is a request: the job notices it between chunks (see :meth:`check_cancel`),
    so the step it is in (decoding the whole track, encoding the result) finishes
    first."""

    def __init__(self, kind: str) -> None:
        self.id = uuid.uuid4().hex
        self.kind = kind
        self.state = "running"
        self.progress: Optional[Dict[str, Any]] = None
        self.result: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelling(self) -> bool:
        return self.state == "running" and self._cancel.is_set()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "cancelling": self.cancelling,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }


def _number(params: Dict[str, Any], key: str, default: float) -> float:
    value = params.get(key)
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"'{key}' must be a number")
    return number


def build_split_config(
    params: Dict[str, Any], output_dir: str
) -> Tuple[SplitConfig, Optional[float]]:
    """Turn the form's values into a :class:`SplitConfig` (and the service
    limit, if one was given). Mirrors the CLI: ``limit`` derives the target
    as ``limit - after-slack - overlap - margin``."""
    fmt = normalize_export_format(str(params.get("format") or "mp3"))
    if fmt not in GUI_FORMATS:
        raise ValueError(f"format must be one of: {', '.join(GUI_FORMATS)}")
    after = _number(params, "after_slack", 3.0)
    overlap = _number(params, "overlap", 4.0)
    before = _number(params, "before_slack", 20.0)

    limit: Optional[float] = None
    if params.get("limit") not in (None, ""):
        limit = _number(params, "limit", 0.0)
        target = limit - after - overlap - default_margin_s(fmt)
        if target <= 0:
            raise ValueError("The limit is too short for this overlap and after-slack")
    else:
        target = _number(params, "target", 52.0)
    if 0 < target <= before:
        before = target / 2  # a short limit must not make the default slack invalid

    config = SplitConfig(
        target_s=target,
        before_slack_s=before,
        after_slack_s=after,
        overlap_s=overlap,
        output_dir=output_dir,
        export_format=fmt,
    )
    return config, limit


REPORT_SUFFIXES = ("_report.md", "_report.json")


def service_presets() -> Dict[str, Any]:
    """The service presets the page offers (``services``): the built-in ones and the
    user's. A user's file that cannot be read is reported (``services_error``) and left out,
    so one typo in it does not take the page down."""
    error = None
    try:
        presets = available_presets()
    except (OSError, ValueError) as exc:
        presets, error = dict(BUILTIN_PRESETS), str(exc)
    return {
        "services": [
            {
                "name": p.name,
                "limit_s": p.limit_s,
                "format": p.format,
                "overlap_s": p.overlap_s,
                "note": p.note,
            }
            for _, p in sorted(presets.items())
        ],
        "services_error": error,
    }


def write_report_files(report: MergeReport, output: Path) -> List[str]:
    """Write the per-seam report next to the merged file, as Markdown to read and
    JSON for tools (what ``seamsplit merge --report`` writes). Returns the names."""
    md = output.with_name(f"{output.stem}{REPORT_SUFFIXES[0]}")
    js = output.with_name(f"{output.stem}{REPORT_SUFFIXES[1]}")
    md.write_text(report.to_markdown(), encoding="utf-8")
    js.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return [md.name, js.name]


def summarize_merge(report: MergeReport) -> Dict[str, Any]:
    """The few numbers worth showing after a merge."""
    rated = [b for b in report.boundaries if b.seam_match_db is not None]
    worst = min((b.seam_match_db for b in rated if b.seam_match_db is not None), default=None)
    flagged = [
        format_time(b.seam_ms) if b.seam_ms is not None else f"seam {b.boundary}"
        for b in report.boundaries
        if b.flagged
    ]
    aligned = [b for b in report.boundaries if b.aligned]
    return {
        "seams": len(report.boundaries),
        "worst_match_db": None if worst is None else round(worst, 1),
        "flagged": flagged,
        "largest_offset_ms": (
            round(max(aligned, key=lambda b: abs(b.lag_ms)).lag_ms, 1) if aligned else None
        ),
        "warnings": list(report.warnings),
    }


def summarize_anchor(report: MergeReport) -> Optional[Dict[str, Any]]:
    """What comparing the chunks with their originals found, in a few numbers
    (``None`` when the merge was not anchored)."""
    if not report.chunks:
        return None
    timed = [c for c in report.chunks if c.lag_ms is not None]
    leveled = [c for c in report.chunks if c.gain_db is not None]
    rated = [c for c in report.chunks if c.recon_db is not None]
    worst = min(rated, key=lambda c: c.recon_db if c.recon_db is not None else 0.0, default=None)
    return {
        "chunks": len(report.chunks),
        "in_time": len(timed),
        "in_level": len(leveled),
        "largest_offset_ms": (
            round(max(timed, key=lambda c: abs(c.lag_ms or 0.0)).lag_ms or 0.0, 1)
            if timed
            else None
        ),
        "worst_recon_db": None if worst is None else round(worst.recon_db or 0.0, 1),
        "worst_recon_chunk": None if worst is None else worst.part,
        "flagged": [c.part for c in report.chunks if c.flagged],
    }


def open_folder(path: Path) -> bool:
    """Show a folder in the system's file manager."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        return False
    return True


def _explain(exc: Exception) -> str:
    from .cli import _explain as explain  # cli imports this module lazily, so no cycle

    return explain(exc)


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, workdir: Path, port: int = 0) -> None:
        self.workspace = Workspace(workdir)
        self.token = secrets.token_urlsafe(24)
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()
        super().__init__(("127.0.0.1", port), _Handler)

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?t={self.token}"

    # -- jobs ----------------------------------------------------------------

    def start_job(self, kind: str, work: Callable[[Job], None]) -> Job:
        with self.lock:
            if any(j.state == "running" for j in self.jobs.values()):
                raise RuntimeError("Another job is still running")
            job = Job(kind)
            self.jobs[job.id] = job

        def run() -> None:
            try:
                work(job)
                job.state = "done"
            except JobCancelled:
                job.state = "cancelled"
            except (OSError, ValueError, CouldntDecodeError, CouldntEncodeError) as exc:
                job.error, job.state = _explain(exc), "error"
            except Exception as exc:  # never leave the page waiting on a dead thread
                job.error, job.state = f"{type(exc).__name__}: {exc}", "error"

        threading.Thread(target=run, daemon=True).start()
        return job

    def split_work(self, params: Dict[str, Any]) -> Callable[[Job], None]:
        ws = self.workspace
        inputs = ws.files("input")
        if not inputs:
            raise ValueError("Drop an audio file first")
        source = ws.path("input", inputs[0]["name"])
        config, limit = build_split_config(params, str(ws.path("chunks")))

        def work(job: Job) -> None:
            for area in ("chunks", "manifest", "processed", "merged", "audition"):
                ws.clear(area)  # a new split starts a new job

            def progress(i: int, n: int, name: str) -> None:
                job.progress = {"i": i, "n": n, "name": name}
                job.check_cancel()

            try:
                manifest = split_audio(str(source), config, progress=progress)
            except JobCancelled:
                ws.clear("chunks")  # no manifest was written: do not leave half a split behind
                raise
            warnings: List[str] = []
            if limit is not None:
                from .cli import encoded_over_limit

                paths = [str(ws.path("chunks", n)) for n in manifest["parts"]]
                _, over = encoded_over_limit(paths, limit)
                if over:
                    listed = ", ".join(f"{n} ({s:.2f}s)" for n, s in over)
                    warnings.append(f"Longer than the {limit:g}s limit as exported: {listed}")
            job.result = {"warnings": warnings}

        return work

    def merge_work(self, params: Dict[str, Any]) -> Callable[[Job], None]:
        ws = self.workspace
        uploaded = ws.path("manifest", "manifest.json")
        manifest_path = uploaded if uploaded.is_file() else ws.path("chunks", "manifest.json")
        if not manifest_path.is_file():
            raise ValueError("No manifest.json: split a track first, or drop its manifest.json")
        if not ws.files("processed"):
            raise ValueError("Drop the processed chunks first")
        method = str(params.get("method") or DEFAULT_METHOD)
        if method not in METHODS:
            raise ValueError(f"method must be one of: {', '.join(METHODS)}")
        fmt = normalize_export_format(str(params.get("format") or "mp3"))
        if fmt not in GUI_FORMATS:
            raise ValueError(f"format must be one of: {', '.join(GUI_FORMATS)}")
        samplerate = int(_number(params, "samplerate", 44100))
        if samplerate <= 0:
            raise ValueError("samplerate must be greater than 0")
        match = str(params.get("match") or "").strip() or None
        other_match = str(params.get("other_match") or "").strip() or None
        anchor = params.get("anchor") in (True, 1, "true", "1")
        seam_search = params.get("seam_search") in (True, 1, "true", "1")
        seam_fade_ms = _number(params, "seam_fade_ms", DEFAULT_SEAM_FADE_MS)
        if not seam_fade_ms > 0:
            raise ValueError("seam_fade_ms must be greater than 0")
        bitrate = str(params.get("bitrate") or "320k")
        if bitrate not in BITRATES:
            raise ValueError(f"bitrate must be one of: {', '.join(BITRATES)}")
        manifest = load_manifest(str(manifest_path))
        if anchor:  # said now, not after the first chunk has been decoded
            find_original_chunks(str(ws.path("chunks")), manifest["parts"])
        which = str(params.get("audition") or "off")
        if which not in ("off", "flagged", "all"):
            raise ValueError("audition must be one of: off, flagged, all")
        source_name = str(manifest.get("source_file") or "merged")
        out_name = f"{Path(safe_name(source_name)).stem}_merged.{fmt}"
        other_name = other_output_path(out_name, other_match) if other_match else None

        def work(job: Job) -> None:
            for area in ("merged", "audition"):
                ws.clear(area)
            report = MergeReport()
            out_path = ws.path("merged", out_name)
            audition = None
            if which != "off":

                def audition_progress(i: int, n: int, name: str) -> None:
                    job.progress = {"i": i, "n": n, "name": name, "phase": "audition"}
                    job.check_cancel()

                audition = AuditionOptions(
                    out_dir=str(ws.path("audition")),
                    seams=parse_seams(which),
                    progress=audition_progress,
                )

            def progress(i: int, n: int, name: str) -> None:
                job.progress = {"i": i, "n": n, "name": name}
                job.check_cancel()

            try:
                merged = merge_audio(
                    processed_dir=str(ws.path("processed")),
                    manifest_path=str(manifest_path),
                    output_path=str(out_path),
                    match=match,
                    method=method,
                    bitrate=bitrate,
                    samplerate=samplerate,
                    seam_search=seam_search,
                    seam_fade_ms=seam_fade_ms,
                    report=report,
                    progress=progress,
                    anchor=anchor,
                    originals_dir=str(ws.path("chunks")),
                    other_match=other_match,
                    other_output=None if other_name is None else str(ws.path("merged", other_name)),
                    audition=audition,
                )
                job.check_cancel()  # asked for while the result was being encoded
            except JobCancelled:
                for area in ("merged", "audition"):
                    ws.clear(area)
                raise
            kit_path = ws.path("audition", KIT_FILE)
            seams = []
            if kit_path.is_file():
                seams = [
                    s["seam"] for s in json.loads(kit_path.read_text(encoding="utf-8"))["seams"]
                ]
            job.result = {
                "file": out_name,
                "other_file": other_name,
                "length_ms": len(merged),
                "reports": write_report_files(report, out_path),
                "anchor": summarize_anchor(report),
                "audition": seams,
                **summarize_merge(report),
            }

        return work


class _Handler(BaseHTTPRequestHandler):
    server: GuiServer

    def log_message(self, format: str, *args: Any) -> None:  # quiet: the page is the UI
        pass

    # -- plumbing ------------------------------------------------------------

    def _reply(
        self, status: int, body: bytes, content_type: str, extra: Optional[Dict[str, str]] = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._reply(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _fail(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _guard(self) -> Optional[Dict[str, List[str]]]:
        """Refuse anything that is not this page talking to this server."""
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.port}", f"localhost:{self.server.port}"}
        origin = self.headers.get("Origin")
        if host not in allowed or (
            origin is not None and origin not in {f"http://{h}" for h in allowed}
        ):
            self._fail(HTTPStatus.FORBIDDEN, "Forbidden")
            return None
        query = parse_qs(urlsplit(self.path).query)
        supplied = (query.get("t") or [""])[0]
        if not secrets.compare_digest(supplied, self.server.token):
            self._fail(HTTPStatus.FORBIDDEN, "Open the address printed by 'seamsplit gui'")
            return None
        return query

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1_000_000:
            raise ValueError("Request too large")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data

    def _send_file(self, path: Path, download_name: str) -> None:
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Disposition", f"attachment; filename*=UTF-8''{quote(download_name)}"
        )
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1024 * 1024)

    # -- GET -----------------------------------------------------------------

    def do_GET(self) -> None:
        query = self._guard()
        if query is None:
            return
        route = unquote(urlsplit(self.path).path)
        ws = self.server.workspace

        if route == "/":
            page = resources.files("seamsplit").joinpath("static/index.html").read_text("utf-8")
            page = page.replace("/*AUDITION_JS*/", player_script().replace("</", "<\\/"))
            self._reply(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
        elif route == "/api/info":
            self._json(
                {
                    "version": __version__,
                    "audio_engine": engine_info(),
                    "bitrates": list(BITRATES),
                    "seam_fade_ms": DEFAULT_SEAM_FADE_MS,
                    "workdir": str(ws.root),
                    "methods": list(METHODS),
                    "default_method": DEFAULT_METHOD,
                    **service_presets(),
                }
            )
        elif route == "/api/state":
            self._json(self._state())
        elif route == "/api/job":
            job = self.server.jobs.get((query.get("id") or [""])[0])
            if job is None:
                self._fail(HTTPStatus.NOT_FOUND, "Unknown job")
            else:
                self._json(job.to_dict())
        elif route == "/api/audition":
            self._json(self._audition())
        elif route.startswith("/audition/"):
            name = route[len("/audition/") :]
            try:
                path = ws.path("audition", name)
            except ValueError:
                path = ws.root
            if name == path.name and path.suffix.lower() in MEDIA_TYPES and path.is_file():
                self._send_media(path)
            else:
                self._fail(HTTPStatus.NOT_FOUND, "Not found")
        elif route == "/download/chunks.zip":
            self._send_zip("chunks", "chunks.zip")
        elif route == "/download/audition.zip":
            self._send_zip("audition", "listening-kit.zip")
        elif route.startswith("/download/"):
            parts = route.split("/")[2:]
            if len(parts) == 2 and parts[0] in ("chunks", "merged", "audition"):
                try:
                    path = ws.path(parts[0], parts[1])
                except ValueError:
                    path = ws.root  # not a file: falls through to "not found"
                if parts[1] == path.name and path.is_file():
                    self._send_file(path, path.name)
                    return
            self._fail(HTTPStatus.NOT_FOUND, "Not found")
        else:
            self._fail(HTTPStatus.NOT_FOUND, "Not found")

    def _state(self) -> Dict[str, Any]:
        ws = self.server.workspace
        chunk_files = [f for f in ws.files("chunks") if f["name"] != "manifest.json"]
        split: Optional[Dict[str, Any]] = None
        manifest_file = ws.path("chunks", "manifest.json")
        if manifest_file.is_file():
            try:
                split = load_manifest(str(manifest_file))
            except (OSError, ValueError):
                split = None
        uploaded = ws.path("manifest", "manifest.json").is_file()
        merged_files = ws.files("merged")
        return {
            "input": ws.files("input"),
            "chunks": chunk_files,
            "split": split,
            "processed": ws.files("processed"),
            "manifest": "uploaded" if uploaded else ("split" if split else None),
            "merged": [f for f in merged_files if not f["name"].endswith(REPORT_SUFFIXES)],
            "reports": [f["name"] for f in merged_files if f["name"].endswith(REPORT_SUFFIXES)],
            "audition": ws.path("audition", KIT_FILE).is_file(),
            "busy": any(j.state == "running" for j in self.server.jobs.values()),
        }

    def _audition(self) -> Dict[str, Any]:
        """The listening kit with the answers given so far (``kit`` is ``None`` when
        the last merge made none)."""
        ws = self.server.workspace
        kit_path = ws.path("audition", KIT_FILE)
        if not kit_path.is_file():
            return {"kit": None, "choices": [], "summary": None}
        kit = json.loads(kit_path.read_text(encoding="utf-8"))
        answers = ws.path("audition", CHOICES_FILE)
        doc = load_choices(str(answers)) if answers.is_file() else new_choices(kit)
        return {"kit": kit, "choices": doc["choices"], "summary": summarize_choices([doc])}

    def _record_choice(self, body: Dict[str, Any]) -> Dict[str, Any]:
        ws = self.server.workspace
        seam, choice = body.get("seam"), body.get("choice")
        if isinstance(seam, bool) or not isinstance(seam, int) or not isinstance(choice, str):
            raise ValueError("Expected a seam number and a choice")
        with self.server.lock:  # two answers arriving together must not lose one
            kit_path = ws.path("audition", KIT_FILE)
            if not kit_path.is_file():
                raise ValueError("There is no listening kit to answer")
            kit = json.loads(kit_path.read_text(encoding="utf-8"))
            answers = ws.path("audition", CHOICES_FILE)
            doc = load_choices(str(answers)) if answers.is_file() else new_choices(kit)
            record_choice(doc, kit, seam, choice, bool(body.get("blind")))
            save_choices(doc, str(ws.path("audition")))
        return {"summary": summarize_choices([doc])}

    def _send_media(self, path: Path) -> None:
        """A clip for the page's player, which seeks: that needs ``Range`` requests."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        wanted = re.fullmatch(r"bytes=(\d*)-(\d*)", (self.headers.get("Range") or "").strip())
        if wanted and (wanted.group(1) or wanted.group(2)):
            if wanted.group(1):
                start = int(wanted.group(1))
                end = min(int(wanted.group(2)), size - 1) if wanted.group(2) else size - 1
            else:  # the last N bytes
                start = max(0, size - int(wanted.group(2)))
            if start >= size or start > end:
                self._reply(
                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    b"",
                    "text/plain",
                    {"Content-Range": f"bytes */{size}"},
                )
                return
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", MEDIA_TYPES[path.suffix.lower()])
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        remaining = end - start + 1
        try:
            with open(path, "rb") as f:
                f.seek(start)
                while remaining > 0:
                    block = f.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass  # a player that has seen enough hangs up; that is not an error

    def _send_zip(self, area: str, download_name: str) -> None:
        ws = self.server.workspace
        names = [f["name"] for f in ws.files(area)]
        if not names:
            self._fail(HTTPStatus.NOT_FOUND, "Nothing to download yet")
            return
        with tempfile.TemporaryFile() as tmp:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
                for name in names:
                    zf.write(ws.path(area, name), name)
            size = tmp.tell()
            tmp.seek(0)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f"attachment; filename={download_name}")
            self.end_headers()
            shutil.copyfileobj(tmp, self.wfile, 1024 * 1024)

    # -- POST ----------------------------------------------------------------

    def do_POST(self) -> None:
        query = self._guard()
        if query is None:
            return
        route = unquote(urlsplit(self.path).path)
        ws = self.server.workspace
        try:
            if route == "/api/upload":
                self._upload(query)
            elif route == "/api/clear":
                area = (query.get("area") or [""])[0]
                if area not in ("input", "processed", "manifest", "merged"):
                    raise ValueError("Unknown area")
                ws.clear(area)
                self._json({"ok": True})
            elif route == "/api/split":
                job = self.server.start_job("split", self.server.split_work(self._read_json()))
                self._json({"job": job.id})
            elif route == "/api/merge":
                job = self.server.start_job("merge", self.server.merge_work(self._read_json()))
                self._json({"job": job.id})
            elif route == "/api/audition-choice":
                self._json(self._record_choice(self._read_json()))
            elif route == "/api/cancel":
                target = self.server.jobs.get((query.get("id") or [""])[0])
                if target is None:
                    self._fail(HTTPStatus.NOT_FOUND, "Unknown job")
                else:
                    target.cancel()  # a finished job ignores it
                    self._json({"ok": True})
            elif route == "/api/open-folder":
                self._json({"ok": open_folder(ws.root)})
            elif route == "/api/quit":
                self._json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._fail(HTTPStatus.NOT_FOUND, "Not found")
        except RuntimeError as exc:
            self._fail(HTTPStatus.CONFLICT, str(exc))
        except (ValueError, OSError, CouldntDecodeError, CouldntEncodeError) as exc:
            self._fail(HTTPStatus.BAD_REQUEST, _explain(exc))

    def _upload(self, query: Dict[str, List[str]]) -> None:
        ws = self.server.workspace
        area = (query.get("area") or [""])[0]
        if area not in UPLOAD_AREAS:
            raise ValueError("Unknown area")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or not raw_length.isdigit():
            raise ValueError("Missing Content-Length")
        length = int(raw_length)
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("File too large")
        name = "manifest.json" if area == "manifest" else safe_name((query.get("name") or [""])[0])
        dest = ws.path(area, name)
        partial = dest.with_name(dest.name + ".part")
        remaining = length
        try:
            with open(partial, "wb") as f:
                while remaining > 0:
                    block = self.rfile.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("Upload was cut short")
                    f.write(block)
                    remaining -= len(block)
            if area == "manifest":
                load_manifest(str(partial))  # reject anything that is not a seamsplit manifest
            os.replace(partial, dest)
            if area in ("input", "manifest"):
                # These hold a single file. Drop the previous one only now that the new
                # file is complete (and, for a manifest, valid): a failed or cut-short
                # upload must leave what the user already had untouched.
                ws.clear(area, keep=dest)
        finally:
            if partial.exists():
                partial.unlink()
        self._json({"name": name, "size": length})


def create_server(workdir: Optional[Path] = None, port: int = 0) -> GuiServer:
    return GuiServer(Path(workdir) if workdir else DEFAULT_WORKDIR, port)


def run(workdir: Optional[str] = None, port: int = 0, open_browser: bool = True) -> int:
    """Start the GUI and block until interrupted (or the page's Quit button)."""
    server = create_server(Path(workdir) if workdir else None, port)
    # flushed: when the output goes to a pipe or a file (a launcher, a frozen program)
    # the address would otherwise not appear until the server stops
    print(f"SeamSplit {__version__} — open {server.url}", flush=True)
    print(f"Work folder: {server.workspace.root}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
