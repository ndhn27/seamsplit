"""The ``run`` and ``status`` commands: a job taken from the audio file to the merged stems
(see :mod:`seamsplit.job`), and a look at how far it has got (:mod:`seamsplit.status`).

Exit codes: 0 when the job is done (or complete, for ``status``), 3 when it is waiting for
something (results that have not come back, chunks a backend failed on), 1 for an error."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

from .backends import backend_from_options
from .job import (
    CHUNKS_DIR,
    JOB_FILE,
    PROCESSED_DIR,
    RunResult,
    job_stems,
    load_job,
    run_job,
)
from .merger import DEFAULT_METHOD, METHODS
from .pairing import PAIRING_MODES
from .splitopts import add_split_options, apply_service, build_config
from .splitter import SplitConfig
from .status import MANIFEST_FILE, compute_status, format_status

EXIT_WAITING = 3
# What a local backend is given when nothing says how long a chunk may be: no web service
# limit applies, and fewer chunks mean fewer seams. A convenience, not a measured optimum.
LOCAL_TARGET_S = 120.0


def add_parsers(subparsers: Any) -> None:
    run_p = subparsers.add_parser(
        "run",
        help="Split a song, separate it (here or by hand) and merge it: one job folder, resumable.",
        description=(
            "Take a song from the audio file to merged stems in one folder (--job): split it, "
            "get the chunks separated, merge. With --backend the chunks are separated on this "
            "computer; without one, send the chunks to your service, put what it returns in "
            "JOB/processed and run the command again. Run it as often as you like: it only does "
            "what is left. Exit status 3 means it is waiting. Nothing is uploaded."
        ),
    )
    run_p.add_argument("input_file", nargs="?", help="The audio file (needed to start a job)")
    run_p.add_argument(
        "--job",
        metavar="DIR",
        help="The job folder (default: INPUT_FILE with '.job' added, next to it).",
    )
    add_split_options(run_p)
    run_p.add_argument(
        "--backend",
        metavar="NAME",
        help="Separate on this computer with a program that is installed: demucs, "
        "audio-separator (uvr is the same). Without --service, --limit or --target the chunks "
        f"are {LOCAL_TARGET_S:g} s long: no web service limit applies.",
    )
    run_p.add_argument("--backend-model", metavar="MODEL", help="The model the backend is to use.")
    run_p.add_argument(
        "--backend-cmd",
        metavar="COMMAND",
        help="A command line of your own for --backend: {input} is the chunk, {output} a "
        "folder for what it writes. Needs --backend-stem.",
    )
    run_p.add_argument(
        "--backend-stem",
        action="append",
        default=[],
        metavar="STEM=REGEX",
        help="With --backend-cmd: which output file is which stem (repeat for each; one or two).",
    )
    run_p.add_argument(
        "--backend-timeout",
        type=float,
        metavar="SECONDS",
        help="Give up on a chunk the backend takes longer than this on.",
    )
    run_p.add_argument(
        "--match", metavar="WORD", help="Without --backend: the word in the names of the main stem."
    )
    run_p.add_argument(
        "--other-match", metavar="WORD", help="Without --backend: the word of the second stem."
    )
    run_p.add_argument(
        "--out-dir", metavar="DIR", help="Where the merged files go (default: JOB/output)."
    )
    run_p.add_argument(
        "--out-format", default="wav", help="Format of the merged files (default: wav)."
    )
    run_p.add_argument("--method", choices=list(METHODS), default=DEFAULT_METHOD)
    run_p.add_argument(
        "--samplerate",
        type=int,
        default=None,
        help="Output sample rate in Hz (default: the chunks' own, so nothing is resampled)",
    )
    run_p.add_argument(
        "--anchor", action="store_true", help="Anchor the merge to the original chunks (see merge)."
    )
    run_p.add_argument(
        "--pairing", choices=PAIRING_MODES, default="auto", help="See 'seamsplit merge --help'."
    )
    run_p.add_argument(
        "--report", metavar="FILE", help="Write the seam report to FILE (JSON if .json)."
    )

    status_p = subparsers.add_parser(
        "status",
        help="Which chunks of a job (or of a split) have a result, and which are missing.",
        description=(
            "List the chunks that have come back and those that are missing. Files whose names "
            "were lost are compared with the original chunks to tell which chunk they are. Exit "
            f"status {EXIT_WAITING} if something is missing."
        ),
    )
    status_p.add_argument(
        "folder", help="A job folder, or the folder 'split' wrote the chunks and manifest to."
    )
    status_p.add_argument(
        "--processed", metavar="DIR", help="Where the results are (default: JOB/processed)."
    )
    status_p.add_argument("--match", metavar="WORD", help="The word in the names of the main stem.")
    status_p.add_argument(
        "--other-match", metavar="WORD", help="The word in the names of the second stem."
    )
    status_p.add_argument("--json", action="store_true", help="Print the status as JSON.")


def _default_job(args: argparse.Namespace) -> str:
    if args.job:
        return str(args.job)
    if not args.input_file:
        raise ValueError("name the job folder with --job, or the audio file to start one from")
    return os.path.splitext(os.path.abspath(args.input_file))[0] + ".job"


def run_command(args: argparse.Namespace) -> int:
    folder = _default_job(args)
    backend = backend_from_options(
        args.backend, args.backend_model, args.backend_cmd, args.backend_stem
    )
    if backend is None and (args.backend_timeout is not None):
        raise ValueError("--backend-timeout goes with --backend")

    def make_config(chunks: str) -> Tuple[SplitConfig, Dict[str, Any]]:
        if backend is not None and args.format is None:
            args.format = "wav"  # nothing is uploaded, so nothing needs to be small
        if backend is not None and not (args.service or args.limit or args.target):
            args.target = LOCAL_TARGET_S
        apply_service(args)
        return build_config(args, chunks), {
            "service": args.service,
            "limit_s": args.limit,
        }

    merge_options: Dict[str, Any] = {
        "method": args.method,
        "samplerate": args.samplerate,
        "bitrate": args.bitrate,
        "anchor": args.anchor,
        "pairing": args.pairing,
    }
    result = run_job(
        folder,
        args.input_file,
        make_config,
        backend,
        match=args.match,
        other_match=args.other_match,
        out_dir=args.out_dir,
        out_format=args.out_format,
        merge_options=merge_options,
        timeout_s=args.backend_timeout,
        say=print,
    )
    return _after_run(folder, result, args)


def _after_run(folder: str, result: RunResult, args: argparse.Namespace) -> int:
    from .cli import _print_anchor_summary, _print_seam_summary, _write_report

    if result.failures:
        print(f"\n{len(result.failures)} chunk(s) failed:", file=sys.stderr)
        for name, why in result.failures.items():
            print(f"  {name}: {why.splitlines()[0]}", file=sys.stderr)
        print("Run the command again to retry them.", file=sys.stderr)
    if not result.merged:
        pending = result.status.pending_parts()
        processed = os.path.join(folder, PROCESSED_DIR)
        print(
            f"\nNot complete: {len(pending)} of {result.status.parts} chunk(s) still lack a result."
        )
        job = load_job(folder)
        if job is not None and job.backend is None and not result.failures:
            print(
                f"Send the files in {os.path.join(folder, CHUNKS_DIR)} to your service (up to "
                f"its limit each), put what it returns in {processed}, and run this command "
                f"again. 'seamsplit status {folder}' lists what is missing."
            )
        return EXIT_WAITING
    report = result.report
    assert report is not None
    _print_anchor_summary(report, len(result.merged) > 1)
    _print_seam_summary(report, searched=False)
    for note in report.notes:
        print(f"Note: {note}")
    for message in report.warnings:
        print(f"Warning: {message}", file=sys.stderr)
    if args.report:
        _write_report(report, args.report)
        print(f"Seam report: {args.report}")
    print("\nDone. Merged: " + ", ".join(result.merged))
    return 0


def status_command(args: argparse.Namespace) -> int:
    folder = args.folder
    job = load_job(folder)
    if job is not None:
        chunks = os.path.join(folder, CHUNKS_DIR)
        if not os.path.isfile(os.path.join(chunks, MANIFEST_FILE)):
            raise ValueError(
                f"The split of the job in '{folder}' was interrupted (there is no "
                f"{MANIFEST_FILE} in {chunks}): run the same `run --job {folder}` again to "
                "redo it"
            )
        processed = args.processed or os.path.join(folder, PROCESSED_DIR)
        stems = job_stems(job, args.match, args.other_match)
    elif os.path.isfile(os.path.join(folder, MANIFEST_FILE)):
        chunks = folder
        if not args.processed:
            raise ValueError("give the folder of the results with --processed")
        processed = args.processed
        words = [w for w in (args.match, args.other_match) if w]
        stems = list(words) or [None]
    else:
        raise ValueError(
            f"'{folder}' is neither a job folder (no {JOB_FILE}) nor a split folder (no "
            f"{MANIFEST_FILE})"
        )
    status = compute_status(chunks, processed, stems)
    if args.json:
        print(json.dumps(status.to_dict(), indent=2))
    else:
        print(format_status(status))
    return 0 if status.complete else EXIT_WAITING
