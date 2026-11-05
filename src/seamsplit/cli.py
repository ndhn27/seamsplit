"""Command-line interface for SeamSplit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

from . import __version__
from .anchor import DEFAULT_RECON_WARN_DB
from .audio import CouldntDecodeError, CouldntEncodeError
from .audition import (
    AUDITION_FORMATS,
    DEFAULT_MAX_SEAMS,
    DEFAULT_WINDOW_S,
    KIT_FILE,
    KIT_PAGE,
    AuditionOptions,
    parse_seams,
)
from .cli_jobs import add_parsers as add_job_parsers, run_command, status_command
from .choices import CHOICES_FILE, describe_summary, load_choices, summarize_choices
from .cutpoints import MIN_BAR_CONFIDENCE
from .merger import (
    DEFAULT_MAX_GAIN_DB,
    DEFAULT_MAX_LAG_MS,
    DEFAULT_METHOD,
    DEFAULT_SEAM_FADE_MS,
    DEFAULT_SEAM_WARN_DB,
    METHODS,
    BoundaryInfo,
    MergeReport,
    format_time,
    merge_audio,
)
from .pairing import PAIRING_MODES
from .services import available_presets, default_services_file
from .splitopts import (  # noqa: F401  (the defaults are part of this module's interface)
    DEFAULT_FORMAT,
    DEFAULT_OVERLAP_S,
    DEFAULT_TARGET_S,
    add_split_options,
    apply_service,
    build_config,
    resolve_margin,
    resolve_target,
)
from .splitter import (
    BAR_REACH_MS,
    probe_duration_s,
    split_audio,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seamsplit",
        description=(
            "Split audio into overlapping chunks for AI stem separation, then "
            "merge the processed chunks back together with a crossfade."
        ),
    )
    parser.add_argument("--version", action="version", version=f"seamsplit {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_p = subparsers.add_parser(
        "split", help="Split an audio file into overlapping chunks at stable quiet points."
    )
    split_p.add_argument("input_file", help="Input audio file (mp3, wav, ...)")
    add_split_options(split_p)
    split_p.add_argument("--out", default="chunks", help="Output directory (default: chunks)")
    split_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse the file and show the planned chunks without writing anything.",
    )

    merge_p = subparsers.add_parser(
        "merge", help="Merge processed chunks back together with a crossfade."
    )
    merge_p.add_argument("processed_dir", help="Directory containing the AI-processed chunk files")
    merge_p.add_argument(
        "--manifest", required=True, help="manifest.json produced by 'seamsplit split'"
    )
    merge_p.add_argument("--out", required=True, help="Output audio file path")
    merge_p.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Override the overlap length (seconds). Default: read from the manifest.",
    )
    merge_p.add_argument(
        "--match",
        default=None,
        help=(
            "Filter which processed file to use when a part maps to several files "
            "(e.g. 'instrumental', 'no_vocals')"
        ),
    )
    merge_p.add_argument(
        "--method",
        choices=list(METHODS),
        default=DEFAULT_METHOD,
        help=(
            "Crossfade curve: adaptive (shaped by how similar the two overlap regions "
            "are: linear for identical audio, equal-power for unrelated audio, "
            "constant power in between), or the fixed curves linear / equalpower "
            "(equalpower adds up to +3 dB mid-fade on identical audio). "
            f"(default: {DEFAULT_METHOD})"
        ),
    )
    merge_p.add_argument(
        "--no-align",
        action="store_true",
        help="Don't measure and cancel timing offsets between neighboring chunks.",
    )
    merge_p.add_argument(
        "--max-lag-ms",
        type=float,
        default=DEFAULT_MAX_LAG_MS,
        help=(
            "Largest timing offset (ms) the alignment step will look for "
            f"(default: {DEFAULT_MAX_LAG_MS:g})"
        ),
    )
    merge_p.add_argument(
        "--no-match-level",
        action="store_true",
        help=(
            "Don't scale each chunk so its overlap is as loud as the previous chunk's "
            "(a service may normalize every chunk differently)."
        ),
    )
    merge_p.add_argument(
        "--max-gain-db",
        type=float,
        default=DEFAULT_MAX_GAIN_DB,
        help=(
            "Largest level correction (dB, either direction) applied to one chunk "
            f"(default: {DEFAULT_MAX_GAIN_DB:g})"
        ),
    )
    merge_p.add_argument(
        "--seam-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "EXPERIMENTAL, off by default: make a short crossfade where the two chunks "
            "agree best instead of fading across the whole overlap. Not validated on real "
            "separator output; compare both by ear."
        ),
    )
    merge_p.add_argument(
        "--seam-fade-ms",
        type=float,
        default=DEFAULT_SEAM_FADE_MS,
        help=f"Length of the crossfade at the seam (default: {DEFAULT_SEAM_FADE_MS:g})",
    )
    merge_p.add_argument(
        "--seam-warn-db",
        type=float,
        default=DEFAULT_SEAM_WARN_DB,
        help=(
            "Flag a seam where the two chunks agree to less than this many dB (signal "
            f"power vs power of their difference) (default: {DEFAULT_SEAM_WARN_DB:g}, an "
            "unvalidated heuristic; it only flags, it never changes the audio)"
        ),
    )
    merge_p.add_argument(
        "--anchor",
        action="store_true",
        help=(
            "Compare every processed chunk with the original chunk it was made from (the "
            "files 'split' wrote) and take its place in time from that, not from its "
            "neighbours. With --other-match it takes the level from it too and reports how "
            "well the two stems add up to the original."
        ),
    )
    merge_p.add_argument(
        "--originals",
        metavar="DIR",
        default=None,
        help="With --anchor: where the original chunks are (default: next to the manifest).",
    )
    merge_p.add_argument(
        "--pairing",
        choices=PAIRING_MODES,
        default="auto",
        help=(
            "Which file is which chunk. auto (default): when the file names were lost and "
            "the files can only be paired with the chunks by order, compare each with the "
            "original chunk (next to the manifest, or --originals) and put right what is "
            "out of order. verify: do that for files paired by name too. off: never."
        ),
    )
    merge_p.add_argument(
        "--other-match",
        metavar="WORD",
        default=None,
        help=(
            "The second stem the service returned, named the way --match names the first "
            "(--match instrumental --other-match vocals). Both are merged together, with the "
            "same offsets, gains and fades, and the second is written next to --out."
        ),
    )
    merge_p.add_argument(
        "--other-out",
        metavar="FILE",
        default=None,
        help="Where the second stem goes (default: --out with the --other-match word added).",
    )
    merge_p.add_argument(
        "--recon-warn-db",
        type=float,
        default=DEFAULT_RECON_WARN_DB,
        help=(
            "With --anchor and --other-match: flag a chunk whose two stems add up to the "
            f"original to less than this many dB (default: {DEFAULT_RECON_WARN_DB:g}, an "
            "unvalidated heuristic; it only flags, it never changes the audio)"
        ),
    )
    merge_p.add_argument(
        "--audition",
        metavar="DIR",
        default=None,
        help=(
            "Write a listening kit to DIR: for the seams worth a listen, the audio around "
            "each one joined in three ways (straight, crossfade, seam search), and a page "
            "(index.html) to compare them blind and record which sounds best."
        ),
    )
    merge_p.add_argument(
        "--audition-seams",
        metavar="WHICH",
        default=None,
        help=(
            "With --audition: 'flagged' (default), 'all', or seam numbers such as 3,7,9 "
            "(seam 3 is the one between chunk 3 and chunk 4)."
        ),
    )
    merge_p.add_argument(
        "--audition-window",
        type=float,
        default=None,
        metavar="SECONDS",
        help=f"With --audition: seconds either side of the seam (default: {DEFAULT_WINDOW_S:g})",
    )
    merge_p.add_argument(
        "--audition-format",
        choices=AUDITION_FORMATS,
        default=None,
        help="With --audition: format of the clips (default: flac)",
    )
    merge_p.add_argument(
        "--audition-max",
        type=int,
        default=None,
        metavar="N",
        help=(
            "With --audition: at most this many seams, the worst agreeing first "
            f"(default: {DEFAULT_MAX_SEAMS}; seams named by number are all made)"
        ),
    )
    merge_p.add_argument(
        "--report",
        metavar="FILE",
        default=None,
        help="Write the per-seam report to FILE (JSON if it ends in .json, else Markdown).",
    )
    merge_p.add_argument(
        "--bitrate",
        default="320k",
        help="Output bitrate for lossy formats such as MP3; ignored for .wav/.flac (default: 320k)",
    )
    merge_p.add_argument(
        "--samplerate", type=int, default=44100, help="Output sample rate (default: 44100)"
    )

    add_job_parsers(subparsers)

    services_p = subparsers.add_parser(
        "services", help="List the service presets (--service NAME) and where yours go."
    )
    services_p.add_argument(
        "--services-file", metavar="FILE", help="Read your presets from FILE (JSON)."
    )

    summary_p = subparsers.add_parser(
        "audition-summary",
        help="Count the answers recorded in one or more listening kits.",
    )
    summary_p.add_argument(
        "answers",
        nargs="+",
        metavar="PATH",
        help=f"A {CHOICES_FILE} or the folder of a listening kit that holds one",
    )
    summary_p.add_argument("--json", action="store_true", help="Print the summary as JSON.")

    gui_p = subparsers.add_parser(
        "gui",
        help="Open a drag-and-drop page in your browser (runs on this computer only).",
    )
    gui_p.add_argument(
        "--workdir",
        default=None,
        metavar="DIR",
        help=(
            "Folder for the input, chunks and results; use one that is only for this "
            "(default: ~/SeamSplit). A new split replaces the chunks of the previous one."
        ),
    )
    gui_p.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to listen on, on 127.0.0.1 (default: any free one)",
    )
    gui_p.add_argument(
        "--no-browser", action="store_true", help="Print the address but don't open a browser."
    )

    return parser


def encoded_over_limit(
    paths: List[str], limit_s: float
) -> Tuple[Optional[float], List[Tuple[str, float]]]:
    """Longest encoded duration among ``paths`` (``None`` if none could be read)
    and the files that exceed ``limit_s``."""
    longest: Optional[float] = None
    over: List[Tuple[str, float]] = []
    for path in paths:
        duration = probe_duration_s(path)
        if duration is None:
            continue
        longest = duration if longest is None else max(longest, duration)
        if duration > limit_s:
            over.append((os.path.basename(path), duration))
    return longest, over


def _print_cut_summary(manifest: dict) -> None:
    """Say how the cuts were chosen, and how well they sit on the beat and the bar."""
    if manifest.get("cut_by") == "voice":
        print("\nCuts chosen by voice-band activity.")
    snap_bar = bool(manifest.get("snap_bar"))
    if not (manifest.get("snap_beat") or snap_bar):
        return
    tempo, offsets = manifest.get("tempo_bpm"), manifest.get("cut_beat_offsets_ms")
    if tempo is None or not offsets:
        print("Warning: no steady beat was found, so the cuts were not moved onto beats.")
        return
    close = sum(1 for o in offsets if o <= 50)
    print(
        f"Tempo about {tempo:g} BPM (a half or double tempo can be reported for the real one); "
        f"{close} of {len(offsets)} cuts are within 50 ms of a beat "
        f"(distances in ms: {', '.join(str(o) for o in offsets)})."
    )
    if not snap_bar:
        return
    bar_offsets, confidence = manifest.get("cut_bar_offsets_ms"), manifest.get(
        "downbeat_confidence"
    )
    if not bar_offsets:
        why = (
            "too few beats to tell where the bars start"
            if confidence is None
            else f"the first beat of a bar does not stand out enough (confidence {confidence:g}, "
            f"needs {MIN_BAR_CONFIDENCE:g})"
        )
        print(f"Warning: {why}, so the cuts were moved onto beats only.")
        return
    on_bar = sum(1 for o in bar_offsets if o <= BAR_REACH_MS)
    print(
        f"Bars of {manifest.get('bar_beats')} beats found (confidence {confidence:g}); "
        f"{on_bar} of {len(bar_offsets)} cuts are on a bar line, within {BAR_REACH_MS:g} ms "
        f"(distances in ms: {', '.join(str(o) for o in bar_offsets)})."
    )


def _run_services(args: argparse.Namespace) -> None:
    presets = available_presets(args.services_file)
    for name in sorted(presets):
        preset = presets[name]
        origin = "built in" if preset.builtin else "yours"
        print(f"{name} ({origin}): {preset.describe()}")
    print(f"\nYour own presets: {args.services_file or default_services_file()}")


def _run_split(args: argparse.Namespace) -> None:
    apply_service(args)
    config = build_config(args, args.out)
    if args.limit is not None:
        margin = resolve_margin(args)
        print(
            f"Target chunk length: {config.target_s:g}s (from --limit {args.limit:g}s - "
            f"after-slack {args.after_slack:g}s - overlap {args.overlap:g}s"
            f"{f' - margin {margin:g}s' if margin else ''})"
        )
    print(f"Reading {args.input_file} ...")

    def show_progress(i: int, n: int, name: str) -> None:
        if not args.dry_run and sys.stderr.isatty():
            print(f"\r  exporting {i}/{n}: {name}\033[K", end="", file=sys.stderr, flush=True)

    manifest = split_audio(args.input_file, config, dry_run=args.dry_run, progress=show_progress)
    if not args.dry_run and sys.stderr.isatty():
        print("\r\033[K", end="", file=sys.stderr)

    print("\nChunks:")
    for seg in manifest["segments"]:
        print(
            f"  {seg['name']}: {seg['start_s']:7.2f}s -> {seg['end_s']:7.2f}s "
            f"({seg['duration_s']:5.2f}s)"
        )

    _print_cut_summary(manifest)
    longest = manifest["max_export_ms"] / 1000
    if args.dry_run:
        print(
            f"\nDry run: {len(manifest['parts'])} chunk(s) would be written to '{args.out}/', "
            f"overlap={manifest['overlap_ms'] / 1000:.1f}s. Nothing was saved."
        )
        print(f"Longest chunk would be {longest:.1f}s.")
        return

    print(
        f"\nDone. {len(manifest['parts'])} chunk(s) saved to '{args.out}/', "
        f"overlap={manifest['overlap_ms'] / 1000:.1f}s."
    )
    if args.limit is None:
        print(
            f"Longest exported chunk: {longest:.1f}s of audio (a lossy file reads a little "
            "longer). Pass --limit SECONDS to have it checked against your AI service's limit."
        )
        return

    print(f"Longest exported chunk: {longest:.2f}s of audio.")
    if longest > args.limit:
        print(
            f"Warning: longest chunk ({longest:.2f}s) exceeds --limit ({args.limit:g}s).",
            file=sys.stderr,
        )
    names = manifest["parts"]
    encoded, over = encoded_over_limit([os.path.join(args.out, n) for n in names], args.limit)
    if encoded is None:
        print(
            "Note: could not measure the exported files (is ffprobe installed?).", file=sys.stderr
        )
        return
    print(f"Longest chunk as exported: {encoded:.2f}s (limit {args.limit:g}s).")
    if over:
        listed = ", ".join(f"{name} ({secs:.2f}s)" for name, secs in over)
        print(
            f"Warning: {len(over)} file(s) exceed --limit {args.limit:g}s as exported: {listed}. "
            "Raise --margin.",
            file=sys.stderr,
        )


def _print_seam_summary(report: MergeReport, searched: bool) -> None:
    rated = [b for b in report.boundaries if b.seam_match_db is not None]
    if rated:

        def agreement(b: BoundaryInfo) -> float:
            value = b.seam_match_db if searched else b.overlap_match_db
            return 0.0 if value is None else value

        values = sorted(agreement(b) for b in rated)
        worst = min(rated, key=agreement)
        where = "at the best-matching point" if searched else "across the whole overlap"
        fades = sorted(b.fade_ms for b in rated if b.fade_ms is not None)
        fade = f"{fades[len(fades) // 2]:.0f} ms " if fades else ""
        print(
            f"Seams: {len(rated)} {fade}crossfade(s) {where}; the chunks agree to "
            f"{agreement(worst):.1f} dB at worst (seam {worst.boundary}/{worst.boundary + 1}"
            f"{'' if worst.seam_ms is None else ', at ' + format_time(worst.seam_ms)}), "
            f"median {values[len(values) // 2]:.1f} dB."
        )
    # includes seams whose overlaps did not correspond at all (nothing to rate)
    flagged = [b for b in report.boundaries if b.flagged]
    if flagged:
        spots = ", ".join(
            format_time(b.seam_ms) if b.seam_ms is not None else f"seam {b.boundary}"
            for b in flagged
        )
        print(f"{len(flagged)} seam(s) worth a listen: {spots}")


def _write_report(report: MergeReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if path.lower().endswith(".json"):
            json.dump(report.to_dict(), f, indent=2)
            f.write("\n")
        else:
            f.write(report.to_markdown())


def _print_anchor_summary(report: MergeReport, both_stems: bool) -> None:
    chunks = report.chunks
    if not chunks:
        return
    timed = [c for c in chunks if c.lag_ms is not None]
    if timed:
        worst = max(timed, key=lambda c: abs(c.lag_ms or 0.0))
        print(
            f"Anchored {len(timed)} of {len(chunks)} chunk(s) to the originals in time; "
            f"largest offset {worst.lag_ms:+.1f} ms (chunk {worst.part})."
        )
    leveled = [c for c in chunks if c.gain_db is not None]
    if leveled:
        worst = max(leveled, key=lambda c: abs(c.gain_db or 0.0))
        print(
            f"Took the level of {len(leveled)} chunk(s) from the originals; largest "
            f"correction {worst.gain_db:+.1f} dB (chunk {worst.part})."
        )
    elif not both_stems:
        print(
            "The level was not taken from the originals: one stem alone is quieter than "
            "the mix it came from, so it says nothing about the level (give the other "
            "stem with --other-match for that)."
        )
    rated = [c for c in chunks if c.recon_db is not None]
    if rated:
        values = sorted(c.recon_db for c in rated if c.recon_db is not None)
        worst = min(rated, key=lambda c: c.recon_db if c.recon_db is not None else 0.0)
        print(
            f"The two stems add up to the original to {worst.recon_db:.1f} dB at worst "
            f"(chunk {worst.part}), median {values[len(values) // 2]:.1f} dB."
        )
    flagged = [str(c.part) for c in chunks if c.flagged]
    if flagged:
        print(f"{len(flagged)} chunk(s) worth a listen: {', '.join(flagged)}")


def _print_audition_summary(directory: str, wanted: str) -> None:
    path = os.path.join(directory, KIT_FILE)
    if not os.path.isfile(path):
        print(
            f"No listening kit: no seam was selected ({wanted}). "
            "Use --audition-seams all to hear them all."
        )
        return
    with open(path, "r", encoding="utf-8") as f:
        kit = json.load(f)
    numbers = ", ".join(str(seam["seam"]) for seam in kit["seams"])
    print(
        f"Listening kit: seam(s) {numbers} in '{directory}'. Open "
        f"{os.path.join(directory, KIT_PAGE)} in a browser to compare them."
    )


def _audition_options(args: argparse.Namespace) -> Optional[AuditionOptions]:
    if args.audition is None:
        return None
    return AuditionOptions(
        out_dir=args.audition,
        seams=parse_seams(args.audition_seams or "flagged"),
        window_s=args.audition_window if args.audition_window is not None else DEFAULT_WINDOW_S,
        audio_format=args.audition_format or "flac",
        max_seams=args.audition_max if args.audition_max is not None else DEFAULT_MAX_SEAMS,
    )


def _run_merge(args: argparse.Namespace) -> None:
    if args.overlap is not None:
        print(f"Overlap override for crossfade: {args.overlap:.2f}s (method={args.method})")
    else:
        print(f"Crossfade method: {args.method}")

    report = MergeReport()
    audition = _audition_options(args)
    merged = merge_audio(
        processed_dir=args.processed_dir,
        manifest_path=args.manifest,
        output_path=args.out,
        overlap_s=args.overlap,
        match=args.match,
        method=args.method,
        bitrate=args.bitrate,
        samplerate=args.samplerate,
        align=not args.no_align,
        max_lag_ms=args.max_lag_ms,
        match_level=not args.no_match_level,
        max_gain_db=args.max_gain_db,
        seam_search=args.seam_search,
        seam_fade_ms=args.seam_fade_ms,
        seam_warn_db=args.seam_warn_db,
        report=report,
        anchor=args.anchor,
        originals_dir=args.originals,
        other_match=args.other_match,
        other_output=args.other_out,
        recon_warn_db=args.recon_warn_db,
        audition=audition,
        pairing=args.pairing,
    )

    applied = [b for b in report.boundaries if b.aligned]
    if applied:
        worst = max(applied, key=lambda b: abs(b.lag_ms))
        print(
            f"Aligned {len(applied)} boundary(ies); largest offset "
            f"{worst.lag_ms:+.1f} ms (boundary {worst.boundary}/{worst.boundary + 1})."
        )
    leveled = [b for b in report.boundaries if b.gain_db != 0.0]
    if leveled:
        worst = max(leveled, key=lambda b: abs(b.gain_db))
        print(
            f"Matched level on {len(leveled)} boundary(ies); largest correction "
            f"{worst.gain_db:+.1f} dB (boundary {worst.boundary}/{worst.boundary + 1})."
        )
    _print_anchor_summary(report, args.other_match is not None)
    _print_seam_summary(report, searched=args.seam_search)
    for message in report.notes:
        print(f"Note: {message}")
    for message in report.warnings:
        print(f"Warning: {message}", file=sys.stderr)
    if args.report:
        _write_report(report, args.report)
        print(f"Seam report: {args.report}")
    if audition is not None:
        _print_audition_summary(audition.out_dir, str(args.audition_seams or "flagged"))
    print(f"\nDone. Saved: {args.out}  (length: {len(merged) / 1000:.1f}s)")
    if args.other_match is not None:
        from .anchor import other_output_path

        print(f"Second stem: {args.other_out or other_output_path(args.out, args.other_match)}")


def _run_audition_summary(args: argparse.Namespace) -> None:
    docs = []
    for path in args.answers:
        if os.path.isdir(path):
            path = os.path.join(path, CHOICES_FILE)
        docs.append(load_choices(path))
    summary = summarize_choices(docs)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\n".join(describe_summary(summary)))


def _check_merge_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Options that only mean something together."""
    if args.originals is not None and not args.anchor and args.pairing == "off":
        parser.error("--originals only means something with --anchor, or when --pairing checks")
    if args.other_out is not None and args.other_match is None:
        parser.error("--other-out needs --other-match: which files are the other stem?")
    if args.audition is None:
        extras = ("audition_seams", "audition_window", "audition_format", "audition_max")
        given = [
            f"--{name.replace('_', '-')}" for name in extras if getattr(args, name) is not None
        ]
        if given:
            parser.error(f"{', '.join(given)} only mean something with --audition DIR")


def _explain(exc: Exception) -> str:
    if isinstance(exc, CouldntDecodeError):
        return f"could not read the audio (is the file valid?): {exc}"
    if isinstance(exc, CouldntEncodeError):
        return f"could not write the audio (is the format one of wav, flac, mp3 ...?): {exc}"
    return str(exc)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "merge":
        _check_merge_args(parser, args)

    try:
        if args.command == "split":
            _run_split(args)
        elif args.command == "merge":
            _run_merge(args)
        elif args.command == "run":
            return run_command(args)
        elif args.command == "status":
            return status_command(args)
        elif args.command == "services":
            _run_services(args)
        elif args.command == "audition-summary":
            _run_audition_summary(args)
        elif args.command == "gui":
            from .gui import run as run_gui  # imported here: it needs nothing for the CLI

            return run_gui(args.workdir, args.port, open_browser=not args.no_browser)
    except (OSError, ValueError, CouldntDecodeError, CouldntEncodeError) as exc:
        print(f"Error: {_explain(exc)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
