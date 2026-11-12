"""The options that decide how a file is cut into chunks, shared by ``split`` and ``run``:
how they are declared, how ``--service``, ``--limit`` and ``--target`` settle the chunk
length, and the :class:`SplitConfig` they make."""

from __future__ import annotations

import argparse

from .cutpoints import CUT_BY, DEFAULT_BEAT_TOLERANCE_DB, DEFAULT_BEATS_PER_BAR
from .services import apply_preset, find_preset
from .splitter import DEFAULT_LOSSY_MARGIN_S, SplitConfig, default_margin_s

DEFAULT_TARGET_S = 52.0
DEFAULT_OVERLAP_S = 4.0
DEFAULT_FORMAT = "flac"


def add_split_options(parser: argparse.ArgumentParser) -> None:
    """Declare the options of a split on ``parser`` (everything but where the chunks go)."""
    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument(
        "--target",
        type=float,
        default=None,
        help=(
            "Target length of each chunk in seconds, before overlap is added. Tune so "
            "target + after-slack + overlap stays under your AI service's limit. "
            f"(default: {DEFAULT_TARGET_S:g})"
        ),
    )
    length_group.add_argument(
        "--limit",
        "--max-duration",
        dest="limit",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Your AI service's per-file limit in seconds. Sets --target to limit - "
            "after-slack - overlap - margin so no exported chunk exceeds it, and checks "
            "the length of the exported files. Alternative to --target. "
            "(--max-duration is the same option.)"
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Room to keep under --limit for the padding a lossy encoder adds (an MP3 of "
            f"59.000s reads 59.037s). Default: {DEFAULT_LOSSY_MARGIN_S:g} for lossy "
            "formats, 0 for wav/flac."
        ),
    )
    parser.add_argument(
        "--service",
        metavar="NAME",
        help=(
            "A service preset (see 'seamsplit services'): its length limit, and its format "
            "and overlap if it has any, unless you give them yourself. A preset's limit is "
            "a --limit: it cannot be combined with --target."
        ),
    )
    parser.add_argument(
        "--services-file",
        metavar="FILE",
        help="Your own presets (JSON); default: services.json in the configuration folder.",
    )
    parser.add_argument(
        "--before-slack",
        type=float,
        default=20.0,
        help=(
            "How much shorter than --target a chunk may end up; must be smaller than "
            "--target (default: 20)"
        ),
    )
    parser.add_argument(
        "--after-slack",
        type=float,
        default=3.0,
        help="How much longer than --target a chunk may end up (default: 3)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help=(
            "Seconds shared between neighboring chunks, for crossfading later. "
            f"0 disables overlap. (default: {DEFAULT_OVERLAP_S:g}, or the --service's)"
        ),
    )
    parser.add_argument(
        "--smooth-ms",
        type=float,
        default=None,
        help=(
            "Moving-average window (ms) applied before searching for the quietest point, "
            "favoring sustained silence over a single noisy sample. 0 disables smoothing. "
            "(default: 250, or 1000 with --cut-by voice, whose measurement is noisier)"
        ),
    )
    parser.add_argument(
        "--cut-by",
        choices=CUT_BY,
        default="rms",
        help=(
            "What decides where it is quiet enough to cut. rms (default): the loudness of "
            "the whole signal. voice: the activity in the voice band (300 Hz-3.4 kHz; for "
            "stereo only what is the same in both channels, where lead vocals usually sit), "
            "so a bass line or a drum loop does not hide a sung phrase. A measurement, not "
            "a vocal detector: check the cuts with --dry-run."
        ),
    )
    parser.add_argument(
        "--snap-beat",
        action="store_true",
        help=(
            "Track the beat and, within --beat-tolerance-db of the quietest point of each "
            "window, cut on a beat. Needs a track with a fairly steady tempo; says so when "
            "it finds none."
        ),
    )
    parser.add_argument(
        "--beat-tolerance-db",
        type=float,
        default=DEFAULT_BEAT_TOLERANCE_DB,
        help=(
            "With --snap-beat: how much louder than the quietest point a cut may be to land "
            f"on a beat (default: {DEFAULT_BEAT_TOLERANCE_DB:g})"
        ),
    )
    parser.add_argument(
        "--snap-bar",
        action="store_true",
        help=(
            "Like --snap-beat, and where the track clearly marks its bars (the first beat "
            "of a bar struck harder than the others), prefer a point on a bar line within "
            "the quiet stretch. Says so when it cannot tell where the bars start; the beat "
            "rule is used then."
        ),
    )
    parser.add_argument(
        "--bar-beats",
        type=int,
        default=DEFAULT_BEATS_PER_BAR,
        metavar="N",
        help=(
            "With --snap-bar: beats in a bar as the beat tracker counts them "
            f"(default: {DEFAULT_BEATS_PER_BAR}; 3 for a waltz; 8 if it reports a double tempo)"
        ),
    )
    parser.add_argument(
        "--fade-ms",
        type=float,
        default=8.0,
        help=(
            "Fade in/out (ms) applied only at the very start/end of the whole track. "
            "Internal chunk edges are left raw for crossfading on merge. (default: 8)"
        ),
    )
    parser.add_argument(
        "--min-final",
        type=float,
        default=5.0,
        help=(
            "If the last chunk would be shorter than this (seconds), re-divide the last "
            "two chunks so none exceeds the length limit (default: 5)"
        ),
    )
    parser.add_argument(
        "--format",
        default=None,
        help=(
            f"Export format of the chunks (default: {DEFAULT_FORMAT}, or the --service's). "
            "flac and wav are lossless: no quality loss and no encoder delay/padding, which "
            "an MP3 adds at every seam (flac is about half the size of wav). Use mp3 only "
            "if your AI service rejects both."
        ),
    )
    parser.add_argument(
        "--bitrate",
        default="320k",
        help="Export bitrate for lossy formats; ignored for wav/flac (default: 320k)",
    )


def resolve_margin(args: argparse.Namespace) -> float:
    margin = args.margin if args.margin is not None else default_margin_s(args.format)
    if margin < 0:
        raise ValueError("--margin must not be negative")
    return margin


def resolve_target(args: argparse.Namespace) -> float:
    if args.limit is None:
        return args.target if args.target is not None else DEFAULT_TARGET_S
    reserved = args.after_slack + args.overlap + resolve_margin(args)
    target = args.limit - reserved
    if target <= 0:
        raise ValueError(
            "--limit is too small: it must exceed --after-slack + --overlap + --margin "
            f"({reserved:g}s)"
        )
    return target


def apply_service(args: argparse.Namespace) -> None:
    """Settle ``limit``, ``format`` and ``overlap`` of a split: what was given, else
    what ``--service`` says, else the default."""
    resolved = apply_preset(
        find_preset(args.service, args.services_file) if args.service else None,
        target=args.target,
        limit=args.limit,
        fmt=args.format,
        overlap=args.overlap,
    )
    args.limit = resolved["limit"]
    args.format = resolved["fmt"] or DEFAULT_FORMAT
    args.overlap = DEFAULT_OVERLAP_S if resolved["overlap"] is None else resolved["overlap"]
    if resolved["notes"]:
        print(f"Service {args.service}: " + ", ".join(resolved["notes"]) + ".")


def build_config(args: argparse.Namespace, output_dir: str) -> SplitConfig:
    """The split ``args`` describe (after :func:`apply_service`), writing to ``output_dir``."""
    return SplitConfig(
        target_s=resolve_target(args),
        before_slack_s=args.before_slack,
        after_slack_s=args.after_slack,
        overlap_s=args.overlap,
        smooth_ms=args.smooth_ms,
        fade_ms=args.fade_ms,
        min_final_s=args.min_final,
        output_dir=output_dir,
        export_format=args.format,
        bitrate=args.bitrate,
        cut_by=args.cut_by,
        snap_beat=args.snap_beat,
        beat_tolerance_db=args.beat_tolerance_db,
        snap_bar=args.snap_bar,
        bar_beats=args.bar_beats,
    )
