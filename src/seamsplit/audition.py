"""
The listening kit: for the seams of a merge that are worth a listen, the audio around
each one (``window_s`` seconds either side) joined in each of three ways, so the
question "does this seam, or the seam search, sound better?" is answered by ear.

* **straight**: the two chunks cut at the point the splitter chose and glued together,
  with no alignment, no level matching and no fade. This is what you would get without
  SeamSplit, and shows what the seam would have been.
* **crossfade**: what ``merge`` does by default, a fade over the whole overlap with the
  chosen ``--method`` (``adaptive`` unless told otherwise), after alignment and level
  matching.
* **seam search**: the experimental short crossfade at the best-matching point of the
  overlap (see :mod:`seamsplit.crossfade`).

The two crossfades are made by running the *same* merge (:class:`seamsplit.plan.MergePlan`:
the same files, anchoring, stems and options) over just the two chunks of the seam, and
every clip is cut around the same point, so switching between them while playing keeps
the music in place and only the join changes. The first chunk gets the level it has in
the whole merge, so a clip is as loud as the merged file is there.

A kit is a folder: one audio file per seam and variant, ``audition.json`` (what each one
is, and what was measured at the seam) and ``index.html``, a page that plays them and
records which one you prefer, blind by default, without a server (see
:mod:`seamsplit.choices` for what is done with the answers). Every choice is a data point
for the questions the docs say are open: whether the seam search is worth having, and
what agreement in dB a seam needs to need no listening.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .audio import AudioSegment

from . import __version__
from .plan import MergePlan
from .report import BoundaryInfo, MergeReport, format_time
from .splitter import export_segment, normalize_export_format

AUDITION_FORMATS = ("flac", "wav", "mp3")
KIT_FILE = "audition.json"
KIT_PAGE = "index.html"
DEFAULT_WINDOW_S = 5.0
DEFAULT_MAX_SEAMS = 12
FORMAT_VERSION = 1


@dataclass
class AuditionOptions:
    """Which seams to make a kit for, and how. ``seams`` is ``"flagged"`` (the seams
    the report says are worth a listen, the default), ``"all"``, or a list of seam
    numbers (``7`` is the one between chunk 7 and chunk 8). At most ``max_seams`` are
    made, the worst agreeing first, unless the numbers are listed. ``progress(i, n,
    name)`` is called before each seam; an exception it raises stops the kit."""

    out_dir: str
    seams: Union[str, Sequence[int]] = "flagged"
    window_s: float = DEFAULT_WINDOW_S
    audio_format: str = "flac"
    max_seams: int = DEFAULT_MAX_SEAMS
    bitrate: str = "192k"
    progress: Optional[Callable[[int, int, str], None]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not (isinstance(self.window_s, (int, float)) and 0 < self.window_s < math.inf):
            raise ValueError("the listening window must be a number of seconds above 0")
        self.audio_format = normalize_export_format(self.audio_format)
        if self.audio_format not in AUDITION_FORMATS:
            raise ValueError(f"listening kit format must be one of: {', '.join(AUDITION_FORMATS)}")
        if not isinstance(self.max_seams, int) or self.max_seams < 1:
            raise ValueError("the largest number of seams in a listening kit must be at least 1")
        if isinstance(self.seams, str):
            if self.seams not in ("flagged", "all"):
                raise ValueError("seams must be 'flagged', 'all' or a list of seam numbers")
        elif not self.seams or any(not isinstance(n, int) or n < 1 for n in self.seams):
            raise ValueError("seams must be 'flagged', 'all' or a list of seam numbers (1, 2, ...)")


def parse_seams(text: str) -> Union[str, List[int]]:
    """``"flagged"``, ``"all"`` or ``"3,7,9"`` as :class:`AuditionOptions` wants it."""
    word = text.strip().lower()
    if word in ("flagged", "all"):
        return word
    try:
        numbers = [int(piece) for piece in word.split(",")]
    except ValueError:
        raise ValueError(
            f"'{text}' is not 'flagged', 'all' or a list of seam numbers such as 3,7,9"
        ) from None
    if any(n < 1 for n in numbers):
        raise ValueError("seam numbers start at 1")
    return numbers


def variants_for(method: str) -> List[Dict[str, str]]:
    """The ways a seam is joined, in the order the kit lists them."""
    return [
        {
            "id": "straight",
            "kind": "straight",
            "label": "Straight join",
            "description": "cut where the splitter cut and glued: no alignment, levelling or fade",
        },
        {
            "id": method,
            "kind": "crossfade",
            "label": f"Crossfade ({method})",
            "description": "faded over the whole overlap, after alignment and level matching",
        },
        {
            "id": "seam_search",
            "kind": "seam_search",
            "label": "Seam search",
            "description": "a short fade where the two chunks agree best (experimental)",
        },
    ]


def select_seams(report: MergeReport, options: AuditionOptions) -> List[BoundaryInfo]:
    """The boundaries of ``report`` the kit is made for, in track order."""
    by_number = {b.boundary: b for b in report.boundaries}
    if isinstance(options.seams, str):
        chosen = [b for b in report.boundaries if b.flagged or options.seams == "all"]

        def agreement(b: BoundaryInfo) -> float:
            # a seam whose overlaps did not correspond has nothing rated: the worst
            return -math.inf if b.overlap_match_db is None else b.overlap_match_db

        chosen = sorted(chosen, key=agreement)[: options.max_seams]
    else:
        chosen = [by_number[n] for n in dict.fromkeys(options.seams) if n in by_number]
    return sorted(chosen, key=lambda b: b.boundary)


def _straight_join(plan: MergePlan, index: int, extra_db: float) -> Tuple[AudioSegment, int]:
    """Chunks ``index`` and ``index + 1`` (0-based) cut where the splitter cut them and
    glued, as they came back. Returns the audio and the cut's position in it."""
    prev = AudioSegment.from_file(plan.paths[index])
    nxt = AudioSegment.from_file(plan.paths[index + 1])
    if extra_db:
        prev = prev.apply_gain(extra_db)
    if (nxt.frame_rate, nxt.channels, nxt.sample_width) != (
        prev.frame_rate,
        prev.channels,
        prev.sample_width,
    ):
        nxt = nxt.set_sample_width(prev.sample_width).set_frame_rate(prev.frame_rate)
        nxt = nxt.set_channels(prev.channels)
    overlap = int(round(plan.overlap_ms))
    lead = overlap // 2  # the part of the overlap the splitter took from the previous chunk's end
    cut = max(0, len(prev) - (overlap - lead))
    return prev[:cut] + nxt[lead:], cut


def _metrics(b: BoundaryInfo, plan_report: MergeReport) -> Dict[str, Any]:
    """What the merge measured at the seam, and how well each of its two chunks adds up
    to its original when that was measured."""
    data: Dict[str, Any] = {
        "lag_ms": b.lag_ms,
        "correlation": b.correlation,
        "gain_db": b.gain_db,
        "overlap_match_db": b.overlap_match_db,
        "seam_match_db": b.seam_match_db,
        "flagged": b.flagged,
    }
    recon = {c.part: c.recon_db for c in plan_report.chunks}
    if recon.get(b.boundary) is not None or recon.get(b.boundary + 1) is not None:
        data["recon_db"] = [recon.get(b.boundary), recon.get(b.boundary + 1)]
    return data


def _write(clip: AudioSegment, out_dir: str, name: str, options: AuditionOptions) -> None:
    export_segment(clip, os.path.join(out_dir, name), options.audio_format, options.bitrate)


def _make_seam(
    plan: MergePlan,
    report: MergeReport,
    b: BoundaryInfo,
    options: AuditionOptions,
    variants: List[Dict[str, str]],
    total: int,
) -> Dict[str, Any]:
    index = b.boundary - 1  # the first of the seam's two chunks, 0-based
    # a chunk has, in the whole merge, the level the seam before it gave it
    extra_db = next((x.gain_db for x in report.boundaries if x.boundary == b.boundary - 1), 0.0)

    local = MergeReport()
    chunks = list(plan.stream([index, index + 1], local, extra_db=extra_db))
    crossfade_id, search_id = variants[1]["id"], variants[2]["id"]
    audio: Dict[str, AudioSegment] = {}
    infos: Dict[str, Optional[BoundaryInfo]] = {}
    for vid, search in ((crossfade_id, False), (search_id, True)):
        run = MergeReport()
        merged = plan.merge(None, run, seam_search=search, chunks=chunks)
        audio[vid] = plan.split(merged)[0]
        infos[vid] = run.boundaries[0] if run.boundaries else None
    straight, cut = _straight_join(plan, index, extra_db)
    audio["straight"] = straight

    # every clip is cut around the same point: the middle of the overlap, where the
    # whole-overlap crossfade has its centre (and the splitter its cut)
    whole = infos[crossfade_id]
    centre = {"straight": float(cut)}
    centre[crossfade_id] = centre[search_id] = (
        whole.seam_ms if whole is not None and whole.seam_ms is not None else float(cut)
    )
    half = max(options.window_s * 1000.0, plan.overlap_ms / 2.0 + 500.0)
    left = min([half] + [centre[v] for v in audio])
    right = min([half] + [len(audio[v]) - centre[v] for v in audio])
    right = max(right, 0.0)

    prefix = f"seam{b.boundary:0{max(2, len(str(total)))}d}"
    clips: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        vid = variant["id"]
        start = centre[vid] - left
        clip = audio[vid][int(round(start)) : int(round(centre[vid] + right))]
        name = f"{prefix}_{vid}.{options.audio_format}"
        _write(clip, options.out_dir, name, options)
        entry: Dict[str, Any] = {"file": name, "length_ms": len(clip), "match_db": None}
        info = infos.get(vid)
        if vid == "straight":
            entry["seam_ms"], entry["fade_ms"] = left, 0.0
        elif info is not None and info.seam_ms is not None:
            entry["seam_ms"] = info.seam_ms - start
            entry["fade_ms"] = info.fade_ms
            entry["match_db"] = info.seam_match_db
        clips[vid] = entry
    # a seam search that found nothing better than the whole-overlap fade (or could not
    # run, for overlaps that do not correspond) makes the same audio: say so
    same = audio[crossfade_id].raw_data == audio[search_id].raw_data
    clips[search_id]["identical_to"] = crossfade_id if same else None

    return {
        "seam": b.boundary,
        "of": total,
        "at_ms": b.seam_ms,
        "at": None if b.seam_ms is None else format_time(b.seam_ms),
        "flagged": b.flagged,
        "centre_ms": left,
        "metrics": _metrics(b, report),
        "clips": clips,
    }


def export_auditions(
    plan: MergePlan,
    report: MergeReport,
    options: AuditionOptions,
    source_file: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Make the listening kit of a merge that has run (``report`` is what it found) in
    ``options.out_dir``, and return what was written to ``audition.json``; ``None`` when
    no seam was selected, and then nothing is written."""
    selected = select_seams(report, options)
    if not selected:
        return None
    os.makedirs(options.out_dir, exist_ok=True)
    variants = variants_for(plan.method)
    total = len(plan.paths) - 1
    seams = []
    for k, b in enumerate(selected, start=1):
        if options.progress is not None:
            options.progress(k, len(selected), f"seam {b.boundary}")
        seams.append(_make_seam(plan, report, b, options, variants, total))

    kit = {
        "format": FORMAT_VERSION,
        "seamsplit": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": source_file or plan.manifest.get("source_file"),
        "window_s": options.window_s,
        "audio_format": options.audio_format,
        "settings": {
            "method": plan.method,
            "align": plan.align,
            "max_lag_ms": plan.max_lag_ms,
            "match_level": plan.match_level,
            "max_gain_db": plan.max_gain_db,
            "seam_fade_ms": plan.seam_fade_ms,
            "seam_warn_db": plan.seam_warn_db,
            "anchored": plan.original_paths is not None,
            "both_stems": plan.other_paths is not None,
            "merged_with_seam_search": plan.seam_search,
        },
        "variants": variants,
        "seams": seams,
    }
    with open(os.path.join(options.out_dir, KIT_FILE), "w", encoding="utf-8") as f:
        json.dump(kit, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_kit_page(options.out_dir, kit)
    return kit


def player_script() -> str:
    """The A/B player (``static/audition.js``), shared by the page of a kit and the GUI."""
    return resources.files("seamsplit").joinpath("static/audition.js").read_text(encoding="utf-8")


def write_kit_page(out_dir: str, kit: Dict[str, Any]) -> str:
    """``index.html`` in ``out_dir``: a page that plays the kit from the files next to it
    (it needs no server, so the folder can be opened, shared or published as it is)."""
    template = resources.files("seamsplit").joinpath("static/audition_page.html").read_text("utf-8")
    # "</" must not end the script block the data sits in
    data = json.dumps(kit, ensure_ascii=False).replace("</", "<\\/")
    page = template.replace("/*PLAYER*/", player_script().replace("</", "<\\/"))
    page = page.replace("/*KIT*/null", data)
    path = os.path.join(out_dir, KIT_PAGE)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
