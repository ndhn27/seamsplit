"""
What is done with the answers of a listening kit (:mod:`seamsplit.audition`).

Each answer says which of the ways of joining a seam sounded best, or that none did; it is
kept next to what was measured at that seam, so a pile of them can tell two things the
docs call open: whether the seam search is worth having (the premise behind it, that a
chunk is at its worst near its own edges, has not been checked on real separator output)
and how well two chunks have to agree, in dB, for the seam not to need a listen (the
default threshold is a guess). :func:`summarize_choices` counts them and puts one number
on each question, an exact sign test, so a handful of answers is not read as more than it
is. It can take the answers of several kits (``seamsplit audition-summary``): the more
songs and services, the more it is worth.

Only **blind** answers count towards the tests: when the page shows which clip is which, the
answer is as much about what you expect of it as about what you hear.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

CHOICES_FILE = "choices.json"
TIE = "tie"
FORMAT_VERSION = 1

# Fewer decided seams than this cannot reach p < 0.05 even when every one goes the same
# way (2 * 0.5**6 = 0.031 is the first that can).
MIN_DECIDED = 6
ALPHA = 0.05


def new_choices(kit: Dict[str, Any]) -> Dict[str, Any]:
    """An empty answer sheet for ``kit`` (what ``audition.json`` holds)."""
    return {
        "format": FORMAT_VERSION,
        "source_file": kit.get("source_file"),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": kit.get("settings", {}),
        "window_s": kit.get("window_s"),
        "choices": [],
    }


def record_choice(
    doc: Dict[str, Any], kit: Dict[str, Any], seam: int, choice: str, blind: bool
) -> Dict[str, Any]:
    """Put the answer for ``seam`` (its number in the report) into ``doc``, replacing an
    earlier one for the same seam. ``choice`` is the id of one of the kit's variants or
    ``"tie"`` (no audible difference). Raises ``ValueError`` for a seam the kit does not
    have or a choice it does not offer."""
    by_id = {v["id"]: v for v in kit.get("variants", [])}
    entry = next((s for s in kit.get("seams", []) if s["seam"] == seam), None)
    if entry is None:
        raise ValueError(f"The kit has no seam {seam}")
    if choice != TIE and choice not in by_id:
        raise ValueError(f"choice must be one of: {', '.join([*by_id, TIE])}")
    search = next((v["id"] for v in by_id.values() if v["kind"] == "seam_search"), None)
    record = {
        "seam": seam,
        "choice": choice,
        "kind": TIE if choice == TIE else by_id[choice]["kind"],
        "blind": bool(blind),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": entry.get("metrics", {}),
        "seam_search_identical": bool(
            search and entry["clips"].get(search, {}).get("identical_to")
        ),
    }
    doc["choices"] = [c for c in doc["choices"] if c["seam"] != seam] + [record]
    doc["choices"].sort(key=lambda c: c["seam"])
    return doc


def binomial_two_sided(k: int, n: int) -> float:
    """The exact two-sided p-value of ``k`` successes in ``n`` fair coin tosses."""
    if n <= 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / 2.0**n)


def _verdict(n: int, wins: int, a: str, b: str) -> str:
    """``a`` or ``b`` when one is preferred beyond chance, else why not."""
    if n < MIN_DECIDED:
        return "too_few"
    if binomial_two_sided(wins, n) >= ALPHA:
        return "no_clear_preference"
    return b if wins > n - wins else a


def _spread(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "n": len(values),
        "min": round(min(values), 1),
        "median": round(statistics.median(values), 1),
        "max": round(max(values), 1),
    }


def summarize_choices(docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Count the answers of ``docs`` (answer sheets, as :func:`record_choice` fills them)
    and say what they show about the seam search and about the straight join.

    ``seam_search_vs_crossfade`` leaves out the seams where the seam search made the same
    audio as the crossfade (there is nothing to prefer) and every answer that was not
    blind; ``straight_vs_processed`` asks the same of the straight join against the two
    others together. ``verdict`` is the name of the preferred way when the sign test says
    so (``p_value`` is two-sided, below 0.05), ``no_clear_preference``, or ``too_few`` when
    fewer than 6 seams were decided. ``agreement_db`` is the agreement the merge measured
    over the overlap at the seams where nothing was audible against those where
    something was, which is what ``--seam-warn-db`` is about."""
    choices = [c for doc in docs for c in doc.get("choices", [])]
    counts = {"straight": 0, "crossfade": 0, "seam_search": 0, TIE: 0}
    for c in choices:
        counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    blind = [c for c in choices if c.get("blind")]

    pairs = [c for c in blind if c["kind"] in ("crossfade", "seam_search")]
    pairs = [c for c in pairs if not c.get("seam_search_identical")]
    s = sum(1 for c in pairs if c["kind"] == "seam_search")
    sv = {
        "n": len(pairs),
        "seam_search": s,
        "crossfade": len(pairs) - s,
        "p_value": binomial_two_sided(s, len(pairs)),
        "verdict": _verdict(len(pairs), s, "crossfade", "seam_search"),
    }

    decided = [c for c in blind if c["kind"] != TIE]
    st = sum(1 for c in decided if c["kind"] == "straight")
    sp = {
        "n": len(decided),
        "straight": st,
        "processed": len(decided) - st,
        "p_value": binomial_two_sided(st, len(decided)),
        "verdict": _verdict(len(decided), st, "processed", "straight"),
    }

    def agreement(c: Dict[str, Any]) -> Optional[float]:
        value = c.get("metrics", {}).get("overlap_match_db")
        return None if value is None else float(value)

    same = [agreement(c) for c in choices if c["kind"] == TIE]
    differ = [agreement(c) for c in choices if c["kind"] != TIE]
    return {
        "choices": len(choices),
        "blind": len(blind),
        "counts": counts,
        "seam_search_vs_crossfade": sv,
        "straight_vs_processed": sp,
        "agreement_db": {
            "no_difference": _spread([v for v in same if v is not None]),
            "heard_difference": _spread([v for v in differ if v is not None]),
        },
    }


def describe_summary(summary: Dict[str, Any]) -> List[str]:
    """The summary as lines of text, for the command line."""
    if not summary["choices"]:
        return ["No answers yet."]
    c = summary["counts"]
    lines = [
        f"{summary['choices']} answer(s), {summary['blind']} of them blind: "
        f"straight join {c['straight']}, crossfade {c['crossfade']}, "
        f"seam search {c['seam_search']}, no audible difference {c[TIE]}."
    ]
    sv, sp = summary["seam_search_vs_crossfade"], summary["straight_vs_processed"]
    texts = {
        "too_few": f"too few decided seams (at least {MIN_DECIDED} are needed to tell)",
        "no_clear_preference": "no clear preference",
        "seam_search": "seam search preferred",
        "crossfade": "the crossfade over the whole overlap preferred",
        "straight": "the straight join preferred (merging is not helping here)",
        "processed": "the processed joins preferred to the straight one",
    }
    lines.append(
        f"Seam search against the crossfade: {sv['seam_search']} to {sv['crossfade']} "
        f"(p = {sv['p_value']:.3f}): {texts[sv['verdict']]}."
    )
    lines.append(
        f"Straight join against the other two: {sp['straight']} to {sp['processed']} "
        f"(p = {sp['p_value']:.3f}): {texts[sp['verdict']]}."
    )
    for key, label in (("no_difference", "nothing audible"), ("heard_difference", "a difference")):
        spread = summary["agreement_db"][key]
        if spread:
            lines.append(
                f"Agreement at the seams with {label}: {spread['min']:g} to {spread['max']:g} dB, "
                f"median {spread['median']:g} ({spread['n']} seams)."
            )
    return lines


def load_choices(path: str) -> Dict[str, Any]:
    """Read an answer sheet written by :func:`save_choices`."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"'{path}' is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("choices"), list):
        raise ValueError(f"'{path}' is not an answer sheet of a seamsplit listening kit")
    for c in doc["choices"]:
        if not isinstance(c, dict) or "kind" not in c or "seam" not in c:
            raise ValueError(f"'{path}' has an answer without a seam or a kind")
    return doc


def save_choices(doc: Dict[str, Any], folder: str) -> str:
    """Write ``doc`` to ``choices.json`` in ``folder`` (replacing it, all at once)."""
    path = os.path.join(folder, CHOICES_FILE)
    partial = path + ".part"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(partial, path)
    return path
