"""RESULTS.md, written from the JSON files in ``results/``: the numbers as they were
measured, in tables, with what each column is. What they mean is in the README."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .chunks import DESCRIPTIONS


def _load(folder: str, name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(folder, name + ".json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _f(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value != value:
        return "-"
    return f"{value:.{digits}f}{suffix}"


def _ci(row: Dict[str, Any]) -> str:
    low, high = row.get("seam_gain_mean_ci95_db", [None, None])
    return f"{_f(row.get('seam_gain_mean_db'), 2)} [{_f(low, 2)}, {_f(high, 2)}]"


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out + [""]


def _environment(env: Dict[str, Any]) -> List[str]:
    sep = env.get("separator", {})
    lines = ["## What it was measured on", ""]
    rows = [
        ["date", env.get("date", "-")],
        [
            "machine",
            f"{env.get('cpu', '-')}, {env.get('cpu_threads', '-')} threads, {env.get('memory_gb', '-')} GB",
        ],
        ["system", env.get("platform", "-")],
        ["python / numpy", f"{env.get('python', '-')} / {env.get('numpy', '-')}"],
        ["audio layer", ", ".join(f"{k} {v}" for k, v in env.get("audio_engine", {}).items())],
        ["separator", sep.get("name", "-")],
        ["model sha256", (sep.get("model_sha256") or "-")[:16] + "..."],
        [
            "separator versions",
            ", ".join(f"{k} {v}" for k, v in sep.get("versions", {}).items()) or "-",
        ],
    ]
    lines += _table(["", ""], [[a, b] for a, b in rows])
    songs = env.get("songs", {})
    lines += ["Songs:", ""]
    lines += _table(
        ["id", "title", "seconds", "ground-truth stems"],
        [
            [k, v["title"], _f(v["seconds"]), "yes" if v["has_truth"] else "no"]
            for k, v in songs.items()
        ],
    )
    return lines


def _seams(data: Dict[str, Any]) -> List[str]:
    lines = ["## 1. Seams: chunks separated for real, joined in different ways", ""]
    lines += [
        "All figures are signal-to-error ratios in dB (higher is better; 99 = identical) of the "
        "merged stem against the same song separated in one go (**whole**), averaged over both "
        "stems and all songs, in the 2 s windows around the seams (**seams**) and everywhere "
        "else (**away**). **truth** is against the ground-truth stems (three songs only; its "
        "ceiling is 18-23 dB). **click** is the reference-free step at the join (dB above the "
        "typical change around it). **better** is the number of seams (both stems) where the "
        "error is lower than with a hard cut of the same seam, **median gain** how much lower.",
        "",
    ]
    for set_name, rows in data["summary"].items():
        lines += [f"### `{set_name}`: {DESCRIPTIONS.get(set_name, '')}", ""]
        body = []
        for variant, r in rows.items():
            body.append(
                [
                    f"`{variant}`",
                    _f(r["vs_whole_db"]),
                    _f(r["vs_whole_seams_db"]),
                    _f(r["vs_whole_away_db"]),
                    _f(r.get("vs_truth_seams_db")),
                    _f(r["click_db_mean"]),
                    (
                        f"{r['seams_better_than_hardcut']}/{r['seams_compared']}"
                        if "seams_compared" in r
                        else "-"
                    ),
                    _f(r.get("seam_gain_median_db"), 2) if "seams_compared" in r else "-",
                    _ci(r) if "seams_compared" in r else "-",
                ]
            )
        lines += _table(
            [
                "joined as",
                "whole",
                "seams",
                "away",
                "truth (seams)",
                "click",
                "better",
                "median gain",
                "mean gain [95 % interval]",
            ],
            body,
        )
    worst = data.get("worst_seams")
    if worst:
        lines += [
            "A median that is a gain can hide seams that got worse. The six where the default "
            "(`adaptive`) did worst against a hard cut of the same seam (dB; below 0 is a loss):",
            "",
        ]
        lines += _table(
            ["song", "chunk set", "stem", "at (s)", "hard cut", "adaptive", "gain"],
            [
                [
                    w["song"],
                    f"`{w['set']}`",
                    w["stem"],
                    _f(w["at_s"]),
                    _f(w["hardcut_db"]),
                    _f(w["variant_db"]),
                    _f(w["gain_db"]),
                ]
                for w in worst
            ],
        )
    lines += [
        "Song by song, in the windows around the seams of `quiet2` "
        "(dB against the whole-song separation):",
        "",
    ]
    song_rows = []
    for song_id, song in data["songs"].items():
        variants = song["sets"].get("quiet2", {}).get("variants", {})
        row = [song_id, str(song["sets"].get("quiet2", {}).get("chunks", "-"))]
        for variant in ("hardcut", "adaptive", "equalpower"):
            scores = variants.get(variant, {}).get("stems", {})
            values = [v["vs_whole_seams_db"] for v in scores.values()]
            row.append(_f(sum(values) / len(values)) if values else "-")
        song_rows.append(row)
    lines += _table(["song", "chunks", "hardcut", "adaptive", "equalpower"], song_rows)
    ref_rows = []
    for song_id, song in data["songs"].items():
        ref = song["reference"]
        ref_rows.append(
            [
                song_id,
                _f(ref.get("vocals_vs_truth_db")),
                _f(ref.get("instrumental_vs_truth_db")),
                _f(ref.get("truth_recon_db")),
                _f(ref["recon_db"]),
            ]
        )
    lines += [
        "What the separation costs by itself (the whole song, in one go, against the ground "
        "truth), which is the ceiling for everything above, and how well two stems add up to the mix:",
        "",
    ]
    lines += _table(
        [
            "song",
            "vocals vs truth",
            "instrumental vs truth",
            "truth stems add up to mix",
            "separated stems add up to mix",
        ],
        ref_rows,
    )
    return lines


def _cuts(data: Dict[str, Any]) -> List[str]:
    lines = ["## 2. Cut points", ""]
    lines += [
        "**below song** is how many dB below the whole song's level the mix is in the 200 ms "
        "around a cut; **quieter places** the share of the song's 200 ms windows that are "
        "quieter than the cut (0 % = the quietest place there is). **hard-cut error** is the "
        "ratio against the whole-song separation in the 2 s around a cut when the chunks are "
        "just joined there, which is what the choice of the cut alone decides.",
        "",
    ]
    rows = []
    for set_name, r in data["summary"].items():
        rows.append(
            [
                f"`{set_name}`",
                str(r["cuts"]),
                _f(r["level_below_song_db_mean"]),
                _f(r["percent_of_song_quieter_mean"], 0, " %"),
                _f(r["hardcut_seam_db_mean"]),
                _f(r["hardcut_seam_db_median"]),
            ]
        )
    return lines + _table(
        [
            "chunk set",
            "cuts",
            "below song (dB)",
            "quieter places",
            "hard-cut error mean (dB)",
            "median (dB)",
        ],
        rows,
    )


def _pairing(data: Dict[str, Any]) -> List[str]:
    lines = ["## 3. Pairing files whose names were lost", ""]
    lines += [
        "Real stems of every chunk of the overlapping sets, scored against the original chunks "
        "(the thresholds of `seamsplit.pairing` were set on synthetic songs), then renamed "
        "`download (k)` in a random order and merged with the check on (`auto`) and off.",
        "",
    ]
    rows = []
    for stem, r in data["summary"].items():
        rows.append(
            [
                stem,
                str(r["chunks"]),
                f"{r['right_part_of_best_score']}/{r['chunks']}",
                f"{r['sure_by_thresholds']}/{r['chunks']}",
                _f(r["true_score_min"], 2),
                _f(r["true_score_median"], 2),
                _f(r["runner_up_max"], 2),
                _f(r["margin_min"], 2),
                str(r["pairing_auto"]),
                str(r["pairing_off"]),
            ]
        )
    lines += _table(
        [
            "stem",
            "chunks",
            "best score is the right part",
            "sure by the thresholds",
            "true score min",
            "median",
            "runner-up max",
            "margin min",
            "renamed, check on",
            "renamed, check off",
        ],
        rows,
    )
    t = next(iter(data["summary"].values()), {}).get("thresholds", {})
    if t:
        lines += [
            f"Thresholds: score at least {t['min_ncc']}, margin at least {t['min_margin']}.",
            "",
        ]
    return lines


def _degraded(data: Dict[str, Any]) -> List[str]:
    lines = ["## 4. What services do to their answers (the damage is SIMULATED)", ""]
    lines += [
        "Real separated chunks, damaged as described, merged three ways, and compared with the merge of "
        "the undamaged chunks (after the best single gain and shift). dB, higher is better. "
        "`off` = no alignment and no level matching; `default`; `anchor` = also tied to the original chunks.",
        "",
    ]
    rows = []
    for kind, handling in data["summary"].items():
        for name, r in handling.items():
            rows.append([kind, f"`{name}`", _f(r["vs_clean_db"]), _f(r["vs_clean_seams_db"])])
    lines += _table(["damage", "merged with", "whole file", "around the seams"], rows)
    return lines


def _perf(data: Dict[str, Any]) -> List[str]:
    lines = ["## 5. Speed and memory", ""]
    lines += [
        f"About {data['input_seconds']:.0f} s of stereo 44.1 kHz music; median of {data['repeats']} runs, each in "
        "a fresh process. Seconds; peak memory in MB.",
        "",
    ]
    legacy = data.get("legacy")
    header = ["phase", "seconds", "peak MB"] + (
        ["other version s", "other version peak MB"] if legacy else []
    )
    rows = []
    for name, r in data["current"].items():
        row = [name, _f(r.get("seconds_median"), 2), _f(r.get("peak_rss_mb"), 0)]
        if legacy:
            old = legacy.get(name, {})
            if old.get("not_available"):
                row += ["not in it", "not in it"]
            else:
                row += [_f(old.get("seconds_median"), 2), _f(old.get("peak_rss_mb"), 0)]
        rows.append(row)
    lines += _table(header, rows)
    exe = data.get("executable")
    if exe:
        s = exe["split_with_empty_path"]
        lines += [
            f"One-file executable `{exe['path']}`: {exe['size_mb']:.0f} MB, starts in "
            f"{exe['start_seconds_median']:.2f} s (median; the first start {exe['start_seconds_first']:.2f} s); "
            f"a split with an empty PATH took {s['seconds']:.1f} s, exit code {s['exit_code']}, "
            f"{s['chunks_written']} files written.",
            "",
        ]
    return lines


def _backend(data: Dict[str, Any]) -> List[str]:
    lines = ["## 6. The whole program with a real backend", ""]
    if "skipped" in data:
        return lines + [f"Not run: {data['skipped']}.", ""]
    lines += [f"`{data['command']}`", ""]
    first, second = data["first_run"], data["second_run"]
    lines += [
        f"First run: {first['seconds']:.0f} s, exit code {first['exit_code']}. Second run (nothing left "
        f"to do): {second['seconds']:.1f} s, exit code {second['exit_code']}.",
        "",
    ]
    if "error" in data:
        return lines + ["Error: " + " / ".join(data["error"]), ""]
    rows = [
        [stem, _f(s.get("vs_whole_db")), _f(s.get("vs_truth_db")), _f(s.get("whole_vs_truth_db"))]
        for stem, s in data["scores"].items()
    ]
    lines += _table(
        ["stem", "vs whole-song separation", "vs truth", "(whole-song separation vs truth)"], rows
    )
    return lines


def write(folder: str) -> str:
    env = _load(folder, "environment")
    lines = ["# SeamSplit benchmark results", ""]
    lines += [
        "Generated by `python benchmarks/run.py report` from the JSON files next to this one. "
        "Read `../README.md` first: it says what is measured, what the numbers are not, and what to "
        "conclude from them.",
        "",
    ]
    if env:
        lines += _environment(env)
    for name, writer in (
        ("seams", _seams),
        ("cuts", _cuts),
        ("pairing", _pairing),
        ("degraded", _degraded),
        ("perf", _perf),
        ("backend", _backend),
    ):
        data = _load(folder, name)
        if data is not None:
            lines += writer(data)
    path = os.path.join(folder, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path
