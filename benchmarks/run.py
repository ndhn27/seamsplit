#!/usr/bin/env python3
"""Benchmarks of SeamSplit on real music. Run from the repository root:

    python benchmarks/run.py fetch                      # download the songs (checked by hash)
    python benchmarks/run.py all --separator-python /path/to/venv/bin/python
    python benchmarks/run.py seams|cuts|pairing|degraded|perf|backend|report

``--separator-python`` is the Python of an environment that has ``audio-separator``
installed (``pip install "audio-separator[cpu]"``); the model it uses is downloaded into
``benchmarks/cache/models`` on first use. Separated files are kept in ``benchmarks/cache``
and used again. ``--fake`` replaces the model by a trivial filter, to try the harness out in
seconds; its numbers go to ``results-fake/`` and mean nothing.

The numbers are written to ``benchmarks/results/*.json`` and ``RESULTS.md``; see
``benchmarks/README.md`` for what they are and what they are not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EXPERIMENTS = ("seams", "cuts", "pairing", "degraded", "perf", "backend")


def _write(out_dir: str, name: str, data) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=False, default=float)
        f.write("\n")
    print(f"  wrote {name}.json", flush=True)


def _read(out_dir: str, name: str):
    path = os.path.join(out_dir, name + ".json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=("fetch", "all", "report") + EXPERIMENTS + ("_phase",))
    p.add_argument("rest", nargs="*", help=argparse.SUPPRESS)
    p.add_argument("--separator-python", help="Python with audio-separator installed")
    p.add_argument(
        "--model", default=None, help="UVR model file (default: UVR-MDX-NET-Voc_FT.onnx)"
    )
    p.add_argument("--fake", action="store_true", help="use a trivial filter instead of a model")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--corpus-dir", default=os.path.join(HERE, "corpus"))
    p.add_argument("--out", default=None, help="where the results go (default: benchmarks/results)")
    p.add_argument("--songs", default=None, help="comma-separated song ids (default: all)")
    p.add_argument("--trials", type=int, default=4, help="renamings per stem in the pairing trial")
    p.add_argument("--exe", default=None, help="one-file executable to measure (perf)")
    p.add_argument(
        "--legacy-src",
        default=None,
        help="src/ of another version of the package, timed on the same phases (perf)",
    )
    p.add_argument("--repeats", type=int, default=3, help="repeats of each timing (perf)")
    return p


def _separator(args):
    if args.fake:
        from bench.fake import FakeSeparator

        return FakeSeparator(args.cache_dir)
    from bench.separation import DEFAULT_CACHE_DIR, DEFAULT_MODEL, LocalSeparator

    if not args.separator_python:
        sys.exit("--separator-python is needed (or --fake, to try the harness out)")
    return LocalSeparator(
        args.separator_python,
        args.model or DEFAULT_MODEL,
        args.cache_dir or DEFAULT_CACHE_DIR,
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "_phase":  # run by perf.measure in a child process
        from bench import perf

        name, inputs, scratch = args.rest
        print(json.dumps(perf.run_phase_here(name, json.loads(inputs), scratch)))
        return 0

    from bench import corpus as corpus_mod  # (not before: a phase may run without seamsplit.audio)

    corpus = corpus_mod.load_corpus()
    if args.command == "fetch":
        corpus_mod.fetch(corpus, args.corpus_dir)
        print("every file is there and matches its hash")
        return 0

    out_dir = args.out or os.path.join(HERE, "results-fake" if args.fake else "results")
    cache_dir = args.cache_dir or os.path.join(HERE, "cache-fake" if args.fake else "cache")
    args.cache_dir = cache_dir
    only = args.songs.split(",") if args.songs else None

    if args.command == "report":
        from bench import report

        report.write(out_dir)
        print(f"  wrote {os.path.join(out_dir, 'RESULTS.md')}")
        return 0

    wanted = list(EXPERIMENTS) if args.command == "all" else [args.command]
    corpus_mod.fetch(corpus, args.corpus_dir)
    songs = corpus_mod.load_songs(corpus, args.corpus_dir, only)

    from bench import envinfo, workset

    work_dir = os.path.join(cache_dir, "work")
    needs_stems = [e for e in wanted if e in ("seams", "cuts", "pairing", "degraded")]
    works = None
    separator = None
    if needs_stems or "backend" in wanted:
        separator = _separator(args)
    if needs_stems:
        print("cutting the songs and separating the chunks", flush=True)
        works = workset.prepare(songs, work_dir)
        workset.separate_all(works, separator)

    env = envinfo.environment()
    if separator is not None:
        from bench.separation import model_fingerprint, separator_versions

        env["separator"] = {
            "name": separator.name,
            "model_sha256": model_fingerprint(separator),
            "load_seconds": getattr(separator, "loaded_s", None),
        }
        if works is not None:
            env["separator"]["speed"] = workset.separation_speed(works)
        if not args.fake:
            env["separator"]["versions"] = separator_versions(args.separator_python)
    env["songs"] = {
        s.id: {"title": s.title, "seconds": round(s.seconds, 2), "has_truth": s.vocals is not None}
        for s in songs
    }
    _write(out_dir, "environment", env)

    scratch_out = os.path.join(cache_dir, "merged")
    seams_result = None
    if works is not None and ("seams" in wanted or "cuts" in wanted):
        from bench import seams

        print("seams", flush=True)
        seams_result = seams.run(works, work_dir, scratch_out)  # the cuts are judged by it too
        if "seams" in wanted:
            _write(
                out_dir,
                "seams",
                {
                    "songs": seams_result,
                    "summary": seams.summarise(seams_result),
                    "worst_seams": seams.worst_seams(seams_result),
                },
            )
    if "cuts" in wanted and works is not None:
        from bench import cuts

        print("cuts", flush=True)
        result = cuts.run(works, seams_result or {})
        _write(out_dir, "cuts", {"songs": result, "summary": cuts.summarise(result)})
    if "pairing" in wanted and works is not None:
        from bench import pairing

        print("pairing", flush=True)
        result = pairing.run(works, work_dir, trials=args.trials)
        _write(out_dir, "pairing", {"songs": result, "summary": pairing.summarise(result)})
    if "degraded" in wanted and works is not None:
        from bench import degraded

        print("degraded (simulated service damage)", flush=True)
        result = degraded.run(works, work_dir, scratch_out)
        _write(out_dir, "degraded", {"songs": result, "summary": degraded.summarise(result)})
    if "perf" in wanted:
        from bench import perf

        print("perf", flush=True)
        result = perf.run(songs, cache_dir, args.legacy_src, args.repeats)
        if args.exe:
            result["executable"] = perf.run_executable(args.exe, songs[0], cache_dir)
        _write(out_dir, "perf", result)
    if "backend" in wanted:
        from bench import backend

        print("backend", flush=True)
        _write(out_dir, "backend", backend.run(songs, separator, cache_dir, args.separator_python))

    if args.command == "all":
        from bench import report

        report.write(out_dir)
        print(f"  wrote {os.path.join(out_dir, 'RESULTS.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
