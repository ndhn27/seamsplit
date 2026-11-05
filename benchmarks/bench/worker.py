"""Runs in the environment where ``audio-separator`` is installed (not in SeamSplit's own):
loads one model once and separates every file of a job list, so that the model is not
loaded again for each chunk.

    python worker.py JOBS.json

``JOBS.json``: ``{"model": "UVR-MDX-NET-Voc_FT.onnx", "model_dir": "...", "jobs": [{"input":
"a.flac", "vocals": "a_vocals.flac", "instrumental": "a_instrumental.flac"}, ...]}``. One
line ``RESULT {json}`` is printed per file, with the seconds the separation took.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import time


def main(jobs_file: str) -> int:
    with open(jobs_file, "r", encoding="utf-8") as f:
        spec = json.load(f)
    from audio_separator.separator import Separator

    work = tempfile.mkdtemp(prefix="seamsplit-bench-")
    started = time.time()
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=spec["model_dir"],
        output_dir=work,
        output_format="FLAC",
        # no re-scaling of the stems, so that chunks and the whole song are treated alike
        normalization_threshold=1.0,
    )
    separator.load_model(model_filename=spec["model"])
    print("RESULT " + json.dumps({"loaded_s": round(time.time() - started, 2)}), flush=True)
    for job in spec["jobs"]:
        if os.path.isfile(job["vocals"]) and os.path.isfile(job["instrumental"]):
            # done by an earlier run that was stopped before it could say so
            print("RESULT " + json.dumps({"input": job["input"], "seconds": None}), flush=True)
            continue
        t0 = time.time()
        names = {"Vocals": "vocals_stem", "Instrumental": "instrumental_stem"}
        written = separator.separate(job["input"], custom_output_names=names)
        by_name = {os.path.basename(p): p for p in written}
        for stem, key in (("vocals_stem", "vocals"), ("instrumental_stem", "instrumental")):
            produced = by_name.get(stem + ".flac")
            if produced is None:
                raise SystemExit(f"{job['input']}: no {stem}.flac among {sorted(by_name)}")
            produced = produced if os.path.isabs(produced) else os.path.join(work, produced)
            os.makedirs(os.path.dirname(os.path.abspath(job[key])), exist_ok=True)
            shutil.move(produced, job[key])
        print(
            "RESULT " + json.dumps({"input": job["input"], "seconds": round(time.time() - t0, 2)}),
            flush=True,
        )
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
