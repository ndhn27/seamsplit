"""Stand-ins for the separators ``seamsplit run --backend`` starts, to test the backends
without Demucs or UVR: they take the command line of the real program, split the chunk
into the two stems of ``fake_service.separate`` and write them under the names the real
program uses.

    python fake_programs.py demucs --two-stems=vocals -o OUT [-n MODEL] FILE
    python fake_programs.py audio-separator FILE --output_dir OUT [--model_filename M] ...

What they do can be changed with environment variables: ``FAKE_FAIL_ON`` (a part of a
file name: exit with an error on those), ``FAKE_BAD_OUTPUT`` (write a file that is neither
stem), ``FAKE_SLEEP`` (seconds to wait first) and ``FAKE_LOG`` (a file to append each file
name that was processed to)."""

import os
import sys
import time
import zlib

import numpy as np
from seamsplit.audio import AudioSegment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_service import separate, to_array, to_segment  # noqa: E402


def _write(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_segment(x).export(path, format="wav")


def main(argv):
    mode, args = argv[0], argv[1:]
    out = model = source = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-o", "--output_dir"):
            out, i = args[i + 1], i + 2
        elif a in ("-n", "--model_filename"):
            model, i = args[i + 1], i + 2
        elif a == "--output_format":
            i += 2
        elif a.startswith("-"):
            i += 1
        else:
            source, i = a, i + 1
    name = os.path.basename(source)
    time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
    if os.environ.get("FAKE_LOG"):
        with open(os.environ["FAKE_LOG"], "a") as f:
            f.write(name + "\n")
    fail = os.environ.get("FAKE_FAIL_ON")
    if fail and fail in name:
        print("CUDA out of memory (pretend)", file=sys.stderr)
        return 1
    chunk = to_array(AudioSegment.from_file(source))
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    instrumental, vocals = separate(chunk, rng)
    stem = os.path.splitext(name)[0]
    if os.environ.get("FAKE_BAD_OUTPUT"):
        _write(os.path.join(out, "something_else.wav"), chunk)
    elif mode == "demucs":
        folder = os.path.join(out, model or "htdemucs", stem)
        _write(os.path.join(folder, "vocals.wav"), vocals)
        _write(os.path.join(folder, "no_vocals.wav"), instrumental)
    else:
        tag = (model or "default").replace(".", "_")
        _write(os.path.join(out, f"{stem}_(Vocals)_{tag}.wav"), vocals)
        _write(os.path.join(out, f"{stem}_(Instrumental)_{tag}.wav"), instrumental)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
