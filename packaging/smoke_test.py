"""Run a built program through a split and a merge with no ffmpeg anywhere on PATH.

    python packaging/smoke_test.py dist/seamsplit        (dist\\seamsplit.exe on Windows)

It writes a short song to a temporary folder, splits it into FLAC and MP3 chunks, merges
the FLAC chunks and checks that the result is as long as the song. The program is run with
an empty PATH, so that if it still needed an ffmpeg program it would fail here.
"""

import math
import os
import struct
import subprocess
import sys
import tempfile
import wave

RATE = 22050
SECONDS = 50


def write_song(path: str) -> None:
    frames = bytearray()
    for n in range(RATE * SECONDS):
        t = n / RATE
        beat = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t)  # a pulse, so cuts have somewhere to go
        tone = beat * 0.3 * math.sin(2 * math.pi * 220 * t)
        sample = tone + 0.1 * math.sin(2 * math.pi * 660 * t)
        frames += struct.pack("<h", int(sample * 32767))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))


def run(exe: str, *args: str) -> str:
    env = {
        "PATH": "",
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "HOME": tempfile.gettempdir(),
    }
    done = subprocess.run([exe, *args], env=env, capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        sys.exit(f"FAILED: {' '.join(args)}\n{done.stdout}\n{done.stderr}")
    return done.stdout


def wav_seconds(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    exe = os.path.abspath(sys.argv[1])
    with tempfile.TemporaryDirectory() as work:
        os.chdir(work)
        write_song("song.wav")
        print(run(exe, "--version").strip())
        run(
            exe,
            "split",
            "song.wav",
            "--target",
            "20",
            "--before-slack",
            "5",
            "--overlap",
            "2",
            "--format",
            "flac",
            "--out",
            "flac_chunks",
        )
        run(
            exe,
            "split",
            "song.wav",
            "--limit",
            "30",
            "--before-slack",
            "5",
            "--format",
            "mp3",
            "--bitrate",
            "192k",
            "--out",
            "mp3_chunks",
        )
        flacs = [n for n in os.listdir("flac_chunks") if n.endswith(".flac")]
        mp3s = [n for n in os.listdir("mp3_chunks") if n.endswith(".mp3")]
        assert len(flacs) >= 2 and len(mp3s) >= 2, (flacs, mp3s)
        run(
            exe,
            "merge",
            "flac_chunks",
            "--manifest",
            os.path.join("flac_chunks", "manifest.json"),
            "--out",
            "merged.wav",
        )
        length = wav_seconds("merged.wav")
        assert abs(length - SECONDS) < 0.01, f"merged is {length}s, the song is {SECONDS}s"
        print(
            f"ok: {len(flacs)} flac and {len(mp3s)} mp3 chunks, "
            f"merged {length:.2f}s of {SECONDS}s"
        )


if __name__ == "__main__":
    main()
