"""Say what is inside a build: the versions of PyAV and FFmpeg, the licence FFmpeg reports
for itself, the options it was configured with and the libraries that travel with it.

    python packaging/build_info.py > BUILD_INFO.txt

The release workflow runs it next to each one-file program and attaches the output, so what
a download contains, and which sources belong to it, can be read from the release itself.
The licence and the options are read from the bytes of the bundled libavutil, so nothing is
loaded and it works the same on Linux, Windows and macOS.
"""

from __future__ import annotations

import os
import platform
import re
import sys
from typing import List, Optional, Tuple

LICENCE = re.compile(rb"(?:A|L)?GPL version [0-9](?:\.[0-9])? or later")
CONFIGURATION = re.compile(rb"(?:--[A-Za-z0-9_=/.,:'+-]+ ){4,}--[A-Za-z0-9_=/.,:'+-]+")


def library_folders(av_dir: str) -> List[str]:
    """Where PyAV's wheel keeps the native libraries: ``av.libs`` next to the package on
    Linux and Windows, ``av/.dylibs`` on macOS."""
    candidates = [os.path.join(os.path.dirname(av_dir), "av.libs"), os.path.join(av_dir, ".dylibs")]
    return [folder for folder in candidates if os.path.isdir(folder)]


def bundled_libraries(av_dir: str) -> List[str]:
    names: List[str] = []
    for folder in library_folders(av_dir):
        names += os.listdir(folder)
    return sorted(names)


def ffmpeg_licence_and_options(av_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """The licence line and the configure options stored in the bundled libavutil, or
    ``None`` for what is not found (PyAV built against a system FFmpeg has no bundled one)."""
    for folder in library_folders(av_dir):
        for name in sorted(os.listdir(folder)):
            if "avutil" not in name:
                continue
            with open(os.path.join(folder, name), "rb") as handle:
                data = handle.read()
            licence = LICENCE.search(data)
            options = sorted(CONFIGURATION.findall(data), key=len, reverse=True)
            return (
                licence.group().decode("ascii") if licence else None,
                options[0].decode("ascii") if options else None,
            )
    return None, None


def describe() -> str:
    import av

    av_dir = os.path.dirname(os.path.abspath(av.__file__))
    licence, options = ffmpeg_licence_and_options(av_dir)
    try:
        import seamsplit

        program = f"SeamSplit {seamsplit.__version__}"
    except ImportError:
        program = "SeamSplit (not installed)"
    lines = [
        program,
        f"Python {platform.python_version()} on {platform.platform()} ({platform.machine()})",
        f"PyAV {av.__version__}",
        f"FFmpeg {av.ffmpeg_version_info}",
        f"FFmpeg licence, as its own build reports it: {licence or 'not found'}",
        f"FFmpeg configuration: {options or 'not found'}",
        "",
        "Libraries carried by PyAV's wheel:",
    ]
    lines += [f"  {name}" for name in bundled_libraries(av_dir)] or ["  none found"]
    return "\n".join(lines) + "\n"


def main() -> int:
    sys.stdout.write(describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
