# Third-party notices

SeamSplit's own code is under the MIT License ([LICENSE](LICENSE)).

`pip install seamsplit` installs numpy and PyAV as separate packages, each under its own
licence. The one-file programs attached to a release (built from `packaging/seamsplit.spec`)
carry copies of them, and of the native libraries inside PyAV's wheel, so the notices below
apply to those programs. Every program comes with a `BUILD_INFO-<system>.txt` that names the
PyAV and FFmpeg versions it was built with and lists the libraries it carries. The tables
below were read from the PyAV 19.0.2 wheels for Linux, Windows and macOS; the licences are the
ones their projects state.

## In every program

| Component | What it does here | Licence | Source |
|---|---|---|---|
| CPython | the interpreter | PSF-2.0 | <https://www.python.org/downloads/source/> |
| NumPy (with the libraries its wheel carries; its own licence file lists them) | the samples | BSD-3-Clause | <https://github.com/numpy/numpy> |
| PyAV | Python bindings to FFmpeg | BSD-3-Clause | <https://github.com/PyAV-Org/PyAV> |
| PyInstaller (its bootloader starts the program) | packaging | GPL-2.0-or-later with an exception that lets the program it bundles be distributed under any licence | <https://github.com/pyinstaller/pyinstaller> |
| FFmpeg: libavcodec, libavformat, libavutil, libavfilter, libavdevice, libswresample, libswscale | decoding and encoding | LGPL-3.0-or-later, as the build reports itself (see below) | <https://ffmpeg.org/download.html> |
| LAME (built into libavcodec) | MP3 encoding | LGPL-2.0-or-later | <https://lame.sourceforge.io/> |
| libx264 | H.264 encoder; linked into FFmpeg, not used by SeamSplit | GPL-2.0-or-later | <https://code.videolan.org/videolan/x264> |
| libx265 | HEVC encoder; linked into FFmpeg, not used by SeamSplit | GPL-2.0-or-later (its authors also sell a commercial licence) | <https://bitbucket.org/multicoreware/x265_git> |
| Opus | audio codec | BSD-3-Clause | <https://opus-codec.org/> |
| libvpx | VP8/VP9 codec | BSD-3-Clause | <https://chromium.googlesource.com/webm/libvpx> |
| dav1d | AV1 decoder | BSD-2-Clause | <https://code.videolan.org/videolan/dav1d> |
| SVT-AV1 | AV1 encoder | BSD-3-Clause-Clear, with the AOMedia patent licence | <https://gitlab.com/AOMediaCodec/SVT-AV1> |
| libvmaf | video quality metric | BSD-2-Clause-Patent | <https://github.com/Netflix/vmaf> |
| libwebp, libsharpyuv | WebP | BSD-3-Clause | <https://chromium.googlesource.com/webm/libwebp> |
| oneVPL dispatcher (libvpl) | hardware video access | MIT | <https://github.com/intel/libvpl> |
| zlib | compression | Zlib | <https://zlib.net/> |

## Linux programs also carry

| Component | Licence | Source |
|---|---|---|
| GnuTLS | LGPL-2.1-or-later | <https://www.gnutls.org/> |
| Nettle and Hogweed | LGPL-3.0-or-later or GPL-2.0-or-later | <https://git.lysator.liu.se/nettle/nettle> |
| GMP | LGPL-3.0-or-later or GPL-2.0-or-later | <https://gmplib.org/> |
| libunistring | LGPL-3.0-or-later or GPL-2.0-or-later | <https://www.gnu.org/software/libunistring/> |
| ALSA library | LGPL-2.1-or-later | <https://www.alsa-project.org/> |
| libdrm, libxcb, libXau | MIT | <https://gitlab.freedesktop.org/mesa/drm>, <https://gitlab.freedesktop.org/xorg/lib/libxcb>, <https://gitlab.freedesktop.org/xorg/lib/libxau> |

## Windows programs also carry

| Component | Licence | Source |
|---|---|---|
| libiconv | LGPL-2.1-or-later | <https://www.gnu.org/software/libiconv/> |
| libstdc++ and libgcc (MinGW-w64 runtime) | GPL-3.0-or-later with the GCC Runtime Library Exception | <https://gcc.gnu.org/> |
| winpthreads | MIT-style, from MinGW-w64 | <https://www.mingw-w64.org/> |

## FFmpeg, libx264 and libx265

The FFmpeg in PyAV's wheels is configured with `--enable-version3`, `--enable-libx264` and
`--enable-libx265`, and without `--enable-gpl`; its libraries report themselves as "LGPL
version 3 or later". The Linux, Windows and macOS (Intel and Apple silicon) wheels all say the
same. libx264 and libx265 are not LGPL: they are GPL-licensed libraries, and in these builds
libavcodec is linked against them. SeamSplit only reads and writes audio and does not use the
video encoders, but the libraries are inside the programs.

This file does not decide what that means for redistributing the programs. It lists every part
with its licence so that the question can be answered, and says where the sources are.

## Source code

The native libraries come from PyAV's own build project,
[pyav-ffmpeg](https://github.com/PyAV-Org/pyav-ffmpeg), which holds the build recipe and the
version of every library it builds (FFmpeg, x264, x265 and the others). The project of each
component is linked above. The PyAV and FFmpeg versions of a given program are in its
`BUILD_INFO` file. If a source cannot be found at these places, ask in
<https://github.com/ndhn27/seamsplit/issues> and it will be provided.
