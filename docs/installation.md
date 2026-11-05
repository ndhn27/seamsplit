# Installation

[← Back to the README](../README.md)

Requires Python 3.10+. Nothing else has to be installed: audio files (WAV, FLAC, MP3, Ogg, Opus,
AAC/M4A, AIFF, ...) are read and written inside the process by [PyAV](https://pyav.org/), whose
wheels carry the FFmpeg libraries, so there is no `ffmpeg` or `ffprobe` to find on your `PATH`
(and none is used if there is one). WAV is written with Python's own `wave` module and read by
PyAV like the other formats (24 bit audio is held as 32 bit). The newest PyAV needs Python 3.12
or later; on older Pythons pip picks the newest one that supports them (SeamSplit is tested with
14.0 and with the current release). PyAV's wheels carry FFmpeg together with libx264 and
libx265, which are GPL-licensed. SeamSplit uses them only for audio and its own code stays
MIT-licensed, but a program that bundles them has to follow their licences (the
[one-file programs](packaging.md) are such a program): see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

```bash
git clone https://github.com/ndhn27/seamsplit.git
cd seamsplit
pip install -e .
```
