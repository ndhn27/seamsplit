# One-file programs

[← Back to the README](../README.md)

`packaging/seamsplit.spec` builds a single program per system with
[PyInstaller](https://pyinstaller.org/) (Python, numpy, the pages of the interface and PyAV's
FFmpeg inside; nothing to install, nothing on the `PATH` needed):

```bash
pip install . pyinstaller
pyinstaller packaging/seamsplit.spec --noconfirm     # dist/seamsplit, dist/seamsplit.exe on Windows
python packaging/smoke_test.py dist/seamsplit        # splits and merges with an empty PATH
```

The release workflow (`.github/workflows/release.yml`) builds them for Linux, Windows and macOS
(Apple silicon and Intel) when a tag `v*` is pushed, or on demand, and runs the smoke test on
each. A release gets the programs with their SHA-256 sums, a `BUILD_INFO` file per system (the
PyAV and FFmpeg versions, licence and options inside the program) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The v1.1.0 release has all four: each was
built by the release workflow and passed its smoke test (a split and a merge with an empty
`PATH`) on a GitHub-hosted runner of its own system. The programs are about 74 MB on Linux, 48 MB
on Windows, 31 MB on macOS (Apple silicon) and 37 MB on macOS (Intel). The start-up times in the
[benchmarks](../benchmarks/README.md) (about 1.3 s, 7 s the first time, while it unpacks) were
measured on a 61 MB Linux build, so they may differ for the release.
