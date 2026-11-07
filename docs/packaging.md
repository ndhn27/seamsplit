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

## Running a downloaded program

A release has one program per system: `seamsplit-linux-x86_64`, `seamsplit-windows-x86_64.exe`,
`seamsplit-macos-arm64` (Apple silicon) and `seamsplit-macos-x86_64` (Intel). Each takes the
same commands as the installed `seamsplit` (`gui`, `split`, `merge`, `run`, ...). Give it a
command: started with none, as by a double-click, it prints a usage error and its window closes
at once. The programs are not code-signed, so Windows and macOS warn about them the first time.

**Check the download.** Put the program, its `BUILD_INFO-<system>.txt` and its
`SHA256SUMS-<system>.txt` in one folder, then:

```bash
sha256sum -c SHA256SUMS-linux-x86_64.txt           # Linux
shasum -a 256 -c SHA256SUMS-macos-arm64.txt        # macOS
```

```powershell
Get-FileHash .\seamsplit-windows-x86_64.exe -Algorithm SHA256   # Windows
```

Linux and macOS print `OK` for each file. On Windows, compare the hash with the line for the
`.exe` in `SHA256SUMS-windows-x86_64.txt`.

**Linux.** A downloaded file is not executable:

```bash
chmod +x seamsplit-linux-x86_64
./seamsplit-linux-x86_64 gui
```

**macOS.** Make it executable the same way. Gatekeeper blocks a downloaded program that Apple has
not signed and notarized ("cannot be opened because the developer cannot be verified"). Remove
the quarantine mark the browser put on it, once, and run it:

```bash
chmod +x seamsplit-macos-arm64
xattr -d com.apple.quarantine seamsplit-macos-arm64
./seamsplit-macos-arm64 gui
```

Instead of `xattr`, you can press *Open Anyway* in *System Settings, Privacy & Security* after
the first blocked attempt.

**Windows.** Open PowerShell in the folder and run `.\seamsplit-windows-x86_64.exe gui`. If
SmartScreen says "Windows protected your PC", choose *More info*, then *Run anyway*. To skip the
warning for this file, run `Unblock-File .\seamsplit-windows-x86_64.exe` first.

The first start takes several seconds while the program unpacks itself; the next ones are faster.

