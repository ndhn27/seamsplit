# Changelog

All notable changes to SeamSplit are written here, newest first, from the point of view of
someone who uses or develops it. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [1.1.1] - 2026-11-12

### Fixed

- A job could not be resumed after a `run` was killed in the middle of its split: `job.json` was
  only written once the split had finished, so the folder held chunks but no job and the next run
  refused it until it was emptied by hand. `job.json` is now written before the split. Running
  again redoes an interrupted split (it removes only the chunk files the split made, and the
  folder's other files still stop it), and `manifest.json`, the file that says a split is
  complete, is written all at once. `status` on such a job says it was interrupted.
- The GUI could delete the user's work too easily. *Split* wiped the processed chunks, the merged
  result, the listening kit (with its answers) and a dropped manifest without asking, and the
  docs only mentioned the chunks. The page now lists what is about to go and asks; the server
  answers `409` with the counts unless the request says `replace: true`. *Clear* on the processed
  chunks asks too.
- Clearing, uploading (a file or a manifest) or recording a listening answer while a split or
  merge was running could pull its files out from under it: a `/api/clear` in the middle of a
  merge ended the job with `FileNotFoundError`. They are now refused with `409` until the job
  ends, and the check and the job start are made under one lock, so none can slip in between.
- `merge`, `run` and the GUI no longer resample to 44.1 kHz by default: the result keeps the sample
  rate of the chunks, so a 48 kHz source stays at 48 kHz. `--samplerate` (and the page's *Sample
  rate*) still resamples when you ask for a rate. Chunks that disagree on rate or channel count
  are now reported when they are converted to match the first one.
- `estimate_lag` no longer settles for a shift of 0 when the true one is a few samples away.
  It treated every shift within 0.01 of the best correlation as a tie and took the one closest
  to 0, which is meant for periodic signals but on bass-heavy material (or after a lossy
  encode) swallows an offset of up to several samples. Neighbouring chunks were then merged
  with that offset left in, it added up along the track (a merge of 20 minutes ended 11 samples
  longer than its source, with a seam 30 dB worse than the rest), and `--anchor`, which measures
  the offset the same way, did not catch it. The tie is now 1e-4 (`LAG_TIE_TOLERANCE` in
  `dsp.py`); a pure tone still gets the lag nearest to 0. Covered by new tests on both paths.

### Changed

- The page's *Crossfade curve* no longer lists `equalpower` (still accepted by the command line
  and the API): in the benchmark it is a median 16 dB worse than a hard cut at the seams of real
  separated chunks. The README, the `--method` help and the merge docs say so, and the README
  now says what the benchmark does and does not support (the overlap; not the crossfade shape,
  seam search, `--anchor` or voice-band cuts) and how small it is.
- Chunks are now split to **FLAC by default** (`--format`, `SplitConfig.export_format` and the
  split format of the interface). The default was MP3, which adds encoder delay and padding and
  cost the seams about 10 dB of SER against FLAC on percussive music, even at 320k. Pass
  `--format mp3` (or pick MP3 in the interface, or set `"format": "mp3"` in a service preset)
  if your service accepts nothing lossless; the length `--margin` then goes back to its lossy
  default of 0.25 s. `--limit` no longer reserves that margin by default.
- `seamsplit.merger` no longer re-exports the 32 private helpers (names with a leading
  underscore, such as `_crossfade_merge`, `_adaptive_gains` or `_check_number`) of the modules it
  ties together. Before, a test pinned all of them as if they were its interface. Its interface
  is now `merge_audio` and the public names listed in `seamsplit.merger.__all__`, which still
  import from there as before; code that needs a private helper imports it from the module that
  defines it (`seamsplit.dsp`, `seamsplit.crossfade`, `seamsplit.checks`, ...). The five that
  `merge_audio` itself calls are still imported there, but are not part of its interface.

### Documentation

- The docs said the `audio-separator` preset had not been run against the real program. It has:
  the benchmark's experiment 6 runs `seamsplit run` with audio-separator 0.48.0 and the
  `UVR-MDX-NET-Voc_FT.onnx` model. `docs/jobs-and-backends.md` and the docstring of
  `backends.py` now say that (one song and one model) and that only the `demucs` preset is
  still tested against a stand-in alone.
- `docs/packaging.md` says what the SHA-256 sums do (show a download is whole) and what they do
  not (show it was not replaced), and how to check a program's build attestation.

### Development

- The version is now **1.1.1**. `1.1.0a` was written after `1.1.0`, but under PEP 440 it is
  older than `1.1.0`, so pip would never have upgraded to it (and `1.1.0a0` is a pre-release,
  which does not go with the *Production/Stable* classifier). The `1.1.0a` entry below stays as
  the record of what was changed after `1.1.0`. `tests/test_version.py` now fails when
  `pyproject.toml` and `seamsplit.__version__` differ, when the version is not spelled the way
  PEP 440 normalizes it, or when it sorts before a release listed in this file.
- The merge code is split into steps. `_crossfade_merge` was one function of over 200 lines
  whose nested helpers shared state through `nonlocal`; it is now a loop of about 70 lines over
  small steps (`_measure`, `_place_fade`, `_blend`, and `_Settled`, which keeps the audio that is
  final). The body of `merge_audio` (about 130 lines, now about 60) calls one function for each stage: checking
  the numbers, checking the options that name files, the overlap, pairing the stems,
  checking the pairing, the length check, exporting. Its 25 keyword arguments are unchanged, as
  are their order, their defaults and its docstring: the CLI, the GUI, jobs and library users
  all call it by keyword, and changing that is a decision for a major version. The merged audio,
  the report, the warnings (and where a warning without a report points) and the errors (and
  which of several is raised first) were checked to be identical, by running 568 cases before
  and after: 400 random merges over different chunk formats, offsets, levels, curves and
  options, the error cases, and `merge_audio` end to end (one and two stems, anchoring, pairing
  by order, listening kits, a manifest that disagrees with the audio). The peak memory of a
  merge is 5–13 % lower, because the measuring arrays are freed after each boundary.
  `tests/test_merge_stages.py` pins what the split must not change.
- Release workflow: every action is pinned to a commit instead of a tag (the version is in a
  comment next to it), PyInstaller to the version in `packaging/requirements-build.txt`, and
  `.github/dependabot.yml` proposes updates to both. The same major version of each action is
  kept as before.
- The programs of a release get a signed build attestation (`actions/attest`, GitHub's
  Sigstore signing), which `gh attestation verify` checks. The SHA-256 sums next to the files
  could not do that: whoever replaces a file can replace its sum.

## [1.1.0a] - 2026-11-07

### Fixed

- In dark mode the checkboxes, the progress bar, the spin buttons of the number fields and the
  scrollbars now follow the dark theme too (`color-scheme` is set on both the interface and the
  listening page).
- The two drop zones can be reached with the Tab key and opened with Enter or Space; the one
  with the focus is highlighted, and other controls get a visible focus ring. Before, the file
  inputs were `hidden`, which takes them out of the tab order.
- The Merge card no longer ends in a blank strip: an empty status line takes no room.
- The line under the Split and Merge cards (for example "Done: 5 chunks.") now changes
  language with the page. Before, it stayed in the language it was written in.
- The Vietnamese page calls a chunk "chunk" everywhere, as the English page and the file names
  do (it mixed it with "đoạn"), and the two split options "Before-slack" and "After-slack" are
  now "Rút ngắn tối đa" and "Kéo dài tối đa".
- The page has an icon of its own in the browser tab. Before, the browser asked the server for
  `/favicon.ico` without the token and got a `403`, which showed as a failed request in the
  browser's developer tools.

### Documentation

- `docs/packaging.md`: how to run a downloaded program (the executable bit, macOS Gatekeeper,
  Windows SmartScreen) and how to check its SHA-256.
- This file now has the 1.0.0 entry and the comparison links at the bottom.

### Development

- `pyproject.toml` has the `Development Status` and `Intended Audience` classifiers and links
  to the documentation and this changelog.
- The description of a GitHub release is now that version's section of this file. The release
  workflow stops if the section is missing.
- `SHA256SUMS-*.txt` no longer lists itself (its own sum could never match, so `sha256sum -c`
  reported it as FAILED). The release build now checks the file it has just written.

## [1.1.0] - 2026-11-07

### Fixed

- Exporting a WAV file to a place that cannot be written (for example a folder that does not
  exist) still fails with `CouldntEncodeError`, but no longer prints an
  `Exception ignored in: Wave_write.__del__` warning on the way. The file is now opened by
  SeamSplit instead of by `wave.open`, which could not clean up after a failed open.

### Development

- The listening page is now tested in a real browser on every CI run. Before, `playwright` was
  not a development dependency and no browser was installed, so that test was always skipped
  in CI. CI now installs Chromium, and `playwright` is part of the `dev` extras.
- When the `CI` environment variable is set, a missing Playwright or browser makes that test
  fail instead of skip, so it cannot silently stop running again. On a local machine it still
  skips; run `python -m playwright install chromium` once to let it run (see
  [CONTRIBUTING.md](https://github.com/ndhn27/seamsplit/blob/main/CONTRIBUTING.md)).
- A regression test fails if the `Wave_write.__del__` warning comes back.
- The tests no longer leave files open (11 `ResourceWarning`s removed), so a run with
  `-W default` is clean.
- The release smoke test (`packaging/smoke_test.py`) no longer changes its own working
  directory into its temporary folder. Windows cannot delete a folder that is some process's
  current directory, so the cleanup failed there and the Windows release build with it.

## [1.0.0]

### Added

- First release.

[Unreleased]: https://github.com/ndhn27/seamsplit/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/ndhn27/seamsplit/compare/v1.1.0a...v1.1.1
[1.1.0a]: https://github.com/ndhn27/seamsplit/compare/v1.1.0...v1.1.0a
[1.1.0]: https://github.com/ndhn27/seamsplit/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ndhn27/seamsplit/releases/tag/v1.0.0
