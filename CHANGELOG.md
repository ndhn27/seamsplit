# Changelog

All notable changes to SeamSplit are written here, newest first, from the point of view of
someone who uses or develops it. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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

[Unreleased]: https://github.com/ndhn27/seamsplit/compare/v1.1.0a...HEAD
[1.1.0a]: https://github.com/ndhn27/seamsplit/compare/v1.1.0...v1.1.0a
[1.1.0]: https://github.com/ndhn27/seamsplit/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ndhn27/seamsplit/releases/tag/v1.0.0
