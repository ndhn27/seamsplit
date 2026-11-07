# Changelog

All notable changes to SeamSplit are written here, newest first, from the point of view of
someone who uses or develops it. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
  [CONTRIBUTING.md](CONTRIBUTING.md)).
- A regression test fails if the `Wave_write.__del__` warning comes back.
- The tests no longer leave files open (11 `ResourceWarning`s removed), so a run with
  `-W default` is clean.
