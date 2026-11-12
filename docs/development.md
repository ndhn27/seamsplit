# Development

[← Back to the README](../README.md)

```bash
pip install -e ".[dev]"
pytest --cov
flake8 src tests benchmarks
black --check src tests benchmarks
mypy
```

The code is split by what it does, each module small enough to read:

| Module | What is in it |
|---|---|
| `splitter` | Cutting a track: the loudness curve, the search for cut points, the chunks and the manifest. |
| `cutpoints` | What the search can know besides loudness: voice-band activity and beat tracking. |
| `merger` | `merge_audio`, the entry point of a merge: it checks the options, pairs the files and exports, with a function for each of those stages; it also re-exports the public names the other modules define for a merge (`__all__` lists them). Names with a leading underscore are private to their module: import them from there. |
| `manifest`, `checks` | Reading `manifest.json`; checking the numbers a caller passes. |
| `matching` | Pairing the manifest's parts with the files that came back, by name or by order. |
| `pairing` | Telling which chunk a file is by comparing it with the original chunks. |
| `services`, `splitopts` | Service presets; the options that decide how a file is cut, shared by `split` and `run`. |
| `status`, `job`, `backends`, `cli_jobs` | A job folder: what is missing, the resumable run, the programs that separate on this computer, and the `run` and `status` commands. |
| `crossfade` | The merge itself: alignment, level matching, seam search, fades. `_crossfade_merge` is a loop of one step per boundary (`_measure`, `_place_fade`, `_blend`, `_Settled`). What is to be reported is returned by those steps and emitted by `_crossfade_merge` (and, in `merger`, by `merge_audio`) itself: without a report it becomes a warning that `_emit` points at the caller of the function it is called from, which has to be the caller of the merge. `tests/test_merge_stages.py` pins it. |
| `anchor` | Comparing each chunk with its original, and merging both stems of a service as one. |
| `plan` | `MergePlan`: a merge that has been set up (files, stems, options), which can be run over all the chunks or over just the two of a seam. |
| `audition`, `choices` | The listening kit of a merge, and what is done with the answers given to it. |
| `dsp` | The signal processing it is built from, on plain arrays. |
| `report` | The seam report. |
| `audio`, `audio_io` | `AudioSegment` (the sample arrays everything else works on) and reading and writing files through PyAV: formats, sample widths, resampling, padding-aware durations. |
| `cli`, `gui` | The command line and the browser page. |

CI runs this on Python 3.10 to 3.14. black is pinned to one major version
(`black>=26,<27`, and the same in `.pre-commit-config.yaml`): its style only
changes once a year, so a new black release cannot turn CI red on its own.
When you move to the next major, change both places and run `black src tests`.

Optional: `pip install pre-commit && pre-commit install` runs black, flake8 and
mypy before each commit.
