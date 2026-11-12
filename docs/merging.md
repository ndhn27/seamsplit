# Merging the results

[← Back to the README](../README.md)

## Sending the chunks to your separator

Process every file in `chunks/` through your vocal/instrumental separation
service of choice and save the results into a new folder (e.g.
`processed/`). Don't drop or reorder any of them. `merge` finds each part's
file in one of two ways:

- **By name.** A file belongs to a part when its name contains the part's
  name without the extension (`song_part03` matches `song_part03_instrumental.wav`;
  `song_part10` does not match `song_part100`). This is how services that append
  a suffix such as `_instrumental` or `_vocals` work.
- **By order, if you renamed them.** When *no* file contains any part name
  (say, you renamed them `Song - 1.wav`, `Song - 2.wav`, …), `merge` pairs the
  audio files with the parts in natural sort order (`Song - 2` before
  `Song - 10`). That needs exactly one audio file per part, and it says which
  file it took as the first and the last part; check that they sort the way the
  parts were split. If only some parts are found by name, it stops instead of
  guessing, and lists **every** part that has no file. Pairing by order is a guess,
  so when the original chunks are next to the manifest (or in `--originals`) `merge`
  checks it: see [Which file is which chunk](#which-file-is-which-chunk).

When a service returns several files per chunk (vocals and instrumental),
narrow them down with `--match` — it applies to both ways.

## The `merge` command

```bash
seamsplit merge processed/ --manifest chunks/manifest.json --out "song_instrumental_full.mp3"
```

If a single source chunk produced more than one output file (e.g. both a
vocals and an instrumental version), narrow it down with `--match`:

```bash
seamsplit merge processed/ --manifest chunks/manifest.json \
    --match instrumental --out "song_instrumental_full.mp3"
```

Memory use is about the size of the merged audio itself (roughly 20 MB per
minute of 16-bit stereo at 44.1 kHz, plus about the same again while the result
is assembled), so a 20-minute song needs well under 1 GB. Chunks are decoded one
at a time, and only the few seconds around the seam being worked on are held as
floating point.

Key options:

| Option | Default | Description |
|---|---|---|
| `--manifest` | *(required)* | Path to the `manifest.json` produced by `seamsplit split`. |
| `--overlap` | *(from manifest)* | Override the overlap length used for crossfading. |
| `--match` | – | Filter which processed file to use when a part maps to several files. |
| `--method` | `adaptive` | Crossfade curve: `adaptive` (shaped by how alike the two overlaps are), `linear` or `equalpower` (a median 16 dB worse than a hard cut at the seams in the benchmark; the page does not offer it) — see [Choosing a crossfade curve](how-the-merge-works.md#choosing-a-crossfade-curve). |
| `--no-align` | off | Skip the timing alignment between neighboring chunks. |
| `--max-lag-ms` | `50` | Largest timing offset the alignment will look for. |
| `--no-match-level` | off | Skip the loudness matching between neighboring chunks. |
| `--max-gain-db` | `6` | Largest level correction applied to one chunk, in either direction. |
| `--seam-search` | off | **Experimental.** Make a short crossfade at the best-matching point of the overlap instead of fading across all of it. Not validated on real separator output — see [Seam search](how-the-merge-works.md#seam-search-experimental). (`--no-seam-search`, the default, is accepted too.) |
| `--seam-fade-ms` | `250` | Length of the crossfade at the seam. Only used with `--seam-search`. |
| `--seam-warn-db` | `15` | Flag a seam where the two chunks agree to less than this (see [The seam report](#the-seam-report)). The 15 dB is an unvalidated heuristic. |
| `--anchor` | off | Compare every chunk with the original chunk it was made from, not only with its neighbour — see [Anchoring to the original chunks](anchoring-and-listening.md#anchoring-to-the-original-chunks). |
| `--originals` | *(next to the manifest)* | The folder that holds the original chunks (for `--anchor`, and for `--pairing`). |
| `--pairing` | `auto` | `auto`: when file names were lost and files are paired by order, compare each with its original chunk and put right what is out of order. `verify`: do that for files paired by name too. `off`: never. See [Which file is which chunk](#which-file-is-which-chunk). |
| `--other-match` | – | Name of the **second** stem the way `--match` names the first (`--match instrumental --other-match vocals`). Both are merged together and the second is written next to the output. |
| `--other-out` | `--out` + the word | Where the second stem goes (`song.mp3` becomes `song_vocals.mp3`). |
| `--recon-warn-db` | `15` | With `--anchor` and `--other-match`: flag a chunk whose two stems add up to the original to less than this. An unvalidated heuristic, like `--seam-warn-db`. |
| `--audition` | – | Write a listening kit for the seams worth a listen to this folder — see [The listening kit](anchoring-and-listening.md#the-listening-kit). |
| `--audition-seams` | `flagged` | Which seams: `flagged`, `all`, or numbers such as `3,7,9`. |
| `--audition-window` | `5` | Seconds of audio either side of the seam in each clip. |
| `--audition-format` | `flac` | Format of the clips: `flac`, `wav` or `mp3`. |
| `--audition-max` | `12` | At most this many seams, the worst-agreeing first (not applied to a list of numbers). |
| `--report` | – | Write the per-seam report to a file (JSON if it ends in `.json`, else Markdown). |
| `--bitrate` | `320k` | Output MP3 bitrate. |
| `--samplerate` | the chunks' own | Output sample rate in Hz. Left out, nothing is resampled: a 48 kHz result stays at 48 kHz. |

## Which file is which chunk

A service often renames what it gives back (`download (3).mp3`), and pairing those with the
chunks by order is a guess. The original chunks are still on your computer, and a processed
file is, whatever the service did to it (a stem, another level, an offset of some
milliseconds), a piece of the chunk it came from. So when the files can only be paired by
order and the original chunks are at hand (next to the manifest, or `--originals`), `merge`
compares every file with the chunk it was paired with:

- all of them are that chunk: a note says so (`Note: ... all are that chunk (lowest
  similarity 0.41)`) and the merge goes on;
- some clearly belong to another chunk: they are put in order, and a warning says which;
- some cannot be told apart from the others: they stay where they were, and a warning lists
  them;
- two files are the same chunk, so another has none: the merge stops before writing anything
  (`--pairing off` merges them in the order they were paired anyway).

`--pairing verify` does the same for files paired by name, which catches names that are wrong
(a file renamed to the wrong part). The comparison whitens both signals (every 2 Hz band gets
the same weight, so a sustained bass note that every chunk shares does not dominate), takes the
peak of the normalised cross-correlation within ±1 s, and calls a file identified when it is
at least 0.15 like its best chunk and ahead of the next one by at least 0.08. **Those two
numbers come from synthetic songs** (one and two channels, instrumental and vocal stems, against
the other chunks of the same song: true pairs scored 0.2-0.7, wrong ones up to about 0.15), not
from real recordings. Chunks whose content repeats exactly (a loop played unchanged) come out as
"cannot tell" instead of as a wrong answer, which is the behaviour to want from it.

## The seam report

How well the two chunks agree at a seam is measured as
`10·log10(signal power / power of the difference)` in dB, after alignment and
level matching: 99 means identical, about 40 dB means a 1 % amplitude
difference, 0 dB means the difference is as loud as the signal. `merge` prints a
summary and flags every seam below `--seam-warn-db` (default 15 dB) with its
position in the merged audio, so you know where to listen (the numbers here only
illustrate the format):

```
Seams: 11 4000 ms crossfade(s) across the whole overlap; the chunks agree to 21.4 dB at worst (seam 7/8, at 6:12.4), median 33.0 dB.
1 seam(s) worth a listen: 6:12.4
```

`--report seams.md` (or `seams.json`) writes the full table — position, lag,
level correction, agreement at the seam and over the whole overlap, fade
length — for every boundary. The agreement figure also catches a level step
that `--no-match-level` or `--max-gain-db` left in place.

**The 15 dB default is a heuristic** (roughly a correlation of 0.985 between
chunks of similar level), chosen without access to real separator output. It only
flags a seam for a listen and never changes the audio. Treat it as a starting
point and adjust it to what your service's good seams look like.

**The sign of the lag.** The `Lag (ms)` column (`lag_ms` in the JSON report) is
negative when the next chunk's audio arrives *later* than the manifest says (the
service put silence in front of it: e.g. −13.0 for 13 ms of added silence) and
positive when it arrives *earlier* (the service trimmed the start of the chunk).
