# Benchmarks

What this folder is for: SeamSplit says that overlapping chunks, cut at quiet places and joined
with a crossfade, give a better result than cutting a song into pieces and gluing the pieces
back. That is a claim about real music and a real separator, and a script that cuts and joins
makes the same claim. Here it is measured, on a few real songs and a few real chunks, and the
numbers are kept in [`results/`](results/) ([`RESULTS.md`](results/RESULTS.md) has all the
tables) so that anyone can check them, rerun them, and see what they do not show.

Everything below was measured with the code in this repository, on one machine (a 2-thread
cloud sandbox), once. Nothing is rounded in the direction that helps.

## What was run

| # | Experiment | Music | Separation | Damage |
|---|---|---|---|---|
| 1 | **Seams**: every song cut into chunks four ways, every chunk separated, the chunks joined in six ways | 5 real songs | **real** (UVR MDX-Net, CPU) | none |
| 2 | **Cut points**: is a cut at the "quietest point" in a quiet place, and does that leave a smaller seam? | same | real | none |
| 3 | **Pairing**: files renamed `download (k)` in random order, put back by what they sound like | same | real | renaming only |
| 4 | **What a service does to its answers**: levels, offsets and a lossy file | same | real | **simulated** |
| 5 | **Speed and memory** of the audio layer, split and merge on 5 minutes; the one-file program | the same songs end to end | none | none |
| 6 | **The whole program**: `seamsplit run --backend audio-separator` from the song to merged stems | 1 song | real | none |

The songs ([`corpus.json`](corpus.json)) are named by repository, commit and SHA-256 and are
downloaded from where they are published (`python benchmarks/run.py fetch`); none of the audio
is part of SeamSplit, and none is redistributed here. Three are MUSDB18 excerpts of 30 s
(Cristina Vane - So Easy, The Mountaineering Club - Mallory, Triviul - Widow) with the
ground-truth vocals and accompaniment; the other two (a 10.9 s and a 20 s clip) have no truth.

The four chunk sets ([`bench/chunks.py`](bench/chunks.py)) are cut with SeamSplit's own
splitter into pieces of about 8 s (the clips are short; real songs would be cut into 30-60 s):

| set | cuts | overlap |
|---|---|---|
| `fixed0` | equal pieces | none: a plain cut-and-join script |
| `fixed2` | the same equal pieces | 2 s |
| `quiet2` | the quietest point of a window (`--cut-by rms`) | 2 s |
| `voice2` | where the voice band is quietest (`--cut-by voice`) | 2 s |

and joined as `hardcut` (the overlap thrown away: each chunk is used up to the seam, no
fade), `linear`, `equalpower`, `adaptive` (the default), `adaptive+seam` (`--seam-search`) and
`adaptive+anchor` (both stems tied to the original chunks).

## What the numbers say

All of this is on the five songs above. "Seams" are the 2 s windows around a join; 24-26 joins
(two stems each) per chunk set. Figures in dB are signal-to-error ratios, higher is better.

**The overlap helps more than the crossfade.** With the same cut points, chunks cut without
overlap and simply joined (`fixed0`) are 21.2 dB from the same song separated in one go, in the
seam windows; with 2 s of overlap, each chunk sees the music on both sides of the cut, and
joining them *hard* (`fixed2`, `hardcut`) gives 25.9 dB (+4.7); crossfading the overlap
(`adaptive`) 26.7 (+5.5 over the plain script). Away from the seams the chunked results are
24-25 dB from the whole-song separation in every set, which is what a model that does not see
the whole song loses by itself.

**The crossfade is a small gain that most seams get and a few lose heavily.** Against a hard cut
of the same seam, `adaptive` lowers the error in 19 of 24, 21 of 26 and 19 of 26 seams (`fixed2`,
`quiet2`, `voice2`), by a median of 0.96, 0.69 and 0.55 dB. But the *mean* gain's 95 % interval
(resampling the seams) includes zero in all three, because a few seams get clearly worse, by up
to ten dB (the six worst are listed in `RESULTS.md`: *Widow* near 25.5 s in both sets that cut
there, *So Easy*'s vocals at 24 s). Song by song, over the windows of `quiet2` pooled (so loud
seams count for more), it is better than a hard cut for two songs (*Mallory* +3.3 dB, *Slow
Motion Dream* +0.8), even for two (*So Easy* -0.3, *Demucs test* -0.1) and worse for one
(*Widow*, -2.0). So what is established is the overlap (previous paragraph); that the crossfade
adds to it on average is only weakly shown. A reading that fits (not tested here): in a 2 s
overlap, the half of each chunk next to its own edge is where a model is worst, and a fade that
long lets that half into the result where a hard cut at the middle would not. `linear` is
indistinguishable from `adaptive` here; `adaptive` is the default for what it does when the two
chunks do *not* agree, which this data does not exercise (experiment 4 does).

**Equal-power is the wrong fade for these chunks.** It is the textbook curve for unrelated
signals; two separations of the same music are nearly the same signal, so the two halves add up
and the seam is up to 3 dB too loud in the middle: a median 16 dB worse than a hard cut, and
worse in all but one or two seams of each set (`equalpower` in `RESULTS.md`). `adaptive`
measures the correlation and does not do this.

**Searching for a seam (`--seam-search`) did not help**: median gain 0.09 / 0.06 / -0.03 dB, and
it was better than a hard cut in only 12-17 of 24-26 seams. It is off by default and this keeps it
experimental. **`--anchor` did not help on clean chunks either** (the local model neither moves
nor rescales them): 0.4-0.6 dB worse than the default on the whole file, though the two stems add
up to the mix better (38 dB against 34-36), which is what it is for: telling a good return from a
damaged one.

**Quiet cuts are quiet, but that is not where the error is.** The `rms` cuts are 6.3 dB below
the song's level, in places only 12 % of the song is quieter than (fixed cuts: 0.4 dB, 50 %),
and the reference-free click measure at the join is 5-6 dB lower. The error of a hard cut at
those places is not smaller (25.3 dB against 26.0 for fixed cuts): the cut position changes
what the ear hears at the join more than what the separator got wrong. The voice-band cuts
look for singing, not for loudness (1.8 dB below the level, 38 %); their seam error is
25.7 dB.

**Against the ground-truth stems** (three songs, both stems, the whole file) the whole song
separated in one go scores 12.3 dB. Chunked, the plain cut-and-join script (`fixed0`) scores 11.2,
and the best joined result (`voice2`, `adaptive` or `linear`) 11.4: chunking costs about 0.9 dB
against the truth and the overlap and crossfade get back about a quarter of a dB of it, which is
as much as can be said on three songs. Equal-power scores 10.5-10.7. The separator's own error
against the truth (8.8-17.1 dB over the six stems) is much larger than anything the joining does,
and the truth is a 128 kbit/s MP3 whose stems add up to the mix only to 18-23 dB, so the
ceiling is low. Joining well keeps a chunked result close to the whole-song separation; it cannot
make it better than the model.

**Pairing by sound works on real stems.** For all 94 stems of 47 chunks (two stems each, three
overlapping sets), the original chunk the file correlates with best was the right one. 90 of
the 94 clear the thresholds (score 0.15, margin 0.08, which were set on synthetic songs); the
four that do not are vocal stems of the mostly instrumental 20 s clip (a true score of 0.11-0.15
where there is hardly any singing). Renamed `download (k)` in random order, 88 of 96 trials were
put back and merged identically, 8 were refused with an error, none was merged wrong; with the
check off, all 96 were merged wrong. The refusals are all the same case: one file the check was
not sure of is left where it was and clashes with the place of a file it was sure of, while one
part has no file; in those the part that is left over could be given to the file that is left
over, which the check does not do yet. It is safe and a little cautious.

**What services do (simulated damage, real chunks).** Each chunk louder or quieter by up to
3 dB: merged without alignment or level matching 16.4 dB from the undamaged merge, with the
default 53.6, anchored 60.1. Each chunk later or earlier by up to 30 ms: 0.8 dB without, 27.5
with the default alignment, 23.8 anchored (the ceiling is the audio that a shift pushes out of
the chunk). A 128 kbit/s MP3 in place of the file: 24.4 dB with every handling, since nothing in
a merge can undo an encoder. All three together: 1.6, 20.3 and 18.5. Whether any given service
does this, and by how much, is not measured here.

**Speed and memory** (5 minutes of stereo 44.1 kHz, median of three runs, each in its own
process): decoding takes 0.15 s for WAV, 0.48 s for FLAC and 0.86 s for MP3; encoding FLAC 0.60 s
and MP3 (320k) 4.6 s; the default split 6.1 s and the merge 1.5 s. Peak memory is 203-236 MB. The
one-file Linux program is 61 MB, starts in 1.3 s (7 s the first time, while it unpacks itself)
and splits a song with an empty `PATH`.

**The whole program with a real backend** (`seamsplit run --backend audio-separator --anchor`,
one command, the MDX-Net model, `So Easy`): exit status 0, four chunks, three seams, none
flagged, both stems written; 77 s on two CPU threads for a 30 s song (the separation is the
whole of it); run again it does nothing and returns in 1.1 s. The merged stems are 21.9
(instrumental) and 17.9 (vocals) dB from the whole-song separation and 13.0 / 10.1 dB from the
ground truth (the whole-song separation: 14.5 / 11.7), about 1.5 dB less. Why is not separated
out: the command line normalises each chunk's peak by itself (the benchmark's own runs do not),
the chunks are short, and this is one song.

## What these numbers are not

- **A local model is not a web service.** The separator is `UVR-MDX-NET-Voc_FT` run by
  `audio-separator` on the CPU. A web service has its own model, its own normalisation, a lossy
  output, and does things to timing and level that this folder only *simulates* (experiment 4).
  How SeamSplit does on any one service has to be measured with that service, using
  `seamsplit merge`'s report; nothing here uploads anything.
- **The clips are short.** 10-30 s, so the chunks are 4-14 s, where a service's limit makes
  them 30-60 s. The overlap is 2 s of an 8 s chunk, where the defaults overlap 4 s of 52 s, so
  the edges of a chunk, where a model is worst, are a larger share of every chunk here.
- **Five songs, three with truth, 24-26 seams a set.** The medians are real; the intervals are
  wide; seams of one song and the two stems of one seam are not independent, so even the
  bootstrap intervals shown are too narrow. None of this was tested for significance.
- **The measure is plain signal-to-error, not BSS-eval SDR**, and not what anyone hears. It
  compares the chunked result with the same model run once on the whole song, which is the
  best that chunking could do, not the truth. The truth (MUSDB) is a lossy MP3 with a 18-23 dB
  ceiling. The reference-free "click" figure is a heuristic. **No listening test was done**:
  `seamsplit merge --audition` is the tool for it, and the ear is the final judge of a seam.
- **Part of what a crossfade gains is averaging.** In an overlap, two separations of the same
  music are averaged, which lowers the error of each; a hard cut uses one. That is a real
  advantage of overlap-and-fade, but it is not the same as hiding a click.
- **Speed is on one machine** (two threads of a cloud CPU, Linux), and the one-file program was
  measured on Linux only. Windows and macOS builds come from the same recipe in CI and have not
  been run.
- **Simulated damage is simple.** One gain and one offset per chunk, the same for both stems.
  Real services may do other things (change speed slightly, drift, clip, fade, resample).

## Running it

You need the Python environment SeamSplit runs in (`pip install -e .`), and for the real
separation another one with [`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator)
(`pip install "audio-separator[cpu]"`), which downloads the model on first use. From the root of
the repository:

```bash
python benchmarks/run.py fetch                                    # the songs, checked by SHA-256
python benchmarks/run.py all --separator-python /path/to/separator-env/bin/python
python benchmarks/run.py seams --separator-python ...             # or one experiment at a time
python benchmarks/run.py perf --exe dist/seamsplit
python benchmarks/run.py report                                   # RESULTS.md from results/*.json
python benchmarks/run.py all --fake                               # the harness in seconds, no model
```

On two CPU threads the separation of the 75 files took about half an hour: for the 59 whose
time was recorded (the others were done by a run that was stopped), 542 s of music took 1041 s,
1.9 s for each second of music (0.52 times real time, with other work sometimes running on the
same machine; the model loads in 18 s). The experiments after it take some minutes. What the
separator wrote is kept in
`benchmarks/cache/` (not in git), keyed by the SHA-256 of each file, and a stopped run is taken
up where it was. `--fake` replaces the model by a low-pass filter and writes to `results-fake/`:
those numbers are about the harness and mean nothing about SeamSplit. `perf --legacy-src PATH`
times another version of the package (its `src` folder) on the same phases.

Another separator can be used by writing an object with a `name`, and `stems_for(paths)` returning
`{"vocals": path, "instrumental": path, "seconds": s}` for each path; see
[`bench/fake.py`](bench/fake.py) and [`bench/separation.py`](bench/separation.py). Another song:
add it to `corpus.json` with its source and hash.

## Layout

```
benchmarks/
  run.py              the command line
  corpus.json         the songs: where from, which commit, which hash, which licence
  bench/
    corpus.py chunks.py workset.py separation.py worker.py fake.py   songs, chunk sets, separating
    metrics.py                                                        the measures
    seams.py cuts.py pairing.py degraded.py perf.py backend.py        the six experiments
    envinfo.py report.py                                              the machine; RESULTS.md
  results/            environment.json, seams.json, cuts.json, pairing.json, degraded.json,
                      perf.json, backend.json, RESULTS.md
```

The harness itself is tested in `tests/test_benchmarks.py` (synthetic songs and the stand-in
separator, no network).
