# Anchoring and the listening kit

[← Back to the README](../README.md)

## Anchoring to the original chunks

Without `--anchor`, `merge` only ever compares a chunk with its neighbour, in the few
seconds they share. That is a *relative* measurement: each chunk is corrected by
what the one before it came out as, so the first chunk's own timing and level become
the track's, a small bias at every seam adds up, and where the overlaps do not
correspond (a vocals stem that is silent there) there is nothing to measure. But the
original chunks are on your computer, so `--anchor` compares every processed chunk
with **its own original** and gets an absolute reference:

```bash
seamsplit merge processed/ --manifest chunks/manifest.json --anchor \
    --match instrumental --other-match vocals --out song_instrumental.wav
```

- **Time.** The offset of the processed chunk against its original is measured (the same
  cross-correlation as between neighbours, within `--max-lag-ms`) and cancelled, so every
  chunk, the first included, sits where the manifest says it was cut. `--no-align` switches
  this off.
- **Level, with both stems.** The two stems add up to (about) the original, so one gain per
  chunk, the least-squares amplitude at which the sum reproduces the original, is applied to
  **both** stems and their balance is untouched. `--no-match-level` switches it off. It cannot
  come from one stem: an instrumental alone is quieter than the mix by exactly what the voice
  carried, so matching it to the original would make it too loud. With one stem only the timing
  is anchored, and the level is left to the neighbour matching. The correction is
  skipped, and said so in the report, for a chunk whose stems do not add up to its original
  (correlation below 0.9) or that would need more than 24 dB.
- **Both stems, one set of decisions.** Merged one at a time, the stems would each get their
  own offsets, gains and fades from their own overlaps, and what they measure differs. They are
  merged **together** instead, as the channels of one signal analysed as the sum of the stems,
  so both outputs have the same timing, gains and fades and still add up to the original.
- **A consistency figure.** What is left of the original after the best offset and gain,
  `10·log10(power of the original / power of what is left)`, is reported per chunk
  (`Stems add up to the original (dB)`, and `recon_db` in the JSON report). A chunk below
  `--recon-warn-db` is flagged and is the first to listen to.

**What is and is not known.** This was checked against a simulated service (a synthetic song,
spectral-mask stems with their own noise, a different offset of up to 18 ms and a different
level of up to 4 dB for every chunk): the offsets and gains it applied are found to within a
fraction of a millisecond and 0.5 dB, and the two outputs add up to the source to about 33 dB,
where the merge without the originals carries the first chunk's offset and level through the
whole track. A simulation like this one says that the arithmetic is right; it says nothing about
how a real separator behaves. In particular:

- `recon_db` is a **consistency check, not a separation score**. A separator can reproduce the
  mix perfectly and still leave the voice in the instrumental, and one that does not conserve
  energy can separate well and reproduce it badly. What it does catch is a chunk where the
  service did something that is neither of the two stems (normalized each stem on its own, a
  dropout, a different take).
- The 15 dB threshold is a **heuristic**; look at your service's typical value first.
- If your service adds effects or processes the stems differently, the sum will not be the
  original and the level will not be taken (and the report says so).

## The listening kit

`--seam-search` and `--seam-warn-db` are open questions that only ears can answer, so
`merge --audition DIR` turns each seam worth a listen into a small test: the audio around the
seam (`--audition-window`, 5 s either side) joined three ways, which you compare by ear.

```bash
seamsplit merge processed/ --manifest chunks/manifest.json --out song.flac \
    --audition kit/ --audition-seams flagged
```

- **Straight join:** the two chunks cut where the splitter cut and glued, with no alignment,
  levelling or fade. What you would get without SeamSplit.
- **Crossfade:** what `merge` does by default (the chosen `--method`, after alignment and level
  matching), a fade over the whole overlap.
- **Seam search:** the experimental short crossfade at the best-matching point. When it makes
  the same audio as the crossfade, only two are offered.

The clips are made by running the *same* merge (same files, anchoring, stems and options) over
just the two chunks of the seam, and are all cut around the same point, so switching between them
while playing keeps the music in place and only the join changes. The first chunk gets the level it
has in the whole merge, so a clip is as loud as the merged file is there. `kit/` holds one audio
file per seam and variant, `audition.json` and `index.html`: **open the page in a browser**, no
server needed. It plays the clips in step (loop ±2 s around the seam, keys `1`–`3` to switch,
space to play), shuffles and hides which button is which until you have answered ("blind"), and
keeps your answers in the browser; *Download* gives `choices.json`. In the GUI, the third card
does the same and keeps the answers in the work folder.

Each answer is stored with what was measured at that seam. `seamsplit audition-summary` counts the
answers of one or several kits (files or folders):

```bash
seamsplit audition-summary kit/ other_song_kit/choices.json
```

It reports, with an exact two-sided sign test, whether the seam search was preferred to the
crossfade, and the straight join to the two others; only **blind** answers and seams that
actually differ count, and fewer than 6 decided seams cannot show anything (the result says "too
few"). It also shows the agreement (in dB) at the seams where you heard nothing against those
where you did, which is the evidence for `--seam-warn-db`. A handful of seams from one song,
mixed from one service, is a start, not a verdict: the more songs and services, the more it is
worth.
