# How the merge works

[← Back to the README](../README.md)

## Timing alignment and integrity checks

AI services and MP3 encoders often add or trim a few milliseconds at the start
of a file. Crossfading two copies of the same audio that are offset by even a
millisecond causes comb filtering (a hollow, phasey sound at the seam), and an
offset also shifts everything after it. So before each crossfade `merge`
measures the offset between the two overlap regions with a cross-correlation
(up to `--max-lag-ms`) and lengthens or shortens the overlap to cancel it.
Offsets are only applied when the two regions clearly match.

`merge` prints warnings (to stderr) when it finds a problem:

- a processed chunk's length differs from the manifest by more than 60 ms;
- the merged length differs from the source's length;
- two overlap regions barely correlate (nothing to align on or to match the level
  with — for example, the separator changed the material a lot), or the offset hit
  the search limit;
- a level correction hit `--max-gain-db`, or the result clips;
- two chunks agree to less than `--seam-warn-db` at a seam;
- `--overlap` differs from the value used when splitting.

A chunk that was padded with silence at the start can't be perfectly restored
(the missing audio isn't there to blend), but the timing of everything after it
is.

## Level matching

An AI service may normalize every chunk to a different loudness. Because both
chunks carry the same music in the overlap, the ratio of their RMS levels there
is exactly the gain needed, so `merge` scales each chunk so that its overlap is
as loud as the previous chunk's (which has itself already been corrected, so
the correction chains along the track). It applies one constant gain to the
whole chunk, so the dynamics inside a chunk are untouched.

- Only overlaps that clearly match (correlation ≥ 0.5) are used; unrelated
  material says nothing about level.
- A correction is limited to `--max-gain-db` (default 6 dB) and a warning is
  printed when the limit was hit — a bigger difference usually means the
  chunks differ in more than loudness.
- Differences under 0.05 dB are left alone, so chunks that already match are
  not touched. `--no-match-level` turns the step off.
- If a boost pushes a peak over full scale, `merge` says so (samples are
  clipped, not wrapped).

## Choosing a crossfade curve

Across each overlap region the outgoing chunk's tail is multiplied by a
falling gain and the incoming chunk's head by a rising gain, with `t` going
from 0 to 1. The right curve depends on how alike the two regions are, which
is measured by their normalized correlation `rho` (1 = identical audio,
0 = unrelated):

| Method | Gains | Constant… | Right when |
|---|---|---|---|
| `adaptive` (default) | `sin((1-t)θ)/sin θ`, `sin(tθ)/sin θ`, `θ = arccos(rho)` | **power**, for the measured `rho` | always: it follows the overlap |
| `linear` | `1 - t`, `t` | **amplitude** (gains sum to 1) | both sides are *identical* (`rho = 1`) |
| `equalpower` | `sqrt(1 - t)`, `sqrt(t)` | **power** (squares sum to 1) | both sides are *unrelated* (`rho = 0`) |

The overlap holds the same stretch of music in both chunks, but after
independent separation the two versions are alike, not identical, so `rho` is
somewhere below 1. A fixed `linear` fade then dips mid-seam (about −1.2 dB
at `rho = 0.5`), and a fixed `equalpower` fade rises (+1.8 dB at `rho = 0.5`,
+3 dB at `rho = 1`). `adaptive` removes the guesswork: it is exactly `linear`
at `rho = 1`, the sine/cosine equal-power law at `rho = 0`, and keeps
`gain_out² + gain_in² + 2·rho·gain_out·gain_in = 1` (constant power) for every
value in between. The correlation is measured per boundary, after alignment,
inside the fade window (the whole overlap unless `--seam-search` is given), on the
whole signal, so it reflects the dominant (usually low-frequency) content
rather than every band separately.

Use `--method linear` if you want the plain, fully predictable curve.

## Seam search (experimental)

`--seam-search` is **off by default** because its premise has not been checked
against real separator output.

The overlap is several seconds long, and blending two independently separated
versions of it over its whole length smears both versions' artifacts across all
of it. The idea behind seam search is that each chunk is at its worst near its own
edges (where a separator has the least context). So after alignment and level
matching, `merge --seam-search` scans the overlap for the stretch where the two
chunks agree best and makes only a short crossfade (`--seam-fade-ms`, 250 ms)
there; before it you hear the outgoing chunk, after it the incoming one.

Every possible position of the fade window is scored by the energy of the
difference between the two chunks inside it, relative to the energy of the
signal inside it (with a floor, so near-silence with tiny absolute differences
doesn't look bad). The first and last 10 % of the overlap are never used, since
that is where the chunks are closest to their own edges. Positions within 5 %
of the best score tie, and the one nearest the middle wins — so identical chunks
get a centered fade, and the middle of the overlap is where the splitter put
its quiet cut point. The crossfade curve (`--method adaptive`) is then shaped
by the correlation *inside the fade window*, not over the whole overlap.

Seam search applies only where the overlaps clearly match (correlation ≥ 0.5);
where they don't, `merge` falls back to a fade across the whole overlap and says
so.

**What is and is not known.** The test suite includes a case where the error of
each chunk grows towards its own edge, and there seam search wins by more than
10 dB. That case is built on the premise, so it shows that the algorithm exploits
such an error profile, not that real separators have one. When both chunks carry
about the same error everywhere in the overlap, a fade over the whole overlap
does slightly better, because it averages the two errors while a short fade
cannot (the suite has a test for this case too). Neither case says anything about
real material, and neither size of difference should be read as a result. Before
relying on `--seam-search`, compare by ear at the seams that `--report` lists:
`--audition` makes the clips and the page for that, and `audition-summary` says
whether your answers amount to a preference.
