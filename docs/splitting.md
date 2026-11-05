# Splitting a track

[← Back to the README](../README.md)

## The `split` command

```bash
seamsplit split "song.mp3" --target 52 --overlap 4 --out chunks/
```

If you know your service's per-file limit, give that instead of a target:

```bash
seamsplit split "song.mp3" --limit 59 --out chunks/   # target = 59 - after-slack - overlap - margin
```

`--limit` also measures the files it wrote and warns if any of them is longer
than the limit. That check matters for lossy formats: an encoder pads the
stream, so an MP3 holding exactly 59.000 s of audio is reported as 59.037 s
(44.1 kHz) or 59.063 s (22.05 kHz) by `ffprobe`. A limit-sized chunk therefore
passes a check on the audio and still reads over the limit as a file. `--margin`
reserves room for that padding (0.25 s by default for lossy formats, 0 for
wav/flac), so with `--limit 59` the target is 51.75 s, not 52 s.

The defaults (`--target 52`, `--after-slack 3`, `--overlap 4`) can produce a
59.000 s chunk, right at the edge of a 59 s limit; use `--limit` rather than
`--target` to be safe.

Add `--dry-run` to see where the cuts would go without writing any files.

Key options:

| Option | Default | Description |
|---|---|---|
| `--target` | `52` | Target length of each chunk in seconds, **before** overlap is added. Tune so `target + after-slack + overlap` stays under your AI service's limit — no chunk ever exceeds it. |
| `--before-slack` | `20` | How much shorter than `--target` a chunk may end up, while searching for a quiet cut point. Must be smaller than `--target`. |
| `--limit` | – | Your AI service's per-file limit (seconds; `--max-duration` is the same option). Sets the target to `limit − after-slack − overlap − margin` and checks the exported files. Use instead of `--target`. |
| `--margin` | `0.25` (lossy), `0` (wav/flac) | Room kept under `--limit` for the padding a lossy encoder adds. |
| `--after-slack` | `3` | How much longer than `--target` a chunk may end up. |
| `--overlap` | `4` | Seconds shared between two neighboring chunks, for crossfading later. Must be shorter than `--target` − `--before-slack`. |
| `--smooth-ms` | `250`; `1000` with `--cut-by voice` | Moving-average window (ms) applied to the curve before searching for the quietest point, to favor sustained silence over a single noisy sample. The voice-band curve is a noisier measurement and needs more of it. |
| `--cut-by` | `rms` | What decides where it is quiet enough to cut: `rms` is the loudness of the whole signal, `voice` the activity in the voice band. See [Choosing where to cut](#choosing-where-to-cut). |
| `--snap-beat` | off | Track the beat and, among the points nearly as quiet as the quietest, cut on a beat. |
| `--beat-tolerance-db` | `6` | With `--snap-beat`: how much louder than the quietest tenth of the window a point may be and still be chosen for lying on a beat. |
| `--fade-ms` | `8` | Fade in/out (ms) applied only at the very start and end of the whole track — internal chunk edges are left raw since they'll be crossfaded during merge. |
| `--min-final` | `5` | If the last chunk would be shorter than this (seconds), the last two chunks are re-divided so neither exceeds the length limit. Must be at most half of `--target` + `--after-slack`. |
| `--format` | `mp3` | Export format of the chunks. `wav` or `flac` keep them lossless — see below. |
| `--bitrate` | `320k` | Bitrate of lossy formats; ignored for `wav` / `flac`. |
| `--out` | `chunks` | Output directory. |
| `--dry-run` | off | Show the planned chunks (and the longest one) without writing anything. |

**Lossless chunks.** MP3 is lossy and every encode shifts the audio: the encoder
adds delay and padding (an MP3 of exactly 59.000 s reads 59.037 s), and the
service decodes your chunk, separates it, and re-encodes it on top of that. If
your AI service accepts them, split to `wav` or `flac` instead:

```bash
seamsplit split "song.mp3" --limit 59 --format flac --out chunks/
```

A lossless chunk lasts exactly as long as the audio in it, so `--margin` is `0`
and `--limit` can use the full limit. `flac` is about half the size of `wav` and
decodes to the same samples (use `wav` if the service rejects `flac`). `--bitrate`
does not apply. `merge` reads whatever the service returns, and writes `.wav` or
`.flac` when `--out` ends in that extension.

This produces `chunks/<name>_part01.mp3`, `chunks/<name>_part02.mp3`, ... and
`chunks/manifest.json`, which records the source file and length, the overlap
length, the part order and each part's exact position — `seamsplit merge`
needs it later. The output also tells you the longest chunk (and, with
`--limit`, the longest exported file) to check against your service's limit.

## Choosing where to cut

By default each cut goes at the quietest stretch of the **whole signal** inside its
window. In a mix that is often the wrong place: a bass line or a kick drum sets the
overall level, so the quietest moment can fall in the middle of a sung phrase, and
nothing ties it to the beat. Two options give the search more to go on, both measured
from the audio with numpy alone (no model, no extra dependency):

- **`--cut-by voice`** looks at the energy between 300 Hz and 3.4 kHz, where the
  fundamentals and formants of a singing voice sit. In a stereo track only what the two
  channels have in common counts, because a lead vocal is usually panned to the centre
  while guitars, keyboards and reverb are spread wider. That common part is noisy from
  one 30 ms frame to the next, which is why this mode averages over a second instead of
  a quarter of a second.
- **`--snap-beat`** tracks the beat (tempo from the autocorrelation of the onset
  strength, then dynamic programming for the beat positions, as in Ellis 2007), which
  carries on through a rest and follows a slow drift of tempo. Within
  `--beat-tolerance-db` of the quietest tenth of the window, the cut goes to the point
  nearest a beat. It works with either `--cut-by`.
- **`--snap-bar`** goes one step further: it works out which beat starts a bar (the one
  struck harder than the others: the kick lands on it, the bass changes note) and, when
  a bar line lies inside the quiet stretch, cuts on that. It says how sure it is
  (`downbeat_confidence` in the manifest, `--dry-run` prints it) and, when it cannot tell
  where the bars start, it says so and falls back to the beat rule. `--bar-beats N` sets
  the beats per bar as the tracker counts them (4 by default; 3 for a waltz).

```bash
seamsplit split "song.mp3" --limit 59 --cut-by voice --snap-bar --dry-run
```

**What these are not.** `voice` is a measurement, not a vocal detector: it cannot tell a
voice from a guitar that plays in the same range and sits in the same place in the stereo
image, and it loses a short rest when the wide instruments are louder than the voice. The
beat tracker finds nothing in a track that is not rhythmic (it says so and leaves the cuts
alone), is sometimes a half or a double off in tempo (the beats it reports are still beats),
and only extrapolates through a long rest while the tempo is drifting (by up to about 50 ms
in a four second rest). `--snap-bar` is a heuristic with blind spots of its own: it needs
six bars or more, it takes the strongest beat of the bar for the first one (a track whose
first beat is not the strongest gets the wrong phase, with a confidence to match), a drum
pattern that strikes beats 1 and 3 alike cannot tell them apart, and a beat tracker that
skips a beat moves every later bar line by one (they are still beats). It never trades
silence for a bar line: it only picks one that is already inside the quiet stretch. Always
check with `--dry-run`: it prints how well the cuts sit on the beat and on the bar, and the
manifest records the distance of every cut (`cut_beat_offsets_ms`, `cut_bar_offsets_ms`),
the tempo, the confidence and the options used.

**No voice-activity detector.** An obvious next step is to cut where a VAD says there is no
voice. I tried one (Silero, through `pysilero-vad`) and it barely reacted to the synthetic
singing and speech-like signals I can make here, so I do not know whether it works on real
vocals and could not check; shipping it would only have added a dependency and a claim.
`--cut-by voice` stays what it is: a measurement of activity in the voice band.

**How well it works.** Measured only on synthetic stereo mixes (loud bass with a dip, a
kick on every beat, a wide pad, a centred voice with a rest), not on real recordings. In
them the loudness search never cut inside the vocal rest: the mix is built so that the
quietest overall moment is a bass dip in the middle of a phrase. `--cut-by voice` cut inside
the rest in all of 40 random mixes at every pad level from far below the voice up to about
twice its loudness, and `--snap-beat` then put the cut a median 5 ms from a beat (98 to 100 %
still inside the rest). With a pad three times as loud as the voice it still found rests of
two seconds or more, but only 87 % of one second ones and a third of half second ones.
Beats were found within 30 to 40 ms for 98 to 100 % of the true beats on click tracks and
drum patterns, and the periodicity test separates noise from rhythm by a wide margin. How it
fares on your music is for you to see with `--dry-run`; if it does worse than the default,
the default is still there.

`--snap-bar` was measured on 40 random synthetic mixes (80-140 BPM, a bass note and a
harder kick on the first beat of the bar, with accents of 1, 0.5 and 0.25 times the kick
strength, a four second vocal rest at a random place): every cut was inside the rest, 70 %
of them within 50 ms of a true bar line, against 28 % (chance is one in four) for
`--snap-beat` alone. The bars were claimed in all mixes with the two stronger accents and in
81 % of those with the weakest. The cuts that missed the bar line did so because no bar line
was inside the quiet stretch: it stays quiet first. This is a simulation that shares its
assumptions with the detector; how often real music marks its first beat that way, I have not
measured.

The analysis takes about two seconds for ten minutes of stereo audio; the default search
about a third of a second.

## Service presets

A preset stores what you know about a service so that you need not retype it:

```bash
seamsplit services                              # what there is, and where yours go
seamsplit split song.mp3 --service xminus       # --limit 59, the target worked out from it
seamsplit split song.mp3 --service xminus --limit 45    # what you give always wins
```

A preset fills in the length limit, and the format and overlap if it has them, for
whatever you did not give. Its limit is a `--limit`, so `--service` cannot be combined
with `--target` (that is an error, not a silent choice). Your own presets go in
`services.json` in `.config/seamsplit/` under your home folder (under `XDG_CONFIG_HOME`
instead, if that is set; `seamsplit services` prints the exact path; `--services-file` or
`SEAMSPLIT_SERVICES` name another file), and a name there replaces a built-in one:

```json
{"myservice": {"limit_s": 120, "format": "wav", "overlap_s": 5, "note": "my plan"}}
```

**Take the numbers with care.** The only built-in preset is `xminus`, with the 59 s limit as
it was given to this project; I have not checked it against the service, which can change it
at any time. `split --limit` measures the files it wrote, so a limit that is wrong shows up
there. SeamSplit does not upload anything to any service and has no way to: automating a web
page is fragile and may break a service's terms, so the files go there by hand.
