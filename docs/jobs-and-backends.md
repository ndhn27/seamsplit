# Jobs and local separation

[← Back to the README](../README.md)

## One folder for the whole job: `run` and `status`

`seamsplit run` keeps a song's split, the results and the merged files in one folder
(`song.job/` by default, or `--job DIR`) and does what is still to be done, so it can be run
again after every interruption:

```text
song.job/job.json     what the job is: the file, how it was split, the backend
song.job/chunks/      what `split` writes (the chunks and manifest.json)
song.job/processed/   what came back: from a backend, or put there by hand
song.job/output/      the merged files
```

```bash
seamsplit run song.mp3 --service xminus --match instrumental --other-match vocals
#   splits into song.job/chunks, says what to send, and stops with exit status 3
seamsplit status song.job --match instrumental --other-match vocals
#   lists the chunks that have no result yet (exit status 3 while any is missing)
seamsplit run --job song.job --match instrumental --other-match vocals
#   when everything is there: merges into song.job/output
```

`run` takes the split options of `split` (`--limit`, `--service`, `--cut-by`, `--snap-bar` ...) and
the main options of `merge` (`--method`, `--anchor`, `--pairing`, `--samplerate`, `--report`).
Exit status is 0 when the job is done, 3 when it waits for something, 1 for an error. A job
belongs to one song and one backend: another song, or another backend, needs another folder.

`job.json` is written before the split starts and `manifest.json` after it, so a `run` that was
killed in the middle of splitting (a power cut, `kill -9`) leaves a job with no manifest. Running
the same command again says so, drops the half-written chunks (only files a split makes) and splits
again from the start; it does not need the folder to be emptied by hand. Because nothing was
separated yet, that restart may also name another song or backend. `status` on such a folder tells
you to run it again.

`status` also works on a plain `split` folder (`seamsplit status chunks/ --processed
processed/`). Files are found by chunk name; a file whose name was lost is compared with the
original chunks, and if it is clearly one of the chunks that has no file, it counts for that chunk.
It also points out files whose length is not the chunk's (by more than 0.3 s), files that are
none of the chunks, and chunks with several files. `--json` gives the same as data.

## Separating on this computer

With a separator installed, no web service is needed, and so no limit of 59 seconds:

```bash
seamsplit run song.mp3 --backend demucs                     # Demucs, two stems (vocals + the rest)
seamsplit run song.mp3 --backend audio-separator --backend-model MODEL   # the UVR models
seamsplit run song.mp3 --backend-cmd "mytool {input} -o {output}" \
    --backend-stem "vocals=^vocals" --backend-stem "instrumental=^no_vocals"
```

Each chunk is given to the program in a work folder of its own, the stems are found in what it
wrote by file name, and moved to `processed/` as `<chunk>_<stem>.<ext>`, which is what `merge`
finds; a chunk counts as done only when both stems are there, so a run that is stopped (or a chunk
the program fails on) is continued by running the same command again. When no `--service`,
`--limit` or `--target` is given, chunks are cut at 120 s in WAV (no web service limit applies, and
fewer chunks mean fewer seams); that default is a convenience of mine, not a measured optimum.
`--backend-timeout` gives up on a chunk that takes too long.

This turns SeamSplit into a way to separate a song of any length with whatever engine is installed:
the program decides the quality, SeamSplit does the cutting, resuming and merging.
**What is and is not checked:** the command lines and output names of the `demucs` and
`audio-separator` presets follow those programs' documentation. The `audio-separator` preset has
been run against the real program: the [benchmark](../benchmarks/README.md) (experiment 6) runs
`seamsplit run` with audio-separator 0.48.0 and the `UVR-MDX-NET-Voc_FT.onnx` model on one song,
from the audio file to merged stems, and again to see that nothing is left to do. Only one song
and one model were tried, so another model that names its files differently may need
`--backend-cmd`. The `demucs` preset has been tested only against a stand-in program that writes
the same names, not against the real Demucs. If a real program names its files differently, the
error lists what it wrote, and `--backend-cmd` with `--backend-stem` takes over completely.
