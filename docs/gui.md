# Drag-and-drop interface

[← Back to the README](../README.md)

If you would rather not use a terminal for the work itself, start the interface:

```bash
seamsplit gui          # or: python -m seamsplit gui
```

It prints an address and opens it in your browser. **Split:** drop a track, enter
your AI service's per-file limit, pick a chunk format (`flac` by default, `wav` if the service
rejects it; `mp3` only as a last resort), press *Split*, and download the chunks as a `.zip`. **Merge:**
after the service has processed them, drop the results on the second card —
type a word from the file name (e.g. `instrumental`) if the service returns
several files per chunk — and press *Merge*. The seam report is summarized on
the page and can be downloaded as Markdown or JSON next to the merged file.
Under *Advanced* you can pick the crossfade curve, the sample rate and the
experimental seam search (with its fade length); the MP3 bitrate is next to the
output format. If the service gave back both stems, name the second one
(*Other stem*, e.g. `vocals`) and tick *Anchor to the original chunks*; a third
card then lets you listen to the seams and answer which join sounds best (see
[The listening kit](anchoring-and-listening.md#the-listening-kit)). A running split or merge
can be stopped with *Cancel*: it stops
before the next chunk, so the step it is in (reading the whole track, or
encoding the result) finishes first, and nothing half-done is left behind.
The page is available in English and Vietnamese.

Everything runs on your computer: the server listens on `127.0.0.1` only, uses
nothing beyond the Python standard library, and ignores any request that does not
carry the random token in the printed address (so another website open in your
browser cannot use it). Files are kept in `~/SeamSplit` (`--workdir DIR` to
change it — use a folder that is only for this). `--port N` fixes the port and `--no-browser` only prints
the address. Stop it with Ctrl+C or the page's *Quit* button.

**What gets deleted.** A new split replaces the chunks of the previous one. If the work
folder also holds your own files — the processed chunks the service gave back, a merged
result and its reports, the listening kit with your answers, or a `manifest.json` you
dropped — the page lists them and asks before *Split* deletes them (the server refuses to
split over them until it is told to, so a script cannot do it by accident either). *Clear*
on the processed chunks asks too. While a split or merge is running, nothing can be cleared,
added or replaced (the server answers `409`), so a job never loses its files half-way:
cancel it first, or wait for it.

The page covers the common options; the command line has all of them (see [Splitting](splitting.md) and [Merging](merging.md)).
