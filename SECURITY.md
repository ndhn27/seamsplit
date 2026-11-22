# Security policy

## Supported versions

Security fixes go into the latest release and into the `main` branch. Older
releases are not patched.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately through
GitHub: open the repository's **Security** tab and choose **Report a vulnerability**
(<https://github.com/ndhn27/seamsplit/security/advisories/new>).

Say what you found and how to reproduce it, and include the output of `seamsplit --version`,
your operating system and your Python version. A small manifest or audio file that triggers
the problem helps more than a description of it.

SeamSplit is maintained by one person, so there is no fixed response time. Reports are read
and answered as soon as possible; please give a fix reasonable time before you publish the
details.

## What is in scope

- **The local web interface (`seamsplit gui`).** It is meant to be reachable only from the
  same computer and only by the page that was opened for you: the server listens on
  `127.0.0.1`, and a request is refused unless it carries the random token from the printed
  address and its `Host` (and its `Origin`, when it sends one) names this server. A way
  around any of these, or a way to read or write a file outside the work folder through an
  upload name or a download path, is a vulnerability, and so is a request that keeps a
  thread of the server for as long as the client likes (a body length that is not a number
  of bytes is refused, and a client that stops sending or reading is dropped after two
  minutes).
- **Reading untrusted input.** A crafted `manifest.json`, `job.json` or audio file that makes
  SeamSplit hang, exhaust memory, write outside the folder it was given, read a file it was
  not given, or run something it was not asked to run. The names of the parts in a manifest are
  joined to a folder, so on Windows one with a drive (`D:evil.flac`, which replaces the folder
  it is joined to) or the name of a device (`NUL.wav`) is refused. A file that says it is audio
  is not trusted to be only that: one that lists other files or addresses (an ffconcat or HLS
  playlist, which FFmpeg detects by looking inside, whatever the file is called) is refused,
  and no input opens a network protocol. A job folder is made to be carried to another computer and picked up
  there, so opening one (`status`, `run --job`) must never run a command that is written in it:
  `job.json` records a preset's name and model, and for a command of your own only a hash (see
  [Job folders from someone else](docs/jobs-and-backends.md#job-folders-from-someone-else)).
  Audio is held in memory in full, and a few kilobytes of FLAC silence decode to gigabytes,
  so a file is read only up to a limit (`--max-input-duration`, two hours by default), checked
  against its header before it is decoded and again while it is: a file that gets past that,
  or that makes the memory grow with something other than its length, is a vulnerability. A
  file inside the limit still takes memory in proportion to its length (up to about 1.3 GB
  for two hours of stereo at 44.1 kHz, 2.8 GB for the heaviest ordinary format, and a split
  needs about 1.4 times that); on a small computer, lower the limit. See
  [Long files and memory](docs/splitting.md#long-files-and-memory).

## What is not

- Problems in FFmpeg or PyAV themselves (SeamSplit reads and writes audio through the
  FFmpeg libraries that PyAV carries). Report those to [PyAV](https://github.com/PyAV-Org/PyAV)
  or [FFmpeg](https://ffmpeg.org/security.html); it is useful to also tell us, so that a fixed
  version can be required.
- What the separator programs you choose with `--backend` (`demucs`, `audio-separator`) do:
  SeamSplit runs them as they are installed on your computer.
- Attacks that need access to your user account, or to a work folder of your own, to begin
  with. A job folder or a manifest that someone else made and you open is in scope, above.
