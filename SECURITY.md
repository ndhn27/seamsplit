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
  upload name or a download path, is a vulnerability.
- **Reading untrusted input.** A crafted `manifest.json` or audio file that makes SeamSplit
  hang, exhaust memory, write outside the folder it was given, or run something it was not
  asked to run.

## What is not

- Problems in FFmpeg or PyAV themselves (SeamSplit reads and writes audio through the
  FFmpeg libraries that PyAV carries). Report those to [PyAV](https://github.com/PyAV-Org/PyAV)
  or [FFmpeg](https://ffmpeg.org/security.html); it is useful to also tell us, so that a fixed
  version can be required.
- What the separator programs you choose with `--backend` (`demucs`, `audio-separator`) do:
  SeamSplit runs them as they are installed on your computer.
- Attacks that need access to your user account or to the work folder to begin with.
