# Contributing to SeamSplit

Bug reports, questions and pull requests are welcome. For a security problem, do not open
a public issue: see [SECURITY.md](SECURITY.md).

## Setting up

SeamSplit needs Python 3.10 or newer; CI runs 3.10 to 3.14.

```bash
git clone https://github.com/ndhn27/seamsplit.git
cd seamsplit
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install pre-commit && pre-commit install
```

Audio is read and written by [PyAV](https://pyav.org/), whose wheels carry FFmpeg, so there
is nothing else to install. The tests build their own audio and use stand-ins for the
separation services and programs (`tests/fake_service.py`, `tests/fake_programs.py`), so they
need neither a model nor an account. The one exception is the test of the listening page
in `tests/test_audition.py`, which runs the page in a browser: run
`python -m playwright install chromium` once to let it run. Without it the test is skipped on
your machine; CI installs the browser and fails if it is missing.

## Before you open a pull request

These are the checks CI runs; all of them must pass:

```bash
pytest --cov --cov-report=term-missing --cov-fail-under=90
flake8 src tests benchmarks
black --check src tests benchmarks
mypy
```

`black src tests benchmarks` formats the code (line length 100, set in `pyproject.toml`).
`pre-commit run --all-files` runs the same linters, plus a few whitespace checks, and also
covers `packaging/`.

Then:

- Add or change tests with the code. The coverage gate is 90 %.
- In the pull request, say what changes for someone who uses SeamSplit, not which files were
  touched: that text becomes the release notes.
- If an option, a default or a behaviour changes, change the README or the page in `docs/`
  where it is described.
- Keep a pull request to one change, and say in it how you checked it.

## Where things are

[`docs/development.md`](docs/development.md) has a table of what each module does.

The benchmarks in `benchmarks/` download music and run a real separation model, so CI does not
run them; their code is linted and tested like the rest. Read
[`benchmarks/README.md`](benchmarks/README.md) before changing them or the numbers in
`benchmarks/results/`.

## Making a release

1. Set the same version in `pyproject.toml` and `src/seamsplit/__init__.py`.
2. On GitHub, run *Actions, Release programs, Run workflow*. It builds and smoke-tests the
   program on Linux, Windows and macOS (Apple silicon and Intel) and publishes nothing. Download
   the artifacts and read the `BUILD_INFO` files: PyAV and FFmpeg versions, licence line,
   options and libraries.
3. Push a tag `vX.Y.Z`, for example `v1.0.0`. The workflow stops if the tag is not the package
   version; otherwise it attaches the programs, their SHA-256 sums, the `BUILD_INFO` files,
   `LICENSE` and `THIRD_PARTY_NOTICES.md` to the release.
4. After PyAV is updated, check that `THIRD_PARTY_NOTICES.md` still matches the libraries
   in `BUILD_INFO`.

## License

SeamSplit is under the [MIT License](LICENSE). By sending a contribution you agree that it is
licensed under the same terms.
