import pytest

from fake_service import build_service


@pytest.fixture(scope="session")
def service(tmp_path_factory):
    """A song split into chunks and handed back, as instrumental and vocals, by a
    pretend separation service. Read-only: tests merge from it and write elsewhere."""
    return build_service(str(tmp_path_factory.mktemp("service")))


@pytest.fixture
def make_service(tmp_path):
    """``make_service(**options)``: a service of its own (see ``fake_service``)."""

    def make(**options):
        return build_service(str(tmp_path / f"svc{len(list(tmp_path.glob('svc*')))}"), **options)

    return make


@pytest.fixture
def fake_programs(tmp_path_factory, monkeypatch):
    """``demucs`` and ``audio-separator`` on the PATH, as stand-ins that write what the
    real ones write (see ``fake_programs``); returns the folder they are in."""
    import os
    import stat
    import sys

    if sys.platform == "win32":
        pytest.skip("the stand-in programs are shell scripts")
    folder = tmp_path_factory.mktemp("bin")
    script = os.path.join(os.path.dirname(__file__), "fake_programs.py")
    for name in ("demucs", "audio-separator"):
        path = folder / name
        path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" {name} "$@"\n')
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    for var in ("FAKE_FAIL_ON", "FAKE_BAD_OUTPUT", "FAKE_SLEEP", "FAKE_LOG"):
        monkeypatch.delenv(var, raising=False)
    return folder
