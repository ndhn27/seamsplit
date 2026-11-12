"""The version is written in ``pyproject.toml``, in ``seamsplit.__version__`` and in the
changelog, and what pip does with it follows PEP 440, not the order the releases were made
in: a version that sorts before a listed release (``1.1.0a`` is older than ``1.1.0``) is
never offered as an upgrade."""

import pathlib
import re

import pytest
from packaging.version import Version

import seamsplit

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("[project]", 1)[1]
    match = re.search(r'^version = "([^"]+)"$', project, re.MULTILINE)
    assert match, "no version in the [project] table"
    return match.group(1)


def _changelog_versions():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    return re.findall(r"^## \[(\d[^\]]*)\]", text, re.MULTILINE)


def test_the_two_places_the_version_is_written_agree():
    assert seamsplit.__version__ == _pyproject_version()


def test_the_version_is_written_the_way_pep_440_normalizes_it():
    # "1.2.0a" is accepted but normalizes to "1.2.0a0": two spellings of one version
    assert str(Version(seamsplit.__version__)) == seamsplit.__version__


def test_the_version_is_not_older_than_any_release_in_the_changelog():
    listed = _changelog_versions()
    assert listed, "the changelog lists no release"
    newest = max(Version(v) for v in listed)
    assert Version(seamsplit.__version__) >= newest, (
        f"{seamsplit.__version__} sorts before {newest} under PEP 440, so pip would never "
        "upgrade to it"
    )


@pytest.mark.parametrize("version", _changelog_versions())
def test_every_changelog_release_is_a_valid_version(version):
    Version(version)
