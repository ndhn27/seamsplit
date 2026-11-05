"""The real music the benchmarks use: where it comes from, fetching it, reading it.

``corpus.json`` names every file by repository, commit and path (so the audio is fetched
from where it is published, at a fixed version) and by SHA-256 (so a changed or damaged
file is noticed). The audio itself is not part of SeamSplit.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from seamsplit.audio import AudioSegment

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(HERE)
CORPUS_FILE = os.path.join(BENCH_DIR, "corpus.json")
DEFAULT_CORPUS_DIR = os.path.join(BENCH_DIR, "corpus")  # not in the repository (see .gitignore)


def load_corpus(path: str = CORPUS_FILE) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def raw_url(entry: Dict[str, Any]) -> str:
    """Where ``entry`` is published: the file at its commit, as GitHub serves it raw."""
    quoted = urllib.parse.quote(entry["path"])
    return f"https://raw.githubusercontent.com/{entry['repo']}/{entry['commit']}/{quoted}"


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(song: Dict[str, Any], role: str, corpus_dir: str) -> str:
    return os.path.join(corpus_dir, song["id"], f"{role}.mp3")


def fetch(corpus: Dict[str, Any], corpus_dir: str = DEFAULT_CORPUS_DIR, say=print) -> List[str]:
    """Download every file that is not there (or whose hash is wrong) and check it. Returns
    the paths. Raises :class:`ValueError` for a download that does not match its hash."""
    paths = []
    for song in corpus["songs"]:
        for role, entry in song["files"].items():
            path = local_path(song, role, corpus_dir)
            paths.append(path)
            if os.path.isfile(path) and sha256_of(path) == entry["sha256"]:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            say(f"fetching {song['id']}/{role} from {entry['repo']}@{entry['commit'][:8]}")
            with urllib.request.urlopen(raw_url(entry), timeout=120) as reply:
                data = reply.read()
            with open(path, "wb") as f:
                f.write(data)
            if sha256_of(path) != entry["sha256"]:
                os.remove(path)
                raise ValueError(f"{song['id']}/{role}: the download does not match its sha256")
    return paths


@dataclass
class Song:
    """A song of the corpus, decoded. ``vocals`` and ``instrumental`` are the ground-truth
    stems, ``None`` when the song has none."""

    id: str
    title: str
    path: str
    mix: AudioSegment
    vocals: Optional[AudioSegment]
    instrumental: Optional[AudioSegment]
    redistributable: bool

    @property
    def seconds(self) -> float:
        return len(self.mix) / 1000.0


def load_songs(
    corpus: Dict[str, Any], corpus_dir: str = DEFAULT_CORPUS_DIR, only: Optional[List[str]] = None
) -> List[Song]:
    songs = []
    for entry in corpus["songs"]:
        if only and entry["id"] not in only:
            continue
        files = entry["files"]
        paths = {role: local_path(entry, role, corpus_dir) for role in files}
        missing = [p for p in paths.values() if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"{missing[0]} is missing: run 'python benchmarks/run.py fetch' first"
            )
        songs.append(
            Song(
                id=entry["id"],
                title=entry["title"],
                path=paths["mix"],
                mix=AudioSegment.from_file(paths["mix"]),
                vocals=AudioSegment.from_file(paths["vocals"]) if "vocals" in paths else None,
                instrumental=(
                    AudioSegment.from_file(paths["instrumental"])
                    if "instrumental" in paths
                    else None
                ),
                redistributable=bool(entry.get("redistributable")),
            )
        )
    return songs
