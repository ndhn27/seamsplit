"""Everything that has to be separated, laid out once and shared by the experiments: the
whole songs (the reference) and the chunks of every set, with the stems of each."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .chunks import CHUNK_SETS, make_chunk_set
from .corpus import Song, load_songs

STEM_NAMES = ("vocals", "instrumental")


@dataclass
class ChunkSet:
    name: str
    manifest: Dict[str, Any]
    chunk_dir: str
    song_id: str = ""
    stems: List[Dict[str, Any]] = field(default_factory=list)  # one per chunk, from the separator

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.chunk_dir, "manifest.json")

    def processed_dir(self, work_dir: str, tag: str = "plain") -> str:
        return os.path.join(work_dir, self.song_id, self.name, f"processed_{tag}")


@dataclass
class SongWork:
    song: Song
    reference: Optional[Dict[str, Any]] = None  # the whole song, separated
    sets: Dict[str, ChunkSet] = field(default_factory=dict)


def prepare(
    songs: List[Song], work_dir: str, set_names: Optional[List[str]] = None
) -> List[SongWork]:
    """Cut every song into every chunk set (into ``work_dir``)."""
    works = []
    for song in songs:
        work = SongWork(song)
        for name in set_names or list(CHUNK_SETS):
            chunk_dir = os.path.join(work_dir, song.id, name, "chunks")
            if os.path.isdir(chunk_dir):
                shutil.rmtree(chunk_dir)
            manifest = make_chunk_set(song.path, name, chunk_dir)
            work.sets[name] = ChunkSet(name, manifest, chunk_dir, song_id=song.id)
        works.append(work)
    return works


def everything_to_separate(works: List[SongWork]) -> List[str]:
    paths: List[str] = []
    for work in works:
        paths.append(work.song.path)
        for chunk_set in work.sets.values():
            paths.extend(chunk_set.manifest["paths"])
    return paths


def separate_all(works: List[SongWork], separator: Any) -> None:
    """Separate the songs and every chunk (in one run, so the model is loaded once) and
    attach the stems."""
    paths = everything_to_separate(works)
    stems = separator.stems_for(paths)
    by_path = dict(zip(paths, stems))
    for work in works:
        work.reference = by_path[work.song.path]
        for chunk_set in work.sets.values():
            chunk_set.stems = [by_path[p] for p in chunk_set.manifest["paths"]]


def write_processed(chunk_set: ChunkSet, work_dir: str, tag: str = "plain") -> str:
    """Lay the separated chunks out as a service would hand them back: one folder with a
    ``<part>_vocals`` and a ``<part>_instrumental`` file for every chunk. Returns the folder."""
    folder = chunk_set.processed_dir(work_dir, tag)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    for part, stems in zip(chunk_set.manifest["parts"], chunk_set.stems):
        base = os.path.splitext(part)[0]
        for stem in STEM_NAMES:
            shutil.copyfile(stems[stem], os.path.join(folder, f"{base}_{stem}.flac"))
    return folder


def load_work(corpus: Dict[str, Any], corpus_dir: str, only: Optional[List[str]] = None):
    return load_songs(corpus, corpus_dir, only)


def separation_speed(works: List[SongWork]) -> Dict[str, Any]:
    """How fast the separator was, from the files whose time was recorded: seconds of audio
    separated per second of work (above 1: faster than real time)."""
    audio_s = 0.0
    work_s = 0.0
    timed = 0
    items = []
    for work in works:
        if work.reference and work.reference.get("seconds") is not None:
            items.append((work.song.seconds, work.reference["seconds"]))
        for chunk_set in work.sets.values():
            for seg, stems in zip(chunk_set.manifest["segments"], chunk_set.stems):
                if stems.get("seconds") is not None:
                    items.append((seg["duration_s"], stems["seconds"]))
    for a, w in items:
        audio_s += a
        work_s += w
        timed += 1
    return {
        "files_timed": timed,
        "audio_seconds": round(audio_s, 1),
        "work_seconds": round(work_s, 1),
        "times_real_time": round(audio_s / work_s, 2) if work_s else None,
    }
