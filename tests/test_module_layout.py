"""``seamsplit.merger`` is the entry point of a merge, whose code lives in small modules.
The public names those modules define for a merge stay importable from it, and ``__all__``
is the list: a name with a leading underscore is private to its module and is not part of it."""

import importlib

import pytest

# name -> the module that defines it (public names only: this is the contract)
MOVED = {
    "report": ["BoundaryInfo", "ChunkInfo", "MergeReport", "format_time"],
    "manifest": ["load_manifest"],
    "matching": [
        "AUDIO_EXTENSIONS",
        "find_processed_file",
        "find_original_chunks",
        "find_processed_files",
        "PairedFiles",
        "pair_processed_files",
    ],
    "dsp": [
        "MIN_ALIGN_CORRELATION",
        "LEVEL_DEADBAND_DB",
        "estimate_lag",
        "match_db",
        "find_seam",
    ],
    "crossfade": [
        "METHODS",
        "DEFAULT_METHOD",
        "DEFAULT_MAX_LAG_MS",
        "DEFAULT_MAX_GAIN_DB",
        "DEFAULT_SEAM_FADE_MS",
        "DEFAULT_SEAM_WARN_DB",
        "adaptive_crossfade_merge",
        "linear_crossfade_merge",
        "equal_power_crossfade_merge",
    ],
    "plan": ["DEFAULT_LENGTH_TOLERANCE_MS", "MergePlan"],
    "merger": ["merge_audio"],
}
CASES = [(module, name) for module, names in MOVED.items() for name in names]


@pytest.mark.parametrize("module, name", CASES)
def test_every_public_name_is_importable_from_seamsplit_merger(module, name):
    merger = importlib.import_module("seamsplit.merger")
    defined_in = importlib.import_module(f"seamsplit.{module}")
    assert getattr(merger, name) is getattr(defined_in, name)


def test_all_lists_exactly_the_public_names_and_none_that_is_private():
    merger = importlib.import_module("seamsplit.merger")
    expected = sorted(name for _, name in CASES)
    assert sorted(merger.__all__) == expected
    assert not [name for name in merger.__all__ if name.startswith("_")]


def test_star_import_gives_the_public_names_and_not_the_private_helpers():
    namespace: dict = {}
    exec("from seamsplit.merger import *", namespace)
    names = {name for name in namespace if name != "__builtins__"}
    assert names == {name for _, name in CASES}


def test_the_modules_stay_small_enough_to_read():
    import pathlib

    import seamsplit

    root = pathlib.Path(seamsplit.__file__).parent
    modules = list(MOVED) + [
        "checks",
        "cutpoints",
        "audio",
        "audio_io",
        "splitter",
        "anchor",
        "audition",
        "choices",
        "pairing",
        "services",
    ]
    for module in modules:
        assert len((root / f"{module}.py").read_text().splitlines()) < 600, module
