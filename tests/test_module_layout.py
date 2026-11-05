"""``seamsplit.merger`` is the entry point of a merge, whose code lives in small modules;
the names those modules define must be importable from it."""

import importlib

import pytest

# name -> the module that defines it
MOVED = {
    "report": ["BoundaryInfo", "ChunkInfo", "MergeReport", "format_time", "_emit"],
    "checks": [
        "_JSON_TYPE_NAMES",
        "_describe",
        "_number_problem",
        "_check_number",
        "_check_samplerate",
        "_check_merge_options",
        "_check_threshold",
    ],
    "manifest": ["_part_name_problem", "load_manifest"],
    "matching": [
        "AUDIO_EXTENSIONS",
        "_list_files",
        "_stem_matches",
        "_ambiguous",
        "_filtered_by",
        "find_processed_file",
        "find_original_chunks",
        "_natural_key",
        "_named_list",
        "find_processed_files",
        "PairedFiles",
        "pair_processed_files",
    ],
    "dsp": [
        "MIN_ALIGN_CORRELATION",
        "LEVEL_DEADBAND_DB",
        "_SLERP_EPS",
        "_SEAM_GUARD_FRAC",
        "_SEAM_FLOOR_DB",
        "_SEAM_TIE",
        "_MATCH_CAP_DB",
        "_DTYPES",
        "_to_int_array",
        "_to_mono",
        "estimate_lag",
        "_linear_gains",
        "_equal_power_gains",
        "_adaptive_gains",
        "_fade_gains",
        "_overlapping_parts",
        "_ncc",
        "_level_ratio",
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
        "_crossfade_merge",
        "adaptive_crossfade_merge",
        "linear_crossfade_merge",
        "equal_power_crossfade_merge",
    ],
    "plan": ["DEFAULT_LENGTH_TOLERANCE_MS", "_check_part_length", "MergePlan"],
    "merger": ["merge_audio"],
}
CASES = [(module, name) for module, names in MOVED.items() for name in names]


@pytest.mark.parametrize("module, name", CASES)
def test_every_name_is_still_importable_from_seamsplit_merger(module, name):
    merger = importlib.import_module("seamsplit.merger")
    defined_in = importlib.import_module(f"seamsplit.{module}")
    assert getattr(merger, name) is getattr(defined_in, name)


def test_the_modules_stay_small_enough_to_read():
    import pathlib

    import seamsplit

    root = pathlib.Path(seamsplit.__file__).parent
    modules = list(MOVED) + [
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
