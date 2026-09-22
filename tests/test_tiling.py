"""Tests for src/tiling.py, in particular fold_steel_combined.

RUN THESE IN notebooks/03_tiling.ipynb -- there is no local Python
environment for this project, so nothing here is executed on the machine
that wrote it.

Most of these tests are pure unit tests over synthetic tile records: they
construct a fake ``index`` dict by hand (the same shape ``build_index()``
returns) rather than reading real images or boundary maps, so they run
anywhere, with no data on disk. The one exception is
``test_existing_folds_byte_identical_with_and_without_steel_combined``, which
needs the real audit/extraction/GT pipeline and skips itself when that is
absent, per this project's existing convention (see tests/test_dataset.py).
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

import pytest

from src import tiling


# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------
def _synthetic_tiles():
    """A small, hand-built pool of Steel1 + Steel2 tiles.

    Shaped exactly like ``tiling.all_tiles(index)`` output. Steel1 gets 6
    parents at 10 tiles each (60 tiles); Steel2 gets 4 parents at 25 tiles
    each (100 tiles) -- a deliberately different tiles-per-parent ratio, the
    same shape of imbalance the real datasets have (Steel1 ~47/parent,
    Steel2 ~126/parent), so a test that only passed by coincidence on equal
    parent sizes would fail here.
    """
    tiles = []
    for i in range(6):
        parent = f"steel1_{i:02d}"
        for j in range(10):
            tiles.append({
                "tile_id": f"Steel1/{parent}/x{j}",
                "dataset": "Steel1",
                "parent_id": parent,
                "source_image": f"{parent}.png",
                "x": j, "y": 0,
                "boundary_fraction": 0.05,
                "split": "",
            })
    for i in range(4):
        parent = f"steel2_{i:02d}"
        for j in range(25):
            tiles.append({
                "tile_id": f"Steel2/{parent}/x{j}",
                "dataset": "Steel2",
                "parent_id": parent,
                "source_image": f"{parent}.png",
                "x": j, "y": 0,
                "boundary_fraction": 0.10,
                "split": "",
            })
    return tiles


def _synthetic_index():
    tiles = _synthetic_tiles()
    by_dataset = {}
    for t in tiles:
        by_dataset.setdefault(t["dataset"], []).append(t)
    return {
        "datasets": {name: {"tiles": rows} for name, rows in by_dataset.items()},
        "settings": {},
        "failures": {},
    }


def _settings(**overrides):
    settings = copy.deepcopy(tiling.DEFAULTS)
    settings.update(overrides)
    return settings


def _parent_tiles_from_index(index):
    by_dataset = {}
    for t in tiling.all_tiles(index):
        by_dataset.setdefault(t["dataset"], {})
        by_dataset[t["dataset"]][t["parent_id"]] = (
            by_dataset[t["dataset"]].get(t["parent_id"], 0) + 1)
    return by_dataset


# --------------------------------------------------------------------------
# allocate_parents_by_tile_ratio
# --------------------------------------------------------------------------
def test_no_parent_in_more_than_one_split():
    index = _synthetic_index()
    parent_tiles = _parent_tiles_from_index(index)
    assignment = tiling.allocate_parents_by_tile_ratio(
        parent_tiles, {"train": 0.70, "val": 0.15, "test": 0.15}, seed=0,
        min_parents_per_split=1)

    seen = {}
    for split, per_dataset in assignment.items():
        for ds, parents in per_dataset.items():
            for p in parents:
                key = (ds, p)
                assert key not in seen, (
                    f"parent {key} appears in both {seen[key]} and {split}")
                seen[key] = split

    # every parent from the input is accounted for exactly once
    all_input_parents = {(ds, p) for ds, parents in parent_tiles.items()
                         for p in parents}
    assert set(seen) == all_input_parents


def test_every_split_has_both_datasets():
    index = _synthetic_index()
    parent_tiles = _parent_tiles_from_index(index)
    assignment = tiling.allocate_parents_by_tile_ratio(
        parent_tiles, {"train": 0.70, "val": 0.15, "test": 0.15}, seed=0,
        min_parents_per_split=1)

    for split in ("train", "val", "test"):
        assert assignment[split].get("Steel1"), (
            f"{split} has no Steel1 parent: {assignment[split]}")
        assert assignment[split].get("Steel2"), (
            f"{split} has no Steel2 parent: {assignment[split]}")


def test_too_few_parents_for_the_guarantee_raises():
    # Steel2 has 4 parents; asking for 2 minimum per split needs 6.
    parent_tiles = {"Steel1": {f"p{i}": 10 for i in range(6)},
                    "Steel2": {f"p{i}": 25 for i in range(4)}}
    with pytest.raises(tiling.TilingError):
        tiling.allocate_parents_by_tile_ratio(
            parent_tiles, {"train": 0.70, "val": 0.15, "test": 0.15}, seed=0,
            min_parents_per_split=2)


def test_fracs_must_sum_to_one():
    parent_tiles = {"Steel1": {"p0": 10, "p1": 10, "p2": 10}}
    with pytest.raises(tiling.TilingError):
        tiling.allocate_parents_by_tile_ratio(
            parent_tiles, {"train": 0.5, "val": 0.5, "test": 0.5}, seed=0,
            min_parents_per_split=1)


def test_allocation_is_deterministic_across_runs():
    index = _synthetic_index()
    parent_tiles = _parent_tiles_from_index(index)
    fracs = {"train": 0.70, "val": 0.15, "test": 0.15}
    first = tiling.allocate_parents_by_tile_ratio(
        parent_tiles, fracs, seed=0, min_parents_per_split=1)
    second = tiling.allocate_parents_by_tile_ratio(
        parent_tiles, fracs, seed=0, min_parents_per_split=1)
    assert first == second


def test_allocation_reaches_the_target_ratio_roughly():
    """Not exact -- parents are lumpy -- but the greedy balance should land
    within a wide, generous tolerance of the requested tile-count shares."""
    index = _synthetic_index()
    parent_tiles = _parent_tiles_from_index(index)
    assignment = tiling.allocate_parents_by_tile_ratio(
        parent_tiles, {"train": 0.70, "val": 0.15, "test": 0.15}, seed=0,
        min_parents_per_split=1)

    total = sum(n for ds in parent_tiles.values() for n in ds.values())
    for split, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        n_tiles = sum(parent_tiles[ds][p] for ds, parents in assignment[split].items()
                     for p in parents)
        share = n_tiles / total
        assert abs(share - target) < 0.25, (
            f"{split}: {share:.3f} vs target {target}")


# --------------------------------------------------------------------------
# build_steel_combined_fold
# --------------------------------------------------------------------------
def test_build_steel_combined_fold_rows_partition_train_val_test():
    index = _synthetic_index()
    settings = _settings()
    fold = tiling.build_steel_combined_fold(index, settings)

    assert fold["name"] == "fold_steel_combined"
    assert fold["alias_of"] is None
    # held_out is a descriptive label, never one of the pooled dataset names
    assert fold["held_out"] not in ("Steel1", "Steel2")

    train_parents = {(r["dataset"], r["parent_id"])
                     for r in fold["rows"] if r["split"] == "train"}
    val_parents = {(r["dataset"], r["parent_id"])
                   for r in fold["rows"] if r["split"] == "val"}
    test_parents = {(r["dataset"], r["parent_id"]) for r in fold["test_rows"]}

    assert not (train_parents & val_parents)
    assert not (train_parents & test_parents)
    assert not (val_parents & test_parents)

    all_input_tiles = tiling.all_tiles(index)
    assert len(fold["rows"]) + len(fold["test_rows"]) == len(all_input_tiles)

    for split_name, parents in (("train", train_parents), ("val", val_parents),
                                ("test", test_parents)):
        datasets_present = {ds for ds, _ in parents}
        assert datasets_present == {"Steel1", "Steel2"}, (
            f"{split_name} is missing a dataset: {datasets_present}")


def test_build_steel_combined_fold_missing_dataset_raises():
    index = _synthetic_index()
    del index["datasets"]["Steel2"]
    settings = _settings()
    with pytest.raises(tiling.TilingError):
        tiling.build_steel_combined_fold(index, settings)


def test_build_steel_combined_fold_is_deterministic():
    index = _synthetic_index()
    settings = _settings()
    first = tiling.build_steel_combined_fold(index, settings)
    second = tiling.build_steel_combined_fold(index, settings)
    assert first["parent_allocation"] == second["parent_allocation"]
    assert sorted(r["tile_id"] for r in first["rows"]) == \
        sorted(r["tile_id"] for r in second["rows"])
    assert sorted(r["tile_id"] for r in first["test_rows"]) == \
        sorted(r["tile_id"] for r in second["test_rows"])


# --------------------------------------------------------------------------
# pos_weight: computed fresh, never inherited
# --------------------------------------------------------------------------
def test_pos_weight_is_computed_from_this_folds_own_train_split():
    index = _synthetic_index()
    settings = _settings()
    fold = tiling.build_steel_combined_fold(index, settings)

    folds = {"folds": {fold["name"]: fold},
             "test": {"name": "test", "datasets": [], "rows": []}}
    stats = tiling.fold_statistics(folds, settings)
    combined_pw = stats[fold["name"]]["pos_weight"]

    train_rows = [r for r in fold["rows"] if r["split"] == "train"]
    expected = round(tiling.pos_weight(train_rows), 3)
    assert combined_pw == expected

    # Not the same as either dataset's OWN pos_weight computed in isolation
    # (Steel1 tiles are boundary_fraction=0.05, Steel2's are 0.10 in this
    # fixture) -- proving the fold's value comes from ITS pooled train split,
    # not a value copied from a Steel1-only or Steel2-only computation.
    steel1_only = round(tiling.pos_weight(
        [r for r in train_rows if r["dataset"] == "Steel1"]), 3)
    steel2_only = round(tiling.pos_weight(
        [r for r in train_rows if r["dataset"] == "Steel2"]), 3)
    assert combined_pw != steel1_only
    assert combined_pw != steel2_only


def test_steel_combined_test_slice_statistics_shape():
    index = _synthetic_index()
    settings = _settings()
    fold = tiling.build_steel_combined_fold(index, settings)
    ts = tiling.test_slice_statistics(fold["test_rows"], fold["datasets"], "note")
    assert ts["n_tiles"] == len(fold["test_rows"])
    assert set(ts["datasets"]) == {"Steel1", "Steel2"}
    assert ts["note"] == "note"


# --------------------------------------------------------------------------
# existing folds unaffected -- needs the real pipeline, skips without it
# --------------------------------------------------------------------------
def _real_pipeline_available():
    try:
        from src import boundary_gt
        from src import paths as paths_mod

        resolved = paths_mod.resolve_paths()
        reports_dir = Path(resolved["repo_root"]) / "reports"
        audit = boundary_gt.load_audit(reports_dir)
        extraction = tiling.load_extraction(reports_dir)
        gt_root = Path(resolved["gt_boundaries_root"])
        needed = set(tiling.load_config()["lodo_datasets"]) | {"Steel1", "Steel2"}
        for name in needed:
            folder = Path(audit["datasets"][name]["path"])
            if not folder.is_dir() or not (gt_root / name).is_dir():
                return False, None, None, None
        return True, resolved, audit, extraction
    except Exception:
        return False, None, None, None


def test_existing_folds_byte_identical_with_and_without_steel_combined():
    """The change is additive: building fold_steel_combined must not alter a
    single byte of any other fold's manifest or fold_stats.yaml entry.

    Runs the real step-3 pipeline twice into two throwaway directories --
    once the old way (no fold_steel_combined), once with it added -- and
    diffs every pre-existing manifest plus the non-steel_combined keys of
    fold_stats.yaml between them. Skips if the real audit/extraction/GT
    pipeline is not available on this machine (there is no local Python
    environment for this project; this test is meant to run on the host via
    notebooks/03_tiling.ipynb).
    """
    available, resolved, audit, extraction = _real_pipeline_available()
    if not available:
        pytest.skip("real audit.json/gt_extraction.json/GT boundary maps not "
                    "available in this environment")

    import yaml

    settings = tiling.load_config()
    gt_root = Path(resolved["gt_boundaries_root"])
    index = tiling.build_index(extraction, audit, gt_root, settings)

    tmp_before = Path(tempfile.mkdtemp(prefix="tiling_before_"))
    tmp_after = Path(tempfile.mkdtemp(prefix="tiling_after_"))
    try:
        # before: exactly what build_folds()/fold_statistics() have always done
        folds_before = tiling.build_folds(index, settings)
        stats_before = tiling.fold_statistics(folds_before, settings)
        tiling.write_manifests(folds_before, tmp_before / "manifests")
        tiling.write_fold_stats(stats_before, folds_before, settings, tmp_before)

        # after: the same, plus fold_steel_combined inserted additively
        folds_after = tiling.build_folds(index, settings)
        steel_fold = tiling.build_steel_combined_fold(index, settings)
        folds_after["folds"][steel_fold["name"]] = steel_fold
        stats_after = tiling.fold_statistics(folds_after, settings)
        stats_after[steel_fold["name"]]["test"] = tiling.test_slice_statistics(
            steel_fold["test_rows"], steel_fold["datasets"], "note")
        stats_after[steel_fold["name"]]["parent_allocation"] = \
            steel_fold["parent_allocation"]
        tiling.write_manifests(folds_after, tmp_after / "manifests")
        tiling.write_manifest(steel_fold["test_rows"],
                              tmp_after / "manifests" / "fold_steel_combined_test.csv")
        tiling.write_fold_stats(stats_after, folds_after, settings, tmp_after)

        # every manifest that existed before must be byte-identical after
        before_manifests = sorted((tmp_before / "manifests").glob("*.csv"))
        assert before_manifests, "no manifests were written to diff"
        for path in before_manifests:
            after_path = tmp_after / "manifests" / path.name
            assert after_path.is_file(), f"{path.name} missing after the change"
            assert path.read_bytes() == after_path.read_bytes(), (
                f"{path.name} changed after adding fold_steel_combined")
        # the new fold's manifests must exist and must NOT have existed before
        assert (tmp_after / "manifests" / "fold_steel_combined.csv").is_file()
        assert (tmp_after / "manifests" / "fold_steel_combined_test.csv").is_file()
        assert not (tmp_before / "manifests" / "fold_steel_combined.csv").is_file()

        doc_before = yaml.safe_load((tmp_before / "fold_stats.yaml").read_text())
        doc_after = yaml.safe_load((tmp_after / "fold_stats.yaml").read_text())
        del doc_before["generated_utc"]
        del doc_after["generated_utc"]
        after_folds = dict(doc_after["folds"])
        del after_folds[steel_fold["name"]]
        before_folds = doc_before["folds"]
        assert after_folds == before_folds, (
            "an existing fold's fold_stats.yaml entry changed")
        doc_before_no_folds = {k: v for k, v in doc_before.items() if k != "folds"}
        doc_after_no_folds = {k: v for k, v in doc_after.items() if k != "folds"}
        assert doc_before_no_folds == doc_after_no_folds, (
            "top-level fold_stats.yaml keys (e.g. parent_splits) changed")
    finally:
        shutil.rmtree(tmp_before, ignore_errors=True)
        shutil.rmtree(tmp_after, ignore_errors=True)
