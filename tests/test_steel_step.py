"""The finalized steel step: locked config, the final report's provenance, and the helpers
that make 06d_steel_combined.ipynb safe for a fresh "Run all".

Nothing here is executed on the machine that wrote it (see CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "steel_step_final.json"
TRAIN_REPORT = ROOT / "reports" / "train_fold_steel_combined_modea_colab.json"
GAP = ROOT / "reports" / "gap_closing_fold_steel_combined_modea_colab.json"


# --------------------------------------------------------------------------
# reports/steel_step_final.json: every number is a copy, never a new one
# --------------------------------------------------------------------------
def _entries(obj, path=()):
    if isinstance(obj, dict):
        if {"value", "source", "key"} <= set(obj):
            yield path, obj
            return
        for k, v in obj.items():
            yield from _entries(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _entries(v, path + (i,))


def _walk(obj, key):
    for k in key:
        if isinstance(k, dict):
            hits = [r for r in obj if all(r.get(a) == b for a, b in k.items())]
            assert len(hits) == 1, f"selector {k} matched {len(hits)} rows"
            obj = hits[0]
        else:
            obj = obj[k]
    return obj


def _report():
    if not REPORT.is_file():
        pytest.skip(f"{REPORT} not found")
    return json.loads(REPORT.read_text())


FLOAT_TOLERANCE = 5e-3
INT_TOLERANCE_ABS, INT_TOLERANCE_REL = 3, 0.01

#: Reports whose numbers are computed from a checkpoint's PREDICTIONS (by file-name prefix).
PREDICTION_SOURCES = ("region_metrics_", "postprocess_sweep_", "fn_attribution_",
                      "gap_closing_", "modea_retrain_comparison_")
#: Integers that count the DATA or identify the run, even when a prediction report records
#: them: they cannot move with a re-run, so they are compared exactly.
STRUCTURAL_KEYS = frozenset({"tiles", "n_tiles", "parents", "n_parents", "files", "n_files",
                             "epoch", "best_epoch", "folds", "fold_size", "size", "seed"})


def _is_prediction_derived(source, key):
    last = key[-1] if key else None
    return (Path(source).name.startswith(PREDICTION_SOURCES)
            and not (isinstance(last, str) and last in STRUCTURAL_KEYS))


def _allowed(expected, actual, source, key):
    """The absolute difference a recorded value may have from its source, or 0 for exact."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return 0
    if isinstance(expected, float) or isinstance(actual, float):
        # a float above 1 from a prediction report is a COUNT-scale quantity (fn_per_tile ~ 24),
        # not a 0..1 score: it gets the integers' relative allowance. Scores keep the absolute one.
        magnitude = max(abs(expected), abs(actual))
        if magnitude > 1 and _is_prediction_derived(source, key):
            return max(FLOAT_TOLERANCE, INT_TOLERANCE_REL * magnitude)
        return FLOAT_TOLERANCE
    if isinstance(expected, int) and isinstance(actual, int) and _is_prediction_derived(
            source, key):
        return max(INT_TOLERANCE_ABS, INT_TOLERANCE_REL * max(abs(expected), abs(actual)))
    return 0


def _mismatches(expected, actual, source, key, where=()):
    """Every disagreement between a recorded value and its source, as dicts (empty = none)."""
    here = ".".join(str(w) for w in where) or "<value>"
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for k in sorted(set(expected) | set(actual), key=str):
            if k not in expected or k not in actual:
                out.append({"path": ".".join(str(w) for w in where + (k,)), "recorded": expected.get(k, "<absent>"),
                            "source": actual.get(k, "<absent>"), "allowed": 0})
            else:
                out += _mismatches(expected[k], actual[k], source, key, where + (k,))
        return out
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return [{"path": here, "recorded": f"len {len(expected)}",
                     "source": f"len {len(actual)}", "allowed": 0}]
        out = []
        for i, (x, y) in enumerate(zip(expected, actual)):
            out += _mismatches(x, y, source, key, where + (i,))
        return out
    allowed = _allowed(expected, actual, source, key)
    numeric = (isinstance(expected, (int, float)) and isinstance(actual, (int, float))
               and not isinstance(expected, bool) and not isinstance(actual, bool))
    same = abs(expected - actual) <= allowed if numeric else (
        expected == actual and type(expected) is type(actual))
    if same:
        return []
    return [{"path": here, "recorded": expected, "source": actual, "allowed": allowed}]


def test_every_number_in_the_final_report_matches_its_committed_source():
    """The report copies each number from a committed source report; this checks it still
    agrees. Re-running 06d_steel_combined.ipynb rewrites the region / post-processing /
    attribution reports and evaluating one checkpoint on the GPU is not bit-reproducible, so
    what is PRODUCED BY THE MODEL moves a little on every "Run all" and exact matching would
    fail with nothing really changed. The rule:

    * floats (scores in 0..1): absolute tolerance 5e-3; a float above 1 from a prediction report
      (a per-tile count such as fn_per_tile) gets the integers' allowance below instead;
    * integers derived from predictions (any source named region_metrics_*, postprocess_sweep_*,
      fn_attribution_*, gap_closing_*, modea_retrain_comparison_*: region counts, FN-class
      counts): max(3, 1% of the larger value);
    * STRUCTURAL integers (tile counts, parents, files, epochs, fold sizes -- ``STRUCTURAL_KEYS``)
      and everything else -- strings, booleans, verdict labels, hashes, commit ids, a recorded
      settings dict: EXACT. A verdict flipping MISPLACED -> OFFSET is a real change, not noise.

    Every mismatch is collected and reported together, so one run shows all of them."""
    entries = [(p, e) for p, e in _entries(_report()) if "commit" not in e]
    assert len(entries) > 40
    bad = []
    for path, e in entries:
        source = json.loads((ROOT / e["source"]).read_text())
        for m in _mismatches(e["value"], _walk(source, e["key"]), e["source"], e["key"]):
            bad.append({"entry": ".".join(str(x) for x in path), "source_file": e["source"],
                        "key": e["key"], **m})
    assert not bad, f"{len(bad)} recorded number(s) differ from their source:\n" + "\n".join(
        f"  {m['entry']} [{m['path']}]: recorded {m['recorded']!r} vs source {m['source']!r} "
        f"(allowed +-{m['allowed']:g}) <- {m['source_file']} {m['key']}" for m in bad)


def test_the_provenance_comparison_tolerates_model_noise_but_not_real_differences():
    fn = "reports/fn_attribution_fold_steel_combined_modea_colab.json"
    rm = "reports/region_metrics_fold_steel_combined_modea_colab.json"
    cfg = "reports/gt_extraction.json"

    def bad(expected, actual, source, key):
        return _mismatches(expected, actual, source, key)

    assert not bad(0.4095, 0.4121, rm, ["x"])                      # float noise 0.0026
    assert bad(0.4095, 0.4146, rm, ["x"])                          # past 5e-3
    assert not bad(2096, 2094, fn, ["Steel2", "counts", "MERGED"])  # the Colab failure
    assert not bad(2096, 2096 + 20, fn, ["Steel2", "counts", "MERGED"])   # 20 <= 1% of 2116
    assert bad(2096, 2096 + 22, fn, ["Steel2", "counts", "MERGED"])
    assert not bad(23.904761904761905, 23.896825396825395, fn, ["Steel2", "fn_per_tile"])
    assert bad(23.9, 24.2, fn, ["Steel2", "fn_per_tile"])          # 0.3 > 1% of 24.2
    assert bad(23.9, 23.9 + 0.01, cfg, ["fn_per_tile"])            # not a prediction report
    assert not bad(10, 13, fn, ["a", "n"]) and bad(10, 14, fn, ["a", "n"])  # floor of 3
    # structural integers stay exact even inside a prediction report
    assert bad(96, 95, rm, ["decomposition", "Steel1", "tiles"])
    assert bad(32, 33, rm, ["epoch"])
    assert bad(2, 3, "reports/modea_retrain_comparison_colab.json",
               ["test_tiles", "Steel1", "parents"])
    # a source that is not a prediction report is exact
    assert bad(2096, 2094, cfg, ["settings", "n"])
    # strings, booleans, labels
    assert bad("MISPLACED", "OFFSET", rm, ["decomposition", "Steel1", "label"])
    assert bad(True, False, rm, ["flag"]) and bad(1, True, fn, ["Steel2", "counts", "X"])
    # nested values, with the path of each mismatch named
    got = bad({"a": [1, 0.1], "b": "x"}, {"a": [1, 0.2], "b": "y"}, rm, ["k"])
    assert sorted(m["path"] for m in got) == ["a.1", "b"]
    assert [m["path"] for m in bad({"a": 1}, {"a": 1, "z": 2}, cfg, ["k"])] == ["z"]


def test_numbers_taken_from_git_history_match_that_commit():
    entries = [(p, e) for p, e in _entries(_report()) if "commit" in e]
    assert entries, "the connectivity=1 evidence is quoted from git history"
    if shutil.which("git") is None:
        pytest.skip("git not available")
    for path, e in entries:
        out = subprocess.run(["git", "show", f"{e['commit']}:{e['source']}"], cwd=ROOT,
                             capture_output=True, text=True)
        if out.returncode != 0:
            pytest.skip(f"commit {e['commit']} not in this clone (shallow?): {out.stderr.strip()}")
        assert _walk(json.loads(out.stdout), e["key"]) == e["value"], (path, e["commit"])


def test_the_markdown_report_states_the_numbers_of_the_json_twin():
    payload = _report()
    md = (ROOT / "reports" / "steel_step_final.md").read_text()
    for d in ("Steel1", "Steel2"):
        row = payload["final_test"][d]
        for key in ("pixel_dice", "skeleton_dice", "pq", "sq", "rq"):
            assert f"{row[key]['value']:.4f}" in md, (d, key)
        assert row["verdict"]["value"] in md
    for source in {e["source"] for _, e in _entries(payload)}:
        assert Path(source).name in md, f"{source} is used but never named in the markdown"


# --------------------------------------------------------------------------
# the locked config
# --------------------------------------------------------------------------
def test_load_steel_step_reads_the_committed_decisions():
    pytest.importorskip("torch")
    from src import train as train_mod

    step = train_mod.load_steel_step()
    assert step["default_fold"] == "fold_steel_combined_modea"
    assert step["region_path"] == {"partition": "watershed_prob",
                                   "watershed_marker_threshold": {"Steel1": 0.6, "Steel2": 0.6},
                                   "postprocess": "none"}
    # the expected fingerprint is that of the committed extraction record
    blob = (ROOT / "reports" / "gt_extraction.json").read_bytes()
    assert step["expected_gt_sha256"] == hashlib.sha256(blob).hexdigest()[:16]
    # and the markers are the ones VAL selected
    gap = json.loads(GAP.read_text())
    for d, m in step["region_path"]["watershed_marker_threshold"].items():
        assert gap["marker"]["selected"][d]["marker_threshold"] == m


def test_the_committed_ground_truth_resolves_to_the_decided_fold():
    from src import tiling

    settings = tiling.load_config()
    extraction = tiling.load_extraction(ROOT / "reports")
    assert tiling.apply_gt_variant(settings, extraction)["steel_combined"]["name"] == \
        "fold_steel_combined_modea"


def test_min_region_area_px_is_off_in_config_and_in_the_recorded_ground_truth():
    from src import boundary_gt

    assert boundary_gt.load_config()["min_region_area_px"] == {}
    recorded = json.loads((ROOT / "reports" / "gt_extraction.json").read_text())
    assert recorded["settings"]["min_region_area_px"] == {}
    assert boundary_gt.load_config()["mode_a_line_class"] == {"Steel2": 8}


@pytest.mark.parametrize("section, message", [
    ("steel_step:\n  default_fold: f\n  region_path: {}\n", "missing"),
    ("steel_step:\n  default_fold: f\n  expected_gt_sha256: xyz\n  region_path: {}\n",
     "16-hex-digit"),
    ("steel_step:\n  default_fold: f\n  expected_gt_sha256: 53155b40c2b6ebf8\n"
     "  region_path:\n    partition: flood\n    watershed_marker_threshold: {A: 0.6}\n"
     "    postprocess: none\n", "partition"),
    ("steel_step:\n  default_fold: f\n  expected_gt_sha256: 53155b40c2b6ebf8\n"
     "  region_path:\n    partition: watershed_prob\n    watershed_marker_threshold: {A: 1.6}\n"
     "    postprocess: none\n", r"\(0, 1\)"),
    ("steel_step:\n  default_fold: f\n  expected_gt_sha256: 53155b40c2b6ebf8\n"
     "  region_path:\n    partition: watershed_prob\n    watershed_marker_threshold: {A: 0.6}\n"
     "    postprocess: thin_harder\n", "postprocess"),
    ("train:\n  seed: 0\n", "no steel_step"),
])
def test_load_steel_step_refuses_a_malformed_section(tmp_path, section, message):
    pytest.importorskip("torch")
    from src import train as train_mod

    path = tmp_path / "default.yaml"
    path.write_text(section)
    with pytest.raises(train_mod.TrainError, match=message):
        train_mod.load_steel_step(path)


def test_config_hash_of_the_trained_run_is_unchanged():
    """The hash the fold_steel_combined_modea best.pt was trained under, recomputed from the
    configs in this checkout exactly as Trainer.__init__ does. If this fails, a config change
    has invalidated the existing checkpoint."""
    pytest.importorskip("torch")
    import yaml

    from src import dataset as ds
    from src import losses as losses_mod
    from src import model as model_mod
    from src import paths as paths_mod
    from src import train as train_mod

    recorded = json.loads(TRAIN_REPORT.read_text())
    overlay = ROOT / "configs" / f"{paths_mod.detect_platform()}.yaml"
    if overlay.is_file():
        touched = set((yaml.safe_load(overlay.read_text()) or {})) & {
            "model", "loss", "train", "dataset", "augment", "patch"}
        if touched:
            pytest.skip(f"{overlay.name} overrides hashed sections {sorted(touched)}; the "
                        f"recorded hash is for {recorded['platform']}")
    fold = recorded["fold"]
    settings = train_mod.load_config()
    fold_entry = ds.load_fold_stats(ROOT / "configs")["folds"][fold]
    hashed_train = dict(settings)
    hashed_train["exclude_datasets"] = list(
        train_mod.resolve_exclusions(fold, settings, fold_entry))
    digest, payload = train_mod.config_hash(model_mod.load_config(), losses_mod.load_config(),
                                            hashed_train, ds.load_config())
    assert json.loads(json.dumps(payload, default=str)) == recorded["hashed_config"]
    assert digest == recorded["config_hash"] == "067fecc2e06a7d0a"


# --------------------------------------------------------------------------
# helpers behind the "Run all" safety
# --------------------------------------------------------------------------
def test_run_completion():
    pytest.importorskip("torch")
    from src import train as train_mod

    done = [{"epoch": e, "stopped_early": False} for e in range(40)]
    assert train_mod.run_completion(done, 40)[0] is True
    assert train_mod.run_completion(done[:39], 40) == (False, "39 of 40 epochs done")
    early = [{"epoch": e, "stopped_early": e == 11} for e in range(12)]
    ok, why = train_mod.run_completion(early, 40)
    assert ok and "stopped early" in why, "an early-stopped run must not be resumed"
    assert train_mod.run_completion([], 40)[0] is False
    with pytest.raises(train_mod.TrainError):
        train_mod.run_completion(done, 0)


def test_run_completion_on_the_committed_training_history():
    pytest.importorskip("torch")
    from src import train as train_mod

    recorded = json.loads(TRAIN_REPORT.read_text())
    ok, why = train_mod.run_completion(recorded["history"], recorded["summary"]["epochs"])
    assert ok, why


LINE = {"class_index": 1, "colour": [0, 0, 0], "blob_thickness_px": 8.0}


def _gt(tmp_path, record=None, backup=False, files=True):
    from src import boundary_gt

    if backup:
        b = tmp_path / boundary_gt.MODE_B_BACKUP_SUBDIR / "Steel2"
        b.mkdir(parents=True)
        if files:
            (b / "a.png").write_bytes(b"x")
    return {"datasets": {"Steel2": {"mode_a_line_class": record}}}


def _listing(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_regeneration_state_done_todo_and_every_partial_state(tmp_path):
    from src import boundary_gt

    fp = {"gt_extraction_sha256": "53155b40c2b6ebf8"}
    state = boundary_gt.mode_a_regeneration_state
    assert state(_gt(tmp_path / "a", LINE, backup=True), tmp_path / "a", "Steel2", fp,
                 "53155b40c2b6ebf8") == "done"
    assert state(_gt(tmp_path / "b"), tmp_path / "b", "Steel2", fp, "53155b40c2b6ebf8") == "todo"
    partial = [
        (_gt(tmp_path / "c", LINE), tmp_path / "c", fp),                         # no backup
        (_gt(tmp_path / "d", None, backup=True), tmp_path / "d", fp),           # no record
        (_gt(tmp_path / "e", LINE, backup=True, files=False), tmp_path / "e", fp),
        (_gt(tmp_path / "f", LINE, backup=True), tmp_path / "f",
         {"gt_extraction_sha256": "0" * 16}),                                     # other GT
        (_gt(tmp_path / "g", LINE, backup=True), tmp_path / "g", None),           # unreadable
    ]
    for ext, root, fingerprint in partial:
        before = _listing(tmp_path)
        with pytest.raises(boundary_gt.ExtractionError, match="REFUSING"):
            state(ext, root, "Steel2", fingerprint, "53155b40c2b6ebf8")
        assert _listing(tmp_path) == before


def _fold_dirs(tmp_path):
    for n in ("m", "c", "r"):
        (tmp_path / n).mkdir()
    (tmp_path / "c" / "fold_stats.yaml").write_text("folds:\n  other: {}\n")
    return tmp_path / "m", tmp_path / "c", tmp_path / "r"


def test_fold_state_done_todo_and_partial(tmp_path):
    from src import tiling

    m, c, r = _fold_dirs(tmp_path)
    ckpt = tmp_path / "ckpt"
    assert tiling.mode_a_fold_state("f", m, c, r, ckpt) == "todo"
    (m / "f.csv").write_text("x")
    with pytest.raises(tiling.TilingError, match="REFUSING"):
        tiling.mode_a_fold_state("f", m, c, r, ckpt)
    (m / "f_test.csv").write_text("x")
    with pytest.raises(tiling.TilingError, match="fold_stats entry ABSENT"):
        tiling.mode_a_fold_state("f", m, c, r, ckpt)
    (c / "fold_stats.yaml").write_text("folds:\n  other: {}\n  f: {pos_weight: 1}\n")
    ckpt.mkdir()
    (ckpt / "best.pt").write_bytes(b"w")
    assert tiling.mode_a_fold_state("f", m, c, r, ckpt) == "done", (
        "a trained fold is done, checkpoints and all")


def test_fold_state_refuses_traces_without_manifests(tmp_path):
    from src import tiling

    m, c, r = _fold_dirs(tmp_path)
    (r / "tiling_f.md").write_text("x")
    with pytest.raises(tiling.TilingError):
        tiling.mode_a_fold_state("f", m, c, r, None)


def test_training_settings_were_not_changed_by_the_width_and_clDice_work():
    """loss.cldice_iters and the width read-out are measurement concerns here; the training
    setting stays what the recorded run used (it is in HASHED loss settings via the loss
    section of the config hash, so a change would also invalidate the checkpoint)."""
    pytest.importorskip("torch")
    from src import losses

    recorded = json.loads(TRAIN_REPORT.read_text())["hashed_config"]["loss"]
    assert losses.load_config()["cldice_iters"] == recorded["cldice_iters"] == 3
    assert {k: v for k, v in losses.load_config().items() if k in recorded} == recorded
