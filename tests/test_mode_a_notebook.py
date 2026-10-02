"""Tests of notebooks/06d_steel_combined.ipynb's Steel2 MODE A cells, against the REAL file.

Two kinds, neither of which runs the notebook:

- static: the order of the steps in a cell (a refusal comes before every write; a cell
  never touches the pieces it must not);
- guard execution: the real cell's own source, up to and including its refusal line, is
  ``exec``'d against a scratch directory laid out the way a finished run leaves it, so what is
  tested is the code that is actually in the notebook, not a copy of it.

Nothing here is executed on the machine that wrote it (see CLAUDE.md); the notebook's own
pytest gate runs these on the host.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "06d_steel_combined.ipynb"


def _notebook():
    if not NB_PATH.is_file():
        pytest.skip(f"{NB_PATH} not found")
    return json.loads(NB_PATH.read_text())


def _source(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code_cells():
    return [_source(c) for c in _notebook()["cells"] if c["cell_type"] == "code"]


def _the_cell(contains):
    hits = [c for c in _code_cells() if contains in c]
    assert len(hits) == 1, f"expected exactly one code cell containing {contains!r}, found {len(hits)}"
    return hits[0]


def _prefix_through(cell, marker):
    """The cell's source up to and including the line holding ``marker``."""
    lines = cell.splitlines()
    at = next(i for i, line in enumerate(lines) if marker in line)
    return "\n".join(lines[:at + 1])


REGEN_GUARD = "boundary_gt.refuse_if_already_regenerated("
FOLD_GUARD = "tiling.refuse_if_fold_exists("


# --------------------------------------------------------------------------
# 1a. the regeneration cell: guard order (static); skip / refuse / proceed (exec)
# --------------------------------------------------------------------------
def test_regeneration_cell_refuses_before_it_hashes_copies_or_writes_anything():
    cell = _the_cell("boundary_gt.extract_all(")
    guard = cell.index(REGEN_GUARD)
    for write in ("_dir_hashes(GT_STEEL1)", "shutil.copytree(", "boundary_gt.extract_all(",
                  "boundary_gt.write_extraction_report(", "boundary_gt.write_payload_report(",
                  "push_results("):
        assert guard < cell.index(write), f"the refusal must come before {write}"
    assert "try:" not in cell[:guard], "the refusal must not be wrapped where it could be swallowed"
    assert "Steel2" in cell[guard:guard + 120]


def test_regeneration_cell_has_no_path_that_continues_when_the_backup_exists():
    """The previous 'verify-only' and 'backup exists -- NOT overwritten' branches let a
    second run carry on; there must be none left."""
    cell = _the_cell("boundary_gt.extract_all(")
    assert "verify-only" not in cell and "NOT overwritten" not in cell
    assert not re.search(r"^\s*if already\b|^\s*already\s*=", cell, re.M)
    assert cell.count("shutil.copytree(") == 1
    assert "GT_BACKUP.exists()" not in cell, "existence is decided by the refusal, not a branch"


# --------------------------------------------------------------------------
# 1b. the fold-rebuild cell: guard order (static); skip / refuse / proceed (exec)
# --------------------------------------------------------------------------
def test_fold_cell_refuses_before_it_reads_the_index_or_writes_anything():
    cell = _the_cell("tiling.membership_diff(")
    guard = cell.index(FOLD_GUARD)
    for write in ("tiling.build_index(", "tiling.write_manifest(", "tiling.write_fold_report(",
                  "tiling.merge_fold_stats(", "boundary_gt.write_payload_report(", "push_results("):
        assert guard < cell.index(write), f"the refusal must come before {write}"
    assert "try:" not in cell[:guard]
    assert "NEW_CKPT" in cell[guard:guard + 120], "the checkpoint directory is part of the refusal"


def _fingerprint_of(reports_dir):
    """What train.gt_fingerprint reports for the sha, without importing torch."""
    path = Path(reports_dir) / "gt_extraction.json"
    return {"gt_extraction_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}


def _stub_train(expected_sha):
    return SimpleNamespace(gt_fingerprint=_fingerprint_of,
                           load_steel_step=lambda: {"expected_gt_sha256": expected_sha})


def _scratch_gt(tmp_path, record=None, backup=False, backup_files=True, sha="match"):
    """A scratch REPORTS/GT_ROOT laid out as the regeneration leaves them (or not)."""
    reports, gt = tmp_path / "reports", tmp_path / "gt"
    reports.mkdir(exist_ok=True)
    (gt / "Steel2").mkdir(parents=True, exist_ok=True)
    (gt / "Steel1").mkdir(parents=True, exist_ok=True)
    (reports / "gt_extraction.json").write_text(json.dumps(
        {"datasets": {"Steel1": {"mode_a_line_class": None},
                      "Steel2": {"mode_a_line_class": record}}}))
    if backup:
        from src import boundary_gt

        b = gt / boundary_gt.MODE_B_BACKUP_SUBDIR / "Steel2"
        b.mkdir(parents=True)
        if backup_files:
            (b / "a.png").write_bytes(b"the only copy of the original")
    real = _fingerprint_of(reports)["gt_extraction_sha256"]
    return reports, gt, (real if sha == "match" else "0" * 16)


LINE_RECORD = {"class_index": 1, "colour": [0, 0, 0], "blob_thickness_px": 8.0}


def _regen_ns(reports, gt, expected_sha, **extra):
    pytest.importorskip("tqdm")
    from src import boundary_gt, tiling

    ns = {"audit": {}, "line_width": 4.0, "REPORTS": reports, "GT_ROOT": gt, "PATHS": {},
          "tiling": tiling, "boundary_gt": boundary_gt, "train_mod": _stub_train(expected_sha),
          "Path": Path, "json": json, "__name__": "cell"}
    ns.update(extra)
    return ns


def _listing(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _exec_cell(cell, ns, label):
    out = []
    ns["print"] = lambda *a, **k: out.append(" ".join(str(x) for x in a))
    exec(compile(cell, label, "exec"), ns)
    return "\n".join(out)


def test_the_real_regeneration_cell_skips_in_exactly_the_finished_state(tmp_path):
    reports, gt, sha = _scratch_gt(tmp_path, record=LINE_RECORD, backup=True)
    before = _snapshot(tmp_path)
    printed = _exec_cell(_the_cell("boundary_gt.extract_all("), _regen_ns(reports, gt, sha),
                         "<cell 30>")
    assert "Cell 30 already done -- skipped" in printed
    assert _snapshot(tmp_path) == before, "the skip must not write anything"


@pytest.mark.parametrize("state", [
    dict(record=LINE_RECORD, backup=False),                       # record, no backup
    dict(record=None, backup=True),                               # backup, no record
    dict(record=LINE_RECORD, backup=True, backup_files=False),    # empty backup
    dict(record=LINE_RECORD, backup=True, sha="other"),           # foreign fingerprint
    dict(record=None, backup=True, backup_files=False),           # empty backup, no record
])
def test_the_real_regeneration_cell_refuses_any_partial_or_foreign_state(tmp_path, state):
    from src import boundary_gt

    reports, gt, sha = _scratch_gt(tmp_path, **state)
    before = _snapshot(tmp_path)
    with pytest.raises(boundary_gt.ExtractionError, match="REFUSING"):
        _exec_cell(_the_cell("boundary_gt.extract_all("), _regen_ns(reports, gt, sha),
                   "<cell 30>")
    assert _snapshot(tmp_path) == before, "a refusal must not write anything"


def test_the_real_regeneration_cell_proceeds_from_the_clean_state_only_with_cell_29(tmp_path):
    reports, gt, sha = _scratch_gt(tmp_path, record=None, backup=False)
    cell = _the_cell("boundary_gt.extract_all(")
    # without Cell 29's profile it stops, writing nothing
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="RUN_DIAGNOSTICS"):
        _exec_cell(_prefix_through(cell, REGEN_GUARD), _regen_ns(reports, gt, sha), "<cell 30>")
    assert _snapshot(tmp_path) == before
    # with it, it gets past every guard (the heavy work after them is not run here)
    _exec_cell(_prefix_through(cell, REGEN_GUARD),
               _regen_ns(reports, gt, sha, ma_profile={}, ma_line_idx=1), "<cell 30>")


def _fold_ns(tmp_path, expected_sha):
    pytest.importorskip("tqdm")
    pytest.importorskip("yaml")
    from src import boundary_gt, tiling

    for name in ("manifests", "configs"):
        (tmp_path / name).mkdir(exist_ok=True)
    (tmp_path / "configs" / "fold_stats.yaml").write_text("folds:\n  other: {}\n")
    return {"audit": {}, "line_width": 4.0, "REPORTS": tmp_path / "reports",
            "CONFIGS": tmp_path / "configs", "GT_ROOT": tmp_path / "gt",
            "MANIFEST_DIR": tmp_path / "manifests", "DATASETS": ["Steel1", "Steel2"],
            "PATHS": {"persistent_dir": str(tmp_path / "persist")},
            "train_settings": {"checkpoint_subdir": "checkpoints"},
            "train_mod": _stub_train(expected_sha), "Path": Path, "json": json,
            "tiling": tiling, "boundary_gt": boundary_gt, "__name__": "cell"}


def _new_fold():
    from src import tiling

    return tiling.load_config()["steel_combined_mode_a_name"]


def test_the_real_fold_cell_skips_when_ground_truth_and_fold_are_both_done(tmp_path):
    reports, gt, sha = _scratch_gt(tmp_path, record=LINE_RECORD, backup=True)
    ns = _fold_ns(tmp_path, sha)
    new = _new_fold()
    (tmp_path / "manifests" / f"{new}.csv").write_text("x")
    (tmp_path / "manifests" / f"{new}_test.csv").write_text("x")
    (tmp_path / "configs" / "fold_stats.yaml").write_text(
        f"folds:\n  other: {{}}\n  {new}: {{pos_weight: 1}}\n")
    before = _snapshot(tmp_path)
    printed = _exec_cell(_the_cell("tiling.membership_diff("), ns, "<cell 31>")
    assert "Cell 31 already done -- skipped" in printed
    assert _snapshot(tmp_path) == before, "the skip must not write anything"


@pytest.mark.parametrize("what", ["one_manifest", "entry_only", "report_only", "checkpoint_only"])
def test_the_real_fold_cell_refuses_a_partially_built_fold(tmp_path, what):
    from src import tiling

    reports, gt, sha = _scratch_gt(tmp_path, record=LINE_RECORD, backup=True)
    ns = _fold_ns(tmp_path, sha)
    new = _new_fold()
    if what == "one_manifest":
        (tmp_path / "manifests" / f"{new}.csv").write_text("x")
    elif what == "entry_only":
        (tmp_path / "configs" / "fold_stats.yaml").write_text(
            f"folds:\n  other: {{}}\n  {new}: {{pos_weight: 1}}\n")
    elif what == "report_only":
        (tmp_path / "reports" / f"tiling_{new}.json").write_text("{}")
    else:
        ckpt = tmp_path / "persist" / "checkpoints" / new
        ckpt.mkdir(parents=True)
        (ckpt / "last.pt").write_bytes(b"weights trained on the old manifests")
    before = _snapshot(tmp_path)
    with pytest.raises(tiling.TilingError, match="REFUSING"):
        _exec_cell(_the_cell("tiling.membership_diff("), ns, "<cell 31>")
    assert _snapshot(tmp_path) == before, "a refusal must not write anything"


def test_the_real_fold_cell_refuses_when_the_ground_truth_is_not_finished(tmp_path):
    reports, gt, sha = _scratch_gt(tmp_path, record=None, backup=False)
    ns = _fold_ns(tmp_path, sha)
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="run Cell 30 first"):
        _exec_cell(_the_cell("tiling.membership_diff("), ns, "<cell 31>")
    assert _snapshot(tmp_path) == before


def test_the_real_fold_cell_gets_past_its_guards_when_the_fold_does_not_exist(tmp_path):
    reports, gt, sha = _scratch_gt(tmp_path, record=LINE_RECORD, backup=True)
    _exec_cell(_prefix_through(_the_cell("tiling.membership_diff("), FOLD_GUARD),
               _fold_ns(tmp_path, sha), "<cell 31>")      # no raise


# --------------------------------------------------------------------------
# diagnostics gate, Cell 14 completion, and the locked steel step
# --------------------------------------------------------------------------
def _cell_after(title):
    nb = _notebook()
    cells = nb["cells"]
    i = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown"
             and _source(c).startswith(title))
    assert cells[i + 1]["cell_type"] == "code"
    return _source(cells[i + 1])


@pytest.mark.parametrize("n", [25, 26, 27, 28, 29])
def test_diagnostic_cells_are_skipped_when_run_diagnostics_is_false(n):
    """The whole real cell, in a namespace that holds NOTHING else: if anything other
    than the gate ran, it would raise NameError/ImportError here."""
    printed = _exec_cell(_cell_after(f"## Cell {n} "), {"RUN_DIAGNOSTICS": False,
                                                          "__name__": "cell"}, f"<cell {n}>")
    assert printed == f"Cell {n} skipped (diagnostic) -- set RUN_DIAGNOSTICS = True in Cell 2 to run it."


@pytest.mark.parametrize("n", [25, 26, 27, 28, 29])
def test_diagnostic_cells_are_skipped_when_the_flag_was_never_set(n):
    printed = _exec_cell(_cell_after(f"## Cell {n} "), {"__name__": "cell"}, f"<cell {n}>")
    assert "skipped (diagnostic)" in printed


def test_run_diagnostics_is_set_once_in_cell_2_and_defaults_to_false():
    code = _code_cells()
    setters = [c for c in code if re.search(r"^RUN_DIAGNOSTICS\s*=", c, re.M)]
    assert len(setters) == 1 and "FOLD = steel_cfg[" in setters[0]
    assert re.search(r"^RUN_DIAGNOSTICS = False$", setters[0], re.M)
    gated = [c for c in code if 'globals().get("RUN_DIAGNOSTICS", False)' in c]
    assert len(gated) == 5


def test_non_diagnostic_cells_are_not_gated():
    for title in ("## Cell 14 ", "## Cell 20 ", "## Cell 21 ", "## Cell 22 ", "## Cell 23 ",
                  "## Cell 24 ", "## Cell 30 ", "## Cell 31 ", "## Cell 32 "):
        assert "RUN_DIAGNOSTICS\", False)" not in _cell_after(title), title


def test_cell_14_checks_completion_before_it_can_call_fit():
    cell = _cell_after("## Cell 14 ")
    check = cell.index("train_mod.run_completion(trainer.history")
    assert check < cell.index("trainer.fit(")
    branch = cell[check:]
    assert branch.index("if _run_done:") < branch.index("RUN ALREADY COMPLETE") \
        < branch.index("else:") < branch.index("trainer.fit(")
    # fit() is only reachable from the else branch
    fit_line = next(l for l in cell.splitlines() if "trainer.fit(" in l)
    assert fit_line.startswith("        "), "fit() must be inside the else block"
    assert "stopped_early = False" in branch[:branch.index("else:")], (
        "Cell 15 reads stopped_early; the skip branch must set it")


def test_cell_2_refuses_a_fold_other_than_the_decided_one():
    cell = _cell_after("## Cell 2 ")
    assert cell.index("FOLD = steel_cfg[") < cell.index("train_mod.load_steel_step()") \
        < cell.index('if FOLD != STEEL_STEP["default_fold"]:')
    assert "raise SystemExit(" in cell[cell.index('if FOLD != STEEL_STEP["default_fold"]:'):]


# --------------------------------------------------------------------------
# 2. which cell resolves the fold, and why Cell 13 cannot pick up the old checkpoint
# --------------------------------------------------------------------------
def test_one_cell_resolves_the_fold_from_the_recorded_ground_truth_and_everything_else_follows_it():
    cell2 = _the_cell("steel_cfg = settings[\"steel_combined\"]")
    assert cell2.index("tiling.load_extraction(") < cell2.index("tiling.apply_gt_variant(") \
        < cell2.index("steel_cfg = settings[\"steel_combined\"]") < cell2.index("FOLD = steel_cfg[")
    trainer_cell = _the_cell("train_mod.Trainer(fold=FOLD")
    assert "fold=FOLD" in trainer_cell

    # no cell after Cell 2 names the ORIGINAL fold as a literal, which would bypass it
    # (the new cells 30-32 name it as ORIG_FOLD, read from the config, to compare against it)
    for cell in _code_cells():
        if "train_mod.Trainer(fold=FOLD" in cell or "session.stage_resume(" in cell:
            assert not re.search(r"fold_steel_combined(?!_modea)", cell), (
                "a resume/trainer cell hardcodes the original fold's name")


def test_cell_13_resumes_only_this_runs_own_checkpoint():
    cell = _the_cell("session.stage_resume(")
    assert "session.stage_resume(trainer.run_name, fold=trainer.fold)" in cell
    resumes = re.findall(r"\.maybe_resume\(([^)]*)\)", cell)
    assert resumes == [""], (
        f"maybe_resume must be called with NO arguments (its own last.pt, fold and config "
        f"hash are checked), got {resumes}")
    assert "allow_config_change" not in cell


# --------------------------------------------------------------------------
# 3. Cell 32: the retrain comparison
# --------------------------------------------------------------------------
def test_comparison_cell_comes_after_the_fold_cells_and_training():
    nb = _notebook()
    code = [(i, _source(c)) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    at = {key: next(i for i, s in code if key in s) for key in
          ("trainer.fit(", "boundary_gt.extract_all(", "tiling.membership_diff(",
           "checkpoint_test_summary(")}
    assert at["trainer.fit("] < at["checkpoint_test_summary("]
    assert at["boundary_gt.extract_all("] < at["tiling.membership_diff("] < at["checkpoint_test_summary("]


def test_comparison_cell_scores_identical_tiles_and_never_scores_against_the_old_steel2_gt():
    cell = _the_cell("checkpoint_test_summary(")
    # the ONLY test dataset is the new fold's manifest; the backup of the old ground truth is never read
    assert cell.count("ds.TileDataset.from_manifest(") == 1
    assert "from_manifest(test_path" in cell
    assert "_backup_mode_b" not in cell and "MODE_B_BACKUP_SUBDIR" not in cell
    # both checkpoints go through the same scoring function
    assert cell.count("cmp_score(") >= 3          # definition + NEW + OLD
    assert "cmp_score(new_model, new_state)" in cell and "cmp_score(old_model, old_state)" in cell
    # markers come from VAL, through the repo's own selector, before TEST is predicted
    score = cell[cell.index("def cmp_score("):cell.index("print(\"\\nscoring NEW checkpoint")]
    assert (score.index("trainer.val_ds") < score.index("train_mod.select_marker_threshold(")
            < score.index("cmp_test_ds"))
    assert "train_mod.compare_checkpoint_summaries(" in cell
    assert "Trainer(" not in cell and ".train(" not in cell and ".fit(" not in cell


def test_comparison_cell_loads_the_new_checkpoint_against_the_ground_truth_and_the_old_one_as_a_control():
    cell = _the_cell("checkpoint_test_summary(")
    new_load = cell[cell.index("new_model, new_state = "):cell.index("old_entry = ")]
    assert "expected_gt_fingerprint=" in new_load and "fold=FOLD" in new_load
    old_load = cell[cell.index("old_model, old_state = "):cell.index("print(f\"NEW ")]
    assert "fold=ORIG_FOLD" in old_load and "expected_gt_fingerprint" not in old_load
    assert "if FOLD != NEW_FOLD" in cell, "it must refuse to run on the original fold"


def test_comparison_cell_pushes_only_after_its_checks_and_names_the_report_by_platform():
    cell = _the_cell("checkpoint_test_summary(")
    assert ('f"modea_retrain_comparison_{trainer.platform}"' in cell)
    assert (cell.index("boundary_gt.write_payload_report(")
            < cell.index("raise SystemExit(\"NOT pushed") < cell.index("push_results("))
    for required in ("the pipeline reproduces the committed OLD-fold report on Steel1",
                     "test tile membership identical in both folds",
                     "Steel1 test tiles: same ground-truth files and boundary fractions",
                     "the NEW checkpoint was trained on the ground truth on disk"):
        assert required in cell, required
