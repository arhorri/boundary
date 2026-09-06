"""Checkpoint survival across Kaggle sessions.

KAGGLE BRANCH ONLY. This module does not exist on ``main`` and nothing on
``main`` imports it. It is additive: ``git merge main`` never touches it, so
the branch keeps taking pipeline changes cleanly even though it now carries
code as well as configuration.

Why it exists
-------------
``PERSISTENT_DIR`` on Kaggle is ``/kaggle/working``, which is deleted when the
kernel stops. On Colab it is a Drive folder that survives, so ``src/train.py``
is right to treat "write ``last.pt`` after every epoch" as sufficient there.
On Kaggle it is not sufficient, and the gap is not one that configuration can
close:

* a free GPU session is time-limited, so a 40-epoch run is several sessions;
* ``/kaggle/working`` reaches the next session only as a **saved notebook
  version's output**, re-attached as an input dataset;
* an input dataset is mounted **read-only** under ``/kaggle/input``, which is
  not where ``Trainer`` looks for ``last.pt``.

So the three things here are: find a checkpoint in whatever was attached and
stage it where ``Trainer.maybe_resume()`` already looks (:func:`stage_resume`);
stop the run cleanly while a session still has time to save
(:class:`SessionBudget`, :func:`budget_guard`); and say out loud what still has
to be done by hand for the run to survive (:func:`survival_report`).

Nothing here weakens a guard in ``src/train.py``. A staged checkpoint is
verified to be the requested fold *before* it is copied, and it is then handed
to the same ``maybe_resume()`` that checks fold and config hash again. This
module moves files; it never decides that a resume is legitimate.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

#: Overridable under ``session.kaggle.persist`` in ``configs/kaggle.yaml``.
DEFAULTS = {
    # Kaggle's free GPU session limit is 9 hours at the time of writing, and
    # the kernel is killed at it with no grace period. Budget slightly under.
    "session_budget_hours": 8.5,
    # Held back from the budget so the notebook still has time to run its
    # checks, write the report and push after fit() returns. A run that uses
    # the whole session and is then killed during push_results has lost the
    # report, which is the only part git ever sees.
    "reserve_minutes": 25.0,
    # Input dataset slugs to search for a checkpoint, most preferred first.
    # Empty means "scan everything attached", which is what you want when the
    # source is a saved version of this notebook and the slug is generated.
    "resume_input_slugs": [],
    # Refuse to stage a checkpoint bigger than this. A wrong path that happens
    # to contain a last.pt should fail loudly, not fill the session disk.
    "max_stage_mb": 2048.0,
}

CHECKPOINT_NAMES = ("last.pt", "best.pt")


class KagglePersistError(RuntimeError):
    """Raised when checkpoint survival cannot be arranged. Never silent."""


class SessionTimeUp(RuntimeError):
    """Raised by the budget guard to end ``fit()`` while there is time to save.

    Caught by the notebook, not an error condition: ``save()`` has already run
    for the epoch that triggered it, so ``last.pt`` on disk is complete and the
    next session resumes from it.
    """


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
def settings(config: Optional[dict] = None) -> dict:
    """DEFAULTS overlaid with ``session.kaggle.persist`` from the config.

    Reads through ``src.paths.load_config`` so the kaggle overlay is applied
    the same way every other module sees it. Unknown keys are an error rather
    than a typo that silently does nothing.
    """
    if config is None:
        from src import paths as paths_mod

        config = paths_mod.load_config()

    section = (((config.get("session") or {}).get("kaggle") or {})
               .get("persist") or {})
    if not isinstance(section, dict):
        raise KagglePersistError(
            "configs/kaggle.yaml: session.kaggle.persist must be a mapping, "
            f"got {type(section).__name__}.")

    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise KagglePersistError(
            f"configs/kaggle.yaml: session.kaggle.persist has unknown key(s) "
            f"{unknown}. Known keys: {sorted(DEFAULTS)}.")

    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in section.items() if v is not None})
    return merged


def _require_kaggle(resolved: dict) -> None:
    platform = str(resolved.get("platform", "unknown"))
    if platform != "kaggle":
        raise KagglePersistError(
            f"src/kaggle_persist.py is for Kaggle sessions; this one detected "
            f"as {platform!r}. On Colab, PERSISTENT_DIR is Drive and survives "
            "on its own -- use notebooks/06_train.ipynb unchanged.")


# --------------------------------------------------------------------------
# session budget
# --------------------------------------------------------------------------
class SessionBudget:
    """Wall-clock budget for one Kaggle session.

    The clock starts when this object is constructed, which should be as early
    in the notebook as possible: the bootstrap clone, the pip install and the
    tile cache build all happen before the first epoch and all of them spend
    session time that the last epoch will not have.
    """

    def __init__(self, hours: Optional[float] = None,
                 reserve_minutes: Optional[float] = None,
                 config: Optional[dict] = None,
                 started: Optional[float] = None):
        cfg = settings(config)
        self.hours = float(hours if hours is not None
                           else cfg["session_budget_hours"])
        self.reserve_minutes = float(
            reserve_minutes if reserve_minutes is not None
            else cfg["reserve_minutes"])
        if self.hours <= 0:
            raise KagglePersistError(
                f"session_budget_hours must be positive, got {self.hours}.")
        if self.reserve_minutes < 0:
            raise KagglePersistError(
                f"reserve_minutes cannot be negative, got {self.reserve_minutes}.")
        if self.reserve_minutes * 60 >= self.hours * 3600:
            raise KagglePersistError(
                f"reserve_minutes ({self.reserve_minutes}) consumes the entire "
                f"{self.hours} h budget; no time would be left to train.")

        self.started = float(started if started is not None else time.time())
        self.total_seconds = self.hours * 3600.0
        self.usable_seconds = self.total_seconds - self.reserve_minutes * 60.0

    # -- state ------------------------------------------------------------
    def elapsed_seconds(self) -> float:
        return time.time() - self.started

    def remaining_seconds(self) -> float:
        """Seconds left before the reserve must be handed back. May be < 0."""
        return self.usable_seconds - self.elapsed_seconds()

    def exhausted(self) -> bool:
        return self.remaining_seconds() <= 0

    def affords(self, seconds: float) -> bool:
        """Would one more unit of work of this length still fit?"""
        return self.remaining_seconds() >= float(seconds)

    def epochs_affordable(self, seconds_per_epoch: float) -> int:
        if seconds_per_epoch <= 0:
            raise KagglePersistError(
                "seconds_per_epoch must be positive to estimate a budget.")
        return max(0, int(self.remaining_seconds() // float(seconds_per_epoch)))

    def describe(self) -> str:
        remaining = self.remaining_seconds()
        return (f"budget {self.hours:.2f} h "
                f"(reserve {self.reserve_minutes:.0f} min), "
                f"elapsed {self.elapsed_seconds() / 60:.1f} min, "
                f"{remaining / 60:.1f} min usable left")

    def __str__(self) -> str:                        # pragma: no cover - cosmetic
        return self.describe()


def budget_guard(budget: SessionBudget,
                 on_epoch_end: Optional[Callable] = None,
                 verbose: bool = True) -> Callable:
    """Wrap an ``on_epoch_end`` callback so ``fit()`` stops before the kill.

    Passed to ``Trainer.fit(on_epoch_end=...)``. ``Trainer.fit`` writes the
    checkpoint *before* calling the callback, so raising here always leaves a
    complete ``last.pt`` on disk -- that ordering is what makes stopping by
    exception safe rather than lossy.

    The decision uses the slowest epoch seen so far, not the mean. Epoch
    duration on a shared host is not stationary, and a mean lets one slow epoch
    at the end run past the deadline with the checkpoint half written.
    """
    if not isinstance(budget, SessionBudget):
        raise KagglePersistError(
            f"budget must be a SessionBudget, got {type(budget).__name__}.")

    def guard(record, trainer):
        if on_epoch_end is not None:
            on_epoch_end(record, trainer)

        worst = max(float(r["seconds"]) for r in trainer.history)
        remaining = budget.remaining_seconds()
        if verbose:
            print(f"  [budget] {remaining / 60:.1f} min usable left; "
                  f"slowest epoch so far {worst / 60:.1f} min; "
                  f"~{budget.epochs_affordable(worst)} more epoch(s) fit")

        if not budget.affords(worst):
            raise SessionTimeUp(
                f"stopping after epoch {record['epoch']}: "
                f"{remaining / 60:.1f} min usable session time left and the "
                f"slowest epoch so far took {worst / 60:.1f} min. "
                f"last.pt for epoch {record['epoch']} is written and complete; "
                "save this notebook's version, attach its output to the next "
                "run, and it will resume from epoch "
                f"{record['epoch'] + 1}.")

    return guard


# --------------------------------------------------------------------------
# finding a checkpoint in whatever was attached
# --------------------------------------------------------------------------
def input_roots(resolved: Optional[dict] = None,
                config: Optional[dict] = None,
                slugs: Optional[Iterable[str]] = None) -> list:
    """Directories under ``/kaggle/input`` to search, preferred first.

    Configured slugs come first and in order; everything else attached follows,
    so a run works whether the checkpoint came from a hand-uploaded dataset
    with a stable name or from a saved version of this notebook whose slug is
    generated and unpredictable.
    """
    if config is None:
        from src import paths as paths_mod

        config = paths_mod.load_config()

    kaggle_cfg = ((config.get("session") or {}).get("kaggle") or {})
    configured = Path(kaggle_cfg.get("input_root") or "/kaggle/input")

    # Two mount shapes are in play and a session can contain BOTH. Datasets
    # here mount owner-nested (/kaggle/input/datasets/<owner>/<slug>/), which
    # is what session.kaggle.input_root points at; a notebook-output
    # attachment may instead mount flat at /kaggle/input/<slug>/. Searching
    # only the configured root would find the data and miss the checkpoint --
    # silently, as a clean start. So both shapes are enumerated.
    literal = Path(os.sep, "kaggle", "input")
    attached, seen = [], set()

    def offer(path: Path) -> None:
        if not path.is_dir():
            return
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            attached.append(path)

    for base in (configured, literal):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            offer(child)
            # /kaggle/input/datasets/<owner>/<slug> -- one more level down.
            if child.name == "datasets" or base == literal:
                for grandchild in sorted(
                        c for c in child.iterdir() if c.is_dir()):
                    offer(grandchild)

    if not attached:
        raise KagglePersistError(
            f"no attached datasets found under {configured} or {literal}. On "
            "Kaggle at least one is mounted whenever anything is attached; if "
            "there are none, nothing is attached to this notebook.")

    wanted = list(slugs) if slugs is not None else list(
        settings(config)["resume_input_slugs"] or [])

    ordered, claimed = [], set()
    for slug in wanted:
        hits = [p for p in attached if p.name == str(slug)]
        if not hits:
            raise KagglePersistError(
                f"session.kaggle.persist.resume_input_slugs names {slug!r} but "
                "nothing by that name is attached to this notebook. Attached: "
                + (", ".join(sorted(p.name for p in attached)) or "(nothing)"))
        for hit in hits:
            ordered.append(hit)
            claimed.add(str(hit.resolve()))

    ordered.extend(p for p in attached if str(p.resolve()) not in claimed)
    return ordered


def _candidate_paths(root: Path, fold: str, name: str) -> list:
    """Where a checkpoint for ``fold`` could plausibly sit under ``root``.

    Three layouts are accepted, in decreasing confidence: the working-dir shape
    a saved notebook version produces, a dataset built from the checkpoint
    folder alone, and a flat hand-made upload. Anything else falls through to a
    bounded search rather than being guessed at.
    """
    exact = [
        root / "checkpoints" / fold / name,
        root / fold / name,
        root / name,
    ]
    hits = [p for p in exact if p.is_file()]
    if hits:
        return hits

    # Bounded fallback: a saved version may nest the working dir one level.
    for depth in (1, 2):
        pattern = "/".join(["*"] * depth) + f"/checkpoints/{fold}/{name}"
        hits.extend(sorted(p for p in root.glob(pattern) if p.is_file()))
        if hits:
            break
    return hits


def _checkpoint_meta(path: Path) -> dict:
    """Read the identifying header of a checkpoint without building a model."""
    from src import train as train_mod

    state = train_mod.load_checkpoint(path)
    return {
        "path": path,
        "fold": state.get("fold"),
        "epoch": state.get("epoch"),
        "config_hash": state.get("config_hash"),
        "saved_utc": state.get("saved_utc"),
        "held_out": state.get("held_out"),
        "resumable": "optimizer" in state,
        "size_mb": path.stat().st_size / 1024 ** 2,
    }


def find_checkpoints(fold: str, resolved: Optional[dict] = None,
                     config: Optional[dict] = None,
                     name: str = "last.pt") -> list:
    """Every attached checkpoint for ``fold``, newest epoch first.

    Only the file header is read. Files that do not load, or that belong to a
    different fold, are reported with the reason rather than dropped -- a
    checkpoint that is present but unusable is exactly the thing you need to be
    told about before spending a session training from scratch.
    """
    if name not in CHECKPOINT_NAMES:
        raise KagglePersistError(
            f"name must be one of {CHECKPOINT_NAMES}, got {name!r}.")

    found = []
    for root in input_roots(resolved, config):
        for path in _candidate_paths(root, str(fold), name):
            try:
                meta = _checkpoint_meta(path)
            except Exception as exc:                     # noqa: BLE001
                found.append({"path": path, "usable": False,
                              "reason": f"does not load ({type(exc).__name__}: {exc})",
                              "size_mb": path.stat().st_size / 1024 ** 2})
                continue
            if meta["fold"] != str(fold):
                meta.update(usable=False,
                            reason=f"belongs to fold {meta['fold']!r}, not {fold!r}")
            else:
                meta.update(usable=True, reason="fold matches")
            meta["source_dataset"] = root.name
            found.append(meta)

    found.sort(key=lambda m: (m.get("usable", False),
                             m.get("epoch") if m.get("epoch") is not None else -1),
               reverse=True)
    return found


# --------------------------------------------------------------------------
# staging a resume
# --------------------------------------------------------------------------
def stage_resume(fold: str, resolved: dict, config: Optional[dict] = None,
                 overwrite: bool = False, verbose: bool = True) -> dict:
    """Copy an attached checkpoint to where ``maybe_resume()`` looks.

    Returns a report; never raises just because nothing was found, because a
    first session legitimately has nothing to resume from. It *does* raise when
    something is wrong in a way that would otherwise waste a session: a named
    slug that is not attached, a checkpoint too large to be what it claims, a
    copy that does not verify afterwards.

    A checkpoint already in ``PERSISTENT_DIR`` wins unless ``overwrite=True``.
    Within one session that local file is the run in progress, and silently
    replacing it with an older attached copy would throw away finished epochs.
    """
    _require_kaggle(resolved)
    cfg = settings(config)

    checkpoint_dir = (Path(resolved["persistent_dir"]) / "checkpoints" / str(fold))
    report = {
        "fold": str(fold),
        "checkpoint_dir": str(checkpoint_dir),
        "staged": [],
        "skipped": [],
        "candidates": [],
        "resumable": False,
    }

    local_last = checkpoint_dir / "last.pt"
    if local_last.is_file() and not overwrite:
        meta = _checkpoint_meta(local_last)
        report["resumable"] = True
        report["skipped"].append(
            f"{local_last} already exists (epoch {meta['epoch']}); "
            "not overwritten")
        if verbose:
            print(f"  local checkpoint already present: {local_last} "
                  f"(epoch {meta['epoch']}, {meta['size_mb']:.0f} MB)")
            print("  nothing staged; Trainer.maybe_resume() will use it")
        return report

    for name in CHECKPOINT_NAMES:
        candidates = find_checkpoints(fold, resolved, config, name=name)
        report["candidates"].extend(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in c.items()}
            for c in candidates)

        usable = [c for c in candidates if c.get("usable")]
        if not usable:
            for bad in candidates:
                if verbose:
                    print(f"  ! ignoring {bad['path']}: {bad['reason']}")
            report["skipped"].append(f"no usable {name} attached")
            continue

        chosen = usable[0]
        if chosen["size_mb"] > float(cfg["max_stage_mb"]):
            raise KagglePersistError(
                f"{chosen['path']} is {chosen['size_mb']:.0f} MB, over the "
                f"max_stage_mb limit of {cfg['max_stage_mb']:.0f} MB. Either "
                "that is not the checkpoint you think it is, or raise the "
                "limit in configs/kaggle.yaml deliberately.")

        target = checkpoint_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".staging")
        shutil.copy2(chosen["path"], tmp)
        os.replace(tmp, target)

        verified = _checkpoint_meta(target)
        if verified["fold"] != str(fold) or verified["epoch"] != chosen["epoch"]:
            target.unlink(missing_ok=True)
            raise KagglePersistError(
                f"staged {name} did not verify after the copy: expected fold "
                f"{fold!r} epoch {chosen['epoch']}, read back fold "
                f"{verified['fold']!r} epoch {verified['epoch']}. The copy is "
                "removed; do not train on top of it.")

        report["staged"].append({
            "name": name,
            "from": str(chosen["path"]),
            "source_dataset": chosen.get("source_dataset"),
            "to": str(target),
            "epoch": chosen["epoch"],
            "config_hash": chosen["config_hash"],
            "saved_utc": chosen["saved_utc"],
            "size_mb": round(chosen["size_mb"], 1),
        })
        if name == "last.pt":
            report["resumable"] = True
        if verbose:
            print(f"  staged {name} from {chosen['source_dataset']}: "
                  f"epoch {chosen['epoch']}, {chosen['size_mb']:.0f} MB, "
                  f"hash {chosen['config_hash']}")
            print(f"    -> {target}")

    if verbose and not report["staged"]:
        print("  no attached checkpoint for this fold; this session starts clean")
        print("  (that is correct for the FIRST session of a run)")
    return report


# --------------------------------------------------------------------------
# telling the truth about what survives
# --------------------------------------------------------------------------
def working_output_mb(resolved: dict, exclude: Iterable[str] = ()) -> dict:
    """Size of what a Save Version would capture, by top-level entry.

    Kaggle caps a version's output, and the tile cache and the repo clone both
    live in the same directory as the checkpoints. Knowing the split before
    saving is how you find out that 4 GB of regenerable cache is about to be
    stored instead of a 93 MB deliverable.
    """
    working = Path(resolved["persistent_dir"])
    if not working.is_dir():
        raise KagglePersistError(f"PERSISTENT_DIR does not exist: {working}")

    skip = {str(s) for s in exclude}
    entries, total = {}, 0.0
    for child in sorted(working.iterdir()):
        if child.name in skip:
            continue
        size = 0
        if child.is_file():
            size = child.stat().st_size
        elif child.is_dir() and not child.is_symlink():
            for path in child.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    try:
                        size += path.stat().st_size
                    except OSError:
                        pass
        mb = size / 1024 ** 2
        entries[child.name] = round(mb, 1)
        total += mb
    return {"root": str(working), "entries": entries, "total_mb": round(total, 1)}


def survival_report(fold: str, resolved: dict, trainer=None,
                    config: Optional[dict] = None) -> dict:
    """Print exactly what has to happen for this run to reach the next session.

    Deliberately not automatic: saving a version is a UI action that a notebook
    cannot perform for itself. The failure mode this exists to prevent is a
    finished 8-hour session whose author believed the checkpoint was safe
    because the path printed without an error.
    """
    _require_kaggle(resolved)

    checkpoint_dir = Path(resolved["persistent_dir"]) / "checkpoints" / str(fold)
    present = {}
    for name in CHECKPOINT_NAMES:
        path = checkpoint_dir / name
        present[name] = (_checkpoint_meta(path) if path.is_file() else None)

    sizes = working_output_mb(resolved)
    report = {"fold": str(fold), "checkpoint_dir": str(checkpoint_dir),
              "checkpoints": {}, "output": sizes}

    print("=" * 72)
    print("KAGGLE CHECKPOINT SURVIVAL")
    print("=" * 72)
    for name, meta in present.items():
        if meta is None:
            print(f"  {name:<8} MISSING at {checkpoint_dir / name}")
            report["checkpoints"][name] = None
            continue
        print(f"  {name:<8} epoch {meta['epoch']}, {meta['size_mb']:.0f} MB, "
              f"hash {meta['config_hash']}, saved {meta['saved_utc']}")
        report["checkpoints"][name] = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in meta.items()}

    print(f"\n  /kaggle/working currently holds {sizes['total_mb']:.0f} MB:")
    for entry, mb in sorted(sizes["entries"].items(), key=lambda kv: -kv[1]):
        note = ""
        if entry == "tile_cache":
            note = "  <- regenerable; not worth saving"
        elif entry in ("boundary", "logs"):
            note = "  <- comes back from git / is not needed to resume"
        print(f"    {mb:9.1f} MB  {entry}{note}")

    if trainer is not None and getattr(trainer, "history", None):
        done = trainer.history[-1]["epoch"] + 1
        total = int(trainer.settings["epochs"])
        print(f"\n  progress: {done}/{total} epochs done"
              + ("  -- RUN COMPLETE" if done >= total else
                 f"  -- {total - done} epoch(s) still to run"))
        report["epochs_done"] = done
        report["epochs_total"] = total

    print("\n  TO KEEP THIS: File -> Save Version -> Quick Save, with")
    print("  'Save output' enabled. /kaggle/working is deleted when this")
    print("  kernel stops; a saved version is the ONLY thing that survives.")
    print("  Then, for the next session: attach this notebook's output as an")
    print("  input dataset, and stage_resume() will find last.pt inside it.")
    print("=" * 72)
    return report
