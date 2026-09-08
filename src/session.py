"""Host-specific session survival, dispatched in exactly one place.

Most of the pipeline is host-agnostic because `src/paths.py` resolves the
locations and `configs/<platform>.yaml` supplies the differences. One thing
does not reduce to a path: **whether the session outlives the run.**

* On Colab, ``PERSISTENT_DIR`` is a Drive folder. It survives a restart, the
  session has no hard deadline worth planning around, and "write ``last.pt``
  after every epoch" is genuinely sufficient.
* On Kaggle, ``PERSISTENT_DIR`` is ``/kaggle/working``, which is **deleted when
  the kernel stops**. A free GPU session is time-limited, so a 40-epoch run is
  several sessions; the working directory reaches the next one only as a saved
  notebook version's output, re-attached as an input dataset; and an input
  dataset is mounted read-only under ``/kaggle/input``, which is not where
  ``Trainer.maybe_resume()`` looks.

That gap cannot be closed by configuration, so it is closed by code -- and the
code lives here, on ``main``, where both hosts get it, rather than on a branch
where it would drift behind steps 7-9 in silence.

**The dispatch is :func:`for_host` and nowhere else.** Callers ask for a
session once and then call the same methods regardless of platform::

    session = session.for_host(PATHS)          # the only platform decision
    session.stage_resume(trainer.run_name, fold=trainer.fold)   # no-op off Kaggle
    trainer.fit(on_epoch_end=session.guard(cb))
    session.survival_report(trainer.run_name, trainer)

No notebook, script or module may test ``PATHS["platform"]`` to decide which of
these to call. If a host needs behaviour none of these methods expresses, the
method goes on the base class as a no-op and is overridden here -- that is the
whole reason the base class exists.

Nothing here weakens a guard in ``src/train.py``. A staged checkpoint is
verified to be the requested fold *before* it is copied, and is then handed to
the same ``maybe_resume()`` that checks fold and config hash again. This module
moves files; it never decides that a resume is legitimate.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, Optional


class SessionError(RuntimeError):
    """Raised when session survival cannot be arranged. Never silent."""


class SessionStopped(RuntimeError):
    """Raised by a budget guard to end ``fit()`` while there is time to save.

    Caught by the caller, and NOT an error condition: ``Trainer.fit`` writes
    the checkpoint before invoking the epoch callback, so by the time this is
    raised ``last.pt`` on disk is complete and the next session resumes from
    it. That ordering is what makes stopping by exception safe rather than
    lossy.

    On a host whose session has no deadline this is never raised, so a caller
    that catches it costs nothing there.
    """


# --------------------------------------------------------------------------
# settings -- overridable under ``session.kaggle.persist`` in the config
# --------------------------------------------------------------------------
DEFAULTS = {
    # Kaggle kills a free GPU session at its limit with no grace period, so
    # budget slightly under the allowance.
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


def settings(config: Optional[dict] = None) -> dict:
    """DEFAULTS overlaid with ``session.kaggle.persist`` from the config.

    Read through ``src.paths.load_config`` so the platform overlay is applied
    the same way every other module sees it. Unknown keys are an error rather
    than a typo that silently does nothing.
    """
    if config is None:
        from src import paths as paths_mod

        config = paths_mod.load_config()

    section = (((config.get("session") or {}).get("kaggle") or {})
               .get("persist") or {})
    if not isinstance(section, dict):
        raise SessionError(
            "config: session.kaggle.persist must be a mapping, got "
            f"{type(section).__name__}.")
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise SessionError(
            f"config: session.kaggle.persist has unknown key(s) {unknown}. "
            f"Known keys: {sorted(DEFAULTS)}.")

    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in section.items() if v is not None})
    return merged


# --------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------
class SessionBudget:
    """Wall-clock budget for one time-limited session.

    The clock starts when this object is constructed, which is why the session
    object is built as early in a notebook as possible: the bootstrap clone,
    the pip install and the tile cache build all happen before the first epoch
    and all of them spend time the last epoch will not have. A budget measured
    from the first epoch overruns by exactly the setup cost.
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
            raise SessionError(
                f"session_budget_hours must be positive, got {self.hours}.")
        if self.reserve_minutes < 0:
            raise SessionError(
                f"reserve_minutes cannot be negative, got {self.reserve_minutes}.")
        if self.reserve_minutes * 60 >= self.hours * 3600:
            raise SessionError(
                f"reserve_minutes ({self.reserve_minutes}) consumes the entire "
                f"{self.hours} h budget; no time would be left to train.")

        self.started = float(started if started is not None else time.time())
        self.total_seconds = self.hours * 3600.0
        self.usable_seconds = self.total_seconds - self.reserve_minutes * 60.0

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
            raise SessionError(
                "seconds_per_epoch must be positive to estimate a budget.")
        return max(0, int(self.remaining_seconds() // float(seconds_per_epoch)))

    def describe(self) -> str:
        return (f"budget {self.hours:.2f} h "
                f"(reserve {self.reserve_minutes:.0f} min), "
                f"elapsed {self.elapsed_seconds() / 60:.1f} min, "
                f"{self.remaining_seconds() / 60:.1f} min usable left")

    def __str__(self) -> str:                    # pragma: no cover - cosmetic
        return self.describe()


# --------------------------------------------------------------------------
# the generic host: PERSISTENT_DIR survives, nothing to arrange
# --------------------------------------------------------------------------
class SessionSupport:
    """What a host that keeps its own files needs doing: nothing.

    Every method here is a real, honest no-op that returns the same shape its
    Kaggle override returns, so a caller never has to know which one it has.
    """

    name = "generic"
    #: Does PERSISTENT_DIR outlive the session on this host?
    persists = True
    #: Is there a wall-clock deadline the run has to plan around?
    time_limited = False

    def __init__(self, resolved: dict, config: Optional[dict] = None):
        self.resolved = resolved
        self.config = config
        self.platform = str(resolved.get("platform", "unknown"))
        self.persistent_dir = Path(resolved["persistent_dir"])
        self.budget = None

    # -- description ------------------------------------------------------
    def describe(self) -> str:
        return (f"{self.platform}: PERSISTENT_DIR is {self.persistent_dir}, "
                "which survives a session restart. Checkpoints written after "
                "every epoch are safe where they are; nothing has to be staged "
                "or saved by hand.")

    def checkpoint_dir(self, run: str) -> Path:
        """Where Trainer writes, which is named for the RUN, not the fold.

        A run training on a reduced mixture is called ``dev-no-uhcs1`` and
        keeps its checkpoints under that name, so looking under the fold name
        would miss them entirely -- reporting a correctly written checkpoint as
        absent, and on Kaggle staging the wrong one back.
        """
        return self.persistent_dir / "checkpoints" / str(run)

    # -- resume -----------------------------------------------------------
    def stage_resume(self, run: str, fold: Optional[str] = None,
                     overwrite: bool = False, verbose: bool = True) -> dict:
        """Put a resumable checkpoint where ``maybe_resume()`` looks.

        ``run`` names the checkpoint directory; ``fold`` is the identity stored
        inside the checkpoint and defaults to ``run``, which is correct
        whenever nothing is excluded. They differ for a reduced-mixture run and
        conflating them is how a correctly written checkpoint goes missing.

        Here there is nothing to do: whatever the last session wrote is still
        at that path. The report shape matches the Kaggle one so a caller can
        print it either way.
        """
        fold = fold or run
        local = self.checkpoint_dir(run) / "last.pt"
        report = {
            "platform": self.platform, "run": str(run), "fold": str(fold),
            "checkpoint_dir": str(self.checkpoint_dir(run)),
            "staged": [], "skipped": [], "candidates": [],
            "resumable": local.is_file(),
            "reason": ("PERSISTENT_DIR survives on this host; a checkpoint "
                       "from a previous session is already in place"),
        }
        if verbose:
            print(f"  nothing to stage on {self.platform}: "
                  f"{'found' if report['resumable'] else 'no'} checkpoint at "
                  f"{local}")
        return report

    # -- deadline ---------------------------------------------------------
    def guard(self, on_epoch_end: Optional[Callable] = None,
              verbose: bool = True) -> Optional[Callable]:
        """Wrap an epoch callback with a deadline. Here there is no deadline."""
        return on_epoch_end

    # -- what survives ----------------------------------------------------
    def output_sizes(self, exclude: Iterable[str] = ()) -> dict:
        """Size of PERSISTENT_DIR by top-level entry.

        Useful everywhere, and load-bearing on a host with an output size cap:
        knowing the split is how you find out that gigabytes of regenerable
        cache are about to be stored instead of a 93 MB deliverable.
        """
        root = self.persistent_dir
        if not root.is_dir():
            raise SessionError(f"PERSISTENT_DIR does not exist: {root}")
        skip = {str(s) for s in exclude}
        entries, total = {}, 0.0
        for child in sorted(root.iterdir()):
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
        return {"root": str(root), "entries": entries,
                "total_mb": round(total, 1)}

    def survival_report(self, run: str, trainer=None) -> dict:
        """Say where the checkpoints are and what, if anything, must be done."""
        checkpoint_dir = self.checkpoint_dir(run)
        report = {"platform": self.platform, "run": str(run),
                  "checkpoint_dir": str(checkpoint_dir), "checkpoints": {},
                  "action_required": False}
        print("=" * 72)
        print(f"CHECKPOINT SURVIVAL -- {self.platform}  (run {run})")
        print("=" * 72)
        for name in CHECKPOINT_NAMES:
            path = checkpoint_dir / name
            meta = checkpoint_meta(path) if path.is_file() else None
            report["checkpoints"][name] = _stringify(meta)
            if meta is None:
                print(f"  {name:<8} MISSING at {path}")
            else:
                print(f"  {name:<8} epoch {meta['epoch']}, "
                      f"{meta['size_mb']:.0f} MB, hash {meta['config_hash']}, "
                      f"saved {meta['saved_utc']}")
        print(f"\n  {self.describe()}")
        print("=" * 72)
        return report

    # -- host-specific verification --------------------------------------
    def checks(self, run: str, trainer=None) -> list:
        """``(name, ok, detail)`` triples for the notebook's checks cell.

        Host-specific verification lives with the host rather than behind a
        platform test in the notebook: the caller loops over whatever it is
        given. ``run`` is the RUN name -- checking under the fold name would
        fail every reduced-mixture run whose checkpoint is perfectly fine.
        """
        local = self.checkpoint_dir(run) / "last.pt"
        resolved_root = self.persistent_dir.resolve()
        return [(
            "checkpoint is inside PERSISTENT_DIR, which survives on this host",
            local.is_file() and resolved_root in local.resolve().parents,
            f"{local} under {resolved_root}",
        )]


# --------------------------------------------------------------------------
# checkpoint headers -- shared by both hosts
# --------------------------------------------------------------------------
def checkpoint_meta(path: Path) -> dict:
    """The identifying header of a checkpoint, without building a model."""
    from src import train as train_mod

    path = Path(path)
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


def _stringify(meta):
    if meta is None:
        return None
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in meta.items()}


# --------------------------------------------------------------------------
# Kaggle: the working directory is deleted when the kernel stops
# --------------------------------------------------------------------------
class KaggleSession(SessionSupport):
    """Checkpoint survival across Kaggle sessions.

    Three things have to happen that do not have to happen anywhere else: find
    a checkpoint in whatever was attached and stage it where
    ``Trainer.maybe_resume()`` already looks; stop the run cleanly while the
    session still has time to save; and say out loud what remains to be done by
    hand, because saving a version is a UI action a notebook cannot perform for
    itself.
    """

    name = "kaggle"
    persists = False
    time_limited = True

    def __init__(self, resolved: dict, config: Optional[dict] = None):
        super().__init__(resolved, config)
        self.settings = settings(config)
        self.budget = SessionBudget(config=config)

    def describe(self) -> str:
        return (f"kaggle: PERSISTENT_DIR is {self.persistent_dir}, which is "
                "DELETED when this kernel stops. A saved notebook version's "
                "output is the only thing that reaches the next session.\n"
                f"  {self.budget.describe()}\n"
                f"  session_budget_hours {self.settings['session_budget_hours']}"
                f"  reserve_minutes {self.settings['reserve_minutes']}"
                "  (configs/kaggle.yaml session.kaggle.persist)")

    # -- finding a checkpoint in whatever was attached --------------------
    def input_roots(self, slugs: Optional[Iterable[str]] = None) -> list:
        """Directories under ``/kaggle/input`` to search, preferred first.

        Configured slugs come first and in order; everything else attached
        follows, so a run works whether the checkpoint came from a hand-uploaded
        dataset with a stable name or from a saved version of this notebook
        whose slug is generated and unpredictable.
        """
        config = self.config
        if config is None:
            from src import paths as paths_mod

            config = paths_mod.load_config()

        kaggle_cfg = ((config.get("session") or {}).get("kaggle") or {})
        configured = Path(kaggle_cfg.get("input_root") or "/kaggle/input")

        # Two mount shapes are in play and a session can contain BOTH. Datasets
        # may mount owner-nested (/kaggle/input/datasets/<owner>/<slug>/), which
        # is what session.kaggle.input_root points at; a notebook-output
        # attachment may instead mount flat at /kaggle/input/<slug>/. Searching
        # only the configured root would find the data and miss the checkpoint
        # -- silently, as a clean start. So both shapes are enumerated.
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
                if child.name == "datasets" or base == literal:
                    for grandchild in sorted(
                            c for c in child.iterdir() if c.is_dir()):
                        offer(grandchild)

        if not attached:
            raise SessionError(
                f"no attached datasets found under {configured} or {literal}. "
                "On Kaggle at least one is mounted whenever anything is "
                "attached; if there are none, nothing is attached to this "
                "notebook.")

        wanted = list(slugs) if slugs is not None else list(
            self.settings["resume_input_slugs"] or [])

        ordered, claimed = [], set()
        for slug in wanted:
            hits = [p for p in attached if p.name == str(slug)]
            if not hits:
                raise SessionError(
                    f"session.kaggle.persist.resume_input_slugs names {slug!r} "
                    "but nothing by that name is attached to this notebook. "
                    "Attached: "
                    + (", ".join(sorted(p.name for p in attached)) or "(nothing)"))
            for hit in hits:
                ordered.append(hit)
                claimed.add(str(hit.resolve()))

        ordered.extend(p for p in attached if str(p.resolve()) not in claimed)
        return ordered

    @staticmethod
    def _candidate_paths(root: Path, run: str, name: str) -> list:
        """Where a checkpoint for ``run`` could plausibly sit under ``root``.

        Three layouts are accepted, in decreasing confidence: the working-dir
        shape a saved notebook version produces, a dataset built from the
        checkpoint folder alone, and a flat hand-made upload. Anything else
        falls through to a bounded search rather than being guessed at.
        """
        exact = [root / "checkpoints" / run / name, root / run / name,
                 root / name]
        hits = [p for p in exact if p.is_file()]
        if hits:
            return hits
        for depth in (1, 2):
            pattern = "/".join(["*"] * depth) + f"/checkpoints/{run}/{name}"
            hits.extend(sorted(p for p in root.glob(pattern) if p.is_file()))
            if hits:
                break
        return hits

    def find_checkpoints(self, run: str, fold: Optional[str] = None,
                         name: str = "last.pt") -> list:
        """Every attached checkpoint for ``run``, newest epoch first.

        ``run`` locates the directory; ``fold`` is what the checkpoint must
        SAY it is, and defaults to ``run``. The two differ for a
        reduced-mixture run -- ``dev-no-uhcs1`` on disk, ``dev`` inside the
        file -- so validating the directory name against the stored fold would
        reject every one of them.

        Only the file header is read. Files that do not load, or that belong to
        a different fold, are reported with the reason rather than dropped -- a
        checkpoint that is present but unusable is exactly the thing you need to
        be told about before spending a session training from scratch.
        """
        if name not in CHECKPOINT_NAMES:
            raise SessionError(
                f"name must be one of {CHECKPOINT_NAMES}, got {name!r}.")
        fold = fold or run

        found = []
        for root in self.input_roots():
            for path in self._candidate_paths(root, str(run), name):
                try:
                    meta = checkpoint_meta(path)
                except Exception as exc:                 # noqa: BLE001
                    found.append({
                        "path": path, "usable": False,
                        "reason": f"does not load ({type(exc).__name__}: {exc})",
                        "size_mb": path.stat().st_size / 1024 ** 2})
                    continue
                if meta["fold"] != str(fold):
                    meta.update(
                        usable=False,
                        reason=f"belongs to fold {meta['fold']!r}, not {fold!r}")
                else:
                    meta.update(usable=True, reason="fold matches")
                meta["source_dataset"] = root.name
                found.append(meta)

        found.sort(
            key=lambda m: (m.get("usable", False),
                           m.get("epoch") if m.get("epoch") is not None else -1),
            reverse=True)
        return found

    # -- staging ----------------------------------------------------------
    def stage_resume(self, run: str, fold: Optional[str] = None,
                     overwrite: bool = False, verbose: bool = True) -> dict:
        """Copy an attached checkpoint to where ``maybe_resume()`` looks.

        Never raises just because nothing was found: a first session
        legitimately has nothing to resume from. It *does* raise when something
        is wrong in a way that would otherwise waste a session -- a named slug
        that is not attached, a checkpoint too large to be what it claims, a
        copy that does not verify afterwards.

        A checkpoint already in ``PERSISTENT_DIR`` wins unless ``overwrite``.
        Within one session that local file is the run in progress, and silently
        replacing it with an older attached copy would throw away finished
        epochs.
        """
        fold = fold or run
        checkpoint_dir = self.checkpoint_dir(run)
        report = {"platform": self.platform, "run": str(run),
                  "fold": str(fold),
                  "checkpoint_dir": str(checkpoint_dir), "staged": [],
                  "skipped": [], "candidates": [], "resumable": False,
                  "reason": ""}

        local_last = checkpoint_dir / "last.pt"
        if local_last.is_file() and not overwrite:
            meta = checkpoint_meta(local_last)
            report["resumable"] = True
            report["reason"] = (f"{local_last} already exists (epoch "
                                f"{meta['epoch']}); not overwritten")
            report["skipped"].append(report["reason"])
            if verbose:
                print(f"  local checkpoint already present: {local_last} "
                      f"(epoch {meta['epoch']}, {meta['size_mb']:.0f} MB)")
                print("  nothing staged; Trainer.maybe_resume() will use it")
            return report

        for name in CHECKPOINT_NAMES:
            candidates = self.find_checkpoints(run, fold=fold, name=name)
            report["candidates"].extend(_stringify(c) for c in candidates)

            usable = [c for c in candidates if c.get("usable")]
            if not usable:
                for bad in candidates:
                    if verbose:
                        print(f"  ! ignoring {bad['path']}: {bad['reason']}")
                report["skipped"].append(f"no usable {name} attached")
                continue

            chosen = usable[0]
            if chosen["size_mb"] > float(self.settings["max_stage_mb"]):
                raise SessionError(
                    f"{chosen['path']} is {chosen['size_mb']:.0f} MB, over the "
                    f"max_stage_mb limit of {self.settings['max_stage_mb']:.0f} "
                    "MB. Either that is not the checkpoint you think it is, or "
                    "raise the limit in configs/kaggle.yaml deliberately.")

            target = checkpoint_dir / name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".staging")
            shutil.copy2(chosen["path"], tmp)
            os.replace(tmp, target)

            verified = checkpoint_meta(target)
            if (verified["fold"] != str(fold)
                    or verified["epoch"] != chosen["epoch"]):
                target.unlink(missing_ok=True)
                raise SessionError(
                    f"staged {name} did not verify after the copy: expected "
                    f"fold {fold!r} epoch {chosen['epoch']}, read back fold "
                    f"{verified['fold']!r} epoch {verified['epoch']}. The copy "
                    "is removed; do not train on top of it.")

            report["staged"].append({
                "name": name, "from": str(chosen["path"]),
                "source_dataset": chosen.get("source_dataset"),
                "to": str(target), "epoch": chosen["epoch"],
                "config_hash": chosen["config_hash"],
                "saved_utc": chosen["saved_utc"],
                "size_mb": round(chosen["size_mb"], 1)})
            if name == "last.pt":
                report["resumable"] = True
            if verbose:
                print(f"  staged {name} from {chosen['source_dataset']}: "
                      f"epoch {chosen['epoch']}, {chosen['size_mb']:.0f} MB, "
                      f"hash {chosen['config_hash']}")
                print(f"    -> {target}")

        if not report["staged"]:
            report["reason"] = (f"no attached checkpoint for run {run!r}; this "
                                "session starts clean")
            if verbose:
                print(f"  {report['reason']}")
                print("  (that is correct for the FIRST session of a run)")
        return report

    # -- deadline ---------------------------------------------------------
    def guard(self, on_epoch_end: Optional[Callable] = None,
              verbose: bool = True) -> Callable:
        """Stop ``fit()`` before the kernel is killed, on an epoch boundary.

        ``Trainer.fit`` writes the checkpoint *before* calling the callback, so
        raising here always leaves a complete ``last.pt`` on disk.

        The decision uses the SLOWEST epoch seen so far, not the mean. Epoch
        duration on a shared host is not stationary, and a mean lets one slow
        epoch at the end run past the deadline with the checkpoint half
        written.
        """
        budget = self.budget

        def guarded(record, trainer):
            if on_epoch_end is not None:
                on_epoch_end(record, trainer)

            worst = max(float(r["seconds"]) for r in trainer.history)
            remaining = budget.remaining_seconds()
            if verbose:
                print(f"  [budget] {remaining / 60:.1f} min usable left; "
                      f"slowest epoch so far {worst / 60:.1f} min; "
                      f"~{budget.epochs_affordable(worst)} more epoch(s) fit")

            if not budget.affords(worst):
                raise SessionStopped(
                    f"stopping after epoch {record['epoch']}: "
                    f"{remaining / 60:.1f} min usable session time left and "
                    f"the slowest epoch so far took {worst / 60:.1f} min. "
                    f"last.pt for epoch {record['epoch']} is written and "
                    "complete; save this notebook's version, attach its output "
                    "to the next run, and it will resume from epoch "
                    f"{record['epoch'] + 1}.")

        return guarded

    # -- what survives ----------------------------------------------------
    def survival_report(self, run: str, trainer=None) -> dict:
        """Print exactly what has to happen for this run to reach the next session.

        Deliberately not automatic: saving a version is a UI action a notebook
        cannot perform for itself. The failure this exists to prevent is a
        finished 8-hour session whose author believed the checkpoint was safe
        because the path printed without an error.
        """
        checkpoint_dir = self.checkpoint_dir(run)
        present = {}
        for name in CHECKPOINT_NAMES:
            path = checkpoint_dir / name
            present[name] = checkpoint_meta(path) if path.is_file() else None

        sizes = self.output_sizes()
        report = {"platform": self.platform, "run": str(run),
                  "checkpoint_dir": str(checkpoint_dir), "checkpoints": {},
                  "output": sizes, "action_required": True}

        print("=" * 72)
        print(f"KAGGLE CHECKPOINT SURVIVAL  (run {run})")
        print("=" * 72)
        for name, meta in present.items():
            report["checkpoints"][name] = _stringify(meta)
            if meta is None:
                print(f"  {name:<8} MISSING at {checkpoint_dir / name}")
                continue
            print(f"  {name:<8} epoch {meta['epoch']}, {meta['size_mb']:.0f} MB, "
                  f"hash {meta['config_hash']}, saved {meta['saved_utc']}")

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
        print("  Then, for the next session: attach this notebook's output as")
        print("  an input dataset, and stage_resume() will find last.pt in it.")
        print("=" * 72)
        return report

    # -- host-specific verification --------------------------------------
    def checks(self, run: str, trainer=None) -> list:
        out = list(super().checks(run, trainer))
        sizes = self.output_sizes()
        checkpoint_mb = sizes["entries"].get("checkpoints", 0.0)
        cache_mb = sizes["entries"].get("tile_cache", 0.0)
        out.append((
            "checkpoints/ is present in the savable output",
            checkpoint_mb > 0,
            f"{checkpoint_mb:.0f} MB of {sizes['total_mb']:.0f} MB total under "
            f"{sizes['root']}",
        ))
        out.append((
            "the tile cache is not bloating the savable output",
            cache_mb < 100,
            f"tile_cache {cache_mb:.0f} MB -- dataset.persist_cache is off on "
            "Kaggle by configs/kaggle.yaml, because /kaggle/input is a local "
            "SSD with none of Drive's latency",
        ))
        out.append((
            "the session budget left time to save",
            not self.budget.exhausted(),
            self.budget.describe(),
        ))
        return out


# --------------------------------------------------------------------------
# THE dispatch -- the only place a platform decides anything here
# --------------------------------------------------------------------------
#: platform name -> the class that handles its session survival.
HOSTS = {"kaggle": KaggleSession}


def for_host(resolved: Optional[dict] = None,
             config: Optional[dict] = None) -> SessionSupport:
    """The session-survival support this host needs. **The only dispatch.**

    Callers take the object once and then call its methods unconditionally. A
    platform test at a call site is a bug: it means the difference has leaked
    out of this function, and the next host added has to be found in every
    notebook rather than in the table above.
    """
    if resolved is None:
        from src import paths as paths_mod

        resolved = paths_mod.resolve_paths(config)
    platform = str(resolved.get("platform", "unknown"))
    return HOSTS.get(platform, SessionSupport)(resolved, config)
