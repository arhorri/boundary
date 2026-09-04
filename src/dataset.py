"""Tile dataset: manifest-driven reading, synchronized augmentation, sampling.

The manifest is the source of truth. Step 3 decided which tile comes from
which image at which coordinate and on which side of which fold; this module
obeys that and derives nothing. Tiles are cropped on the fly from the full
images -- no coordinate is recomputed, no image is re-tiled, and nothing is
ever resized.

Three invariants are enforced in code rather than trusted:

* **The mask stays binary.** A spatial transform that interpolates a mask
  bilinearly produces values like 0.37, which are neither boundary nor
  background. Masks use nearest-neighbour interpolation and every sample is
  checked before it leaves ``__getitem__``.
* **Image and mask move together.** Spatial augmentation is one Compose
  applied to both; photometric augmentation is a separate Compose that the
  mask is never passed to. The separation is structural, not a flag.
* **Padding is gone.** Step 3 replaced edge padding with clamped tile
  positions, so every manifest row must read pad 0,0. A non-zero value means
  a stale manifest from before that fix, and is refused loudly rather than
  quietly training on fabricated pixels.

Normalization is a per-image z-score over the tile itself. Steel2 is optical
colour and the other four are grayscale SEM; there is no shared mean and
standard deviation that is correct for both, and a dataset-level constant
would encode the very domain difference the held-out fold exists to measure.
"""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from src.paths import REPO_ROOT

# --------------------------------------------------------------------------
# defaults -- overridable from configs/default.yaml under ``dataset:``
# --------------------------------------------------------------------------
DEFAULTS = {
    "patch_size": 256,

    # spatial: applied to image AND mask, identically
    "p_hflip": 0.5,
    "p_vflip": 0.5,
    "p_rot90": 0.5,
    "p_affine": 0.5,
    "affine_shift": 0.0625,    # fraction of the tile
    "affine_scale": 0.10,      # +/- 10%
    "affine_rotate": 15,       # degrees
    "p_elastic": 0.15,
    "elastic_alpha": 20.0,     # low: micrographs must stay physically plausible
    "elastic_sigma": 6.0,

    # photometric: applied to the IMAGE only
    "p_brightness_contrast": 0.5,
    "brightness_limit": 0.25,
    "contrast_limit": 0.25,
    "p_gamma": 0.3,
    "gamma_limit": [70, 140],
    "p_noise": 0.25,
    "noise_std_range": [0.02, 0.10],   # fraction of full scale
    "p_blur": 0.2,
    "blur_limit": 3,
    "p_clahe": 0.2,
    "clahe_clip_limit": 2.0,

    "manifest_subdir": "manifests",
    "gt_subdir": "gt_boundaries",
}

#: Splits that receive augmentation. Everything else returns the raw tile.
AUGMENTED_SPLITS = ("train",)

REQUIRED_COLUMNS = (
    "tile_id", "dataset", "parent_id", "source_image", "x", "y",
    "boundary_fraction", "split", "patch", "image_path", "gt_path",
    "crop_left", "crop_top", "crop_width", "crop_height",
    "pad_right", "pad_bottom",
)

INT_COLUMNS = ("x", "y", "patch", "crop_left", "crop_top", "crop_width",
               "crop_height", "pad_right", "pad_bottom")


class DatasetError(RuntimeError):
    """Raised when a tile cannot be produced. Never fails silently."""


def load_config(config_path: Optional[Path] = None) -> dict:
    """Merge ``dataset:`` from configs/default.yaml over DEFAULTS."""
    from src import paths as paths_mod

    cfg = paths_mod.load_config(config_path)
    settings = dict(DEFAULTS)
    section = cfg.get("dataset") or {}
    if not isinstance(section, dict):
        raise DatasetError(
            f"configs/default.yaml: dataset must be a mapping, got "
            f"{type(section).__name__}")
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise DatasetError(
            f"configs/default.yaml: unknown dataset keys {sorted(unknown)}; "
            f"known keys are {sorted(DEFAULTS)}")
    for key, value in section.items():
        if value is not None:
            settings[key] = value
    return settings


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def load_manifest(
    path: Path,
    split: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
) -> list:
    """Read a fold manifest and validate every row before anything uses it."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(
            f"{path} does not exist. Run notebooks/03_tiling.ipynb: the fold "
            "manifests are what define the dataset.")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise DatasetError(f"{path} has no rows.")

    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise DatasetError(
            f"{path} is missing columns {missing}. It was written by an older "
            "version of src/tiling.py; re-run step 3.")

    out = []
    for i, row in enumerate(rows):
        rec = dict(row)
        for col in INT_COLUMNS:
            try:
                rec[col] = int(rec[col])
            except (TypeError, ValueError) as exc:
                raise DatasetError(
                    f"{path} row {i} ({row.get('tile_id')}): column {col} is "
                    f"{row.get(col)!r}, not an integer") from exc
        rec["boundary_fraction"] = float(rec["boundary_fraction"])
        # Step 3 clamps the last tile to the image edge, so padding no longer
        # exists. A non-zero value here means this manifest predates that fix.
        if rec["pad_right"] or rec["pad_bottom"]:
            raise DatasetError(
                f"{path} row {i} ({rec['tile_id']}) carries padding "
                f"{rec['pad_right']},{rec['pad_bottom']}. Padding was "
                "eliminated in step 3; this manifest is stale. Re-run "
                "notebooks/03_tiling.ipynb and push before training.")
        if split and rec["split"] != split:
            continue
        if datasets and rec["dataset"] not in set(datasets):
            continue
        out.append(rec)

    if not out:
        raise DatasetError(
            f"{path}: no rows left after filtering split={split!r} "
            f"datasets={datasets!r}")
    return out


def load_crops(reports_dir: Optional[Path] = None) -> dict:
    """(dataset, image filename) -> the image_crop step 2 actually applied.

    Read from reports/gt_extraction.json, never recomputed. Two MetalDam pairs
    have a mask one row shorter than the image; the crop that reconciled them
    is a recorded fact, and re-deriving it risks cropping a different row and
    misaligning the pair by one pixel everywhere.
    """
    path = Path(reports_dir or (REPO_ROOT / "reports")) / "gt_extraction.json"
    if not path.is_file():
        raise DatasetError(
            f"{path} does not exist. It records the crops applied in step 2, "
            "which this step must reuse rather than recompute.")
    report = json.loads(path.read_text())
    out = {}
    for name, d in report.get("datasets", {}).items():
        for rec in d.get("reconciled") or []:
            out[(name, rec["pair"][0])] = dict(rec["image_crop"])
    return out


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def _resolve(path_str: str, dataset: str, filename: str,
             roots: Optional[dict] = None, kind: str = "image") -> Path:
    """Use the manifest's path, or re-root it if the manifest came from a host.

    Manifests record absolute paths from the machine that built them. On the
    same host that is exactly right; on another one the dataset is mounted
    elsewhere, so the fallback rebuilds the path from src/paths.py rather than
    failing or hardcoding anything.
    """
    path = Path(path_str)
    if path.is_file():
        return path
    if roots:
        root = roots.get("gt_root") if kind == "gt" else roots.get("data_root")
        if root:
            if kind == "gt":
                candidate = Path(root) / dataset / filename
            else:
                candidate = Path(root) / dataset / "images" / filename
            if candidate.is_file():
                return candidate
    raise DatasetError(
        f"cannot find the {kind} for {dataset}/{filename}. The manifest says "
        f"{path_str}, which does not exist on this host. Pass roots={{'data_root': "
        "..., 'gt_root': ...}} from src.paths.resolve_paths().")


def read_pair(row: dict, crops: Optional[dict] = None,
              roots: Optional[dict] = None) -> tuple:
    """Read one source image and its boundary map, cropped and aligned.

    Returns ``(image, gt)`` as full-size arrays, before tiling. The image is
    reduced to a single channel: Steel2 is optical colour and the rest are
    grayscale SEM, and the network takes one channel for all of them.
    """
    from PIL import Image

    img_path = _resolve(row["image_path"], row["dataset"], row["source_image"],
                        roots, kind="image")
    gt_name = Path(row["source_image"]).stem + ".png"
    gt_path = _resolve(row["gt_path"], row["dataset"], gt_name, roots, kind="gt")

    with Image.open(img_path) as im:
        # "L" is a luminance conversion, not a channel drop: it keeps Steel2's
        # colour information as intensity instead of discarding two thirds.
        image = np.asarray(im.convert("L"))
    gt = np.asarray(Image.open(gt_path))
    if gt.ndim == 3:
        gt = gt[..., 0]

    crop = (crops or {}).get((row["dataset"], row["source_image"]))
    if crop is None and (row["crop_width"] != image.shape[1]
                         or row["crop_height"] != image.shape[0]):
        # The manifest carries the same crop; use it, but only as a fallback,
        # and say so if the two sources disagree.
        crop = {"left": row["crop_left"], "top": row["crop_top"],
                "width": row["crop_width"], "height": row["crop_height"]}
    if crop is not None:
        for key, col in (("left", "crop_left"), ("top", "crop_top"),
                         ("width", "crop_width"), ("height", "crop_height")):
            if int(crop[key]) != int(row[col]):
                raise DatasetError(
                    f"{row['dataset']}/{row['source_image']}: "
                    f"reports/gt_extraction.json says {key}={crop[key]} but the "
                    f"manifest says {col}={row[col]}. Step 2 and step 3 "
                    "disagree about this pair; re-run step 3.")
        image = image[crop["top"]:crop["top"] + crop["height"],
                      crop["left"]:crop["left"] + crop["width"]]

    if image.shape != gt.shape:
        raise DatasetError(
            f"{row['dataset']}/{row['source_image']}: image is "
            f"{image.shape[1]}x{image.shape[0]} after the recorded crop but its "
            f"boundary map is {gt.shape[1]}x{gt.shape[0]}. Shapes must match "
            "exactly before a tile is cut.")
    return image, gt


def crop_tile(array: np.ndarray, x: int, y: int, patch: int) -> np.ndarray:
    """Cut one tile at the manifest's coordinate. No resizing, no padding."""
    tile = array[y:y + patch, x:x + patch]
    if tile.shape[:2] != (patch, patch):
        raise DatasetError(
            f"tile at ({x},{y}) of a {array.shape[1]}x{array.shape[0]} array is "
            f"{tile.shape[1]}x{tile.shape[0]}, not {patch}x{patch}. The "
            "manifest coordinate does not fit this image.")
    return tile


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------
def _supported(cls, **kwargs) -> dict:
    """Keep only the kwargs this albumentations version actually accepts.

    albumentations renames parameters between majors (``var_limit`` ->
    ``std_range``, ``alpha_affine`` dropped, ``ShiftScaleRotate`` deprecated in
    favour of ``Affine``). Filtering by signature keeps one pipeline definition
    working across host versions instead of pinning a version the hosts may
    not have.
    """
    try:
        names = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in names}


def _border_kwargs(cls) -> dict:
    """Ask for reflected borders under whatever name this version uses.

    Geometric transforms default to a CONSTANT border fill, which paints black
    wedges into the corners of every rotated tile. That fill is worse than
    useless here: the seam between real image and flat fill is a hard, dead
    straight, high-contrast intensity step whose mask value is 0, so it teaches
    a boundary detector that strong straight edges are NOT boundaries.

    Reflecting instead is not the tile-edge padding removed in step 3. That was
    a systematic 48% of specific tiles, fabricated identically every epoch, at
    a fixed location. This is a small wedge that moves with every random draw,
    it is real texture from the same specimen, and it carries the mask along
    with it -- so a reflected boundary is still labelled a boundary. Constant
    fill is strictly worse: it invents an edge that exists in no micrograph.
    """
    import cv2

    try:
        names = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return {}
    if "border_mode" in names:
        return {"border_mode": cv2.BORDER_REFLECT_101}
    if "mode" in names:      # albumentations 1.x named it "mode" on Affine
        return {"mode": cv2.BORDER_REFLECT_101}
    return {}


def spatial_transform(settings: dict, image_interpolation: Optional[int] = None):
    """Spatial augmentation applied to the image AND the mask, identically.

    The mask is resampled with NEAREST so it can only ever hold 0 or 1; the
    image is resampled bilinearly. ``image_interpolation`` overrides the image
    side for tests that need both paths to be pixel-identical.
    """
    import albumentations as A
    import cv2

    img_interp = cv2.INTER_LINEAR if image_interpolation is None else image_interpolation
    common = {"interpolation": img_interp, "mask_interpolation": cv2.INTER_NEAREST}

    affine = getattr(A, "Affine", None)
    if affine is not None:
        shift = float(settings["affine_shift"])
        scale = float(settings["affine_scale"])
        affine_op = affine(**_supported(
            affine,
            **_border_kwargs(affine),
            translate_percent=(-shift, shift),
            scale=(1 - scale, 1 + scale),
            rotate=(-float(settings["affine_rotate"]), float(settings["affine_rotate"])),
            fit_output=False,
            p=float(settings["p_affine"]),
            **common))
    else:  # pragma: no cover - very old albumentations
        affine_op = A.ShiftScaleRotate(**_supported(
            A.ShiftScaleRotate,
            **_border_kwargs(A.ShiftScaleRotate),
            shift_limit=float(settings["affine_shift"]),
            scale_limit=float(settings["affine_scale"]),
            rotate_limit=float(settings["affine_rotate"]),
            p=float(settings["p_affine"]),
            **common))

    ops = [
        A.HorizontalFlip(p=float(settings["p_hflip"])),
        A.VerticalFlip(p=float(settings["p_vflip"])),
        A.RandomRotate90(p=float(settings["p_rot90"])),
        affine_op,
        A.ElasticTransform(**_supported(
            A.ElasticTransform,
            **_border_kwargs(A.ElasticTransform),
            alpha=float(settings["elastic_alpha"]),
            sigma=float(settings["elastic_sigma"]),
            p=float(settings["p_elastic"]),
            **common)),
    ]
    # "mask_image" is the mask routed through the IMAGE pipeline: the tests use
    # it to prove both targets receive the same geometric transform.
    return A.Compose(ops, additional_targets={"mask_image": "image"})


def photometric_transform(settings: dict):
    """Photometric augmentation applied to the IMAGE only. The mask is not passed.

    This matters more here than in an ordinary segmentation task. The five
    datasets span SEM and optical microscopy, grayscale and colour, different
    etches, magnifications and detectors -- and the held-out fold is ALWAYS a
    different microscope from the training ones. There is no domain-adaptation
    stage in this pipeline: photometric variety is the only thing standing
    between the model and a detector it has never seen. Brightness, gamma,
    noise, blur and local contrast are exactly the axes along which two
    microscopes disagree about the same specimen.
    """
    import albumentations as A

    return A.Compose([
        A.RandomBrightnessContrast(**_supported(
            A.RandomBrightnessContrast,
            brightness_limit=float(settings["brightness_limit"]),
            contrast_limit=float(settings["contrast_limit"]),
            p=float(settings["p_brightness_contrast"]))),
        A.RandomGamma(**_supported(
            A.RandomGamma,
            gamma_limit=tuple(settings["gamma_limit"]),
            p=float(settings["p_gamma"]))),
        A.GaussNoise(**_supported(
            A.GaussNoise,
            std_range=tuple(settings["noise_std_range"]),
            var_limit=(float(settings["noise_std_range"][0] * 255) ** 2,
                       float(settings["noise_std_range"][1] * 255) ** 2),
            p=float(settings["p_noise"]))),
        A.GaussianBlur(**_supported(
            A.GaussianBlur,
            blur_limit=(3, max(3, int(settings["blur_limit"]))),
            p=float(settings["p_blur"]))),
        A.CLAHE(**_supported(
            A.CLAHE,
            clip_limit=float(settings["clahe_clip_limit"]),
            p=float(settings["p_clahe"]))),
    ])


def verify_interpolation(transform) -> list:
    """Assert mask/image interpolation and border fill are what they claim.

    Configuring interpolation is not the same as having it: this walks the
    composed pipeline and checks the attributes that ended up on the objects.
    Returns ``{"verified": [...], "unverified": [...], "unset": [...]}``.
    ``unverified`` ops are those whose albumentations version exposes no
    ``mask_interpolation`` attribute; they are covered instead by
    :func:`assert_binary` on every sample. ``unset`` ops carry the attribute
    as ``None`` -- containers such as ``Compose`` expose it as an override slot
    and leave it unset, which correctly defers to the transforms inside.

    The ``border_*`` keys report the same three states for the border fill
    mode, which must be ``BORDER_REFLECT_101`` on every op that exposes it:
    a constant fill would paint a hard straight edge into the corner of every
    rotated tile and label it background.
    """
    import cv2

    missing = object()
    verified, unverified, unset = [], [], []
    border_verified, border_unverified, border_unset = [], [], []
    stack = [transform]
    while stack:
        node = stack.pop()
        for child in getattr(node, "transforms", []) or []:
            stack.append(child)
        name = type(node).__name__

        # --- border fill: a constant fill paints a fabricated straight edge
        # into every rotated tile, and labels it "not a boundary".
        border = getattr(node, "border_mode", missing)
        if border is missing:
            border = getattr(node, "mode", missing)
            if not isinstance(border, (int, float, type(None))):
                border = missing      # "mode" means something else on this op
        if border is None:
            border_unset.append(name)
        elif border is missing:
            if getattr(node, "interpolation", None) is not None:
                border_unverified.append(name)
        elif int(border) != int(cv2.BORDER_REFLECT_101):
            raise DatasetError(
                f"{name} fills borders with mode {border}, not "
                f"BORDER_REFLECT_101 ({cv2.BORDER_REFLECT_101}). A constant "
                "fill invents a hard straight edge whose mask says 'not a "
                "boundary' -- the exact opposite of the signal being trained.")
        else:
            border_verified.append(name)

        mask_interp = getattr(node, "mask_interpolation", missing)

        if mask_interp is None:
            # A container (Compose) exposes mask_interpolation as an OVERRIDE
            # slot and leaves it None unless asked to force one on its
            # children. None means "this node decides nothing", which is what
            # we want -- the individual transforms carry the real setting.
            unset.append(name)
            continue
        if mask_interp is missing:
            # Older albumentations hardcode nearest for masks and expose no
            # attribute to check. Not an error, but not verifiable here:
            # assert_binary() in __getitem__ is what actually enforces it.
            if getattr(node, "interpolation", None) is not None:
                unverified.append(name)
            continue

        if int(mask_interp) != int(cv2.INTER_NEAREST):
            raise DatasetError(
                f"{name} resamples masks with {mask_interp}, not INTER_NEAREST "
                f"({cv2.INTER_NEAREST}). A bilinearly resampled mask is no "
                "longer binary.")
        image_interp = getattr(node, "interpolation", None)
        if image_interp is not None and int(image_interp) not in (
                int(cv2.INTER_LINEAR), int(cv2.INTER_NEAREST)):
            raise DatasetError(
                f"{name} resamples images with {image_interp}, which is neither "
                f"INTER_LINEAR ({cv2.INTER_LINEAR}) nor the INTER_NEAREST "
                "override the synchronization test uses.")
        verified.append(name)
    return {"verified": sorted(verified), "unverified": sorted(unverified),
            "unset": sorted(unset),
            "border_verified": sorted(border_verified),
            "border_unverified": sorted(border_unverified),
            "border_unset": sorted(border_unset)}


def assert_binary(mask: np.ndarray, where: str) -> None:
    """A mask that is not exactly {0,1} means interpolation touched it."""
    bad = np.logical_and(mask != 0, mask != 1)
    if bad.any():
        values = np.unique(mask[bad])[:5]
        raise DatasetError(
            f"{where}: mask holds non-binary values {values.tolist()} after "
            "augmentation. The mask was resampled with something other than "
            "nearest-neighbour.")


def constant_border_region(tile: np.ndarray, max_values: int = 3) -> dict:
    """Largest flat region of one value that touches the tile border.

    Constant-fill augmentation leaves a wedge of exactly identical pixels
    against an edge of the tile. Real microstructure does not: even a dark
    phase varies pixel to pixel. So a large, exactly-uniform, border-touching
    component is a reliable fingerprint of fabricated fill -- measured on the
    produced tile rather than trusted from the transform's configuration.

    Returns the fraction of the tile it covers, and the value it is made of.
    """
    from scipy.ndimage import label

    arr = np.asarray(tile)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr[..., 0]
    height, width = arr.shape
    area = float(height * width)

    ring = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    values, counts = np.unique(ring, return_counts=True)
    order = np.argsort(-counts)[:max_values]

    border = np.zeros((height, width), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    worst = {"fraction": 0.0, "value": None, "pixels": 0}
    for idx in order:
        value = values[idx]
        hit = arr == value
        if not hit.any():
            continue
        components, n = label(hit)
        if not n:
            continue
        for comp in set(np.unique(components[border & hit]).tolist()) - {0}:
            pixels = int((components == comp).sum())
            fraction = pixels / area
            if fraction > worst["fraction"]:
                worst = {"fraction": float(fraction), "value": float(value),
                         "pixels": pixels}
    return worst


def normalize(tile: np.ndarray) -> np.ndarray:
    """Per-image z-score over this tile. No dataset or global statistics.

    Applied AFTER photometric augmentation so that a train tile and a val tile
    arrive at the network with the same first two moments; the augmentation
    still varies contrast, gamma, noise and blur, which z-scoring cannot undo.
    """
    arr = tile.astype(np.float32)
    std = float(arr.std())
    if std < 1e-6:
        # A flat tile has no scale to normalise by; centring is all that is
        # meaningful, and dividing would amplify pure noise.
        return arr - float(arr.mean())
    return (arr - float(arr.mean())) / std


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError as exc:  # pragma: no cover - hosts ship torch
    raise DatasetError(
        "torch is not importable. Colab and Kaggle ship it preinstalled; this "
        "module is meant to run on a host, never on a local machine."
    ) from exc


class TileDataset(_TorchDataset):
    """Tiles named by a fold manifest, cropped on the fly from the full images.

    Returns a dict, not a tuple. ``dataset`` and ``parent_id`` travel with
    every sample because validation sets are deliberately mixed -- the held-out
    dataset plus Steel1's validation parents -- and step 6 must report metrics
    per dataset within val. Pooled validation numbers would let Steel1's easy,
    blob-like boundaries mask poor performance on the actually held-out
    microscope, which is the only thing the fold is there to measure.
    """

    def __init__(
        self,
        rows: Sequence,
        settings: Optional[dict] = None,
        augment: Optional[bool] = None,
        crops: Optional[dict] = None,
        roots: Optional[dict] = None,
        cache_images: bool = True,
    ):
        self.rows = list(rows)
        if not self.rows:
            raise DatasetError("TileDataset was given no rows.")
        self.settings = settings or load_config()
        splits = {r["split"] for r in self.rows}
        if augment is None:
            augment = splits <= set(AUGMENTED_SPLITS) and bool(splits)
        self.augment = bool(augment)
        self.crops = crops if crops is not None else {}
        self.roots = roots or {}
        self.patch = int(self.settings["patch_size"])
        self._cache_images = cache_images
        self._cache = {}

        self.spatial = spatial_transform(self.settings) if self.augment else None
        self.photometric = photometric_transform(self.settings) if self.augment else None
        if self.spatial is not None:
            verify_interpolation(self.spatial)

    # -- plumbing ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def _pair(self, row: dict) -> tuple:
        key = (row["dataset"], row["source_image"])
        if key in self._cache:
            return self._cache[key]
        pair = read_pair(row, crops=self.crops, roots=self.roots)
        if self._cache_images:
            # One entry per source image: a MetalDam image serves 28 tiles.
            self._cache[key] = pair
        return pair

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_full, gt_full = self._pair(row)
        patch = int(row["patch"]) or self.patch
        image = crop_tile(image_full, row["x"], row["y"], patch)
        mask = (crop_tile(gt_full, row["x"], row["y"], patch) > 0).astype(np.uint8)

        if self.augment:
            out = self.spatial(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
            assert_binary(mask, f"{row['tile_id']} after spatial augmentation")
            # The mask is not passed to the photometric pipeline at all: it
            # cannot be brightened, blurred or noised because it is not there.
            image = self.photometric(image=image)["image"]

        mask = (np.asarray(mask) > 0).astype(np.float32)
        assert_binary(mask, f"{row['tile_id']} before collation")
        image = normalize(np.asarray(image))

        return {
            "image": image[None, ...].astype(np.float32),
            "mask": mask[None, ...].astype(np.float32),
            "dataset": row["dataset"],
            "parent_id": row["parent_id"],
            "tile_id": row["tile_id"],
        }

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        split: str,
        settings: Optional[dict] = None,
        crops: Optional[dict] = None,
        roots: Optional[dict] = None,
        datasets: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> "TileDataset":
        """Build one split of one fold. Augmentation follows the split."""
        rows = load_manifest(manifest_path, split=split, datasets=datasets)
        if crops is None:
            crops = load_crops()
        return cls(rows, settings=settings, crops=crops, roots=roots,
                   augment=split in AUGMENTED_SPLITS, **kwargs)

    # -- composition ------------------------------------------------------
    def composition(self) -> dict:
        """Tiles and parents per dataset -- what a mixed val split is made of."""
        out = {}
        for row in self.rows:
            entry = out.setdefault(row["dataset"], {"tiles": 0, "parents": set()})
            entry["tiles"] += 1
            entry["parents"].add(row["parent_id"])
        return {ds: {"tiles": v["tiles"], "parents": len(v["parents"])}
                for ds, v in sorted(out.items())}


# --------------------------------------------------------------------------
# sampler -- a training concern, deliberately not baked into the Dataset
# --------------------------------------------------------------------------
def load_fold_stats(configs_dir: Optional[Path] = None) -> dict:
    """Read configs/fold_stats.yaml, written by step 3."""
    import yaml

    path = Path(configs_dir or (REPO_ROOT / "configs")) / "fold_stats.yaml"
    if not path.is_file():
        raise DatasetError(
            f"{path} does not exist. Run notebooks/03_tiling.ipynb: the "
            "sampling weights and pos_weight come from it.")
    doc = yaml.safe_load(path.read_text())
    if not doc or "folds" not in doc:
        raise DatasetError(f"{path} has no folds section.")
    return doc


def build_sampler(dataset: "TileDataset", fold: str,
                  fold_stats: Optional[dict] = None,
                  num_samples: Optional[int] = None,
                  generator=None):
    """WeightedRandomSampler from the per-dataset weights step 3 computed.

    Raw tile counts do not measure independent information: Steel1's ~900
    tiles come from 19 micrographs. ``weight_mode: parents`` gives each dataset
    a share of the epoch equal to its share of the independent scenes, and the
    per-tile weight that achieves it lives in configs/fold_stats.yaml.

    This is a training concern, not a property of the data, so it is a helper
    the training step calls -- the Dataset itself knows nothing about it.
    """
    from torch.utils.data import WeightedRandomSampler

    stats = fold_stats or load_fold_stats()
    if fold not in stats["folds"]:
        raise DatasetError(
            f"fold {fold!r} is not in configs/fold_stats.yaml; available: "
            f"{sorted(k for k in stats['folds'] if k != 'test')}")
    sampling = stats["folds"][fold].get("sampling") or {}
    weights = sampling.get("weights") or {}
    if not weights:
        raise DatasetError(
            f"fold {fold!r} has no sampling weights in configs/fold_stats.yaml")

    missing = sorted({r["dataset"] for r in dataset.rows} - set(weights))
    if missing:
        raise DatasetError(
            f"no sampling weight for {missing} in fold {fold!r}. The manifest "
            "and fold_stats.yaml disagree; re-run step 3.")

    per_sample = [float(weights[r["dataset"]]) for r in dataset.rows]
    return WeightedRandomSampler(
        weights=per_sample,
        num_samples=int(num_samples or len(per_sample)),
        replacement=True,
        generator=generator,
    )


def pos_weight(fold: str, fold_stats: Optional[dict] = None) -> float:
    """n_negative / n_positive on a fold's train split, for BCEWithLogitsLoss."""
    stats = fold_stats or load_fold_stats()
    value = (stats["folds"].get(fold) or {}).get("pos_weight")
    if value is None:
        raise DatasetError(f"fold {fold!r} has no pos_weight in fold_stats.yaml")
    return float(value)
