"""
Albumentations-based augmentation utilities for multispectral tiles.

Augmentation keys:
    "hflip"  - horizontal flip (geom)
    "vflip"  - vertical flip (geom)
    "dflip"  - diagonal flip = transpose + horizontal flip (geom)
    "rot90"  - random 0/90/180/270 degrees (geom)
    "noise"  - Gaussian noise (image only)
    "blur"   - Gaussian blur (image only)
    "iscale" - random intensity scaling (image only)
"""

from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import albumentations as A
import numpy as np

# Public list of augmentation names
AUG_KEYS: List[str] = [
    "hflip",
    "vflip",
    "dflip",
    "rot90",
    "noise",
    "blur",
    "iscale",
]

GEOMETRIC_KEYS = {"hflip", "vflip", "dflip", "rot90"}


class IntensityScale(A.ImageOnlyTransform):
    """
    Simple multiplicative intensity scaling: img' = img * alpha,
    where alpha is sampled uniformly from [1 - scale_limit, 1 + scale_limit].
    """

    def __init__(self, scale_limit: float = 0.1, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.scale_limit = float(scale_limit)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        alpha = 1.0 + random.uniform(-self.scale_limit, self.scale_limit)
        return img * alpha


def _transform_for_name(name: str) -> A.BasicTransform:
    """
    Return the core Albumentations transform (not yet wrapped in a Compose).
    """
    if name == "hflip":
        return A.HorizontalFlip(p=1.0)
    if name == "vflip":
        return A.VerticalFlip(p=1.0)
    if name == "dflip":
        # Option A: transpose then horizontal flip (diagonal flip)
        return A.Compose([
            A.Transpose(p=1.0),
            A.HorizontalFlip(p=1.0),
        ])
    if name == "rot90":
        # Random 0/90/180/270 rotation
        return A.RandomRotate90(p=1.0)
    if name == "noise":
        return A.GaussNoise(std_range = (0.02, 0.08), p=1.0)
    if name == "blur":
        # light blur; tiles are only 48x48
        return A.GaussianBlur(blur_limit=(1, 3), p=1.0)
    if name == "iscale":
        return IntensityScale(scale_limit=0.1, p=1.0)

    raise ValueError(f"Unknown augmentation name: {name!r}")


def apply_augmentation(
    ms_hwc: np.ndarray,
    label_hw: np.ndarray | None,
    mask_hw: np.ndarray | None,
    aug_name: str,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Apply a given augmentation to ms (H,W,C) and optionally label/mask (H,W).

    Geometric transforms:
        - applied to ms, label, mask

    Photometric transforms (noise/blur/iscale):
        - applied to ms only (label/mask untouched)
    """
    core = _transform_for_name(aug_name)

    if aug_name in GEOMETRIC_KEYS:
        additional_targets: Dict[str, str] = {}
        data: Dict[str, np.ndarray] = {"image": ms_hwc}

        if label_hw is not None:
            additional_targets["label"] = "mask"
            data["label"] = label_hw
        if mask_hw is not None:
            additional_targets["valid"] = "mask"
            data["valid"] = mask_hw

        aug = A.Compose([core], additional_targets=additional_targets)
        out = aug(**data)

        ms_out = out["image"]
        label_out = out.get("label", label_hw)
        mask_out = out.get("valid", mask_hw)
        return ms_out, label_out, mask_out

    # Photometric: image only
    aug = A.Compose([core])
    out_img = aug(image=ms_hwc)["image"]
    return out_img, label_hw, mask_hw


# -------------------------------------------------------------------
# AUG_MAP utilities
# -------------------------------------------------------------------

AugMap = Dict[Tuple[str, str], str]


def sample_aug_map(
    tile_ids: Sequence[str],
    versions: Sequence[str] = ("aug1", "aug2"),
    aug_keys: Sequence[str] = AUG_KEYS,
    seed: int = 42,
) -> AugMap:
    """
    Pre-sample ONE augmentation for each (tile_id, version) pair in versions,
    using a fixed RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    aug_map: AugMap = {}

    for tid in tile_ids:
        for ver in versions:
            key = (tid, ver)
            aug_name = rng.choice(list(aug_keys))
            aug_map[key] = aug_name

    return aug_map


def _encode_key(tid: str, version: str) -> str:
    return f"{tid}::{version}"


def _decode_key(key: str) -> Tuple[str, str]:
    tid, version = key.split("::", 1)
    return tid, version


def save_aug_map(aug_map: Mapping[Tuple[str, str], str], path: str | Path) -> None:
    """
    Save AUG_MAP as JSON. Keys are serialized as "tile_id::version".
    """
    path = Path(path)
    serializable = {_encode_key(tid, ver): name for (tid, ver), name in aug_map.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_aug_map(path: str | Path) -> AugMap:
    """
    Load AUG_MAP from JSON saved via save_aug_map.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    aug_map: AugMap = {}
    for key, name in raw.items():
        tid, ver = _decode_key(key)
        aug_map[(tid, ver)] = name
    return aug_map
