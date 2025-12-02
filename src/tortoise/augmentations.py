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
import random
from typing import List

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
        return A.GaussNoise(std_range = (1e-9, 5e-7), p=1.0)
    if name == "blur":
        # light blur; tiles are only 48x48
        return A.GaussianBlur(blur_limit=(1, 3), p=1.0)
    if name == "iscale":
        return IntensityScale(scale_limit=0.1, p=1.0)

    raise ValueError(f"Unknown augmentation name: {name!r}")


def apply_augmentation(
    image_hwc: np.ndarray,
    label_hw: np.ndarray | None,
    mask_hw: np.ndarray | None,
    aug_name: str,
):
    core = _transform_for_name(aug_name)

    
    # GEOMETRIC (apply to all targets)
    if aug_name in GEOMETRIC_KEYS:
        additional = {}
        data = {"image": image_hwc}

        if label_hw is not None:
            data["label"] = label_hw
            additional["label"] = "mask"

        if mask_hw is not None:
            data["mask"] = mask_hw
            additional["mask"] = "mask"

        aug = A.Compose([core], additional_targets=additional)
        out = aug(**data)

        return (
            out["image"],
            out.get("label", label_hw),
            out.get("mask", mask_hw),
        )

    
    # PHOTOMETRIC (image only)
    aug = A.Compose([core])
    out_img = aug(image=image_hwc)["image"]

    return out_img, label_hw, mask_hw
