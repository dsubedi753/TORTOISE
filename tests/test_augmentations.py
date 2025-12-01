import json
import numpy as np
import os
import tempfile
from pathlib import Path

PROJECT_FOLDER = Path(os.getenv("PROJECT_ROOT"))
DATA_FOLDER = PROJECT_FOLDER / "data"
src_path = PROJECT_FOLDER / "src"

import sys
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


from tortoise.augmentations import (
    sample_aug_map,
    save_aug_map,
    load_aug_map,
    apply_augmentation,
    AUG_KEYS,
    GEOMETRIC_KEYS,
)


def test_sample_aug_map_reproducible():
    tile_ids = ["0001", "0002", "0003"]
    map_a = sample_aug_map(tile_ids, versions=("aug1", "aug2"), seed=123)
    map_b = sample_aug_map(tile_ids, versions=("aug1", "aug2"), seed=123)
    assert map_a == map_b


def test_save_and_load_aug_map_roundtrip(tmp_path):
    tile_ids = ["0001", "0002"]
    aug_map = sample_aug_map(tile_ids, versions=("aug1", "aug2"), seed=42)

    p = tmp_path / "aug.json"
    save_aug_map(aug_map, p)
    assert p.exists()

    loaded = load_aug_map(p)
    assert aug_map == loaded


def test_apply_augmentation_shapes_and_label_behavior():
    H, W, C = 6, 8, 3
    # ms: random
    ms = np.arange(H * W * C, dtype=np.float32).reshape((H, W, C))
    # label: single point at (0,0) for alignment tests
    label = np.zeros((H, W), dtype=np.uint8)
    label[0, 0] = 1
    # mask: all ones
    mask = np.ones((H, W), dtype=np.uint8)

    # Photometric transforms should leave label & mask unchanged
    for aug_name in AUG_KEYS:
        ms_out, label_out, mask_out = apply_augmentation(ms, label, mask, aug_name)
        assert ms_out.shape == ms.shape
        assert mask_out.shape == mask.shape
        assert label_out.shape == label.shape
        if aug_name not in GEOMETRIC_KEYS:
            # label should be unchanged
            assert np.array_equal(label_out, label)

    # Geometric test: hflip
    ms_out, label_out, mask_out = apply_augmentation(ms, label, mask, "hflip")
    # label should move from (0,0) to (0, W-1)
    assert label_out[0, W - 1] == 1
    assert label_out[0, 0] == 0

