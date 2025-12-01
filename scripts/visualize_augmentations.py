#!/usr/bin/env python3
"""
Script to visualize a tile and sample augmentations applied to it.
Usage:
    python scripts/visualize_augmentations.py --tile_id 000123 --data_root data/tiles

If tile_id is omitted, picks the first available tile in the data root.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os

PROJECT_FOLDER = Path(os.getenv("PROJECT_ROOT"))
DATA_FOLDER = PROJECT_FOLDER / "data"
src_path = PROJECT_FOLDER / "src"

import sys
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


from tortoise.augmentations import AUG_KEYS, apply_augmentation, sample_aug_map


def read_ms_tile(path: Path):
    with rasterio.open(path) as src:
        arr = src.read()  # (C, H, W)
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
    return arr


def read_label(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1)  # (H, W)
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile_id", type=str, default=None)
    p.add_argument("--data_root", type=str, default="data/tiles")
    p.add_argument("--n_examples", type=int, default=6)
    args = p.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise RuntimeError(f"Data root {data_root} not found")

    # discover tiles
    ms_files = sorted(data_root.glob("tile_ms_*.tif"))
    if len(ms_files) == 0:
        raise RuntimeError(f"No tile_ms_*.tif files found in {data_root}")

    if args.tile_id is None:
        tile_id = ms_files[0].stem.replace("tile_ms_", "")
    else:
        tile_id = args.tile_id

    ms_path = data_root / f"tile_ms_{tile_id}.tif"
    rgb_path = data_root / f"tile_rgb_{tile_id}.png"
    label_path = data_root / f"tile_label_{tile_id}.tif"

    ms = read_ms_tile(ms_path)
    if label_path.exists():
        label = read_label(label_path)
    else:
        label = np.zeros(ms.shape[:2], dtype=np.uint8)

    # choose a subset of augmentations to display (randomly sampled)
    aug_keys = list(AUG_KEYS)
    if len(aug_keys) > args.n_examples:
        aug_keys = aug_keys[: args.n_examples]

    n = 1 + len(aug_keys)  # original + aug versions
    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))

    # Show original as first
    axs[0].imshow(ms[:, :, :3] / np.max(ms[:, :, :3]))
    axs[0].set_title("orig")
    axs[0].imshow(np.ma.masked_where(label == 0, label), alpha=0.5, cmap="Reds")
    axs[0].axis("off")

    for i, aug_name in enumerate(aug_keys, start=1):
        ms_a, label_a, mask_a = apply_augmentation(ms, label, None, aug_name)
        # display using first 3 channels
        img = ms_a[:, :, :3]
        img = img / np.max(img)
        axs[i].imshow(img)
        axs[i].set_title(aug_name)
        axs[i].imshow(np.ma.masked_where(label_a == 0, label_a), alpha=0.5, cmap="Reds")
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
