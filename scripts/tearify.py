# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 01:03:00 2025

@author: Divas
"""


#!/usr/bin/env python
import os
from pathlib import Path
import csv

import rasterio
from rasterio.windows import Window


def extract_window(src_path: Path, window: Window, dst_path: Path):
    """Read a window from src_path and write it to dst_path."""
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    with rasterio.open(src_path) as src:
        # Read window as (bands, height, width)
        data = src.read(window=window)
        mask = src.dataset_mask()[window.row_off:window.row_off+window.height, window.col_off: window.col_off+window.width]
        profile = src.profile.copy()
        profile.update(
            {
                "height": window.height,
                "width": window.width,
                "transform": src.window_transform(window),
            }
        )

        # PNG doesn't like nodata in profile
        if dst_path.suffix.lower() == ".png":
            profile.pop("nodata", None)

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)
            dst.write_mask(mask)


def tearify(dataset_root: Path):

    csv_path: Path = dataset_root / "tile_index.csv"
    imageset_root: Path = dataset_root / "imageset"
    tiles_root: Path = dataset_root / "tiles"


    """Create per-tile MS/RGB/label files from large images using tile_index.csv."""
    tiles_root.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            tile_id = int(row["tile_id"])
            image_id = row["image_id"]

            # Read coordinates and size from CSV
            h0 = int(row["h0"])
            w0 = int(row["w0"])
            height = int(row["height"])
            width = int(row["width"])

            window = Window(col_off=w0, row_off=h0, width=width, height=height)

            tile_tag = f"{tile_id:05d}"  # zero-padded 5 digits

            image_dir = imageset_root / image_id
            ms_src = image_dir / "ms.tif"
            rgb_src = image_dir / "rgb.png"
            label_src = image_dir / "label.tif"

            ms_dst = tiles_root / f"tile_ms_{tile_tag}.tif"
            rgb_dst = tiles_root / f"tile_rgb_{tile_tag}.png"
            label_dst = tiles_root / f"tile_label_{tile_tag}.tif"

            print(f"[{i}] image_id={image_id}, tile_id={tile_id} -> {tile_tag}")

            # MS tile
            extract_window(ms_src, window, ms_dst)

            # RGB tile
            extract_window(rgb_src, window, rgb_dst)

            # Label tile
            extract_window(label_src, window, label_dst)


if __name__ == "__main__":
    root = Path(os.getenv("PROJECT_ROOT"))    # project root
    config_file = root / "configs" / "config.yml"
    dataset_folder = root / "data"   # data folder
    tearify(dataset_folder)