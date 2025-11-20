# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 01:41:44 2025

@author: Divas
"""

# experiment.py
# Explore ms.tif masks: find top-left coordinates of all 255-regions
# and their (height, width), without assuming 48x48 blocks.

from pathlib import Path
import rasterio
import numpy as np


def find_rect_regions(mask):
    """
    Find top-left corners of rectangular 255 regions in a binary mask.
    Assumes regions are roughly axis-aligned rectangles (as your tiles likely are).
    Returns a list of dicts: {"y": y, "x": x, "height": h, "width": w}.
    """
    H, W = mask.shape
    regions = []

    # visited mask so we don't double-count regions
    visited = np.zeros_like(mask, dtype=bool)

    for y in range(H):
        for x in range(W):
            if mask[y, x] != 255 or visited[y, x]:
                continue

            # Check if this pixel is a top-left candidate:
            # above and left are either 0 or out of bounds
            if y > 0 and mask[y - 1, x] == 255:
                continue
            if x > 0 and mask[y, x - 1] == 255:
                continue

            # Estimate height: go down while 255
            h = 0
            while y + h < H and mask[y + h, x] == 255:
                h += 1

            # Estimate width: go right while 255
            w = 0
            while x + w < W and mask[y, x + w] == 255:
                w += 1

            # Mark this rectangle as visited
            visited[y:y + h, x:x + w] = True

            regions.append({"y": y, "x": x, "height": h, "width": w})

    return regions


def main():
    # project_root = Path(__file__).resolve().parent[2]
    # data_root = 
    data_root = Path(r'D:/Divas/Projects/MSCS/7643/Project/TORTOISE/data/')

    if not data_root.exists():
        raise FileNotFoundError(f"Data folder not found: {data_root}")

    # Look for data/<id>/ms.tif
    ms_files = sorted((data_root).glob("*/ms.tif"))

    if not ms_files:
        print(f"No ms.tif files found under {data_root}/<id>/")
        return

    all_sizes = set()

    for ms_path in ms_files:
        image_id = ms_path.parent.name
        print(f"\n=== {image_id} ===")
        print(f"ms.tif: {ms_path}")

        with rasterio.open(ms_path) as src:
            mask = src.dataset_mask()
            H, W = src.height, src.width

        regions = find_rect_regions(mask)
        print(f"Image size: H={H}, W={W}")
        print(f"Found {len(regions)} mask regions:")

        for r in regions:
            y, x, h, w = r["y"], r["x"], r["height"], r["width"]
            all_sizes.add((h, w))
            print(f"  top_left=(x={x}, y={y}), height={h}, width={w}")

    if all_sizes:
        print("\n=== Summary of unique region sizes across all images ===")
        for (h, w) in sorted(all_sizes):
            print(f"  {h} x {w}")
    else:
        print("\nNo 255 regions found in any mask.")


if __name__ == "__main__":
    main()
