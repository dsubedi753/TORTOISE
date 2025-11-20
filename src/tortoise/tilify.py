# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 01:03:00 2025

@author: Divas
@program: tilify.py

Full implementation of global sliding-window tiling:
    - Freestyle mask (no block structure)
    - Global sliding tiles over whole image
    - Valid tile if mask ratio >= threshold
    - Strict tile_size + stride compatibility:
        (48 - tile_size) % stride == 0
        stride < tile_size
        stride ∈ ALLOWED_STRIDES
        tile_size ∈ ALLOWED_TILE_SIZES
    - Writes global tile_index.csv
    - Writes per-image meta.json
    - Public API: run_tilify(config_file, dataset_folder)
"""

from pathlib import Path
import yaml
import json
import csv
from datetime import datetime
import rasterio
import numpy as np
import os
import re


# ============================================================================
# === Allowed parameter sets ================================================
# ============================================================================

ALLOWED_TILE_SIZES = {16, 24, 32, 36, 48}
ALLOWED_STRIDES    = {8, 12, 16, 18, 24, 48}


# ============================================================================
# === Parameter Validation ===================================================
# ============================================================================

def assert_params(tile_size, stride):
    """Validate tile_size and stride with auto-derived compatibility rules."""

    # 1. tile_size allowed
    if tile_size not in ALLOWED_TILE_SIZES:
        raise ValueError(
            f"Invalid tile_size={tile_size}. Allowed: {sorted(ALLOWED_TILE_SIZES)}"
        )

    # 2. stride allowed
    if stride not in ALLOWED_STRIDES:
        raise ValueError(
            f"Invalid stride={stride}. Allowed: {sorted(ALLOWED_STRIDES)}"
        )

    # 3. positive stride
    if stride <= 0:
        raise ValueError("stride must be > 0.")

    # 4. no gaps: stride < tile_size
    if stride >= tile_size:
        raise ValueError(
            f"stride={stride} must be < tile_size={tile_size}"
        )

    # 5. compatibility rule: (48 - tile_size) % stride == 0
    mask_gap = 48 - tile_size
    if mask_gap % stride != 0:
        raise ValueError(
            f"Invalid combination: (48 - {tile_size}) % {stride} != 0.\n"
            "Tile size and stride must align to 48-grid constraint."
        )


# ============================================================================
# === Config Loading =========================================================
# ============================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
# === Folder Scanning ========================================================
# ============================================================================

def find_image_folders(dataset_root):
    """
    Return sorted list of valid image ID folders (3-digit names).
    Example: "013", "014", ...
    """
    folders = []
    for name in os.listdir(dataset_root):
        if re.fullmatch(r"\d{3}", name):
            folder_path = os.path.join(dataset_root, name)
            if os.path.isdir(folder_path):
                folders.append(name)

    return sorted(folders)


# ============================================================================
# === Mask Reading ===========================================================
# ============================================================================

def read_mask(ms_path):
    """
    Read mask and return:
        mask   (H×W uint8, values 0 or 255)
        H
        W
    """
    with rasterio.open(ms_path) as src:
        mask = src.dataset_mask()
        H, W = src.height, src.width
    return mask, H, W


# ============================================================================
# === Global Sliding Tile Generation ========================================
# ============================================================================

def generate_global_tiles(H, W, tile_size, stride):
    """
    Generate all (y, x) top-left coordinates of tiles for sliding window.
    """
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            yield (y, x)


# ============================================================================
# === Tile Validity Check ====================================================
# ============================================================================

def is_tile_valid(mask, y, x, tile_size, threshold):
    """
    Valid if fraction of mask==255 pixels in tile >= threshold.
    threshold ∈ [0,1]
    """
    tile = mask[y:y+tile_size, x:x+tile_size]
    valid_ratio = tile.mean() / 255.0
    return valid_ratio >= threshold


# ============================================================================
# === Atomic JSON Writing ====================================================
# ============================================================================

def write_meta_json(folder, meta):
    """
    Write meta.json atomically: meta.json.tmp → meta.json
    """
    tmp_path = os.path.join(folder, "meta.json.tmp")
    final_path = os.path.join(folder, "meta.json")

    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2)

    os.replace(tmp_path, final_path)


# ============================================================================
# === Main Tiling Logic ======================================================
# ============================================================================

def tilify(dataset_root, tile_size, stride, threshold):
    """
    Process all images in dataset_root:
        - Slide tiles globally across entire image
        - Check mask validity
        - Write global tile_index.csv
        - Write per-image meta.json
    """

    # Prepare global tile index (temp)
    tile_index_tmp = os.path.join(dataset_root, "tile_index.tmp.csv")
    tile_index_final = os.path.join(dataset_root, "tile_index.csv")

    global_tile_id = 0

    folders = find_image_folders(dataset_root)
    print(f"Found {len(folders)} image folders.")

    with open(tile_index_tmp, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tile_id", "tile_key", "image_id", "h0", "w0", "height", "width"])

        # Process each image
        for image_id in folders:
            folder = os.path.join(dataset_root, image_id)
            ms_path = os.path.join(folder, "ms.tif")

            print(f"\nProcessing image {image_id} ...")

            # 1. Read mask
            mask, H, W = read_mask(ms_path)

            # 2. Generate sliding tiles
            valid_tiles = []

            for (y, x) in generate_global_tiles(H, W, tile_size, stride):

                if is_tile_valid(mask, y, x, tile_size, threshold):

                    tile_key = f"{image_id}_{y:05d}_{x:05d}"

                    # Write row to CSV
                    writer.writerow([
                        global_tile_id,
                        tile_key,
                        image_id,
                        y,
                        x,
                        tile_size,
                        tile_size
                    ])

                    # Add to per-image meta
                    valid_tiles.append({
                        "tile_id": global_tile_id,
                        "h0": y,
                        "w0": x
                    })

                    global_tile_id += 1

            # 3. Write meta.json
            meta = {
                "id": image_id,
                "height": H,
                "width": W,
                "tile_size": tile_size,
                "stride": stride,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
                "valid_tiles": valid_tiles
            }

            write_meta_json(folder, meta)

    # Atomic final rename
    os.replace(tile_index_tmp, tile_index_final)

    print("\n====================================================")
    print(f"Tiling complete. {global_tile_id} tiles created.")
    print(f"tile_index.csv written to: {tile_index_final}")
    print("====================================================")


# ============================================================================
# === Public API =============================================================
# ============================================================================

def run_tilify(config_file, dataset_folder):
    """
    Main entry point for external use.
    Parameters:
        config_file: path to config.yml
        dataset_folder: path to dataset root (Option B structure)
    """
    cfg = load_config(config_file)

    if "tiling" not in cfg:
        raise KeyError("config.yml missing 'tiling:' section.")

    tile_size = cfg["tiling"]["tile_size"]
    stride    = cfg["tiling"]["stride"]
    threshold = cfg["tiling"].get("validity_threshold", 1.0)

    dataset_root = Path(dataset_folder).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")

    # Validate parameters
    assert_params(tile_size, stride)

    # Run
    tilify(dataset_root, tile_size, stride, threshold)


# ============================================================================
# === Local Execution ========================================================
# ============================================================================
    

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]   # project root
    config_file = root / "configs" / "config.yml"
    dataset_folder = root / "data"   # data folder
    run_tilify(config_file, dataset_folder)