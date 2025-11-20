# src/tortoise/dataset.py

import os
import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import numpy as np


class TileDataset(Dataset):
    """
    Uses tile_index.csv to load tiles from ms.tif, rgb.png, and label.tif using rasterio.
    RGB is always assumed to be PNG.
    """

    def __init__(
        self,
        dataset_root,
        tile_index_csv,
        load_ms=True,
        load_rgb=False,
        load_label=False,
        transforms=None
    ):
        self.dataset_root = Path(dataset_root)
        self.tile_index_csv = Path(tile_index_csv)

        self.load_ms = load_ms
        self.load_rgb = load_rgb
        self.load_label = load_label
        self.transforms = transforms

        # ------------------------------------------------------------
        # Load tile_index.csv into memory
        # ------------------------------------------------------------
        self.entries = []
        with open(self.tile_index_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append({
                    "tile_id": int(row["tile_id"]),
                    "image_id": row["image_id"],
                    "h0": int(row["h0"]),
                    "w0": int(row["w0"]),
                    "tile_size": int(row["height"])
                })

        self.tile_size = self.entries[0]["tile_size"]

        # ------------------------------------------------------------
        # Cache file paths per image_id
        # ------------------------------------------------------------
        self.ms_paths = {}
        self.rgb_paths = {}
        self.label_paths = {}

        for entry in self.entries:
            img = entry["image_id"]
            folder = self.dataset_root / img

            if img not in self.ms_paths:
                self.ms_paths[img] = folder / "ms.tif"

            if self.load_rgb and img not in self.rgb_paths:
                self.rgb_paths[img] = folder / "rgb.png"

            if self.load_label and img not in self.label_paths:
                self.label_paths[img] = folder / "label.tif"


    # ------------------------------------------------------------
    # PyTorch Methods
    # ------------------------------------------------------------
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]

        tile_id  = entry["tile_id"]
        image_id = entry["image_id"]
        h0       = entry["h0"]
        w0       = entry["w0"]
        tile_sz  = entry["tile_size"]

        sample = {
            "tile_id": tile_id,
            "image_id": image_id,
            "coords": (h0, w0),
        }

        window = Window(w0, h0, tile_sz, tile_sz)

        # --------------------------------------------------------
        # MS tile
        # --------------------------------------------------------
        if self.load_ms:
            with rasterio.open(self.ms_paths[image_id]) as src:
                ms_tile = src.read(window=window)  # (C, H, W)
            sample["ms"] = torch.from_numpy(ms_tile.copy()).float()

        # --------------------------------------------------------
        # RGB tile (from PNG)
        # --------------------------------------------------------
        if self.load_rgb:
            with rasterio.open(self.rgb_paths[image_id]) as src:
                # Usually 3-band RGB → shape (3, H, W)
                rgb = src.read(window=window)
            sample["rgb"] = torch.from_numpy(rgb.copy()).float()

        # --------------------------------------------------------
        # Label tile
        # --------------------------------------------------------
        if self.load_label:
            with rasterio.open(self.label_paths[image_id]) as src:
                lab_tile = src.read(1, window=window)  # (H, W)
            sample["label"] = torch.from_numpy(lab_tile.copy()).long().unsqueeze(0)

        # --------------------------------------------------------
        # Transforms
        # --------------------------------------------------------
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
