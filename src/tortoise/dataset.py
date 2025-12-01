# dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from tortoise.augmentations import apply_augmentation, sample_aug_map, save_aug_map, AugMap  # new



class TileDataset(Dataset):
    """
    Loads tiles from: data/tiles/
        tile_ms_<ID>.tif      (13 bands)
        tile_rgb_<ID>.png     (3 band RGB)
        tile_label_<ID>.tif   (label mask)

    New behavior:
        - Indexes samples as (tile_id, version), where version in {"orig", "aug1", "aug2"}
        - If version != "orig" and aug_map is provided, applies a pre-chosen augmentation.
    """

    def __init__(
        self,
        root,
        tile_ids: Optional[Sequence[str]] = None,
        samples: Optional[Sequence[Tuple[str, str]]] = None,
        use_rgb: bool = False,
        transform=None,
        normalizer=None,
        aug_map: Optional[AugMap] = None,
    ):
        self.root = Path(root)
        self.use_rgb = use_rgb
        self.transform = transform
        self.normalizer = normalizer
        self.aug_map: Optional[AugMap] = aug_map

        # -------------------------------------------------
        # Discover all tile IDs by scanning tile_ms_*.tif
        # -------------------------------------------------
        self.ms_paths = sorted(self.root.glob("tile_ms_*.tif"))
        if len(self.ms_paths) == 0:
            raise RuntimeError(f"No tile_ms_*.tif files found in {self.root}")

        # Extract tile IDs like "00000" from filenames
        if tile_ids is not None:
            self.tile_ids = list(tile_ids)
        else:
            self.tile_ids = [p.stem.replace("tile_ms_", "") for p in self.ms_paths]

        # Build sample list: (tile_id, version)
        # If samples given, use them; otherwise default to (tid, "orig") only.
        if samples is not None:
            self.samples: List[Tuple[str, str]] = list(samples)
        else:
            self.samples = [(tid, "orig") for tid in self.tile_ids]

        # Prebuild file paths for speed
        self.rgb_paths = {tid: self.root / f"tile_rgb_{tid}.png" for tid in self.tile_ids}
        self.label_paths = {tid: self.root / f"tile_label_{tid}.tif" for tid in self.tile_ids}

    def __len__(self):
        return len(self.samples)

    def _read_raster(self, path):
        with rasterio.open(path) as src:
            arr = src.read()        # (C, H, W)
            return torch.from_numpy(arr).float()

    def _read_mask(self, path):
        with rasterio.open(path) as src:
            arr = src.dataset_mask()      # (H, W)
            return torch.from_numpy(arr).float()

    def __getitem__(self, index):
        tid, version = self.samples[index]

        # Read multispectral tile
        ms = self._read_raster(self.root / f"tile_ms_{tid}.tif").float()      # (13, H, W)

        # dataset_mask (valid pixels)
        mask = self._read_mask(self.root / f"tile_ms_{tid}.tif").float() / 255.0  # (H, W)

        # Label tile
        label = self._read_raster(self.root / f"tile_label_{tid}.tif").float() / 65535.0  # (1, H, W) or (C,H,W)

        # Normalize MS if requested (before augmentation)
        if self.normalizer is not None:
            ms = self.normalizer(ms)

        # Optional RGB
        rgb = None
        if self.use_rgb:
            rgb = self._read_raster(self.root / f"tile_rgb_{tid}.png")    # (3, H, W)

        # -------------------------------------------------
        # Apply augmentation if version != "orig" and aug_map is provided
        # -------------------------------------------------
        if self.aug_map is not None and version != "orig":
            aug_name = self.aug_map.get((tid, version), None)
            if aug_name is not None:
                # Convert to numpy (H, W, C) and (H, W)
                # ms: (C,H,W) -> (H,W,C)
                ms_np = ms.permute(1, 2, 0).cpu().numpy()

                # label: (C,H,W) or (H,W) -> (H,W)
                if label.ndim == 3:
                    label_np = label[0].cpu().numpy()
                else:
                    label_np = label.cpu().numpy()

                mask_np = mask.cpu().numpy()

                ms_np_aug, label_np_aug, mask_np_aug = apply_augmentation(
                    ms_hwc=ms_np,
                    label_hw=label_np,
                    mask_hw=mask_np,
                    aug_name=aug_name,
                )

                # Convert back to torch
                ms = torch.from_numpy(ms_np_aug).permute(2, 0, 1).float()
                label = torch.from_numpy(label_np_aug).unsqueeze(0).float()
                mask = torch.from_numpy(mask_np_aug).float()

        sample = {
            "tile_id": tid,
            "version": version,
            "ms": ms,
            "label": label,
            "mask": mask,
        }

        if rgb is not None:
            sample["rgb"] = rgb

        # Optional external transforms (e.g., tensor-level transforms)
        if self.transform:
            sample = self.transform(sample)

        return sample

