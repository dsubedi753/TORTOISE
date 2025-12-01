import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio

class TileDataset(Dataset):
    """
    Loads tiles from: data/tiles/
        tile_ms_XXX.tif      (13 bands)
        tile_rgb_XXX.png     (3 band RGB)  -- optional usage
        tile_label_XXX.tif   (mask)

    No CSV is required.
    """

    def __init__(self, root, use_rgb=False, transform=None, normalize_ms=False, normalizer =None):
        self.root = Path(root)
        self.use_rgb = use_rgb
        self.transform = transform
        self.normalizer = normalizer
        self.normalize_ms = normalize_ms

        # -------------------------------------------------
        # Discover all tile IDs by scanning tile_ms_*.tif
        # -------------------------------------------------
        self.ms_paths = sorted(self.root.glob("tile_ms_*.tif"))
        if len(self.ms_paths) == 0:
            raise RuntimeError(f"No tile_ms_*.tif files found in {self.root}")

        # Extract tile IDs like "000" from filenames
        self.tile_ids = [p.stem.replace("tile_ms_", "") for p in self.ms_paths]

        # Prebuild file paths for speed
        self.rgb_paths   = {tid: self.root / f"tile_rgb_{tid}.png" for tid in self.tile_ids}
        self.label_paths = {tid: self.root / f"tile_label_{tid}.tif" for tid in self.tile_ids}

    def __len__(self):
        return len(self.tile_ids)

    def _read_raster(self, path):
        with rasterio.open(path) as src:
            arr = src.read()        # (C, H, W)
            return torch.from_numpy(arr).float()
        
    def _read_mask(self, path):
        with rasterio.open(path) as src:
            arr = src.dataset_mask()      # (H, W)
            return torch.from_numpy(arr).float()

    def __getitem__(self, index):
        tid = self.tile_ids[index]

        ms   = self._read_raster(self.root / f"tile_ms_{tid}.tif").float()        # (13,H,W)
        mask = self._read_mask(self.root / f"tile_ms_{tid}.tif").float()/255    # (1,H,W) or (H,W)
        label = self._read_raster(self.root / f"tile_label_{tid}.tif").float()/65535    # (1,H,W) or (H,W)
        
        if self.normalize_ms:
            ms = self.normalizer(ms)

        # If use_rgb=True (optional)
        rgb = None
        if self.use_rgb:
            rgb = self._read_raster(self.root / f"tile_rgb_{tid}.png")    # (3,H,W)

        sample = {
            "tile_id": tid,
            "ms": ms,
            "label": label,
            "mask": mask,
        }

        if rgb is not None:
            sample["rgb"] = rgb

        # Optional transforms
        if self.transform:
            sample = self.transform(sample)

        return sample
