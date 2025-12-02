# dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from tortoise.augmentations import apply_augmentation # new



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
        tiles_dir,
        tile_ids: Sequence[Union[str, Tuple[str, str]]],
        use_ms: bool = True,
        use_rgb: bool = False,    
    ):
        self.tiles_dir = Path(tiles_dir)
        self.use_ms = use_ms
        self.use_rgb = use_rgb


        
        # Normalize tile_ids → List[(tid, version)]
        samples: List[Tuple[str, Optional[str]]] = []
        for item in tile_ids:
            if isinstance(item, tuple):
                tid, version = item
            else:
                tid, version = item, None
            samples.append((tid, version))
        self.samples = samples
        
        
        # Unique tile IDs
        self.unique_ids = sorted({tid for (tid, _) in self.samples})

        
        # Build paths
        if use_ms: self.ms_paths    = {tid: self.tiles_dir / f"tile_ms_{tid}.tif"    for tid in self.unique_ids}
        self.label_paths = {tid: self.tiles_dir / f"tile_label_{tid}.tif" for tid in self.unique_ids}
        if use_rgb: self.rgb_paths   = {tid: self.tiles_dir / f"tile_rgb_{tid}.png"   for tid in self.unique_ids}
        
        
        
        # Validate
        for tid in self.unique_ids:
            if self.use_ms:
                if not self.ms_paths[tid].exists():
                    raise FileNotFoundError(f"Missing MS tile: {self.ms_paths[tid]}")
            if not self.label_paths[tid].exists():
                raise FileNotFoundError(f"Missing label tile: {self.label_paths[tid]}")
            if self.use_rgb:
                if not self.rgb_paths[tid].exists():
                    raise FileNotFoundError(f"Missing RGB tile: {self.rgb_paths[tid]}")

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

     
    # Main getitem
    def __getitem__(self, index):
        tid, version = self.samples[index]


        if self.use_ms:                                                     # Load MS if requested
            image = self._read_raster(self.ms_paths[tid]).float()/10000.0   # (13,H,W)
        elif self.use_rgb:                                                  # Load RGB if requested
            image = self._read_raster(self.rgb_paths[tid]).float()          # (3,H,W)
        else:
            raise RuntimeError("Neither MS nor RGB is enabled in dataset.")
        
        # --- valid mask
        mask = self._read_mask(self.label_paths[tid]).float() / 255.0   # (H,W)

        # --- label
        label = self._read_raster(self.label_paths[tid]).float() / 65535.0  # (H,W)
        if label.ndim == 2:
            label = label.unsqueeze(0)  # (1,H,W)

        
        if version is not None:
            # Prepare numpy versions (convert CHW → HWC)
            image_np = image.permute(1, 2, 0).cpu().numpy()   # (H,W,C)
            label_np = label[0].cpu().numpy() if label is not None else None
            mask_np  = mask.cpu().numpy() if mask is not None else None

            
            # Apply augmentation
            image_np_aug, label_np_aug, mask_np_aug = apply_augmentation(
                image_hwc=image_np,
                label_hw=label_np,
                mask_hw=mask_np,
                aug_name=version,
            )

            
            # Convert back to PyTorch tensors
            image = torch.from_numpy(image_np_aug).permute(2, 0, 1).float()

            if label_np_aug is not None:
                label = torch.from_numpy(label_np_aug).unsqueeze(0).float()

            if mask_np_aug is not None:
                mask  = torch.from_numpy(mask_np_aug).float()

        
        # Store back into sample (MS or RGB)
        sample = {
            "tile_id": tid,
            "version": version,
            "label": label,
            "mask": mask,
        }

        # Assign to appropriate key
        if self.use_ms:
            sample["ms"] = image

        if self.use_rgb:
            sample["rgb"] = image
        
        return sample