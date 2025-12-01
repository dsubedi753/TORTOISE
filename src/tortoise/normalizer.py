import json
import numpy as np
import rasterio
import torch
from pathlib import Path
from tqdm import tqdm

class DataNormalizer:
    """Normalizes tensors using min-max normalization parameters from a JSON file."""
    
    def __init__(self, json_path:Path, preloaded:bool=False):
        """
        Initialize normalizer with min-max stats from JSON file.
        
        Args:
            json_path: Path to JSON file containing 'min' and 'max' lists of size 13
        """
        
        self.json_path = json_path
        self.preloaded = preloaded
        
        if self.preloaded:
            self.load_stats()
        
  
    
    def __call__(self, tensor):
        """
        Normalize tensor of shape (13, H, W) using min-max normalization.
        
        Args:
            tensor: Tensor of shape (13, H, W)
            
        Returns:
            Normalized tensor with values in [0, 1]
        """
        # Reshape min/max to (13, 1, 1) for broadcasting
        min_vals = self.min_vals.view(13, 1, 1)
        range_vals = self.range.view(13, 1, 1)
        
        normalized = (tensor - min_vals) / range_vals
        return normalized
    
    def denormalize(self, tensor):
        """
        Reverse min-max normalization.
        
        Args:
            tensor: Normalized tensor of shape (13, H, W)
            
        Returns:
            Denormalized tensor
        """
        min_vals = self.min_vals.view(13, 1, 1)
        range_vals = self.range.view(13, 1, 1)
        
        return tensor * range_vals + min_vals
    
    def load_stats(self):
        
        with open(self.json_path, 'r') as f:
            stats = json.load(f)
        
        self.min_vals = torch.tensor(stats['mins'], dtype=torch.float32)
        self.max_vals = torch.tensor(stats['maxs'], dtype=torch.float32)
        self.range = self.max_vals - self.min_vals
    
    def compute_stats(self, tile_ids, tiles_dir:Path):
        """
        Compute min-max stats over given tile IDs and save to JSON.
        
        Args:
            tile_ids: List of tile IDs to compute stats on
            tiles_dir: Path to directory containing tile_ms_*.tif files


        Returns:
            dict containing per-channel means and count of valid pixels
        """

        ms_files = [tiles_dir / f"tile_ms_{tid}.tif" for tid in tile_ids]
        # Check all exist
        missing = [p for p in ms_files if not p.exists()]
        if missing:
            raise RuntimeError(f"Missing tile_ms files: {missing}")

        global_mins = None
        global_maxs = None

        for f in tqdm(ms_files, desc="Computing masked min/max"):
            with rasterio.open(f) as src:
                ms = src.read().astype(np.float64)               # (13, H, W)
                mask = src.dataset_mask().astype(np.float64) / 255.0  # (H, W)

            C, H, W = ms.shape

            # broadcast mask to (13, H, W)
            mask_3d = mask.reshape(1, H, W)

            # apply mask
            ms_masked = ms * mask_3d

            flat = ms_masked.reshape(C, -1)

            valid = flat > 0

            # masked min/max per channel
            tile_mins = np.where(valid, flat, np.inf).min(axis=1)
            tile_maxs = np.where(valid, flat, -np.inf).max(axis=1)

            if global_mins is None:
                global_mins = tile_mins.copy()
                global_maxs = tile_maxs.copy()
            else:
                global_mins = np.minimum(global_mins, tile_mins)
                global_maxs = np.maximum(global_maxs, tile_maxs)

        # Replace inf (if channel had no valid pixels)
        global_mins = np.where(np.isinf(global_mins), 0.0, global_mins)
        global_maxs = np.where(np.isinf(global_maxs), 0.0, global_maxs)

        stats = {
            "mins": global_mins.tolist(),
            "maxs": global_maxs.tolist(),
        }

        out_path = self.json_path
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=4)

        return stats