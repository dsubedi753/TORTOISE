from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from tortoise.dataset import TileDataset
from tortoise.augmentations import apply_augmentation, sample_aug_map, save_aug_map, AugMap  # new
from pathlib import Path
import re
import numpy as np

# -------------------------------------------------------------------
# Split helpers
# -------------------------------------------------------------------

def split_samples(
    samples: Sequence[Tuple[str, str]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    Split a list of (tile_id, version) tuples into train/val/test.
    This is the core splitter for the augmentation-aware pipeline.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # Convert back to (tid, version)
    samples = list(samples)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    return train_samples, val_samples, test_samples


# -------------------------------------------------------------------
# Tile ID
# -------------------------------------------------------------------


def list_tile_ids(tiles_root):
    ids = []
    for f in Path(tiles_root).glob("tile_ms_*.tif"):
        m = re.match(r"tile_ms_(.+)\.tif", f.name)
        if m:
            ids.append(m.group(1)) 
    return sorted(ids)




# -------------------------------------------------------------------
# Dataloader builder
# -------------------------------------------------------------------


def build_dataloaders(
    data_root,
    batch_size: int,
    normalizer,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    use_rgb: bool = False,
    num_workers: int = 0,
    save_aug_map_path: str | Path | None = None,
):
    """
    High-level helper to build train/val/test dataloaders with pre-sampled
    augmentations.

    Steps:
        1. List tile_ids from data_root.
        2. Build full sample list: (tid, "orig"), (tid, "aug1"), (tid, "aug2").
        3. Pre-sample AUG_MAP for all (tid, "aug1"/"aug2").
        4. Split the sample list into train/val/test.
        5. Create TileDataset instances with the sample lists and AUG_MAP.

    Note:
        - Total number of samples across all splits ~= N_tiles * 3.
        - Each (tile_id, version) is split independently; a given tile_id may
          have 0, 1, 2, or 3 versions in any particular split.
    """
    data_root = Path(data_root)

    # 1. discover tile_ids
    tile_ids = list_tile_ids(data_root)

    # 2. build full sample list (orig/aug1/aug2)
    versions = ["orig", "aug1", "aug2"]
    all_samples: List[Tuple[str, str]] = [
        (tid, ver) for tid in tile_ids for ver in versions
    ]

    # 3. pre-sample augmentations for aug1/aug2
    aug_map: AugMap = sample_aug_map(
        tile_ids=tile_ids,
        versions=("aug1", "aug2"),
        seed=seed,
    )

    # optionally save aug_map for later inspection / reproducibility
    if save_aug_map_path is not None:
        save_aug_map(aug_map, save_aug_map_path)

    # 4. split samples into train/val/test
    train_samples, val_samples, test_samples = split_samples(
        all_samples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # 5. build datasets
    train_ds = TileDataset(
        root=data_root,
        samples=train_samples,
        use_rgb=use_rgb,
        normalizer=normalizer,
        aug_map=aug_map,
    )

    val_ds = TileDataset(
        root=data_root,
        samples=val_samples,
        use_rgb=use_rgb,
        normalizer=normalizer,
        aug_map=aug_map,
    )

    test_ds = TileDataset(
        root=data_root,
        samples=test_samples,
        use_rgb=use_rgb,
        normalizer=normalizer,
        aug_map=aug_map,
    )

    # 6. build dataloaders
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader
