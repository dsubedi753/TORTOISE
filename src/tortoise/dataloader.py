from nbformat import versions
import torch
from torch.utils.data import DataLoader
from tortoise.dataset import TileDataset
from tortoise.augmentations import AUG_KEYS
from pathlib import Path
import re
import numpy as np

# Split helpers
def split_tile_ids(
    tile_ids: list[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float | None = None
):
    rng = np.random.RandomState(seed)
    tile_ids = np.array(tile_ids)
    rng.shuffle(tile_ids)

    n = len(tile_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = tile_ids[:n_train].tolist()
    val_ids   = tile_ids[n_train:n_train+n_val].tolist()
    
    if test_ratio is None:
        test_ids  = tile_ids[n_train+n_val:].tolist()
    else:
        n_test =  int(n * test_ratio)
        test_ids  = tile_ids[n_train+n_val:n_train+n_val+n_test].tolist()

    return train_ids, val_ids, test_ids


# Tile ID
def list_tile_ids(tiles_dir):
    ids = []
    for f in Path(tiles_dir).glob("tile_ms_*.tif"):
        m = re.match(r"tile_ms_(.+)\.tif", f.name)
        if m:
            ids.append(m.group(1)) 
    return sorted(ids)



# Dataloader builder
def build_dataloaders(
    tiles_dir,
    batch_size: int,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float | None = None,
    use_rgb: bool = False,
    use_ms: bool = True,
    num_workers: int = 0,
):
    """
    High-level helper to build train/val/test dataloaders with pre-sampled
    augmentations.

    Steps:
        1. List tile_ids from tiles_dir.
        2. Build full sample list: (tid, "orig"), (tid, "aug1"), (tid, "aug2").
        3. Pre-sample AUG_MAP for all (tid, "aug1"/"aug2").
        4. Split the tile_ids into train/val/test.
        5. Create TileDataset instances with the sample lists and AUG_MAP.

    Note:
        - Total number of samples across all splits ~= N_tiles * 3.
        - Each (tile_id, version) is split independently; a given tile_id may
          have 0, 1, 2, or 3 versions in any particular split.
    """
    
    if use_rgb == use_ms:
        raise ValueError("One and only one of use_rgb or use_ms may be True.")

    tiles_dir = Path(tiles_dir)

    # 1. discover tile_ids
    tile_ids = list_tile_ids(tiles_dir)
    

    # 2. split ids into train/val/test
    train_ids, val_ids, test_ids = split_tile_ids(tile_ids,
                                                    seed=seed,
                                                    train_ratio=train_ratio,
                                                    val_ratio=val_ratio,
                                                    test_ratio=test_ratio,
                                                )
    
    
    def sample_augmentations(ids, aug_keys, seed: int = 42):
        rng = np.random.RandomState(seed)
        out = []
        for tid in ids:
            augs = rng.choice(aug_keys, size=2, replace=True)
            for aug in [None] + list(augs):
                out.append((tid, aug))
        return out
    
    # 3. Expand ids to samples with all versions
    train_samples = sample_augmentations(train_ids, AUG_KEYS, seed=seed)
    val_samples   = sample_augmentations(val_ids, AUG_KEYS, seed=seed)
    test_samples  = sample_augmentations(test_ids, AUG_KEYS, seed=seed)

    # 5. build datasets
    train_ds = TileDataset(
        tiles_dir=tiles_dir,
        tile_ids=train_samples,
        use_ms=use_ms,
        use_rgb=use_rgb,

    )

    val_ds = TileDataset(
        tiles_dir=tiles_dir,
        tile_ids=val_samples,
        use_ms=use_ms,
        use_rgb=use_rgb,
    )

    test_ds = TileDataset(
        tiles_dir=tiles_dir,
        tile_ids=test_samples,
        use_ms=use_ms,
        use_rgb=use_rgb,
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
