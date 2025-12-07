from nbformat import versions
import torch
from torch.utils.data import DataLoader
from tortoise.dataset import TileDataset
from tortoise.augmentations import AUG_KEYS
from pathlib import Path
import re
import numpy as np
import csv

"""
Parameters:
    tile_id_map: a map with the image id as key and a list of corresponding tile_ids as values

Returns:
    train_ids: list(str) - a list of tile_ids for the training set 
    val_ids: list(str)
    test_ids: list(str)
    (train_image_ids_per_tile, val_image_ids_per_tile, test_image_ids_per_tile): tuple[list[str]] - the images associated with each tile 
"""
def split_tile_ids(
    tile_id_map: dict[list[int]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float | None = None
):
    rng = np.random.RandomState(seed)
    image_ids = list(tile_id_map.keys())
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_image_ids = image_ids[:n_train]
    val_image_ids   = image_ids[n_train:n_train+n_val]
    
    if test_ratio is None:
        test_image_ids  = image_ids[n_train+n_val:]
    else:
        n_test =  int(n * test_ratio)
        test_image_ids  = image_ids[n_train+n_val:n_train+n_val+n_test]

    # Now extract tile_ids from map using image_ids
    train_tile_ids = []
    val_tile_ids = []
    test_tile_ids = []

    train_image_ids_per_tile = []
    val_image_ids_per_tile = []
    test_image_ids_per_tile = []

    for train_image in train_image_ids:
        train_tile_ids.extend(tile_id_map[train_image])
        train_image_ids_per_tile.append(train_image)
    for val_image in val_image_ids:
        val_tile_ids.extend(tile_id_map[val_image])
        val_image_ids_per_tile.append(val_image)
    for test_image in test_image_ids:
        test_tile_ids.extend(tile_id_map[test_image])
        test_image_ids_per_tile.append(test_image)
    
    return train_tile_ids, val_tile_ids, test_tile_ids, (train_image_ids_per_tile, val_image_ids_per_tile, test_image_ids_per_tile)


# Tile ID by image
# Returns a map of image_ids -> list of tile_ids
def list_tile_ids(tiles_dir, csv_file):
    ids = []
    for f in Path(tiles_dir).glob("tile_ms_*.tif"):
        m = re.match(r"tile_ms_(\d{5})\.tif", f.name)
        if m:
            ids.append(int(m.group(1))) 
    all_ids = sorted(ids)

    # Make a map of which tiles have which image ids. 
    ids_image_map = {}
    with open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        id_map = {}
        for row in reader:
            tile_id = int(row["tile_id"])
            image_id = row["image_id"]
            id_map[tile_id] = image_id
    
        # Populate ids_image_map with all tile ids.
        for tile_id in all_ids:
            image_id = id_map[tile_id]

            tile_id_str = f"{tile_id:05d}"
            if image_id not in ids_image_map:  # Construct array tile_ids
                ids_image_map[image_id] = [tile_id_str]
            else:
                ids_image_map[image_id].append(tile_id_str)     

    return ids_image_map



# Dataloader builder
def build_dataloaders(
    tiles_dir,
    csv_file: str | Path,
    batch_size: int,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float | None = None,
    use_rgb: bool = False,
    use_ms: bool = True,
    num_workers: int = 4,
):
    """
    High-level helper to build train/val/test dataloaders with pre-sampled
    augmentations.

    csv_file: the location of tiles_index.csv

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
    tile_ids_map = list_tile_ids(tiles_dir, Path(csv_file))
    


    # 2. split ids into train/val/test
    train_ids, val_ids, test_ids, image_map_tuple = split_tile_ids(tile_ids_map,
                                                    seed=seed,
                                                    train_ratio=train_ratio,
                                                    val_ratio=val_ratio,
                                                    test_ratio=test_ratio,
                                                )
    
    
    def sample_augmentations(ids, aug_keys=None, seed: int = 42):
        rng = np.random.RandomState(seed)
        out = []
        for tid in ids:
            if aug_keys is not None:
                augs = rng.choice(aug_keys, size=2, replace=True)
            else:
                augs = []
            for aug in ["orig"] + list(augs):
                out.append((tid, aug))
        return out
    
    # 3. Expand ids to samples with all versions
    train_samples = sample_augmentations(train_ids, AUG_KEYS, seed=seed)
    val_samples   = sample_augmentations(val_ids, seed=seed)
    test_samples  = sample_augmentations(test_ids, seed=seed)

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

    return train_loader, val_loader, test_loader, image_map_tuple
