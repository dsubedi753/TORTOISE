import json
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Optional
from pathlib import Path
import torch.nn as nn
import rasterio


PROJECT_FOLDER = Path(os.getenv("PROJECT_ROOT"))
DATA_FOLDER = PROJECT_FOLDER / "data"
src_path = PROJECT_FOLDER / "src"
tiles_dir = DATA_FOLDER / "tiles"
imageset_dir = DATA_FOLDER / "imageset"


def to_display_rgb(tensors_list, channels=(0, 1, 2), rescale=False, aug_names=None, nrows=2, ncols=4):
    """
    Convert a list of image tensors to displayable RGB/grayscale arrays and plot in subplots grid.
    
    Args:
        tensors_list: List of torch.Tensor, each shape:
                      - (H, W) for grayscale
                      - (1, H, W) for single-channel
                      - (3, H, W) for RGB
                      - (13, H, W) for MS (multispectral)
        channels: 3-tuple of int, indices to select from C dimension
                  - For RGB (C=3): ignored, all 3 channels used
                  - For MS (C=13): e.g., (2, 1, 0) for RGB display
        rescale: bool, if True rescale to [0, 1] using 1st/99th percentile
        aug_names: optional list of strings, names for each subplot (e.g., augmentation names)
        nrows: int, number of rows in subplot grid (default 2)
        ncols: int, number of columns in subplot grid (default 4)
    
    Returns:
        fig, axes from plt.subplots with images displayed
    """
    n = len(tensors_list)
    
    nrows = (n + ncols - 1) // ncols  # Adjust rows if fewer images
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
    axes = axes.flatten()  # Flatten for easy iteration
    
    
    
    for idx, tensor in enumerate(tensors_list):
        # Convert to numpy
        if isinstance(tensor, torch.Tensor):
            img_np = tensor.cpu().numpy()
        else:
            img_np = tensor
        
        # Handle different input shapes
        if img_np.ndim == 2:
            # (H, W) grayscale
            selected = img_np
            cmap = 'gray'
        elif img_np.ndim == 3:
            C = img_np.shape[0]
            
            if C == 1:
                # (1, H, W) -> squeeze to (H, W)
                selected = img_np[0]
                cmap = 'gray'
            elif C == 3:
                # (3, H, W) -> (H, W, 3) RGB
                selected = img_np.transpose(1, 2, 0)
                cmap = None
            else:
                # (C, H, W) MS -> extract channels and convert to (H, W, 3)
                img_hwc = img_np.transpose(1, 2, 0)  # (H, W, C)
                selected = np.stack([img_hwc[:, :, ch] for ch in channels], axis=-1)
                cmap = None
        else:
            raise ValueError(f"Unsupported tensor shape: {img_np.shape}")
        
        # Rescale for display
        if rescale:
            vmin = np.nanpercentile(selected, 1)
            vmax = np.nanpercentile(selected, 99)
            if vmax > vmin:
                selected = (selected - vmin) / (vmax - vmin)
            selected = np.clip(selected, 0.0, 1.0)
        
        axes[idx].imshow(selected, cmap=cmap)
        axes[idx].axis("off")
        
        # Add subplot title if aug_names provided
        if aug_names is not None and idx < len(aug_names):
            axes[idx].set_title(aug_names[idx], fontsize=10, pad=5)
    
    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    return fig, axes




def ensemble_inference(model:nn.Module, image_id: str, use_rgb = False, threshold = 0.5):
    """Get the tears from the image_id

    Args:
        image_id (str): _description_
        use_rgb (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    image_id_dir = imageset_dir / image_id
    
    meta_file = image_id_dir / "meta.json"
    
    if use_rgb:
        image = image_id_dir / "rgb.png"
    else:
        image = image_id_dir / "ms.tif"
 
    with meta_file.open("r") as f:
        meta = json.load(f)
        
    height = int(meta["height"])
    width = int(meta["width"])
    tile_size = int(meta["tile_size"])


    tiles = meta.get("valid_tiles", [])
    
    bsz = len(tiles)
    channels = 3 if use_rgb else 13

    batch = torch.zeros((bsz, channels, tile_size, tile_size), dtype=torch.float32)
    batch_masks = torch.zeros((bsz, 1, tile_size, tile_size), dtype=torch.float32)
    positions: list[tuple[int, int]] = []

    for idx, tile in enumerate(tiles):
        tid = int(tile["tile_id"])
        positions.append((int(tile["h0"]), int(tile["w0"])))

        tile_path = tiles_dir / (f"tile_rgb_{tid:05d}.png" if use_rgb else f"tile_ms_{tid:05d}.tif")
        mask_path = tiles_dir / f"tile_label_{tid:05d}.tif"

        with rasterio.open(tile_path) as src:
            arr = src.read().astype(np.float32)
        arr /= 255.0 if use_rgb else 10000.0

        with rasterio.open(mask_path) as src:
            mask_arr = src.dataset_mask().astype(np.float32) / 255.0

        mask_tensor = torch.from_numpy(mask_arr)
        batch_masks[idx, 0] = mask_tensor
        batch[idx] = torch.from_numpy(arr) * mask_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(batch.to(device))
        probs = torch.sigmoid(logits) * batch_masks.to(device)

    probs = probs.squeeze(1).cpu()
    tile_masks = batch_masks.squeeze(1)

    pred_sum = torch.zeros((height, width), dtype=torch.float32)
    weight_sum = torch.zeros((height, width), dtype=torch.float32)

    for idx, (h0, w0) in enumerate(positions):
        h1, w1 = h0 + tile_size, w0 + tile_size
        pred_sum[h0:h1, w0:w1] += probs[idx] * tile_masks[idx]
        weight_sum[h0:h1, w0:w1] += tile_masks[idx]

    pred_full = pred_sum / (weight_sum + 1e-6)

    with rasterio.open(image_id_dir / "label.tif") as src:
        label_full = torch.from_numpy(src.read(1).astype(np.float32)) / 65535.0
        label_mask = torch.from_numpy(src.dataset_mask().astype(np.float32)) / 255.0

    pred_full = pred_full * label_mask
    label_full = label_full * label_mask

    pred_binary = (pred_full > threshold).float()

    inter = (pred_binary * label_full).sum()
    union = pred_binary.sum() + label_full.sum() - inter
    iou = (inter / (union + 1e-6)).item()

    return pred_full, image, iou
