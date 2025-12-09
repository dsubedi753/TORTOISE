import json
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Optional
from pathlib import Path
import rasterio
import torch.nn as nn

import torch.nn.functional as F


PROJECT_FOLDER = Path(os.getenv("PROJECT_ROOT"))
DATA_FOLDER = PROJECT_FOLDER / "data"
src_path = PROJECT_FOLDER / "src"
tiles_dir = DATA_FOLDER / "tiles"
imageset_dir = DATA_FOLDER / "imageset"


def _binary_boundary(mask: torch.Tensor, dilation_pixel=2):
    """
    Extract thin binary boundary from a binary mask using Laplacian.
    mask: (H, W) binary {0,1}
    """
    dilation = dilation_pixel

    lap = torch.tensor([[0, 1, 0],
                        [1,-4, 1],
                        [0, 1, 0]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)

    # Find boundary pixels
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)
    edges = F.conv2d(mask_f, lap, padding=1).abs()
    edges = (edges > 0).float().squeeze()

    # Dilate boundary
    if dilation > 1:
        kernel = torch.ones((1,1,dilation,dilation), device=mask.device)
        edges = edges.unsqueeze(0).unsqueeze(0)
        dil = F.conv2d(edges, kernel, padding=dilation//2)
        dil = (dil > 0).float().squeeze()
        return dil
    else:
        return edges


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




def ensemble_image(model, image_id, use_rgb=False, use_window_weight=False, output_list = ["pred", "image"]):
    """
    Tile-based inference with logit fusion.

    Returns:
        {
            "labels": (H,W),
            "pred": (H,W), optional
            "logits": (H,W),
            "mask": (H,W),
            "image": (C,H,W), optional
        }
    """
    
    output = {}

    #  Paths 
    project_root = Path(os.getenv("PROJECT_ROOT"))
    imageset_dir = project_root / "data" / "imageset" / image_id
    tiles_dir = project_root / "data" / "tiles"

    #  Load metadata 
    with open(imageset_dir / "meta.json", "r") as f:
        meta = json.load(f)

    H = int(meta["height"])
    W = int(meta["width"])
    tile_size = int(meta["tile_size"])
    tiles = meta.get("valid_tiles", [])

    C = 3 if use_rgb else 13

    #  Load full image 
    if use_rgb:
        with rasterio.open(imageset_dir / "rgb.png") as src:
            img = src.read().astype(np.float32) / 255.0
    else:
        with rasterio.open(imageset_dir / "ms.tif") as src:
            img = src.read().astype(np.float32) / 10000.0

    if "image" in output_list:
        image_full = torch.from_numpy(img)  # (C,H,W)
        output["image"] = image_full


    with rasterio.open(imageset_dir / "label.tif") as src:
        full_label = torch.from_numpy(src.read(1).astype(np.float32)) / 65535.0
        full_mask = torch.from_numpy(src.dataset_mask().astype(np.float32)) / 255.0

    #  Prepare logits tensors 
    device = next(model.parameters()).device
    logit_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((H, W), dtype=torch.float32, device=device)

    #  Optional window weights 
    if use_window_weight:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, tile_size),
            torch.linspace(-1, 1, tile_size),
            indexing="ij"
        )
        dist = torch.sqrt(xx**2 + yy**2)
        window = torch.cos((dist.clamp(0,1)) * np.pi/2)
        window = window.to(device)
    else:
        window = torch.ones((tile_size, tile_size), device=device)


    model.eval()
    with torch.no_grad():
        for tile_info in tiles:
            tid = int(tile_info["tile_id"])
            h0 = int(tile_info["h0"])
            w0 = int(tile_info["w0"])

            tile_path = (
                tiles_dir / f"tile_rgb_{tid:05d}.png"
                if use_rgb else
                tiles_dir / f"tile_ms_{tid:05d}.tif"
            )
            mask_path = tiles_dir / f"tile_label_{tid:05d}.tif"


            with rasterio.open(tile_path) as src:
                arr = src.read().astype(np.float32)
            arr = arr / (255.0 if use_rgb else 10000.0)
            tile_tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,C,h,w)

            with rasterio.open(mask_path) as src:
                tile_valid = torch.from_numpy(src.dataset_mask().astype(np.float32)) / 255.0
            tile_valid = tile_valid.to(device)

            logits = model(tile_tensor).squeeze(0).squeeze(0)  # (h,w)

            tile_weight = tile_valid * window  # (h,w)


            h1, w1 = h0 + tile_size, w0 + tile_size
            logit_sum[h0:h1, w0:w1] += logits * tile_weight
            weight_sum[h0:h1, w0:w1] += tile_weight

    fused_logits = logit_sum / (weight_sum + 1e-6)
    masked_logits = fused_logits * full_mask.to(device)
    
    
    
    if "pred" in output_list: 
        masked_pred = torch.sigmoid(masked_logits)
        output["pred"] = masked_pred.cpu()
        
        
    output["labels"] = (full_label * full_mask).cpu()
    output["logits"] = masked_logits.cpu()    
    output["mask"] = full_mask.cpu()


    #  Prepare final outputs on CPU 
    return output

def evaluate_metrics(
    logits,
    labels,
    mask,
    metrics=["iou", "dice", "brier"],
    threshold=0.5,
):
    out = {}
    logits = logits.float()
    labels = labels.float()
    mask = mask.float()

    valid = (mask > 0)

    probs = torch.sigmoid(logits)
    probs_valid = probs[valid]
    labels_valid = labels[valid]

    if any(m in metrics for m in ["iou","dice","precision","recall","fpr","fnr"]):
        pred_bin = (probs > threshold)
        label_bin = (labels > 0.5)

        pred_bin = pred_bin & valid
        label_bin = label_bin & valid

        TP = (pred_bin & label_bin).sum().float()
        FP = (pred_bin & ~label_bin).sum().float()
        FN = (~pred_bin & label_bin).sum().float()
        TN = (~pred_bin & ~label_bin & valid).sum().float()

        if "iou" in metrics:
            out["iou"] = (TP / (TP + FP + FN + 1e-6)).item()

        if "dice" in metrics:
            out["dice"] = (2 * TP / (2 * TP + FP + FN + 1e-6)).item()

        if "precision" in metrics:
            out["precision"] = (TP / (TP + FP + 1e-6)).item()

        if "recall" in metrics:
            out["recall"] = (TP / (TP + FN + 1e-6)).item()

        if "fpr" in metrics:
            out["fpr"] = (FP / (FP + TN + 1e-6)).item()

        if "fnr" in metrics:
            out["fnr"] = (FN / (FN + TP + 1e-6)).item()

    if "brier" in metrics:
        out["brier"] = ((probs_valid - labels_valid)**2).mean().item()

    if "pixelwise_entropy" in metrics:
        p = probs.clamp(1e-6, 1 - 1e-6)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        out["pixelwise_entropy"] = entropy * mask
        
    if "boundary_iou" in metrics:
        pred_bin = (probs > threshold) & valid
        label_bin = (labels > 0.5) & valid

        # boundary extraction
        b_pred = _binary_boundary(pred_bin.float())
        b_label = _binary_boundary(label_bin.float())

        # intersection and union
        inter = (b_pred * b_label).sum().float()
        union = (b_pred + b_label).clamp(max=1).sum().float()

        out["boundary_iou"] = (inter / (union + 1e-6)).item()

    return out

import pandas as pd

def evaluate_images(
    model,
    image_ids,
    metrics=["iou", "dice", "brier"],
    threshold=0.5,
    use_rgb=False,
    use_window_weight=False,
    output_list=[]
):
    """
    Evaluate model over a list of image_ids.

    Returns:
        {
            "metrics": DataFrame with index=image_id and columns=metrics,
            you can also request "pred,"image",logits","labels","mask" in output_list
        }
    """
    # storage
    metric_rows = []
    preds_out = []
    imgs_out = []
    logits_out = []
    labels_out = []
    masks_out = []

    # loop through each image
    for mid in image_ids:
        
        
        # Run ensemble image (get logits/labels/mask always for metric computation)
        required = set(output_list) | {"logits", "labels", "mask"}
        out_inf = ensemble_image(
            model,
            mid,
            use_rgb=use_rgb,
            use_window_weight=use_window_weight,
            output_list=list(required) 
        )

        logits = out_inf["logits"]
        labels = out_inf["labels"]
        mask   = out_inf["mask"]

        # --- compute metrics ---
        m = evaluate_metrics(
            logits=logits,
            labels=labels,
            mask=mask,
            metrics=metrics,
            threshold=threshold
        )

        # append metric row (scalar values)
        metric_rows.append(m)

        # optional outputs
        if "pred" in output_list:
            preds_out.append(out_inf["pred"])

        if "image" in output_list:
            imgs_out.append(out_inf["image"])

        if "logits" in output_list:
            logits_out.append(logits)

        if "labels" in output_list:
            labels_out.append(labels)

        if "mask" in output_list:
            masks_out.append(mask)

    df = pd.DataFrame(metric_rows, index=image_ids)
    
    out_dict = {"metrics": df}

    if "pred" in output_list:
        out_dict["pred"] = preds_out
    if "image" in output_list:
        out_dict["image"] = imgs_out
    if "logits" in output_list:
        out_dict["logits"] = logits_out
    if "labels" in output_list:
        out_dict["labels"] = labels_out
    if "mask" in output_list:
        out_dict["mask"] = masks_out

    return out_dict




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