from matplotlib import pyplot as plt
import numpy as np
import torch


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
