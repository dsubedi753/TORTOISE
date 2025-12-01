# train.py
"""
Utility training functions for Jupyter Notebook:
    - masked_bce_loss
    - train_one_epoch
    - evaluate

This file intentionally contains NO global training loop
and performs NO work on import.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------
# 1. Loss (BCE with mask)
# ---------------------------------------------------------
_bce = nn.BCEWithLogitsLoss(reduction="none")


def masked_bce_loss(logits, target, mask):
    """
    Compute BCE loss only over valid pixels.

    Args:
        logits: (B,1,H,W) raw model output
        target: (B,1,H,W) ground truth mask (0..1)
        mask:   (B,H,W)    valid mask (0/1)

    Returns:
        Scalar masked BCE loss.
    """
    # (B,1,H,W)
    mask = mask.unsqueeze(1)

    # (B,1,H,W)
    loss_map = _bce(logits, target)

    # average over valid pixels only
    return (loss_map * mask).sum() / (mask.sum() + 1e-8)


# ---------------------------------------------------------
# 2. One training epoch
# ---------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device="cuda"):
    """
    Train model for one epoch.

    Args:
        model: U-Net instance
        loader: training DataLoader
        optimizer: torch optimizer
        device: "cuda" or "cpu"

    Returns:
        Average loss over epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        ms    = batch["ms"].to(device)
        label = batch["label"].to(device)
        mask  = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(ms)

        loss = masked_bce_loss(logits, label, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------
# 3. Evaluation loop
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, desc="Val", device="cuda"):
    """
    Evaluate model on validation or test set.

    Args:
        model: U-Net instance
        loader: DataLoader
        desc: tqdm label
        device: "cuda" or "cpu"

    Returns:
        Average loss over dataset.
    """
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc=desc, leave=False):
        ms    = batch["ms"].to(device)
        label = batch["label"].to(device)
        mask  = batch["mask"].to(device)

        logits = model(ms)
        loss = masked_bce_loss(logits, label, mask)

        total_loss += loss.item()

    return total_loss / len(loader)
