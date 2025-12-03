# tortoise/hparams.py
"""
Hyperparameter loading + model/optimizer builders for TORTOISE.

This file centralizes:
    - YAML hyperparameter loading
    - Model construction (U-Net variants)
    - Optimizer construction

Usage in notebooks:
    from tortoise.hparams import load_hparams, build_model, build_optimizer

    hparams = load_hparams()
    model = build_model(hparams).to(device)
    optimizer = build_optimizer(model, hparams)
"""

import yaml
import os
from pathlib import Path
import torch

from tortoise.model import (
    U_Net, R2U_Net, AttU_Net, R2AttU_Net
)



# Load hyperparameters from configs/hparams.yml
def load_hparams(path=None):
    """
    Load hyperparameters from PROJECT_ROOT/configs/hparams.yml
    
    Args:
        path (Path or str): optional override
        
    Returns:
        dict: loaded hyperparameters
    """
    if path is None:
        root = Path(os.getenv("PROJECT_ROOT"))
        path = root / "configs" / "hyperparams.yml"

    if not Path(path).exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


# Build model from hparams
def build_model(hparams):
    """
    Construct a U-Net model based on hparams['model'] spec.
    """
    cfg = hparams["model"]
    name = cfg["name"]
    ch_in = cfg["in_channels"]
    ch_out = cfg["out_channels"]
    init_type = cfg.get("init_type", None)

    # Choose model
    if name == "U_Net":
        model = U_Net(img_ch=ch_in, output_ch=ch_out)

    elif name == "R2U_Net":
        model = R2U_Net(img_ch=ch_in, output_ch=ch_out)

    elif name == "AttU_Net":
        model = AttU_Net(img_ch=ch_in, output_ch=ch_out)

    elif name == "R2AttU_Net":
        model = R2AttU_Net(img_ch=ch_in, output_ch=ch_out)

    else:
        raise ValueError(f"Unknown model name: {name}")

    # Optional: initialize weights
    # (Your model file already has init_weights)
    
    if init_type is not None:
        from tortoise.model import init_weights
        init_weights(model, init_type)

    return model


# Build optimizer from hparams
def build_optimizer(model, hparams):
    """
    Constructs the optimizer from hparams['optimizer'] and hparams['train'].
    """
    opt_cfg = hparams["optimizer"]
    train_cfg = hparams["train"]

    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    opt_type = opt_cfg.get("type", "adam")

    if opt_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=opt_cfg.get("betas", (0.9, 0.999)),
        )

    elif opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=opt_cfg.get("betas", (0.9, 0.999)),
        )

    elif opt_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=opt_cfg.get("momentum", 0.9),
        )

    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
