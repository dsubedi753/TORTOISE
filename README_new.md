# TORTOISE

Tile-based geospatial segmentation pipeline using PyTorch U-Net variants with Albumentations-driven augmentations. Includes data prep utilities for turning large multispectral/RGB scenes into train-ready tiles plus training/eval helpers.

## Repo Layout
- `src/tortoise/` – core code: datasets/dataloaders, augmentations, U-Net family (`U_Net`, `AttU_Net`, etc.), training loop, metrics/inference utilities.
- `configs/` – `config.yml` (tiling params), `hyperparams.yml` (model/optimizer/dataset settings).
- `scripts/` – data prep: `data_organize.py`, `tilify.py`, `tearify.py`.
- `notebooks/` – exploratory work (includes SAM2 finetuning notebooks).

## Setup
```bash
conda env create -f environment.yml
conda activate tortoise
# Point code to the repo root (needed by scripts/utils)
# Linux/macOS: export PROJECT_ROOT=$(pwd)
# Windows PS:  $env:PROJECT_ROOT = (Get-Location).Path
```

## Data Preparation
Assumes raw files under `data/raw/`:
- Multispectral: `data/raw/training_images_masked/`
- Labels: `data/raw/segmentations_masked/`
- RGB: `data/raw/training_images_RGBs/`

Run in order:
1) Organize raw scenes into `data/imageset/<image_id>/`:
```bash
python scripts/data_organize.py
```
2) Generate sliding-window metadata and `data/tile_index.csv`:
```bash
python scripts/tilify.py           # uses configs/config.yml (tile_size/stride/threshold)
```
3) Crop per-tile MS/RGB/label files into `data/tiles/`:
```bash
python scripts/tearify.py
```

## Training
Example (MS tiles):
```python
import torch
from pathlib import Path
from tortoise.hparams import load_hparams, build_model, build_optimizer
from tortoise.dataloader import build_dataloaders
from tortoise.train import train_model, get_device

hparams = load_hparams(Path("configs/hyperparams.yml"))
train_loader, val_loader, _, _ = build_dataloaders(
    tiles_dir="data/tiles",
    csv_file="data/tile_index.csv",
    batch_size=hparams["train"]["batch_size"],
    seed=hparams["dataset"]["seed"],
    train_ratio=hparams["dataset"]["train_ratio"],
    val_ratio=hparams["dataset"]["val_ratio"],
    test_ratio=hparams["dataset"]["test_ratio"],
    use_ms=hparams["dataset"]["use_ms"],
    use_rgb=hparams["dataset"]["use_rgb"],
    num_workers=hparams["train"]["num_workers"],
)

device = get_device()
model = build_model(hparams).to(device)
opt = build_optimizer(model, hparams)
model, *_ = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=opt,
    scheduler=None,
    device=device,
    pos_weight=torch.tensor([1.0], device=device),  # adjust for class imbalance
    num_epochs=hparams["train"]["epochs"],
    alpha=0.5,  # BCE/Dice mix
    checkpoint_path=Path("checkpoints/best.pth"),
    scaler=torch.cuda.amp.GradScaler() if hparams["train"]["amp"] else None,
)
```

## Evaluation / Inference
- Tile-wise evaluation: `tortoise.train.evaluate(...)`
- Whole-image fusion + metrics: `tortoise.utils.ensemble_image` and `evaluate_images`
- Visualization helpers: `tortoise.utils.to_display_rgb`

## Notes
- Adjust tiling in `configs/config.yml` (`tile_size`, `stride`, `validity_threshold`).
- Choose model/optimizer in `configs/hyperparams.yml` (`name: U_Net | R2U_Net | AttU_Net | R2AttU_Net`).
- SAM2 finetuning examples live in `notebooks/`.
