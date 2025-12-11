# Tortoise Dataset

This project detects small scale gold mining in satellite images of Amazon basin. It uses Sentinel-2 multispectral data that is tiled, normalized, and labeled through a full preprocessing pipeline. Models include Attention U-Net and SAM2.

## Repo Layout
- `src/tortoise/` – core code: datasets/dataloaders, augmentations, U-Net family (`U_Net`, `AttU_Net`, etc.), training loop, metrics/inference utilities.
- `configs/` – `config.yml` (tiling params), `hyperparams.yml` (model/optimizer/dataset settings).
- `scripts/` – data prep: `data_organize.py`, `tilify.py`, `tearify.py`.
- `notebooks/` – SAM2 finetuning notebooks and miscellaneous exploratory work

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

You must run these three scripts in order before using the dataset:

1. **`data_organize`** – Organizes raw data files into the required directory structure
```bash
python scripts/data_organize.py
```
2. **`tilify`** – Processes the organized data and generates `tile_index.csv` and `meta.json`
```bash
python scripts/tilify.py
```
3. **`tearify`** - Uses `tile_index.csv` to extract the tiles and store in file system
```bash
python scripts/tearify.py
```

**The dataset will not load if these preprocessing steps are skipped.** The `tile_index.csv` file is required for the dataset to function.

## After Preprocessing

Once preprocessing is complete, you can load the dataset: using `TileDataSet` and `DataLoader` . This processes is demonstrated in `notebooks/example_dataloader.ipynb`. 

## U-Net

U-Net models are adapted from [attention_unet by sfczekalski](https://github.com/sfczekalski/attention_unet). There are two major changes.
- Parameterization of base channel width, depth, and growth factor (scaling factor that scales up number of channel)
- Randomized spatial dropout inside each convolutional block

Training example for U-Net is shown in `notebooks/example_U-Net_training.ipynb`

# TORTOISE

## SAM 2 Notebooks

Code for training a finetuned SAM2 model is in the notebooks/ folder. 
* SAM2FinetuneNew finetunes a SAM2 model.
* SAM2FT_Validate calculates performance metrics based off of generated finetuned models using a tiling approach.
* DL_Zero_Shot calculates performance metrics based off of generated finetuned models using full images, without tiling.

## Evaluation / Inference
- Tile-wise evaluation: `tortoise.train.evaluate(...)`
- Whole-image fusion + metrics: `tortoise.utils.ensemble_image` and `evaluate_images`
- Visualization helpers: `tortoise.utils.to_display_rgb`
