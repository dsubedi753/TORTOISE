# Tortoise Dataset

A PyTorch dataset for working with tiled geospatial data.

## Environment Setup

Create the conda environment from the provided configuration:

```bash
conda env create -f environment.yml
conda activate tortoise
```

## Preprocessing

**Important:** You must run these two scripts in order before using the dataset:

1. **`data_organize`** – Organizes raw data files into the required directory structure
2. **`tilify`** – Processes the organized data and generates `tile_index.csv`

**The dataset will not load if these preprocessing steps are skipped.** The `tile_index.csv` file is required for the dataset to function.

## After Preprocessing

Once preprocessing is complete, you can load the dataset:

```python
from tortoise.dataset import TileDataset
```

# TORTOISE

## Augmentations

The project uses Albumentations for image augmentations (geometric and photometric transforms). See the `docs/ALBUMENTATIONS.md` file for details about supported transforms, how they are pre-sampled using AUG_MAP, how to reproduce augmentations, and how to add new transforms.

## SAM 2

Code for training a finetuned SAM2 model is in the notebooks/ folder. SAM2FinetuneNew finetunes a SAM2 model. SAM2FT_Validate calculates performance metrics based off of generated finetuned models using a tiling approach.
