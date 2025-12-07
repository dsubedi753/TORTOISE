# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 23:54:05 2025

@author: Divas
"""

import os
import re
import shutil
from pathlib import Path



root = Path(os.getenv("PROJECT_ROOT"))  
DATA_FOLDER = root / "data" 
RAW_DATA_FOLDER = DATA_FOLDER / "raw"

TRAINING_IMAGE_FOLDER = RAW_DATA_FOLDER / "training_images_masked"
SEGMENTATION_FOLDER  = RAW_DATA_FOLDER / "segmentations_masked"
RGB_FOLDER           = RAW_DATA_FOLDER / "training_images_RGBs"

OUTPUT_DATASET = root / "data" / "imageset"  # data folder

# Ensure output directory exists
os.makedirs(OUTPUT_DATASET, exist_ok=True)

# Regex to extract 3-digit ID
ID_REGEX = re.compile(r"(\d{3})")


# Helper: extract ID from filename
def extract_id(filename):
    match = ID_REGEX.search(filename)
    if match:
        return match.group(1)
    return None  # File not valid


# Process folders
def process_folder(src_folder, name_in_output):
    """
    src_folder: TRAINING_IMAGE_FOLDER or SEGMENTATION_FOLDER etc.
    name_in_output: 'ms.tif' or 'label.tif' etc.
    """
    for fname in os.listdir(src_folder):
        fpath = os.path.join(src_folder, fname)

        if not os.path.isfile(fpath):
            continue

        image_id = extract_id(fname)
        if image_id is None:
            print(f"Skipping file without 3-digit ID: {fname}")
            continue

        # Make destination folder for this image_id
        out_dir = os.path.join(OUTPUT_DATASET, image_id)
        os.makedirs(out_dir, exist_ok=True)

        # Output filename depends on the type
        out_path = os.path.join(out_dir, name_in_output)

        print(f"Copying {fpath} → {out_path}")
        shutil.copy2(fpath, out_path)



# Perform reorganization
print("Processing multispectral files...")
process_folder(TRAINING_IMAGE_FOLDER, "ms.tif")

print("Processing segmentation files...")
process_folder(SEGMENTATION_FOLDER, "label.tif")

print("Processing RGB files...")
process_folder(RGB_FOLDER, "rgb.png")

print("Dataset restructure complete!")
print(f"Output at: {OUTPUT_DATASET}")
