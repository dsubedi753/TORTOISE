# Albumentations in TORTOISE

✅ Overview

This project uses Albumentations to define image augmentations for multispectral tiles. Albumentations is a fast, flexible augmentation library for images that works well with deep learning pipelines. In TORTOISE, we use Albumentations primarily to produce geometric and photometric variation for tiles (e.g., flips, rotations, Gaussian noise, blur, and intensity scaling).

---

## 🔧 Installation

Use pip or conda to install Albumentations and its dependencies:

```bash
# pip
pip install albumentations

# or conda-forge
conda install -c conda-forge albumentations
```

If you use certain Albumentations transforms that require OpenCV or scikit-image, ensure those are installed (e.g., opencv-python).

---

## 📁 Where it's used

- Source: `src/tortoise/augmentations.py` (core utilities)
- Dataset integration: `src/tortoise/dataset.py` (applies augmentations to tile samples)
 - Dataloader helper: `src/tortoise/dataloader.py` (pre-sampling augmentations via `sample_augmentations`)

---

## 🧩 Core API and Behavior

### AUG_KEYS

Defined in `augmentations.py`:

- `AUG_KEYS = ["hflip","vflip","dflip","rot90","noise","blur","iscale"]`

These are the named augmentations supported by the project.

`GEOMETRIC_KEYS` defines which transforms are geometric and therefore should be applied to image, label, and mask.

### IntensityScale

A small custom Albumentations `ImageOnlyTransform` defined in `augmentations.py`:

```python
class IntensityScale(A.ImageOnlyTransform):
    def __init__(self, scale_limit: float = 0.1, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.scale_limit = float(scale_limit)

    def apply(self, img: np.ndarray, **params):
        alpha = 1.0 + random.uniform(-self.scale_limit, self.scale_limit)
        return img * alpha
```

It multiplies all pixel values by alpha in range [1-scale_limit, 1+scale_limit].

### _transform_for_name(name)

Maps a name (like `hflip`, `noise`, or `iscale`) to a core Albumentations transform (e.g., `A.HorizontalFlip(p=1.0)`). Geometric transforms use `A.Compose` with `Transpose`+`Flip` where needed.

### apply_augmentation(image_hwc, label_hw, mask_hw, aug_name)

 - Inputs:
   - `image_hwc`: multispectral or RGB tile (H, W, C) as numpy
  - `label_hw`: label mask (H, W) or None
  - `mask_hw`: valid pixel mask (H, W) or None
  - `aug_name`: augmentation name from `AUG_KEYS`

- Behavior:
  - Geometric transforms (`hflip, vflip, dflip, rot90`) are applied to the image AND the label and mask by using `additional_targets` in `A.Compose`.
  - Photometric transforms (noise, blur, iscale) are applied to the image only; label and mask pass through unchanged.

 - Returns a tuple with transformed `image_hwc` (H, W, C), `label` (H, W) or None, and `mask` (H, W) or None.

### sample_augmentations and `build_dataloaders`

The dataloader's `build_dataloaders` function contains a helper `sample_augmentations(ids, aug_keys, seed=42)` that expands a list of tile ids into a list of tuples intended for the dataset's `samples` list. For each tile, two augmentations are sampled and the helper yields `(tid, None)`, `(tid, aug1)`, `(tid, aug2)`.

`build_dataloaders` uses deterministic RNG seeds by default (`seed` arg), so augmentation assignment is reproducible across runs if you fix the seed.

---

## 🧭 Data pipeline details (Important)

- The dataset currently scales multispectral values by dividing by 10000.0 before augmentation (see `TileDataset.__getitem__`). As a result photometric transforms operate on scaled pixel values rather than raw sensor readouts.
  - If you prefer the photometric transforms to operate on raw values (e.g., for physically plausible noise models), perform augmentation prior to the division or apply a different normalizer.

- Geometric transforms are applied to image, label, and mask so alignment is preserved.

- Reproducibility: `build_dataloaders(..., seed=42)` uses a deterministic RNG seed for sampling augmentations, so the set of `(tid, aug)` tuples produced by `sample_augmentations` is reproducible across runs when `seed` is fixed.

---

## ➕ How to add a new augmentation

1. Add a new key to `AUG_KEYS` (and decide whether it's geometric or photometric):

```python
AUG_KEYS.append("myaugment")
```

If the transform is geometric, add it to the `GEOMETRIC_KEYS` set.

2. Implement transform mapping in `_transform_for_name`:
```python
if name == "myaugment":
    return A.RandomBrightnessContrast(brightness_limit=(0.1, 0.3), contrast_limit=(0.1, 0.3), p=1.0)
```

If the transform is image-only, be sure it returns an `A.ImageOnlyTransform` or similar.

3. (Optional) If the transform needs custom behavior not supported by a built-in transform, implement a custom subclass of `A.ImageOnlyTransform` and use it like `IntensityScale`.

4. Update tests & documentation. You may want to add checks that the shape, dtype, and value ranges remain consistent after augmentation.

---

## ⚠️ Tips & Troubleshooting

- Shape/dtype expectations:
  - When using `apply_augmentation`, inputs must be `numpy.ndarray` in HWC format for the image and HW for masks. The dataset converts between `torch.Tensor` and numpy inside `__getitem__`.
  - The dataset converts label masks back to (1,H,W) and uses `float` dtypes.

- If you see misaligned labels after geometric transforms, ensure `additional_targets` is configured and the label/mask are passed as `mask` targets.

- For small tiles (48x48 in this dataset), avoid too-large blur or overly aggressive geometric transforms that might reduce effective learning signal.

- If you want photometric transforms applied on raw pixel values rather than normalized ones, move normalization to after augmentation in `dataset.py`.

---

## 📌 Example usage

Pre-sampled augmentations and dataloaders:

```python
from tortoise.dataloader import build_dataloaders

train_loader, val_loader, test_loader = build_dataloaders(
  tiles_dir= DATA_FOLDER / "tiles",
  batch_size=16,
  seed=42,
  use_rgb=False,
  use_ms=True,
  num_workers=0,
)
print(train_loader.dataset.samples[:6])  # tuples: (tile_id, None | aug_name)
```

Inspecting dataset samples (the augmentation applied per sample):

```python
# The `samples` attribute lists tuples: (tile_id, None | 'augmentation_name')
samples = train_loader.dataset.samples
print(samples[:8])
```

---

## 📚 References

- Albumentations docs: https://albumentations.ai/docs/
- Albumentations GitHub: https://github.com/albumentations-team/albumentations

---
