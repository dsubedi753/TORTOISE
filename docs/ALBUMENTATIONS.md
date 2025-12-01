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
- Dataloder helper: `src/tortoise/dataloader.py` (pre-sampling augmentations via AUG_MAP)

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

### apply_augmentation(ms_hwc, label_hw, mask_hw, aug_name)

- Inputs:
  - `ms_hwc`: multispectral tile (H, W, C) as numpy
  - `label_hw`: label mask (H, W) or None
  - `mask_hw`: valid pixel mask (H, W) or None
  - `aug_name`: augmentation name from `AUG_KEYS`

- Behavior:
  - Geometric transforms (`hflip, vflip, dflip, rot90`) are applied to the image AND the label and mask by using `additional_targets` in `A.Compose`.
  - Photometric transforms (noise, blur, iscale) are applied to the image only; label and mask pass through unchanged.

- Returns a tuple with transformed `ms` (H, W, C), `label` (H, W) or None, and `mask` (H, W) or None.

### sample_aug_map(tile_ids, versions=("aug1","aug2"), aug_keys=AUG_KEYS, seed=42)

Pre-samples a single augmentation for every `(tile_id, version)` pair using the provided RNG seed. This produces an `AugMap` (dictionary mapping `(tid, version)` -> `aug_name`), which enables deterministic augmentation assignments across runs if you use the same seed.

- `save_aug_map(path)` & `load_aug_map(path)` save and load sampled `AugMap` as a JSON with keys serialized as `tile_id::version`.

---

## 🧭 Data pipeline details (Important)

- The dataset normalizes MS channels *before* augmentation (look at `dataset.py`): `self.normalizer` is applied prior to calling `apply_augmentation`.
  - Implication: Photometric transforms (noise, blur, intensity scaling) operate on normalized values (for example, mean-std normalized numbers) rather than raw data. If you prefer to apply photometric transforms before normalization, you can alter the order in `dataset.py`.

- Geometric transforms are applied to images, labels, and valid-mask (so pixel alignment is preserved).

- For reproducibility:
  - `build_dataloaders(..., seed=42)` calls `sample_aug_map` with the same seed, making augmentation deterministic across runs if you keep the seed and `save_aug_map_path` consistent.
  - You can `save_aug_map` to a JSON file and re-load it later.

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
    data_root=DATA_FOLDER / "tiles",
    batch_size=16,
    normalizer=dn,
    use_rgb=False,
    seed=42,
    num_workers=0,
    save_aug_map_path=PROJECT_FOLDER / "aug_map.json",
)
```

Inspecting or re-using saved `aug_map`:

```python
from tortoise.augmentations import load_aug_map
aug_map = load_aug_map(PROJECT_FOLDER / "aug_map.json")
# Pass aug_map into a TileDataset
```

---

## 📚 References

- Albumentations docs: https://albumentations.ai/docs/
- Albumentations GitHub: https://github.com/albumentations-team/albumentations

---

If you'd like, I can also:
- Add a short notebook cell to `notebooks/Training.ipynb` that explains augmentation usage and shows the saved `aug_map` structure, or
- Add unit tests that exercise `apply_augmentation` and `sample_aug_map` for full coverage.

Let me know which additions you'd prefer! 💡
