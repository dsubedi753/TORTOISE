# tortoise/checkpoint.py

import torch
from pathlib import Path
import os

def _get_ckpt_dir():
    """Return PROJECT_ROOT/checkpoints, creating if necessary."""
    root = Path(os.getenv("PROJECT_ROOT"))
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def save_checkpoint(model, optimizer, scaler, epoch, hparams, filename="latest.pth"):
    """
    Save full training state into PROJECT_ROOT/checkpoints/<filename>.
    """
    ckpt_dir = _get_ckpt_dir()
    path = ckpt_dir / filename

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "hparams": hparams,
    }

    torch.save(checkpoint, path)
    print(f"[Checkpoint Saved] {path}")


def load_checkpoint(model, optimizer=None, scaler=None, filename="latest.pth", device="cuda"):
    """
    Load state from PROJECT_ROOT/checkpoints/<filename>.

    Returns:
        epoch (int), hparams (dict)
    """
    ckpt_dir = _get_ckpt_dir()
    path = ckpt_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Load model
    model.load_state_dict(checkpoint["model_state"])
    print(f"[Model Loaded] from {path}")

    # Load optimizer if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("[Optimizer Loaded]")

    # Load scaler if provided
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
        print("[Scaler Loaded]")

    return checkpoint["epoch"], checkpoint.get("hparams", {})
