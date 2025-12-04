import torch
import torch.nn as nn
from tqdm import tqdm


# Device management utilities
def get_device():
    """Safely select GPU device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("[WARN] CUDA not available; will use CPU. Training will be slow.")
        return torch.device("cpu")


def print_device_info():
    """Print GPU info for debugging."""
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("[INFO] CUDA not available; using CPU")

#  Loss Components
def dice_loss(logits, targets, mask, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs * mask
    targets = targets * mask

    p = probs.view(probs.size(0), -1)
    t = targets.view(targets.size(0), -1)

    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)

    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets, mask, bce_loss_fn, alpha = 0.5):
    # BCE (masked)
    bce_raw = bce_loss_fn(logits, targets)             # shape: (B,1,H,W)
    bce     = (bce_raw * mask).sum() / (mask.sum() + 1e-6)

    # Dice
    d_loss = dice_loss(logits, targets, mask)

    return (1-alpha)*bce + alpha * d_loss


def masked_iou(logits, targets, mask, threshold = 0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()

    preds = preds * mask
    targets = targets * mask

    p = preds.view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)

    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter

    return ((inter + eps) / (union + eps)).mean()


#  Training Loop

def train_one_epoch(model, loader, optimizer, device, pos_weight, threshold = 0.5, alpha = 0.5, use_amp=True, scaler=None):
    model.train()
    total_loss = 0.0
    total_iou  = 0.0
    count = 0

    # BCEWithLogits with pos_weight for imbalance
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    for batch in tqdm(loader, desc="Train", leave=False):
        ms    = batch["ms"].to(device)
        label = batch["label"].to(device)
        mask  = batch["mask"].to(device).float()

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(ms)
                loss = combined_loss(logits, label, mask, bce_loss_fn, alpha = alpha)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(ms)
            loss = combined_loss(logits, label, mask, bce_loss_fn, alpha = alpha)
            loss.backward()
            optimizer.step()

        # compute IoU for progress monitoring
        iou_val = masked_iou(logits, label, mask, threshold)

        total_loss += loss.item()
        total_iou  += iou_val.item()
        count += 1

    return total_loss / count, total_iou / count



#  Evaluation Loop

@torch.no_grad()
def evaluate(model, loader, device, pos_weight, threshold = 0.5, alpha = 0.5, use_amp=True):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count = 0

    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    for batch in tqdm(loader, desc="Eval", leave=False):
        ms    = batch["ms"].to(device)
        label = batch["label"].to(device)
        mask  = batch["mask"].to(device).float()

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(ms)
                loss = combined_loss(logits, label, mask, bce_loss_fn, alpha = alpha)
        else:
            logits = model(ms)
            loss = combined_loss(logits, label, mask, bce_loss_fn, alpha = alpha)

        iou_val = masked_iou(logits, label, mask, threshold)

        total_loss += loss.item()
        total_iou  += iou_val.item()
        count += 1

    return total_loss / count, total_iou / count



def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    pos_weight,
    num_epochs,
    threshold = 0.5,
    alpha = 0.5,
    checkpoint_path = None,
    scaler = None,
    use_amp=True,
    early_stopping_patience=None
):
    

    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses   = []
    train_ious   = []
    val_ious     = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # ---- Train ----
        train_loss, train_iou = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pos_weight=pos_weight,
            use_amp=use_amp,
            scaler=scaler,
            threshold=threshold,
            alpha = alpha,
        )

        # ---- Eval ----
        val_loss, val_iou = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            pos_weight=pos_weight,
            use_amp=use_amp,
            threshold=threshold,
        )

        # scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # record
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        print(f"  Train Loss: {train_loss:.6f} | IoU: {train_iou:.4f}")
        print(f"  Val   Loss: {val_loss:.6f} | IoU: {val_iou:.4f}")

        # ---- Checkpoint & Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved new best model to {checkpoint_path}")
        else:
            patience_counter += 1
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print("\nTraining complete.")
    print(f"Best Val Loss: {best_val_loss:.6f}")

    return model, train_losses, val_losses, train_ious, val_ious

