"""
train_localization.py  —  Task 2: VGG11 Bounding-Box Regression

Run from da6401_assignment_2/ root:

    # With pretrained Task-1 backbone:
    python train_localization.py --classifier_ckpt checkpoints/task1_best.pth

    # From scratch (no pretrained backbone):
    python train_localization.py

    # Frozen backbone (W&B §2.3 baseline):
    python train_localization.py \
        --classifier_ckpt checkpoints/task1_best.pth \
        --freeze_encoder \
        --run_name "frozen_encoder"

    # Full fine-tune (default, recommended):
    python train_localization.py \
        --classifier_ckpt checkpoints/task1_best.pth \
        --run_name "full_finetune"

Expected directory layout
--------------------------
da6401_assignment_2/
├── data/
│   ├── annotations/  (trainval.txt, test.txt, xmls/, trimaps/)
│   ├── images/
│   └── pets_dataset.py
├── models/
│   ├── layers.py
│   ├── vgg11.py
│   ├── classification.py
│   └── localization.py
├── losses/
│   ├── __init__.py
│   └── iou_loss.py
└── train_localization.py   ← this file
"""

import argparse
import os
import sys
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.pets_dataset   import OxfordIIITPetDataset, get_transforms
from models.localization import VGG11Localizer
from losses.iou_loss     import IoULoss

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
    print("[WARN] wandb not installed — logging to console only.")


# ===========================================================================
# CONFIG
# ===========================================================================
DEFAULT_DATA_ROOT    = os.path.join(ROOT, "data")
DEFAULT_CKPT_DIR     = os.path.join(ROOT, "checkpoints")
DEFAULT_EPOCHS       = 30
DEFAULT_BATCH_SIZE   = 32
DEFAULT_LR_HEAD      = 1e-3    # learning rate for the regression head
DEFAULT_LR_BACKBONE  = 1e-4    # 10× smaller for the pretrained backbone
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT_P    = 0.5
DEFAULT_IMAGE_SIZE   = 224
DEFAULT_VAL_SPLIT    = 0.1
DEFAULT_NUM_WORKERS  = 4
DEFAULT_SEED         = 42
DEFAULT_WANDB_PROJECT = "da6401_a2_task2"
# ===========================================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="DA6401 A2 Task 2 — VGG11 bounding-box regression",
    )
    p.add_argument("--data_root",         default=DEFAULT_DATA_ROOT)
    p.add_argument("--ckpt_dir",          default=DEFAULT_CKPT_DIR)
    p.add_argument("--classifier_ckpt",   default=None,
                   help="Path to task1_best.pth. If omitted, backbone is "
                        "randomly initialised.")
    p.add_argument("--freeze_encoder",    action="store_true",
                   help="Freeze the VGG11 backbone (feature extractor mode).")
    p.add_argument("--epochs",            type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size",        type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr_head",           type=float, default=DEFAULT_LR_HEAD,
                   help="LR for the regression head.")
    p.add_argument("--lr_backbone",       type=float, default=DEFAULT_LR_BACKBONE,
                   help="LR for the backbone (ignored when --freeze_encoder).")
    p.add_argument("--weight_decay",      type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--dropout_p",         type=float, default=DEFAULT_DROPOUT_P)
    p.add_argument("--image_size",        type=int,   default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--val_split",         type=float, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--num_workers",       type=int,   default=DEFAULT_NUM_WORKERS)
    p.add_argument("--seed",              type=int,   default=DEFAULT_SEED)
    p.add_argument("--wandb_project",     default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--run_name",          default=None)
    p.add_argument("--no_wandb",          action="store_true")
    p.add_argument("--resume",            default=None,
                   help="Resume from a localizer checkpoint.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# IoU metric (not the loss — this one gives a readable number for logging)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_mean_iou(
    pred_px: torch.Tensor,
    tgt_px:  torch.Tensor,
    eps:     float = 1e-6,
) -> float:
    """Return mean IoU over a batch; both tensors in pixel xywh format."""
    px1 = pred_px[:, 0] - pred_px[:, 2] * 0.5
    py1 = pred_px[:, 1] - pred_px[:, 3] * 0.5
    px2 = pred_px[:, 0] + pred_px[:, 2] * 0.5
    py2 = pred_px[:, 1] + pred_px[:, 3] * 0.5

    tx1 = tgt_px[:, 0] - tgt_px[:, 2] * 0.5
    ty1 = tgt_px[:, 1] - tgt_px[:, 3] * 0.5
    tx2 = tgt_px[:, 0] + tgt_px[:, 2] * 0.5
    ty2 = tgt_px[:, 1] + tgt_px[:, 3] * 0.5

    iw = torch.relu(torch.min(px2, tx2) - torch.max(px1, tx1))
    ih = torch.relu(torch.min(py2, ty2) - torch.max(py1, ty1))
    inter = iw * ih

    area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    area_t = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union  = area_p + area_t - inter

    iou = inter / (union + eps)
    return iou.mean().item()


# ---------------------------------------------------------------------------
# Collate: skip samples with no valid bbox (sentinel = all -1)
# ---------------------------------------------------------------------------

def bbox_collate(batch):
    """Filter out samples whose bbox is the sentinel [-1,-1,-1,-1]."""
    valid = [s for s in batch if (s["bbox"] >= 0).all()]
    if not valid:
        return None   # entire mini-batch had no annotations
    images = torch.stack([s["image"] for s in valid])
    bboxes = torch.stack([s["bbox"]  for s in valid])   # normalised [0,1]
    labels = torch.tensor([s["label"] for s in valid])
    return {"image": images, "bbox": bboxes, "label": labels}


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: IoULoss,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    image_size: int,
    epoch:     int,
) -> dict:
    model.train()
    total_loss = total_iou = n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        if batch is None:           # all samples in batch had no bbox
            continue

        images = batch["image"].to(device, non_blocking=True)  # [B,3,H,W]
        # Ground-truth: normalised [0,1] → pixel space
        tgt_norm = batch["bbox"].to(device, non_blocking=True)  # [B,4] ∈[0,1]
        tgt_px   = tgt_norm * image_size                        # [B,4] pixels

        optimizer.zero_grad(set_to_none=True)

        pred_px = model(images)           # [B,4] pixels (Sigmoid×image_size)
        loss    = criterion(pred_px, tgt_px)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_iou  = compute_mean_iou(pred_px.detach(), tgt_px.detach())
        total_loss += loss.item()
        total_iou  += batch_iou
        n_batches  += 1

        if step % 50 == 0:
            print(
                f"  epoch {epoch}  step {step}/{len(loader)} | "
                f"loss={total_loss/n_batches:.4f}  "
                f"mIoU={total_iou/n_batches:.4f}"
            )

    if n_batches == 0:
        return {"loss": float("nan"), "miou": 0.0, "time": 0.0}

    return {
        "loss": total_loss / n_batches,
        "miou": total_iou  / n_batches,
        "time": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: IoULoss,
    device:    torch.device,
    image_size: int,
) -> dict:
    model.eval()
    total_loss = total_iou = n_batches = 0

    for batch in loader:
        if batch is None:
            continue

        images  = batch["image"].to(device, non_blocking=True)
        tgt_px  = batch["bbox"].to(device, non_blocking=True) * image_size

        pred_px    = model(images)
        loss       = criterion(pred_px, tgt_px)
        batch_iou  = compute_mean_iou(pred_px, tgt_px)

        total_loss += loss.item()
        total_iou  += batch_iou
        n_batches  += 1

    if n_batches == 0:
        return {"loss": float("nan"), "miou": 0.0}

    return {
        "loss": total_loss / n_batches,
        "miou": total_iou  / n_batches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Device      : {device}")
    print(f"Data root   : {args.data_root}")
    print(f"Classifier  : {args.classifier_ckpt or 'random init'}")
    print(f"Freeze enc  : {args.freeze_encoder}")
    print(f"LR head     : {args.lr_head}   backbone: {args.lr_backbone}")
    print(f"Epochs      : {args.epochs}   BS: {args.batch_size}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ W&B
    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    # ------------------------------------------------------------ Transforms
    # bbox annotations are in normalised coords → no spatial augmentation
    # that would shift bbox coords (only colour jitter + hflip with bbox flip)
    train_tfm = get_transforms("train", args.image_size)
    val_tfm   = get_transforms("val",   args.image_size)

    # ---------------------------------------------------------------- Dataset
    # load_bbox=True, load_mask=False — we only need bboxes here
    full_trainval = OxfordIIITPetDataset(
        root=args.data_root, split="train",
        transform=train_tfm,
        load_bbox=True, load_mask=False,
    )
    test_dataset = OxfordIIITPetDataset(
        root=args.data_root, split="test",
        transform=val_tfm,
        load_bbox=True, load_mask=False,
    )

    n_val   = int(len(full_trainval) * args.val_split)
    n_train = len(full_trainval) - n_val
    train_ds, val_ds = random_split(
        full_trainval, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Count how many samples actually have a bbox annotation
    bbox_count = sum(
        1 for s in full_trainval.samples
        if (full_trainval.xmls_dir / f"{s[0]}.xml").exists()
    )
    print(f"Samples with bbox annotations: {bbox_count} / {len(full_trainval)}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_dataset)}\n")

    # -------------------------------------------------------------- Loaders
    def make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size   = args.batch_size,
            shuffle      = shuffle,
            num_workers  = args.num_workers,
            pin_memory   = (device.type == "cuda"),
            drop_last    = shuffle,
            collate_fn   = bbox_collate,   # skips unannotated samples
        )

    train_loader = make_loader(train_ds,     shuffle=True)
    val_loader   = make_loader(val_ds,       shuffle=False)
    test_loader  = make_loader(test_dataset, shuffle=False)

    # --------------------------------------------------------------- Model
    model = VGG11Localizer(
        in_channels    = 3,
        dropout_p      = args.dropout_p,
        freeze_encoder = args.freeze_encoder,
        image_size     = args.image_size,
    ).to(device)

    # Load pretrained backbone from Task 1 if provided
    if args.classifier_ckpt and os.path.isfile(args.classifier_ckpt):
        model.load_classifier_backbone(args.classifier_ckpt)
    else:
        if args.classifier_ckpt:
            print(f"[WARN] Classifier checkpoint not found: {args.classifier_ckpt}")
        print("[INFO] Backbone initialised randomly.")

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {total_params:,}  trainable: {trainable_params:,}\n")

    # --------------------------------------------------------- Loss / Optim
    criterion = IoULoss(eps=1e-6, reduction="mean")

    # Differential learning rates: small LR for pretrained backbone,
    # larger LR for the newly initialised regression head.
    if args.freeze_encoder:
        # Only head parameters need updating
        optimizer = torch.optim.AdamW(
            model.regression_head.parameters(),
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(),
                 "lr": args.lr_backbone},
                {"params": model.regression_head.parameters(),
                 "lr": args.lr_head},
            ],
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ------------------------------------------------------------ Resume
    start_epoch    = 1
    best_val_miou  = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_miou = ckpt.get("val_miou", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best mIoU={best_val_miou:.4f}")

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=200)

    # -------------------------------------------------------- Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = scheduler.get_last_lr()[0] if epoch > 1 else args.lr_head

        tr  = train_one_epoch(model, train_loader, criterion,
                               optimizer, device, args.image_size, epoch)
        val = evaluate(model, val_loader, criterion, device, args.image_size)
        scheduler.step()

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} mIoU={tr['miou']:.4f} | "
            f"val loss={val['loss']:.4f} mIoU={val['miou']:.4f} | "
            f"lr={lr_now:.2e}  ({tr['time']:.1f}s)"
        )

        if use_wandb:
            wandb.log({
                "epoch":       epoch,
                "train/loss":  tr["loss"],
                "train/miou":  tr["miou"],
                "val/loss":    val["loss"],
                "val/miou":    val["miou"],
                "lr":          lr_now,
            })

        if val["miou"] > best_val_miou:
            best_val_miou = val["miou"]
            ckpt_path = os.path.join(args.ckpt_dir, "task2_best.pth")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_miou":         best_val_miou,
                "args":             vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}  "
                  f"(val_mIoU={best_val_miou:.4f})")

    # ---------------------------------------------------- Final test eval
    print("\nEvaluating on held-out test set …")
    test = evaluate(model, test_loader, criterion, device, args.image_size)
    print(f"Test  loss={test['loss']:.4f}  mIoU={test['miou']:.4f}")

    if use_wandb:
        wandb.log({"test/loss": test["loss"], "test/miou": test["miou"]})
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()