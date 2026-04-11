"""
train_segmentation.py  —  Task 3: U-Net Style Semantic Segmentation

Run from da6401_assignment_2/ root:

    # Full fine-tune with Task-1 backbone (recommended):
    python train_segmentation.py --classifier_ckpt checkpoints/task1_best.pth

    # W&B §2.3 — Strict Feature Extractor:
    python train_segmentation.py \
        --classifier_ckpt checkpoints/task1_best.pth \
        --freeze_encoder \
        --run_name "strict_frozen"

    # W&B §2.3 — Partial Fine-Tune (freeze early stages 1,2,3):
    python train_segmentation.py \
        --classifier_ckpt checkpoints/task1_best.pth \
        --freeze_stages 1 2 3 \
        --run_name "partial_finetune"

    # W&B §2.3 — Full Fine-Tune:
    python train_segmentation.py \
        --classifier_ckpt checkpoints/task1_best.pth \
        --run_name "full_finetune"

Expected directory layout
--------------------------
da6401_assignment_2/
├── data/
│   ├── annotations/  (trainval.txt, test.txt, trimaps/, xmls/)
│   ├── images/
│   └── pets_dataset.py
├── models/
│   ├── layers.py
│   ├── vgg11.py
│   ├── classification.py
│   └── segmentation.py
└── train_segmentation.py   ← this file
"""

import argparse
import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.pets_dataset    import OxfordIIITPetDataset, get_transforms
from models.segmentation  import VGG11UNet, SegmentationLoss

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
    print("[WARN] wandb not installed — logging to console only.")


# ===========================================================================
# CONFIG
# ===========================================================================
DEFAULT_DATA_ROOT     = os.path.join(ROOT, "data")
DEFAULT_CKPT_DIR      = os.path.join(ROOT, "checkpoints")
DEFAULT_EPOCHS        = 30
DEFAULT_BATCH_SIZE    = 16     # U-Net is memory-heavy; 16 is safer than 32
DEFAULT_LR_DECODER    = 1e-3   # decoder + bridge (newly initialised)
DEFAULT_LR_BACKBONE   = 1e-4   # 10× smaller for pretrained encoder
DEFAULT_WEIGHT_DECAY  = 1e-4
DEFAULT_DROPOUT_P     = 0.5
DEFAULT_IMAGE_SIZE    = 224
DEFAULT_VAL_SPLIT     = 0.1
DEFAULT_NUM_WORKERS   = 4
DEFAULT_DICE_WEIGHT   = 1.0    # λ for Dice in combined loss
DEFAULT_SEED          = 42
DEFAULT_WANDB_PROJECT = "da6401_a2_task3"
NUM_SEG_CLASSES       = 3      # trimap: fg=0, bg=1, unknown=2
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
        description="DA6401 A2 Task 3 — VGG11 U-Net segmentation",
    )
    # Paths
    p.add_argument("--data_root",       default=DEFAULT_DATA_ROOT)
    p.add_argument("--ckpt_dir",        default=DEFAULT_CKPT_DIR)
    p.add_argument("--classifier_ckpt", default=None,
                   help="Path to task1_best.pth. Omit for random init.")

    # Encoder freeze strategy
    freeze_grp = p.add_mutually_exclusive_group()
    freeze_grp.add_argument(
        "--freeze_encoder", action="store_true",
        help="Freeze entire VGG11 backbone (Strict Feature Extractor).",
    )
    freeze_grp.add_argument(
        "--freeze_stages", type=int, nargs="+", metavar="N",
        help="Freeze only these encoder stages, e.g. --freeze_stages 1 2 3 "
             "(Partial Fine-Tune).",
    )

    # Training
    p.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size",      type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr_decoder",      type=float, default=DEFAULT_LR_DECODER)
    p.add_argument("--lr_backbone",     type=float, default=DEFAULT_LR_BACKBONE)
    p.add_argument("--weight_decay",    type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--dropout_p",       type=float, default=DEFAULT_DROPOUT_P)
    p.add_argument("--image_size",      type=int,   default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--val_split",       type=float, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--num_workers",     type=int,   default=DEFAULT_NUM_WORKERS)
    p.add_argument("--dice_weight",     type=float, default=DEFAULT_DICE_WEIGHT,
                   help="Weight λ for the Dice term in combined loss.")

    # Misc
    p.add_argument("--seed",            type=int,   default=DEFAULT_SEED)
    p.add_argument("--wandb_project",   default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--run_name",        default=None)
    p.add_argument("--no_wandb",        action="store_true")
    p.add_argument("--resume",          default=None,
                   help="Resume from a segmentation checkpoint.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,   # [B, C, H, W]
    target: torch.Tensor,   # [B, H, W]  long
    num_classes: int = 3,
    eps: float = 1e-6,
) -> dict:
    """Compute pixel accuracy and per-class + mean Dice score.

    Returns dict with keys:
        'pixel_acc'  — fraction of correctly classified pixels
        'dice_mean'  — mean Dice across all classes
        'dice_fg'    — Dice for class 0 (foreground / pet body)
        'dice_bg'    — Dice for class 1 (background)
        'dice_unk'   — Dice for class 2 (unknown/boundary)
    """
    preds = logits.argmax(dim=1)   # [B, H, W]

    # Pixel accuracy
    pixel_acc = (preds == target).float().mean().item()

    # Per-class Dice
    dices = []
    for c in range(num_classes):
        pred_c = (preds  == c).float()
        tgt_c  = (target == c).float()
        inter  = (pred_c * tgt_c).sum()
        denom  = pred_c.sum() + tgt_c.sum()
        dices.append(((2.0 * inter + eps) / (denom + eps)).item())

    return {
        "pixel_acc": pixel_acc,
        "dice_mean": float(np.mean(dices)),
        "dice_fg":   dices[0],
        "dice_bg":   dices[1],
        "dice_unk":  dices[2] if len(dices) > 2 else 0.0,
    }


# ---------------------------------------------------------------------------
# Collate: skip samples with no trimap mask
# ---------------------------------------------------------------------------

def seg_collate(batch):
    """Drop samples whose mask is empty (no trimap file found)."""
    valid = [s for s in batch if s["mask"].numel() > 0]
    if not valid:
        return None
    images = torch.stack([s["image"] for s in valid])
    masks  = torch.stack([s["mask"]  for s in valid])   # [B, H, W] long
    labels = torch.tensor([s["label"] for s in valid])
    return {"image": images, "mask": masks, "label": labels}


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  SegmentationLoss,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    epoch:      int,
    num_classes: int,
) -> dict:
    model.train()
    sum_loss = sum_acc = sum_dice = n = 0
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        if batch is None:
            continue

        images = batch["image"].to(device, non_blocking=True)   # [B,3,H,W]
        masks  = batch["mask" ].to(device, non_blocking=True)   # [B,H,W] long

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)                  # [B, C, H, W]

        # Resize mask to match logit spatial size if needed
        # (only occurs if image_size is not divisible by 32)
        if logits.shape[2:] != masks.shape[1:]:
            masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=logits.shape[2:],
                mode="nearest",
            ).squeeze(1).long()

        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        m = compute_metrics(logits.detach(), masks.detach(), num_classes)
        bs        = images.size(0)
        sum_loss += loss.item() * bs
        sum_acc  += m["pixel_acc"] * bs
        sum_dice += m["dice_mean"] * bs
        n        += bs

        if step % 30 == 0:
            print(
                f"  epoch {epoch} step {step}/{len(loader)} | "
                f"loss={sum_loss/n:.4f}  "
                f"acc={sum_acc/n:.4f}  dice={sum_dice/n:.4f}"
            )

    if n == 0:
        return {"loss": float("nan"), "pixel_acc": 0.0,
                "dice_mean": 0.0, "time": 0.0}
    return {
        "loss":      sum_loss / n,
        "pixel_acc": sum_acc  / n,
        "dice_mean": sum_dice / n,
        "time":      time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  SegmentationLoss,
    device:     torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    sum_loss = sum_acc = sum_dice = 0.0
    sum_dice_fg = sum_dice_bg = sum_dice_unk = 0.0
    n = 0

    for batch in loader:
        if batch is None:
            continue

        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask" ].to(device, non_blocking=True)

        logits = model(images)

        if logits.shape[2:] != masks.shape[1:]:
            masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=logits.shape[2:],
                mode="nearest",
            ).squeeze(1).long()

        loss = criterion(logits, masks)
        m    = compute_metrics(logits, masks, num_classes)

        bs             = images.size(0)
        sum_loss      += loss.item()       * bs
        sum_acc       += m["pixel_acc"]    * bs
        sum_dice      += m["dice_mean"]    * bs
        sum_dice_fg   += m["dice_fg"]      * bs
        sum_dice_bg   += m["dice_bg"]      * bs
        sum_dice_unk  += m["dice_unk"]     * bs
        n             += bs

    if n == 0:
        return {"loss": float("nan"), "pixel_acc": 0.0, "dice_mean": 0.0,
                "dice_fg": 0.0, "dice_bg": 0.0, "dice_unk": 0.0}
    return {
        "loss":      sum_loss     / n,
        "pixel_acc": sum_acc      / n,
        "dice_mean": sum_dice     / n,
        "dice_fg":   sum_dice_fg  / n,
        "dice_bg":   sum_dice_bg  / n,
        "dice_unk":  sum_dice_unk / n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    freeze_desc = (
        "all stages (strict)"      if args.freeze_encoder else
        f"stages {args.freeze_stages} (partial)" if args.freeze_stages else
        "none (full fine-tune)"
    )
    print(f"\n{'='*60}")
    print(f"Device        : {device}")
    print(f"Data root     : {args.data_root}")
    print(f"Classifier ck : {args.classifier_ckpt or 'random init'}")
    print(f"Frozen stages : {freeze_desc}")
    print(f"LR decoder    : {args.lr_decoder}   backbone: {args.lr_backbone}")
    print(f"Loss          : CE + {args.dice_weight}×Dice")
    print(f"Epochs        : {args.epochs}   BS: {args.batch_size}")
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
    # get_transforms returns an albumentations Compose that handles joint
    # image+mask augmentation when called with image= and mask= kwargs.
    train_tfm = get_transforms("train", args.image_size)
    val_tfm   = get_transforms("val",   args.image_size)

    # ---------------------------------------------------------------- Dataset
    full_trainval = OxfordIIITPetDataset(
        root=args.data_root, split="train",
        transform=train_tfm,
        load_bbox=False, load_mask=True,
    )
    test_dataset = OxfordIIITPetDataset(
        root=args.data_root, split="test",
        transform=val_tfm,
        load_bbox=False, load_mask=True,
    )

    n_val   = int(len(full_trainval) * args.val_split)
    n_train = len(full_trainval) - n_val
    train_ds, val_ds = random_split(
        full_trainval, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
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
            collate_fn   = seg_collate,
        )

    train_loader = make_loader(train_ds,     shuffle=True)
    val_loader   = make_loader(val_ds,       shuffle=False)
    test_loader  = make_loader(test_dataset, shuffle=False)

    # --------------------------------------------------------------- Model
    model = VGG11UNet(
        num_classes    = NUM_SEG_CLASSES,
        in_channels    = 3,
        dropout_p      = args.dropout_p,
        freeze_encoder = args.freeze_encoder,
        freeze_stages  = args.freeze_stages,
    ).to(device)

    if args.classifier_ckpt and os.path.isfile(args.classifier_ckpt):
        model.load_classifier_backbone(args.classifier_ckpt)
    else:
        if args.classifier_ckpt:
            print(f"[WARN] Checkpoint not found: {args.classifier_ckpt}")
        print("[INFO] Backbone initialised randomly.")

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {total_p:,}  trainable: {trainable_p:,}\n")

    # --------------------------------------------------------- Loss / Optim
    criterion = SegmentationLoss(
        num_classes  = NUM_SEG_CLASSES,
        dice_weight  = args.dice_weight,
    )

    # Differential LR: pretrained backbone gets 10× smaller LR than decoder
    decoder_params  = (
        list(model.bridge.parameters())   +
        list(model.up5.parameters())      +
        list(model.up4.parameters())      +
        list(model.up3.parameters())      +
        list(model.up2.parameters())      +
        list(model.up1.parameters())      +
        list(model.seg_head.parameters())
    )
    backbone_params = [
        p for p in model.encoder.parameters() if p.requires_grad
    ]

    param_groups = [{"params": decoder_params, "lr": args.lr_decoder}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ------------------------------------------------------------ Resume
    start_epoch    = 1
    best_val_dice  = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("val_dice", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best Dice={best_val_dice:.4f}")

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=200)

    # -------------------------------------------------------- Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = scheduler.get_last_lr()[0] if epoch > 1 else args.lr_decoder

        tr  = train_one_epoch(model, train_loader, criterion,
                               optimizer, device, epoch, NUM_SEG_CLASSES)
        val = evaluate(model, val_loader, criterion, device, NUM_SEG_CLASSES)
        scheduler.step()

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} acc={tr['pixel_acc']:.4f} "
            f"dice={tr['dice_mean']:.4f} | "
            f"val loss={val['loss']:.4f} acc={val['pixel_acc']:.4f} "
            f"dice={val['dice_mean']:.4f} "
            f"[fg={val['dice_fg']:.3f} bg={val['dice_bg']:.3f} "
            f"unk={val['dice_unk']:.3f}] | "
            f"lr={lr_now:.2e}  ({tr['time']:.1f}s)"
        )

        if use_wandb:
            wandb.log({
                "epoch":           epoch,
                "train/loss":      tr["loss"],
                "train/pixel_acc": tr["pixel_acc"],
                "train/dice":      tr["dice_mean"],
                "val/loss":        val["loss"],
                "val/pixel_acc":   val["pixel_acc"],
                "val/dice":        val["dice_mean"],
                "val/dice_fg":     val["dice_fg"],
                "val/dice_bg":     val["dice_bg"],
                "val/dice_unk":    val["dice_unk"],
                "lr":              lr_now,
            })

        if val["dice_mean"] > best_val_dice:
            best_val_dice = val["dice_mean"]
            ckpt_path = os.path.join(args.ckpt_dir, "task3_best.pth")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_dice":         best_val_dice,
                "args":             vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved → {ckpt_path}  (val_dice={best_val_dice:.4f})")

    # ---------------------------------------------------- Final test eval
    print("\nEvaluating on held-out test set …")
    test = evaluate(model, test_loader, criterion, device, NUM_SEG_CLASSES)
    print(
        f"Test  loss={test['loss']:.4f}  "
        f"pixel_acc={test['pixel_acc']:.4f}  "
        f"dice={test['dice_mean']:.4f}  "
        f"[fg={test['dice_fg']:.3f} bg={test['dice_bg']:.3f} "
        f"unk={test['dice_unk']:.3f}]"
    )

    if use_wandb:
        wandb.log({
            "test/loss":      test["loss"],
            "test/pixel_acc": test["pixel_acc"],
            "test/dice":      test["dice_mean"],
            "test/dice_fg":   test["dice_fg"],
            "test/dice_bg":   test["dice_bg"],
            "test/dice_unk":  test["dice_unk"],
        })
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()