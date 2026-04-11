"""
train_classification.py  —  Task 1: VGG11 Classification on Oxford-IIIT Pet

Run from the da6401_assignment_2/ root directory:

    python train_classification.py

All hyper-parameters are at the top under "CONFIG". Edit them directly or
override via CLI flags (see parse_args()).

Expected directory layout
--------------------------
da6401_assignment_2/
├── data/
│   ├── annotations/
│   │   ├── trimaps/
│   │   ├── xmls/
│   │   ├── trainval.txt
│   │   └── test.txt
│   ├── images/
│   └── pets_dataset.py
├── models/
│   ├── __init__.py
│   ├── layers.py
│   ├── vgg11.py
│   └── classification.py
├── losses/
│   ├── __init__.py
│   └── iou_loss.py
└── train_classification.py    ← this file
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.pets_dataset import OxfordIIITPetDataset, get_transforms
from models.classification import VGG11Classifier

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
    print("[WARN] wandb not installed. Logging to console only.")

try:
    from sklearn.metrics import f1_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ===========================================================================
# CONFIG — edit these before your first run
# ===========================================================================
DEFAULT_DATA_ROOT   = os.path.join(ROOT, "data")   # contains images/ and annotations/
DEFAULT_CKPT_DIR    = os.path.join(ROOT, "checkpoints")
DEFAULT_EPOCHS      = 30
DEFAULT_BATCH_SIZE  = 32
DEFAULT_LR          = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT_P   = 0.5
DEFAULT_IMAGE_SIZE  = 224
DEFAULT_VAL_SPLIT   = 0.1          # fraction of trainval.txt used for validation
DEFAULT_NUM_WORKERS = 4            # set to 0 if you hit multiprocessing errors
DEFAULT_SEED        = 42
DEFAULT_WANDB_PROJECT = "da6401_a2_task1"
# ===========================================================================


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="DA6401 A2 Task 1 — VGG11 classification training",
    )
    p.add_argument("--data_root",    default=DEFAULT_DATA_ROOT)
    p.add_argument("--ckpt_dir",     default=DEFAULT_CKPT_DIR)
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--dropout_p",    type=float, default=DEFAULT_DROPOUT_P)
    p.add_argument("--image_size",   type=int,   default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--val_split",    type=float, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--num_workers",  type=int,   default=DEFAULT_NUM_WORKERS)
    p.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    p.add_argument("--num_classes",  type=int,   default=37)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--run_name",     default=None,
                   help="W&B run name. Useful for ablation runs.")
    p.add_argument("--no_wandb",     action="store_true",
                   help="Disable W&B logging entirely.")
    p.add_argument("--resume",       default=None,
                   help="Path to a checkpoint to resume from.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> dict:
    model.train()
    total_loss = correct = total = 0
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)              # [B, num_classes]
        loss   = criterion(logits, labels)
        loss.backward()
        # Gradient clipping — prevents occasional spikes from large BN updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs          = images.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs

        if step % 50 == 0:
            print(
                f"  epoch {epoch} step {step}/{len(loader)} | "
                f"loss={total_loss/total:.4f}  acc={correct/total:.4f}"
            )

    elapsed = time.time() - t0
    return {
        "loss": total_loss / total,
        "acc":  correct    / total,
        "time": elapsed,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> dict:
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        labels = labels.to(device, non_blocking=True)

        logits      = model(images)
        loss        = criterion(logits, labels)
        preds       = logits.argmax(1)

        bs          = images.size(0)
        total_loss += loss.item() * bs
        correct    += (preds == labels).sum().item()
        total      += bs

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    metrics = {
        "loss": total_loss / total,
        "acc":  correct    / total,
    }

    if _SKLEARN:
        yp = torch.cat(all_preds).numpy()
        yt = torch.cat(all_labels).numpy()
        metrics["macro_f1"] = f1_score(yt, yp, average="macro", zero_division=0)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Device : {device}")
    print(f"Data   : {args.data_root}")
    print(f"Epochs : {args.epochs}  |  BS: {args.batch_size}  |  LR: {args.lr}")
    print(f"Dropout: {args.dropout_p}  |  Label-smooth: {args.label_smoothing}")
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
    train_tfm = get_transforms("train", args.image_size)
    val_tfm   = get_transforms("val",   args.image_size)

    # ---------------------------------------------------------------- Dataset
    # trainval.txt → split into train + val
    full_trainval = OxfordIIITPetDataset(
        root=args.data_root,
        split="train",          # reads trainval.txt
        transform=train_tfm,
        load_bbox=False,
        load_mask=False,
    )
    test_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="test",           # reads test.txt
        transform=val_tfm,
        load_bbox=False,
        load_mask=False,
    )

    n_val   = int(len(full_trainval) * args.val_split)
    n_train = len(full_trainval) - n_val
    train_ds, val_ds = random_split(
        full_trainval,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Samples — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_dataset)}\n")

    # -------------------------------------------------------------- Loaders
    def make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size  = args.batch_size,
            shuffle     = shuffle,
            num_workers = args.num_workers,
            pin_memory  = (device.type == "cuda"),
            drop_last   = shuffle,   # avoid tiny last batch when shuffling
        )

    train_loader = make_loader(train_ds,      shuffle=True)
    val_loader   = make_loader(val_ds,        shuffle=False)
    test_loader  = make_loader(test_dataset,  shuffle=False)

    # --------------------------------------------------------------- Model
    model = VGG11Classifier(
        num_classes=args.num_classes,
        dropout_p=args.dropout_p,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --------------------------------------------------------- Loss / Optim
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Cosine annealing: smoothly reduces LR to 1e-6 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ------------------------------------------------------------ Resume
    start_epoch  = 1
    best_val_acc = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=200)

    # -------------------------------------------------------- Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = scheduler.get_last_lr()[0] if epoch > 1 else args.lr

        tr  = train_one_epoch(model, train_loader, criterion,
                               optimizer, device, epoch)
        val = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        f1_str = f"  macro_f1={val['macro_f1']:.4f}" if "macro_f1" in val else ""
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
            f"val loss={val['loss']:.4f} acc={val['acc']:.4f}{f1_str} | "
            f"lr={lr_now:.2e}  ({tr['time']:.1f}s)"
        )

        if use_wandb:
            log = {
                "epoch":      epoch,
                "train/loss": tr["loss"],
                "train/acc":  tr["acc"],
                "val/loss":   val["loss"],
                "val/acc":    val["acc"],
                "lr":         lr_now,
            }
            if "macro_f1" in val:
                log["val/macro_f1"] = val["macro_f1"]
            wandb.log(log)

        # Save best checkpoint
        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            ckpt_path = os.path.join(args.ckpt_dir, "task1_best.pth")
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state":   optimizer.state_dict(),
                "val_acc":           best_val_acc,
                "args":              vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}  (val_acc={best_val_acc:.4f})")

    # ---------------------------------------------------- Final test eval
    print("\nEvaluating on held-out test set …")
    test = evaluate(model, test_loader, criterion, device)
    f1_str = f"  macro_f1={test['macro_f1']:.4f}" if "macro_f1" in test else ""
    print(f"Test  loss={test['loss']:.4f}  acc={test['acc']:.4f}{f1_str}")

    if use_wandb:
        log = {"test/loss": test["loss"], "test/acc": test["acc"]}
        if "macro_f1" in test:
            log["test/macro_f1"] = test["macro_f1"]
        wandb.log(log)
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()