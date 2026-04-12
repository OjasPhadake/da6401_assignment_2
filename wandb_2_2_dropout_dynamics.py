"""
wandb_2_2_dropout_dynamics.py  —  W&B Report Section 2.2
=========================================================
Trains VGG11 under three dropout conditions and overlays
training vs validation loss curves:
  (1) No Dropout   (p=0.0)
  (2) Dropout p=0.2
  (3) Dropout p=0.5

Run from da6401_assignment_2/:
    python wandb_2_2_dropout_dynamics.py \
        --data_root data \
        --epochs 20 \
        --wandb_project da6401_a2_report
"""

import argparse
import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import wandb
from data.pets_dataset   import OxfordIIITPetDataset, get_transforms
from models.classification import VGG11Classifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, crit, opt, device):
    model.train()
    tot_loss = correct = total = 0
    for batch in loader:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        labels = labels.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss   = crit(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        tot_loss += loss.item() * imgs.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    return tot_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    tot_loss = correct = total = 0
    for batch in loader:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        labels = labels.to(device, non_blocking=True)
        logits    = model(imgs)
        tot_loss += crit(logits, labels).item() * imgs.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    return tot_loss / total, correct / total


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_dropout_experiment(dropout_p, args, tr_loader, val_loader,
                            device, run_name):
    """Train one model with a given dropout probability. Return history."""
    set_seed(args.seed)   # same seed each run for fair comparison
    model = VGG11Classifier(num_classes=37, dropout_p=dropout_p).to(device)

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group="section_2_2_dropout",
        config={"dropout_p": dropout_p, "lr": args.lr,
                "epochs": args.epochs, "batch_size": args.batch_size},
        reinit=True,
    )

    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc   = train_epoch(model, tr_loader,  crit, opt, device)
        val_loss, val_acc = eval_epoch(model, val_loader, crit, device)
        sched.step()

        gap = tr_loss - val_loss   # generalisation gap (negative = underfitting)
        wandb.log({
            "epoch":             epoch,
            "train/loss":        tr_loss,
            "train/acc":         tr_acc,
            "val/loss":          val_loss,
            "val/acc":           val_acc,
            "generalisation_gap": gap,
            "lr":                sched.get_last_lr()[0],
        })

        history.append({
            "epoch":     epoch,
            "tr_loss":   tr_loss,  "val_loss":  val_loss,
            "tr_acc":    tr_acc,   "val_acc":   val_acc,
            "gap":       gap,
        })
        print(f"  [{run_name}] E{epoch:02d}  "
              f"tr={tr_loss:.4f}/{tr_acc:.3f}  "
              f"val={val_loss:.4f}/{val_acc:.3f}  "
              f"gap={gap:+.4f}")

    # Log gradient norm histogram for the last epoch to show dropout effect
    grad_norms = []
    model.train()
    for i, batch in enumerate(tr_loader):
        if i >= 5: break
        imgs   = batch["image"].to(device)
        labels = batch["label"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        labels = labels.to(device)
        opt.zero_grad(set_to_none=True)
        crit(model(imgs), labels).backward()
        for p in model.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())

    wandb.log({"gradient_norms": wandb.Histogram(grad_norms)})
    wandb.finish()
    return history


# ---------------------------------------------------------------------------
# Overlay plot (logged to a summary W&B run)
# ---------------------------------------------------------------------------

def make_overlay_plot(results: dict, project: str):
    """results: dict of {label: history_list}"""
    colours = {"No Dropout (p=0.0)": "#e74c3c",
               "Dropout p=0.2":      "#2ecc71",
               "Dropout p=0.5":      "#3498db"}
    styles  = {"train": "-", "val": "--"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Training loss
    for label, hist in results.items():
        epochs  = [h["epoch"]   for h in hist]
        tr_loss = [h["tr_loss"] for h in hist]
        axes[0].plot(epochs, tr_loss, styles["train"],
                     color=colours[label], label=label, linewidth=2)
    axes[0].set_title("Training Loss", fontsize=12)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # Panel 2: Validation loss overlay
    for label, hist in results.items():
        epochs   = [h["epoch"]    for h in hist]
        val_loss = [h["val_loss"] for h in hist]
        axes[1].plot(epochs, val_loss, styles["val"],
                     color=colours[label], label=label, linewidth=2)
    axes[1].set_title("Validation Loss", fontsize=12)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # Panel 3: Generalisation gap (train - val)
    for label, hist in results.items():
        epochs = [h["epoch"] for h in hist]
        gaps   = [h["gap"]   for h in hist]
        axes[2].plot(epochs, gaps, "-", color=colours[label],
                     label=label, linewidth=2)
    axes[2].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[2].set_title("Generalisation Gap (Train − Val Loss)", fontsize=12)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Gap")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    plt.suptitle("Section 2.2 — Dropout Effect on Training Dynamics",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    wandb.init(project=project, name="section_2_2_overlay", reinit=True)
    wandb.log({"dropout_overlay": wandb.Image(fig)})

    # Summary table
    tbl = wandb.Table(columns=["Dropout p", "Best Val Acc", "Best Val Epoch",
                                "Final Gap"])
    for label, hist in results.items():
        best  = max(hist, key=lambda h: h["val_acc"])
        final = hist[-1]["gap"]
        tbl.add_data(label, round(best["val_acc"], 4),
                     best["epoch"], round(final, 4))
    wandb.log({"dropout_summary_table": tbl})
    wandb.finish()

    plt.savefig("dropout_overlay.png", dpi=120)
    plt.close()
    print("Overlay plot saved → dropout_overlay.png")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     default="data")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--val_split",     type=float, default=0.1)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--wandb_project", default="da6401_a2_report")
    return p.parse_args()


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets (shared across all three runs)
    train_tfm = get_transforms("train", 224)
    val_tfm   = get_transforms("val",   224)
    full_tv   = OxfordIIITPetDataset(args.data_root, "train", train_tfm,
                                     load_bbox=False, load_mask=False)
    n_val     = int(len(full_tv) * args.val_split)
    tr_ds, val_ds = random_split(
        full_tv, [len(full_tv)-n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    tr_ldr  = DataLoader(tr_ds,  args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=True,
                         drop_last=True)
    val_ldr = DataLoader(val_ds, args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # Three runs
    configs = [
        (0.0, "No Dropout (p=0.0)", "dropout_p0.0"),
        (0.2, "Dropout p=0.2",      "dropout_p0.2"),
        (0.5, "Dropout p=0.5",      "dropout_p0.5"),
    ]

    results = {}
    for p_val, label, run_name in configs:
        print(f"\n{'='*55}")
        print(f"  Section 2.2 — {label}")
        print(f"{'='*55}")
        hist = run_dropout_experiment(
            dropout_p=p_val, args=args,
            tr_loader=tr_ldr, val_loader=val_ldr,
            device=device, run_name=run_name,
        )
        results[label] = hist

    # Overlay plot logged to W&B
    make_overlay_plot(results, args.wandb_project)
    print("\n=== Section 2.2 complete ===")


if __name__ == "__main__":
    main()