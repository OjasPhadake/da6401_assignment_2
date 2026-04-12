"""
wandb_2_3_transfer_learning.py  —  W&B Report Section 2.3
==========================================================
Runs all three transfer-learning strategies for semantic segmentation:
  Strategy A — Strict Feature Extractor (freeze entire encoder)
  Strategy B — Partial Fine-Tune       (freeze stages 1,2,3; unfreeze 4,5)
  Strategy C — Full Fine-Tune          (unfreeze entire encoder)

All three runs are logged to the same W&B project so you can overlay
their validation Dice / Loss curves directly in the W&B UI.

Run from da6401_assignment_2/:
    python wandb_2_3_transfer_learning.py \
        --data_root data \
        --classifier_ckpt checkpoints/task1_best.pth \
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
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import wandb
from data.pets_dataset   import OxfordIIITPetDataset, get_transforms
from models.segmentation import VGG11UNet, SegmentationLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def seg_collate(batch):
    valid = [s for s in batch if s["mask"].numel() > 0]
    if not valid:
        return None
    images = torch.stack([s["image"] for s in valid])
    masks  = torch.stack([s["mask"]  for s in valid])
    return {"image": images, "mask": masks}


@torch.no_grad()
def compute_dice(logits, target, nc=3, eps=1e-6):
    preds = logits.argmax(1)
    dices = []
    for c in range(nc):
        p = (preds==c).float(); t = (target==c).float()
        dices.append(((2*(p*t).sum()+eps)/(p.sum()+t.sum()+eps)).item())
    return float(np.mean(dices))


@torch.no_grad()
def compute_pixel_acc(logits, target):
    return (logits.argmax(1) == target).float().mean().item()


def train_epoch(model, loader, crit, opt, device, epoch):
    model.train()
    tot_loss = tot_dice = tot_acc = n = 0
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        if batch is None:
            continue
        imgs  = batch["image"].to(device, non_blocking=True)
        masks = batch["mask" ].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(imgs)
        if logits.shape[2:] != masks.shape[1:]:
            masks = F.interpolate(masks.unsqueeze(1).float(),
                                  size=logits.shape[2:],
                                  mode="nearest").squeeze(1).long()
        loss = crit(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        bs        = imgs.size(0)
        tot_loss += loss.item() * bs
        tot_dice += compute_dice(logits.detach(), masks.detach()) * bs
        tot_acc  += compute_pixel_acc(logits.detach(), masks.detach()) * bs
        n        += bs

    return {"loss": tot_loss/n, "dice": tot_dice/n,
            "pixel_acc": tot_acc/n, "time": time.time()-t0}


@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    tot_loss = tot_dice = tot_acc = n = 0
    for batch in loader:
        if batch is None:
            continue
        imgs  = batch["image"].to(device, non_blocking=True)
        masks = batch["mask" ].to(device, non_blocking=True)
        logits = model(imgs)
        if logits.shape[2:] != masks.shape[1:]:
            masks = F.interpolate(masks.unsqueeze(1).float(),
                                  size=logits.shape[2:],
                                  mode="nearest").squeeze(1).long()
        bs        = imgs.size(0)
        tot_loss += crit(logits, masks).item() * bs
        tot_dice += compute_dice(logits, masks) * bs
        tot_acc  += compute_pixel_acc(logits, masks) * bs
        n        += bs

    return {"loss": tot_loss/n, "dice": tot_dice/n, "pixel_acc": tot_acc/n}


# ---------------------------------------------------------------------------
# Single strategy run
# ---------------------------------------------------------------------------

def run_strategy(
    strategy_name: str,
    freeze_encoder: bool,
    freeze_stages:  list,
    args,
    tr_loader, val_loader,
    device,
    run_name: str,
):
    set_seed(args.seed)

    model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=0.5,
                      freeze_encoder=freeze_encoder,
                      freeze_stages=freeze_stages).to(device)

    # Load pretrained backbone from Task-1 checkpoint
    if args.classifier_ckpt and os.path.isfile(args.classifier_ckpt):
        model.load_classifier_backbone(args.classifier_ckpt)
        print(f"  [{run_name}] Loaded backbone from {args.classifier_ckpt}")
    else:
        print(f"  [{run_name}] No backbone checkpoint — random init")

    # Count trainable params
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group="section_2_3_transfer",
        config={
            "strategy":      strategy_name,
            "freeze_encoder": freeze_encoder,
            "freeze_stages":  freeze_stages,
            "total_params":   total_p,
            "trainable_params": trainable_p,
            "lr_decoder":    args.lr_decoder,
            "lr_backbone":   args.lr_backbone,
            "epochs":        args.epochs,
        },
        reinit=True,
    )

    crit = SegmentationLoss(num_classes=3, dice_weight=1.0)

    # Differential LR: backbone (if unfrozen) gets smaller LR
    decoder_params  = (list(model.bridge.parameters())   +
                       list(model.up5.parameters())      +
                       list(model.up4.parameters())      +
                       list(model.up3.parameters())      +
                       list(model.up2.parameters())      +
                       list(model.up1.parameters())      +
                       list(model.seg_head.parameters()))
    backbone_params = [p for p in model.encoder.parameters()
                       if p.requires_grad]

    param_groups = [{"params": decoder_params, "lr": args.lr_decoder}]
    if backbone_params:
        param_groups.append({"params": backbone_params,
                              "lr": args.lr_backbone})

    opt   = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        tr  = train_epoch(model, tr_loader,  crit, opt, device, epoch)
        val = eval_epoch(model, val_loader, crit, device)
        sched.step()

        wandb.log({
            "epoch":           epoch,
            "train/loss":      tr["loss"],
            "train/dice":      tr["dice"],
            "train/pixel_acc": tr["pixel_acc"],
            "val/loss":        val["loss"],
            "val/dice":        val["dice"],
            "val/pixel_acc":   val["pixel_acc"],
            "time_per_epoch":  tr["time"],
            "lr":              sched.get_last_lr()[0],
        })

        history.append({
            "epoch":      epoch,
            "tr_loss":    tr["loss"],   "val_loss":    val["loss"],
            "tr_dice":    tr["dice"],   "val_dice":    val["dice"],
            "tr_acc":     tr["pixel_acc"], "val_acc":  val["pixel_acc"],
            "epoch_time": tr["time"],
        })

        print(f"  [{run_name}] E{epoch:02d}  "
              f"tr_loss={tr['loss']:.4f} tr_dice={tr['dice']:.4f} | "
              f"val_loss={val['loss']:.4f} val_dice={val['dice']:.4f}  "
              f"({tr['time']:.1f}s)")

    wandb.finish()
    return history


# ---------------------------------------------------------------------------
# Overlay comparison plot
# ---------------------------------------------------------------------------

def make_comparison_plot(results: dict, project: str):
    colours = {
        "Strict Feature Extractor": "#e74c3c",
        "Partial Fine-Tune":        "#f39c12",
        "Full Fine-Tune":           "#27ae60",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (0,0) Validation Dice
    ax = axes[0][0]
    for label, hist in results.items():
        ax.plot([h["epoch"]    for h in hist],
                [h["val_dice"] for h in hist],
                "-o", color=colours[label], label=label,
                linewidth=2, markersize=4)
    ax.set_title("Validation Dice Score", fontsize=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (0,1) Validation Loss
    ax = axes[0][1]
    for label, hist in results.items():
        ax.plot([h["epoch"]    for h in hist],
                [h["val_loss"] for h in hist],
                "--", color=colours[label], label=label, linewidth=2)
    ax.set_title("Validation Loss", fontsize=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (1,0) Pixel Accuracy
    ax = axes[1][0]
    for label, hist in results.items():
        ax.plot([h["epoch"]   for h in hist],
                [h["val_acc"] for h in hist],
                "-s", color=colours[label], label=label,
                linewidth=2, markersize=4)
    ax.set_title("Validation Pixel Accuracy", fontsize=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pixel Acc")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (1,1) Time per epoch (bar chart)
    ax = axes[1][1]
    labels_list = list(results.keys())
    avg_times   = [np.mean([h["epoch_time"] for h in results[l]])
                   for l in labels_list]
    bars = ax.bar(range(len(labels_list)), avg_times,
                  color=[colours[l] for l in labels_list], alpha=0.85)
    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels([l.replace(" ","\n") for l in labels_list], fontsize=9)
    ax.set_title("Avg Time per Epoch (seconds)", fontsize=12)
    ax.set_ylabel("Time (s)")
    for bar, t in zip(bars, avg_times):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{t:.1f}s", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Section 2.3 — Transfer Learning Strategy Comparison",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    wandb.init(project=project, name="section_2_3_summary", reinit=True)
    wandb.log({"transfer_comparison": wandb.Image(fig)})

    # Summary table
    tbl = wandb.Table(columns=["Strategy", "Best Val Dice", "Best Epoch",
                                "Final Val Acc", "Avg Time/Epoch (s)"])
    for label, hist in results.items():
        best     = max(hist, key=lambda h: h["val_dice"])
        avg_time = np.mean([h["epoch_time"] for h in hist])
        tbl.add_data(label,
                     round(best["val_dice"],     4),
                     best["epoch"],
                     round(hist[-1]["val_acc"],  4),
                     round(avg_time, 1))
    wandb.log({"transfer_summary_table": tbl})
    wandb.finish()

    plt.savefig("transfer_learning_comparison.png", dpi=120)
    plt.close()
    print("Comparison plot saved → transfer_learning_comparison.png")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       default="data")
    p.add_argument("--classifier_ckpt", default=None,
                   help="Path to task1_best.pth for encoder init.")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr_decoder",      type=float, default=1e-3)
    p.add_argument("--lr_backbone",     type=float, default=1e-4)
    p.add_argument("--val_split",       type=float, default=0.1)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--wandb_project",   default="da6401_a2_report")
    return p.parse_args()


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets (shared across all three strategies)
    train_tfm = get_transforms("train", 224)
    val_tfm   = get_transforms("val",   224)
    full_tv   = OxfordIIITPetDataset(args.data_root, "train", train_tfm,
                                     load_bbox=False, load_mask=True)
    n_val     = int(len(full_tv) * args.val_split)
    tr_ds, val_ds = random_split(
        full_tv, [len(full_tv)-n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    tr_ldr  = DataLoader(tr_ds,  args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=True,
                         drop_last=True, collate_fn=seg_collate)
    val_ldr = DataLoader(val_ds, args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True,
                         collate_fn=seg_collate)

    # Three strategies
    strategies = [
        {
            "name":           "Strict Feature Extractor",
            "freeze_encoder": True,
            "freeze_stages":  None,
            "run_name":       "strict_feature_extractor",
        },
        {
            "name":           "Partial Fine-Tune",
            "freeze_encoder": False,
            "freeze_stages":  [1, 2, 3],   # freeze early; unfreeze stages 4,5
            "run_name":       "partial_finetune",
        },
        {
            "name":           "Full Fine-Tune",
            "freeze_encoder": False,
            "freeze_stages":  None,
            "run_name":       "full_finetune",
        },
    ]

    results = {}
    for s in strategies:
        print(f"\n{'='*60}")
        print(f"  Section 2.3 — {s['name']}")
        if s["freeze_encoder"]:
            print(f"  Frozen: entire encoder")
        elif s["freeze_stages"]:
            print(f"  Frozen: stages {s['freeze_stages']}  "
                  f"Trainable: stages 4,5 + decoder")
        else:
            print(f"  Frozen: nothing (full fine-tune)")
        print(f"{'='*60}")

        hist = run_strategy(
            strategy_name  = s["name"],
            freeze_encoder = s["freeze_encoder"],
            freeze_stages  = s["freeze_stages"],
            args           = args,
            tr_loader      = tr_ldr,
            val_loader     = val_ldr,
            device         = device,
            run_name       = s["run_name"],
        )
        results[s["name"]] = hist

    # Overlay comparison plot
    make_comparison_plot(results, args.wandb_project)

    print("\n=== Section 2.3 complete ===")
    for name, hist in results.items():
        best = max(hist, key=lambda h: h["val_dice"])
        print(f"  {name:30s}  best Dice={best['val_dice']:.4f}  "
              f"at epoch {best['epoch']}")


if __name__ == "__main__":
    main()