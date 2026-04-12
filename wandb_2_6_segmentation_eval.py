"""
wandb_2_6_segmentation_eval.py  —  W&B Report Section 2.6
==========================================================
For the trained VGG11UNet:
  1. Logs 5 sample images: Original | GT Trimap | Predicted Trimap
  2. Tracks Pixel Accuracy vs Dice Score across a validation run
  3. Plots the imbalance-driven divergence between the two metrics

Run:
    python wandb_2_6_segmentation_eval.py \
        --data_root data \
        --unet_ckpt checkpoints/task3_best.pth \
        --wandb_project da6401_a2_report
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import wandb
from data.pets_dataset   import OxfordIIITPetDataset, get_transforms
from models.segmentation  import VGG11UNet


# ---------------------------------------------------------------------------
# Colour palette for trimap classes
# ---------------------------------------------------------------------------
# class 0 = foreground (pet body)   → green
# class 1 = background              → blue
# class 2 = unknown/boundary        → yellow

PALETTE = np.array([
    [0,   200,   0],    # class 0: foreground — green
    [30,   30, 200],    # class 1: background — blue
    [220, 220,   0],    # class 2: unknown    — yellow
], dtype=np.uint8)


def apply_palette(mask_2d: np.ndarray) -> np.ndarray:
    """Convert (H,W) class-index mask → (H,W,3) RGB array."""
    rgb = PALETTE[mask_2d.clip(0, 2)]
    return rgb


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred == gt).mean())


def dice_score(pred: np.ndarray, gt: np.ndarray,
               num_classes: int = 3, eps: float = 1e-6) -> float:
    dices = []
    for c in range(num_classes):
        p = (pred == c).astype(float)
        t = (gt   == c).astype(float)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dices.append((2 * inter + eps) / (denom + eps))
    return float(np.mean(dices))


def class_pixel_fraction(gt: np.ndarray, num_classes: int = 3) -> dict:
    total = gt.size
    return {c: float((gt == c).sum() / total) for c in range(num_classes)}


# ---------------------------------------------------------------------------
# Visualise one sample: original | GT | predicted
# ---------------------------------------------------------------------------

def make_triplet_panel(img_pil: Image.Image,
                       gt_mask: np.ndarray,
                       pred_mask: np.ndarray,
                       title: str,
                       pa: float, dice: float) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(np.array(img_pil))
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(apply_palette(gt_mask))
    axes[1].set_title("Ground Truth Trimap\n"
                      "■ Green=FG  ■ Blue=BG  ■ Yellow=Unknown",
                      fontsize=8)
    axes[1].axis("off")

    axes[2].imshow(apply_palette(pred_mask))
    axes[2].set_title(f"Predicted Trimap\n"
                      f"Pixel Acc={pa:.3f}  Dice={dice:.3f}", fontsize=8)
    axes[2].axis("off")

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     default="data")
    p.add_argument("--unet_ckpt",     default=None,
                   help="Path to task3_best.pth")
    p.add_argument("--n_samples",     type=int, default=5)
    p.add_argument("--image_size",    type=int, default=224)
    p.add_argument("--wandb_project", default="da6401_a2_report")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(project=args.wandb_project,
               name="section_2_6_seg_eval",
               config=vars(args))

    # ---- Load U-Net -------------------------------------------------------
    model = VGG11UNet(num_classes=3).to(device)
    if args.unet_ckpt and os.path.isfile(args.unet_ckpt):
        ckpt = torch.load(args.unet_ckpt, map_location=device)
        sd   = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded U-Net from {args.unet_ckpt}")
    else:
        print("[WARN] No checkpoint loaded — using random weights")
    model.eval()

    # ---- Dataset ----------------------------------------------------------
    val_tfm = get_transforms("val", args.image_size)
    test_ds = OxfordIIITPetDataset(args.data_root, "test", val_tfm,
                                   load_bbox=False, load_mask=True)

    # Filter to samples with valid masks
    masked_indices = [i for i, (stem, _) in enumerate(test_ds.samples)
                      if (test_ds.trimaps_dir/f"{stem}.png").exists()]
    random.shuffle(masked_indices)
    sample_indices = masked_indices[:args.n_samples]

    # ---- Collect per-class pixel fractions across full val set -----------
    print("Computing pixel class distribution over val set …")
    all_pa   = []
    all_dice = []
    class_fracs = {0: [], 1: [], 2: []}

    all_val = masked_indices  # use all masked test samples for distribution
    for i in all_val[:200]:   # cap at 200 for speed
        sample    = test_ds[i]
        img_t     = sample["image"].unsqueeze(0).to(device)
        gt_mask_t = sample["mask"]
        if gt_mask_t.numel() == 0:
            continue
        gt_np = gt_mask_t.numpy()

        with torch.no_grad():
            logits = model(img_t)
        if logits.shape[2:] != gt_mask_t.shape:
            logits = F.interpolate(logits, size=gt_mask_t.shape,
                                   mode="bilinear", align_corners=False)
        pred_np = logits.argmax(1).squeeze(0).cpu().numpy()

        all_pa.append(pixel_accuracy(pred_np, gt_np))
        all_dice.append(dice_score(pred_np, gt_np))
        for c, v in class_pixel_fraction(gt_np).items():
            class_fracs[c].append(v)

    avg_pa   = float(np.mean(all_pa))
    avg_dice = float(np.mean(all_dice))
    print(f"Val  Pixel Acc={avg_pa:.4f}  Dice={avg_dice:.4f}")
    print(f"Avg pixel fractions — "
          f"FG(0)={np.mean(class_fracs[0]):.3f}  "
          f"BG(1)={np.mean(class_fracs[1]):.3f}  "
          f"Unk(2)={np.mean(class_fracs[2]):.3f}")

    # ---- Log 5 sample triplet panels ------------------------------------
    sample_table = wandb.Table(columns=["Image (Orig|GT|Pred)",
                                         "Class", "Pixel Acc", "Dice",
                                         "BG fraction", "FG fraction"])

    for idx in sample_indices:
        sample    = test_ds[idx]
        stem, lbl = test_ds.samples[idx]
        class_name = OxfordIIITPetDataset.CLASS_NAMES[lbl]

        img_t     = sample["image"].unsqueeze(0).to(device)
        gt_mask_t = sample["mask"]
        if gt_mask_t.numel() == 0:
            continue
        gt_np = gt_mask_t.numpy()

        with torch.no_grad():
            logits = model(img_t)
        if logits.shape[2:] != gt_mask_t.shape:
            logits = F.interpolate(logits, size=gt_mask_t.shape,
                                   mode="bilinear", align_corners=False)
        pred_np = logits.argmax(1).squeeze(0).cpu().numpy()

        pa   = pixel_accuracy(pred_np, gt_np)
        dice = dice_score(pred_np, gt_np)
        fracs = class_pixel_fraction(gt_np)

        # Load un-normalised image
        img_pil = Image.open(test_ds.images_dir/f"{stem}.jpg").convert("RGB")
        img_pil = img_pil.resize((args.image_size, args.image_size))

        fig = make_triplet_panel(img_pil, gt_np, pred_np,
                                  f"{class_name}", pa, dice)
        wb_img = wandb.Image(fig,
                             caption=f"{class_name} | PA={pa:.3f} Dice={dice:.3f}")
        plt.close(fig)

        sample_table.add_data(
            wb_img, class_name,
            round(pa,   4), round(dice, 4),
            round(fracs[1], 3), round(fracs[0], 3)
        )
        print(f"  {stem:<35}  PA={pa:.3f}  Dice={dice:.3f}  "
              f"BG={fracs[1]:.3f}  FG={fracs[0]:.3f}")

    wandb.log({"segmentation_samples": sample_table})

    # ---- Metric divergence plot -----------------------------------------
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: Pixel Acc vs Dice coloured by BG fraction
    bg_fracs = [np.mean(class_fracs[1])] * len(all_pa)   # simplification
    bg_arr   = np.array([class_pixel_fraction(
                    test_ds[all_val[i]]["mask"].numpy())[1]
                for i in range(min(len(all_pa), len(all_val)))])
    sc = axes[0].scatter(all_pa[:len(bg_arr)],
                         all_dice[:len(bg_arr)],
                         c=bg_arr, cmap="RdYlGn_r",
                         alpha=0.7, s=20)
    plt.colorbar(sc, ax=axes[0], label="BG pixel fraction")
    axes[0].plot([0,1],[0,1], "k--", alpha=0.4, label="PA=Dice line")
    axes[0].set_xlabel("Pixel Accuracy")
    axes[0].set_ylabel("Dice Score")
    axes[0].set_title("Pixel Accuracy vs Dice Score\n"
                       "(colour = background pixel fraction)", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Bar: avg class pixel fractions
    labels_c = ["Foreground (FG)", "Background (BG)", "Unknown (Unk)"]
    avgs     = [np.mean(class_fracs[c]) for c in range(3)]
    colours  = ["#27ae60", "#3498db", "#f39c12"]
    axes[1].bar(labels_c, avgs, color=colours, alpha=0.85)
    for i, v in enumerate(avgs):
        axes[1].text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=10)
    axes[1].set_ylabel("Average pixel fraction")
    axes[1].set_title("Class Imbalance in Oxford-IIIT Pet Trimaps\n"
                       "(why Pixel Acc is inflated)", fontsize=10)
    axes[1].set_ylim(0, max(avgs)*1.2)
    axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle("Section 2.6 — Dice vs Pixel Accuracy: Imbalance Effect",
                 fontsize=12)
    plt.tight_layout()
    wandb.log({"metric_divergence": wandb.Image(fig2)})
    fig2.savefig("dice_vs_pixacc.png", dpi=120)
    plt.close(fig2)

    # ---- Explanation table -----------------------------------------------
    # Mathematical explanation of metric divergence
    bg_frac = float(np.mean([np.mean(class_fracs[1])]))
    fg_frac = float(np.mean([np.mean(class_fracs[0])]))

    exp_tbl = wandb.Table(columns=["Scenario",
                                    "All-BG model Pixel Acc",
                                    "All-BG model Dice (FG)",
                                    "Why?"])
    exp_tbl.add_data(
        "Predict ALL pixels as Background",
        f"{bg_frac:.3f}  ({bg_frac*100:.1f}% correct)",
        "~0.000  (zero TP for FG class)",
        "Pixel Acc rewards majority class; Dice penalises missing the minority"
    )
    wandb.log({"imbalance_explanation": exp_tbl})

    # ---- Aggregate metric table ------------------------------------------
    agg_tbl = wandb.Table(columns=["Metric", "Value", "Interpretation"])
    agg_tbl.add_data("Mean Pixel Accuracy", round(avg_pa,   4),
                     "Inflated by BG dominance")
    agg_tbl.add_data("Mean Dice Score",     round(avg_dice, 4),
                     "Robust to class imbalance")
    agg_tbl.add_data("Gap (PA - Dice)",
                     round(avg_pa - avg_dice, 4),
                     "Larger gap = more imbalance")
    agg_tbl.add_data("Avg BG fraction",
                     round(float(np.mean(class_fracs[1])), 3),
                     "Fraction of pixels that are background")
    agg_tbl.add_data("Avg FG fraction",
                     round(float(np.mean(class_fracs[0])), 3),
                     "Fraction of pixels that are foreground")
    wandb.log({"aggregate_metrics": agg_tbl})

    wandb.finish()
    print("\n=== Section 2.6 complete ===")
    print(f"Pixel Acc={avg_pa:.4f}  Dice={avg_dice:.4f}  "
          f"Gap={avg_pa-avg_dice:.4f}")


if __name__ == "__main__":
    main()