"""
wandb_2_4_feature_maps.py  —  W&B Report Section 2.4
=====================================================
Pass a single dog image through the trained Task-1 classifier.
Extract and visualise feature maps from:
  • Layer 1  — stage1's Conv2d  (3→64,  full resolution, localised edges)
  • Layer 10 — stage5's 2nd Conv2d (512→512, 1/16 resolution, semantics)

Logs to W&B:
  - Original image
  - First-layer feature maps grid (64 channels)
  - Last-layer feature maps grid (64 of 512 channels)
  - Side-by-side comparison panel
  - Activation statistics table

Run:
    python wandb_2_4_feature_maps.py \
        --data_root data \
        --clf_ckpt  checkpoints/task1_best.pth \
        --wandb_project da6401_a2_report
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import wandb
from data.pets_dataset    import OxfordIIITPetDataset, get_transforms
from models.classification import VGG11Classifier


# ---------------------------------------------------------------------------
# Feature-map extraction via forward hooks
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Register hooks on two target layers and capture their outputs."""

    def __init__(self, model: nn.Module):
        self.features = {}
        self.handles  = []

        # Layer 1: first Conv2d inside stage1  (3 → 64, 224×224)
        # stage1 = Sequential(_conv_bn_relu(3,64))
        # _conv_bn_relu = Sequential(Conv2d, ReLU, BN)
        # so stage1[0][0] is the Conv2d
        layer1 = model.encoder.stage1[0][0]
        self.handles.append(
            layer1.register_forward_hook(self._make_hook("layer1"))
        )

        # Last conv layer before pooling: stage5's 2nd _conv_bn_relu block
        # stage5 = Sequential(_conv_bn_relu(512,512), _conv_bn_relu(512,512))
        # stage5[1][0] is the last Conv2d  (512→512, 14×14 for 224 input)
        layer_last = model.encoder.stage5[1][0]
        self.handles.append(
            layer_last.register_forward_hook(self._make_hook("layer_last"))
        )

    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach().cpu()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def normalise_fmap(fmap: np.ndarray) -> np.ndarray:
    """Min-max normalise a single feature map to [0, 1]."""
    mn, mx = fmap.min(), fmap.max()
    if mx - mn < 1e-8:
        return np.zeros_like(fmap)
    return (fmap - mn) / (mx - mn)


def make_grid_figure(fmaps: np.ndarray, title: str,
                     n_show: int = 64, ncols: int = 8) -> plt.Figure:
    """
    fmaps: [C, H, W] numpy array
    Show the first n_show channels in a grid.
    """
    n_show  = min(n_show, fmaps.shape[0])
    nrows   = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.5, nrows * 1.5))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx < n_show:
            fm = normalise_fmap(fmaps[idx])
            ax.imshow(fm, cmap="viridis", interpolation="nearest")
            ax.set_title(f"ch {idx}", fontsize=5, pad=1)
        ax.axis("off")

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout(pad=0.3)
    return fig


def make_comparison_panel(img_np, fmaps1, fmaps_last,
                           class_name: str) -> plt.Figure:
    """Side-by-side: original | mean of first-layer | mean of last-layer."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title(f"Input Image\n({class_name})", fontsize=11)
    axes[0].axis("off")

    # Mean activation across channels (shows overall response pattern)
    mean1    = normalise_fmap(fmaps1.mean(axis=0))
    axes[1].imshow(mean1, cmap="hot", interpolation="bilinear")
    axes[1].set_title("Layer 1 (stage1 Conv: 3→64)\n"
                      "Mean activation — localised edges & colours",
                      fontsize=9)
    axes[1].axis("off")

    mean_last = normalise_fmap(fmaps_last.mean(axis=0))
    axes[2].imshow(mean_last, cmap="hot", interpolation="bilinear")
    axes[2].set_title("Last Conv (stage5[1]: 512→512)\n"
                      "Mean activation — high-level semantics",
                      fontsize=9)
    axes[2].axis("off")

    plt.suptitle("Feature Map Comparison: First vs Last Conv Layer",
                 fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     default="data")
    p.add_argument("--clf_ckpt",      default=None,
                   help="Path to task1_best.pth")
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

    wandb.init(project=args.wandb_project, name="section_2_4_feature_maps",
               config=vars(args))

    # ---- Load model -------------------------------------------------------
    model = VGG11Classifier(num_classes=37, dropout_p=0.5).to(device)
    if args.clf_ckpt and os.path.isfile(args.clf_ckpt):
        ckpt = torch.load(args.clf_ckpt, map_location=device)
        sd   = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded classifier from {args.clf_ckpt}")
    else:
        print("[WARN] No checkpoint loaded — using random weights")
    model.eval()

    # ---- Find a dog image from the test split ----------------------------
    val_tfm  = get_transforms("val", args.image_size)
    test_ds  = OxfordIIITPetDataset(args.data_root, "test", val_tfm,
                                    load_bbox=False, load_mask=False)

    # Dog classes are index 12-36 (0-indexed) in CLASS_NAMES
    dog_indices = [i for i, (stem, label) in enumerate(test_ds.samples)
                   if label >= 12]
    random.shuffle(dog_indices)

    chosen_idx   = dog_indices[0] if dog_indices else 0
    sample       = test_ds[chosen_idx]
    img_tensor   = sample["image"].unsqueeze(0).to(device)  # [1,3,H,W]
    true_label   = sample["label"]
    class_name   = OxfordIIITPetDataset.CLASS_NAMES[true_label]
    print(f"Chosen sample: idx={chosen_idx}  class={class_name}")

    # Load original PIL image for display (un-normalised)
    stem = test_ds.samples[chosen_idx][0]
    img_pil = Image.open(test_ds.images_dir / f"{stem}.jpg").convert("RGB")
    img_pil = img_pil.resize((args.image_size, args.image_size))
    img_np  = np.array(img_pil)

    # ---- Register hooks and run forward pass ----------------------------
    extractor = FeatureExtractor(model)
    with torch.no_grad():
        logits = model(img_tensor)
    extractor.remove()

    pred_label = logits.argmax(1).item()
    pred_name  = OxfordIIITPetDataset.CLASS_NAMES[pred_label]
    pred_conf  = torch.softmax(logits, dim=1)[0, pred_label].item()
    print(f"Prediction: {pred_name}  (conf={pred_conf:.3f})  "
          f"GT: {class_name}")

    # fmaps: [C, H, W]
    fmaps1    = extractor.features["layer1"][0].numpy()     # [64, 224, 224]
    fmaps_last= extractor.features["layer_last"][0].numpy() # [512, 14, 14]

    print(f"Layer 1 fmap shape: {fmaps1.shape}")
    print(f"Last  layer shape : {fmaps_last.shape}")

    # ---- Log original image ---------------------------------------------
    wandb.log({"input_image": wandb.Image(img_pil,
               caption=f"GT: {class_name} | Pred: {pred_name} "
                       f"(conf={pred_conf:.3f})")})

    # ---- Layer 1 grid (64 channels) -------------------------------------
    fig1 = make_grid_figure(fmaps1, "Layer 1 Feature Maps — stage1 Conv (3→64)",
                             n_show=64, ncols=8)
    wandb.log({"layer1_feature_maps": wandb.Image(fig1)})
    fig1.savefig("feat_layer1_grid.png", dpi=100, bbox_inches="tight")
    plt.close(fig1)
    print("Layer 1 grid saved")

    # ---- Last conv grid (first 64 of 512 channels) ----------------------
    fig_last = make_grid_figure(fmaps_last,
                                "Last Conv Feature Maps — stage5[1] (512→512)",
                                n_show=64, ncols=8)
    wandb.log({"last_layer_feature_maps": wandb.Image(fig_last)})
    fig_last.savefig("feat_last_grid.png", dpi=100, bbox_inches="tight")
    plt.close(fig_last)
    print("Last conv grid saved")

    # ---- Side-by-side comparison panel ----------------------------------
    fig_cmp = make_comparison_panel(img_np, fmaps1, fmaps_last, class_name)
    wandb.log({"feature_map_comparison": wandb.Image(fig_cmp)})
    fig_cmp.savefig("feat_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig_cmp)

    # ---- Top-5 most activated channels per layer ------------------------
    # Rank channels by mean absolute activation
    layer1_rank = np.argsort(-fmaps1.mean(axis=(1,2)))[:5]
    last_rank   = np.argsort(-fmaps_last.mean(axis=(1,2)))[:5]

    fig_top, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ch in enumerate(layer1_rank):
        ax = axes[0, i]
        ax.imshow(normalise_fmap(fmaps1[ch]), cmap="viridis")
        ax.set_title(f"L1 ch{ch}\nμ={fmaps1[ch].mean():.2f}", fontsize=8)
        ax.axis("off")
    for i, ch in enumerate(last_rank):
        ax = axes[1, i]
        ax.imshow(normalise_fmap(fmaps_last[ch]), cmap="plasma")
        ax.set_title(f"Last ch{ch}\nμ={fmaps_last[ch].mean():.2f}", fontsize=8)
        ax.axis("off")
    axes[0, 0].set_ylabel("Layer 1\n(edges/colour)", fontsize=9)
    axes[1, 0].set_ylabel("Last conv\n(semantics)", fontsize=9)
    plt.suptitle("Top-5 Most Activated Channels per Layer", fontsize=12)
    plt.tight_layout()
    wandb.log({"top5_channels": wandb.Image(fig_top)})
    fig_top.savefig("feat_top5.png", dpi=120, bbox_inches="tight")
    plt.close(fig_top)

    # ---- Statistics table -----------------------------------------------
    tbl = wandb.Table(columns=["Layer", "Shape", "Mean Act",
                                "Std Act", "% Dead (≤0)",
                                "Sparsity (>mean)"])
    for name, fmaps in [("Layer 1 (stage1 Conv 3→64)", fmaps1),
                         ("Last  (stage5[1] Conv 512→512)", fmaps_last)]:
        flat = fmaps.flatten()
        tbl.add_data(
            name,
            str(fmaps.shape),
            round(float(flat.mean()),  4),
            round(float(flat.std()),   4),
            round(float((flat <= 0).mean() * 100), 2),
            round(float((flat > flat.mean()).mean() * 100), 2),
        )
    wandb.log({"activation_statistics": tbl})

    print(f"\nLayer 1    — mean={fmaps1.flatten().mean():.4f}  "
          f"std={fmaps1.flatten().std():.4f}")
    print(f"Last layer — mean={fmaps_last.flatten().mean():.4f}  "
          f"std={fmaps_last.flatten().std():.4f}")

    wandb.finish()
    print("\n=== Section 2.4 complete ===")


if __name__ == "__main__":
    main()