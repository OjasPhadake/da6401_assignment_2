"""
wandb_2_5_bbox_confidence.py  —  W&B Report Section 2.5
=========================================================
Runs the trained Task-2 localizer on ≥10 test images.
Logs a W&B table with:
  - Annotated image (green GT box, red predicted box)
  - Confidence score (max Sigmoid output — proxy for box certainty)
  - IoU between prediction and ground truth
  - Failure case analysis column

Run:
    python wandb_2_5_bbox_confidence.py \
        --data_root  data \
        --loc_ckpt   checkpoints/task2_best.pth \
        --n_images   15 \
        --wandb_project da6401_a2_report

Confidence score rationale
--------------------------
VGG11Localizer outputs (xc, yc, w, h) via Sigmoid, each in (0,1).
Confidence = 1 - mean(|pred_norm - 0.5|) × 2
This is 1 when the model places the box dead-centre with moderate size
(maximum information) and 0 when any coordinate is pushed to 0 or 1
(saturated Sigmoid → uncertain gradient region).
A simpler and more informative confidence proxy uses the spatial
consistency of the box: confidence = min(w_norm, h_norm) / max(w_norm, h_norm)
combined with the centre distance from image centre.
We use the norm-based measure: conf = sigmoid_out.min().item() mapped to [0,1].
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
import matplotlib.patches as patches
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import wandb
from data.pets_dataset  import OxfordIIITPetDataset, get_transforms
from models.localization import VGG11Localizer


# ---------------------------------------------------------------------------
# IoU utility
# ---------------------------------------------------------------------------

def iou_xywh(b1, b2, eps=1e-6):
    """
    Compute IoU between two boxes in (xc, yc, w, h) pixel format.
    b1, b2: numpy arrays of shape (4,)
    """
    x1_1, y1_1 = b1[0]-b1[2]/2, b1[1]-b1[3]/2
    x2_1, y2_1 = b1[0]+b1[2]/2, b1[1]+b1[3]/2
    x1_2, y1_2 = b2[0]-b2[2]/2, b2[1]-b2[3]/2
    x2_2, y2_2 = b2[0]+b2[2]/2, b2[1]+b2[3]/2

    ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
    ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter    = max(0., ix2-ix1) * max(0., iy2-iy1)

    a1 = (x2_1-x1_1) * (y2_1-y1_1)
    a2 = (x2_2-x1_2) * (y2_2-y1_2)
    union = a1 + a2 - inter + eps
    return float(inter / union)


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def confidence_score(norm_box: np.ndarray) -> float:
    """
    norm_box: (xc_norm, yc_norm, w_norm, h_norm) each ∈ (0,1)

    Confidence is high when:
      - The model isn't saturating (all values away from 0 or 1)
      - The predicted size is reasonable (w and h not degenerate)

    conf = geometric_mean(w_norm, h_norm) * (1 - |xc - 0.5| - |yc - 0.5|)
    clamped to [0, 1].
    """
    xc, yc, w, h = norm_box
    size_conf   = np.sqrt(max(w, 0) * max(h, 0))      # 0 if degenerate
    centre_conf = 1.0 - abs(xc - 0.5) - abs(yc - 0.5) # 1 at image centre
    centre_conf = max(0.0, centre_conf)
    return float(min(1.0, size_conf * (1.0 + centre_conf) / 2.0))


# ---------------------------------------------------------------------------
# Draw boxes on PIL image
# ---------------------------------------------------------------------------

def draw_boxes(img_pil: Image.Image, gt_px, pred_px,
               image_size: int) -> np.ndarray:
    """
    Draw green GT box and red predicted box on the image.
    gt_px, pred_px: (xc, yc, w, h) in pixel coordinates.
    Returns numpy array (H, W, 3).
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(np.array(img_pil))

    def _add_box(box, colour, label):
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (x1, y1), box[2], box[3],
            linewidth=2.5, edgecolor=colour, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, max(y1-4, 0), label, color=colour,
                fontsize=7, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.4, pad=1))

    if gt_px is not None:
        _add_box(gt_px,   "lime",  "GT")
    _add_box(pred_px, "red",   "Pred")

    ax.set_xlim(0, image_size); ax.set_ylim(image_size, 0)
    ax.axis("off")
    plt.tight_layout(pad=0)

    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    arr = arr.reshape(h_px, w_px, 3)
    plt.close(fig)
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     default="data")
    p.add_argument("--loc_ckpt",      default=None,
                   help="Path to task2_best.pth")
    p.add_argument("--n_images",      type=int, default=15)
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
               name="section_2_5_bbox_confidence",
               config=vars(args))

    # ---- Load localizer --------------------------------------------------
    model = VGG11Localizer(image_size=args.image_size).to(device)
    if args.loc_ckpt and os.path.isfile(args.loc_ckpt):
        ckpt = torch.load(args.loc_ckpt, map_location=device)
        sd   = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded localizer from {args.loc_ckpt}")
    else:
        print("[WARN] No localizer checkpoint — using random weights")
    model.eval()

    # ---- Dataset (test split, only samples with XML annotations) ---------
    val_tfm = get_transforms("val", args.image_size)
    test_ds = OxfordIIITPetDataset(args.data_root, "test", val_tfm,
                                   load_bbox=True, load_mask=False)

    # Filter to samples that have valid GT bboxes
    annotated = [(i, stem, label)
                 for i, (stem, label) in enumerate(test_ds.samples)
                 if (test_ds.xmls_dir / f"{stem}.xml").exists()]
    random.shuffle(annotated)
    chosen = annotated[:max(args.n_images, 10)]
    print(f"Found {len(annotated)} annotated test images, using {len(chosen)}")

    # ---- W&B table -------------------------------------------------------
    tbl_cols = ["Image", "Class", "Confidence", "IoU",
                "GT Box (px)", "Pred Box (px)", "Failure?", "Failure Reason"]
    tbl = wandb.Table(columns=tbl_cols)

    all_ious = []
    failure_rows = []

    for idx, stem, label in chosen:
        sample    = test_ds[idx]
        img_tensor = sample["image"].unsqueeze(0).to(device)
        bbox_norm  = sample["bbox"]   # [4] normalised GT (may be sentinel)
        class_name = OxfordIIITPetDataset.CLASS_NAMES[label]

        has_gt = (bbox_norm >= 0).all().item()
        gt_px  = (bbox_norm.numpy() * args.image_size
                  if has_gt else None)

        # Load original PIL for rendering
        img_pil = Image.open(test_ds.images_dir/f"{stem}.jpg").convert("RGB")
        img_pil = img_pil.resize((args.image_size, args.image_size))

        # Get raw normalised prediction (before pixel scaling) for confidence
        with torch.no_grad():
            pred_px_t  = model(img_tensor)          # [1,4] pixels
            # Get norm boxes by dividing back
            norm_pred  = (pred_px_t / args.image_size).squeeze(0).cpu().numpy()
        pred_px = pred_px_t.squeeze(0).cpu().numpy()

        conf = confidence_score(norm_pred)
        iou  = iou_xywh(pred_px, gt_px) if has_gt else float("nan")

        if has_gt:
            all_ious.append(iou)

        # Determine failure type
        is_failure    = False
        failure_reason = "—"
        if has_gt:
            if iou < 0.3 and conf > 0.5:
                is_failure     = True
                failure_reason = "High conf, low IoU — likely scale/pose mismatch"
            elif iou < 0.1:
                is_failure     = True
                failure_reason = "Very low IoU — possible occlusion or background"

        # Render annotated image
        img_arr = draw_boxes(img_pil, gt_px, pred_px, args.image_size)
        wb_img  = wandb.Image(img_arr,
                              caption=f"{class_name} | IoU={iou:.3f} "
                                      f"Conf={conf:.3f}")

        # Format box strings
        def fmt(box): return (f"xc={box[0]:.1f} yc={box[1]:.1f} "
                              f"w={box[2]:.1f} h={box[3]:.1f}")
        gt_str   = fmt(gt_px)   if has_gt   else "N/A"
        pred_str = fmt(pred_px)

        row = [wb_img, class_name, round(conf, 4),
               round(iou, 4) if has_gt else "N/A",
               gt_str, pred_str,
               "✗" if is_failure else "✓", failure_reason]
        tbl.add_data(*row)

        if is_failure:
            failure_rows.append((stem, class_name, conf, iou, failure_reason))

        print(f"  {stem:<35s}  class={class_name:<30s}  "
              f"conf={conf:.3f}  IoU={iou:.3f}  "
              f"{'FAIL' if is_failure else 'ok'}")

    wandb.log({"bbox_predictions_table": tbl})

    # ---- IoU distribution plot -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_ious, bins=20, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(all_ious), color="red", linestyle="--",
               label=f"Mean IoU = {np.mean(all_ious):.3f}")
    ax.axvline(0.5, color="green", linestyle=":", label="IoU=0.5 threshold")
    ax.set_xlabel("IoU"); ax.set_ylabel("Count")
    ax.set_title("IoU Distribution on Test Set")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    wandb.log({"iou_distribution": wandb.Image(fig)})
    plt.savefig("iou_distribution.png", dpi=120)
    plt.close()

    # ---- Summary metrics -------------------------------------------------
    if all_ious:
        iou_arr = np.array(all_ious)
        metrics_tbl = wandb.Table(columns=["Metric", "Value"])
        metrics_tbl.add_data("Mean IoU",   round(iou_arr.mean(), 4))
        metrics_tbl.add_data("Median IoU", round(float(np.median(iou_arr)), 4))
        metrics_tbl.add_data("Acc@IoU=0.5",
                             round(float((iou_arr >= 0.5).mean()), 4))
        metrics_tbl.add_data("Acc@IoU=0.75",
                             round(float((iou_arr >= 0.75).mean()), 4))
        metrics_tbl.add_data("Failure cases", len(failure_rows))
        wandb.log({"detection_summary": metrics_tbl})

        print(f"\nMean IoU       : {iou_arr.mean():.4f}")
        print(f"Acc@IoU=0.5    : {(iou_arr>=0.5).mean():.4f}")
        print(f"Acc@IoU=0.75   : {(iou_arr>=0.75).mean():.4f}")
        print(f"Failure cases  : {len(failure_rows)}")

    if failure_rows:
        print("\nFailure cases:")
        for stem, cls, conf, iou, reason in failure_rows:
            print(f"  {stem} | {cls} | conf={conf:.3f} | "
                  f"IoU={iou:.3f} | {reason}")

    wandb.finish()
    print("\n=== Section 2.5 complete ===")


if __name__ == "__main__":
    main()