"""
train_multitask.py  —  Task 4: Unified Multi-Task Learning Pipeline

Run from da6401_assignment_2/ root:

    # Standard run (loads all three task checkpoints):
    python train_multitask.py \
        --classifier_ckpt checkpoints/task1_best.pth \
        --localizer_ckpt  checkpoints/task2_best.pth \
        --unet_ckpt       checkpoints/task3_best.pth

    # Resume:
    python train_multitask.py \
        --classifier_ckpt checkpoints/task1_best.pth \
        --localizer_ckpt  checkpoints/task2_best.pth \
        --unet_ckpt       checkpoints/task3_best.pth \
        --resume          checkpoints/task4_best.pth

Expected directory layout
--------------------------
da6401_assignment_2/
├── data/
│   ├── annotations/  (trainval.txt, test.txt, trimaps/, xmls/)
│   ├── images/
│   └── pets_dataset.py
├── models/
│   ├── layers.py  classification.py  localization.py
│   ├── segmentation.py  vgg11.py  multitask.py
├── losses/
│   ├── __init__.py  iou_loss.py
└── train_multitask.py   ← this file
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

from data.pets_dataset   import OxfordIIITPetDataset, get_transforms
from models.multitask    import MultiTaskPerceptionModel
from models.segmentation import SegmentationLoss
from losses.iou_loss     import IoULoss

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
    print("[WARN] wandb not installed — logging to console only.")

try:
    from sklearn.metrics import f1_score as sk_f1
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ===========================================================================
# CONFIG
# ===========================================================================
DEFAULT_DATA_ROOT      = os.path.join(ROOT, "data")
DEFAULT_CKPT_DIR       = os.path.join(ROOT, "checkpoints")
DEFAULT_EPOCHS         = 20
DEFAULT_BATCH_SIZE     = 16
DEFAULT_LR_HEADS       = 1e-3    # cls + loc + seg decoder
DEFAULT_LR_BACKBONE    = 1e-4    # shared encoder (10× smaller)
DEFAULT_WEIGHT_DECAY   = 1e-4
DEFAULT_DROPOUT_P      = 0.5
DEFAULT_IMAGE_SIZE     = 224
DEFAULT_VAL_SPLIT      = 0.1
DEFAULT_NUM_WORKERS    = 4
DEFAULT_SEED           = 42
DEFAULT_WANDB_PROJECT  = "da6401_a2_task4"
NUM_BREEDS             = 37
NUM_SEG_CLASSES        = 3
# Loss weights — balance the three task losses
W_CLS  = 1.0   # classification
W_LOC  = 1.0   # localization (IoU)
W_SEG  = 1.0   # segmentation (CE + Dice)
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
        description="DA6401 A2 Task 4 — Unified multi-task training",
    )
    # Paths
    p.add_argument("--data_root",        default=DEFAULT_DATA_ROOT)
    p.add_argument("--ckpt_dir",         default=DEFAULT_CKPT_DIR)
    p.add_argument("--classifier_ckpt",  default=None,
                   help="Path to task1_best.pth")
    p.add_argument("--localizer_ckpt",   default=None,
                   help="Path to task2_best.pth")
    p.add_argument("--unet_ckpt",        default=None,
                   help="Path to task3_best.pth")

    # Training
    p.add_argument("--epochs",           type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size",       type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr_heads",         type=float, default=DEFAULT_LR_HEADS)
    p.add_argument("--lr_backbone",      type=float, default=DEFAULT_LR_BACKBONE)
    p.add_argument("--weight_decay",     type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--dropout_p",        type=float, default=DEFAULT_DROPOUT_P)
    p.add_argument("--image_size",       type=int,   default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--val_split",        type=float, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--num_workers",      type=int,   default=DEFAULT_NUM_WORKERS)
    p.add_argument("--seed",             type=int,   default=DEFAULT_SEED)

    # Loss weights
    p.add_argument("--w_cls",  type=float, default=W_CLS,
                   help="Weight for classification loss.")
    p.add_argument("--w_loc",  type=float, default=W_LOC,
                   help="Weight for localization IoU loss.")
    p.add_argument("--w_seg",  type=float, default=W_SEG,
                   help="Weight for segmentation CE+Dice loss.")

    # W&B / misc
    p.add_argument("--wandb_project",    default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--run_name",         default=None)
    p.add_argument("--no_wandb",         action="store_true")
    p.add_argument("--resume",           default=None,
                   help="Resume from a multitask checkpoint.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Combined multi-task loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """Weighted sum of the three task losses.

    L_total = w_cls · L_CE_cls
            + w_loc · L_IoU
            + w_seg · (L_CE_seg + L_Dice)

    Loss weight justification
    -------------------------
    All three weights default to 1.0 so each task contributes equally to
    the shared encoder gradient.  If one task dominates (e.g. segmentation
    loss is 10× larger than classification loss in early epochs), the encoder
    will be biased toward that task's representation.  Equal weights keep the
    multi-task gradient balanced.  Weights can be tuned via CLI flags.

    Args:
        w_cls (float): Weight for classification CE loss.
        w_loc (float): Weight for localization IoU loss.
        w_seg (float): Weight for segmentation CE+Dice loss.
        num_seg_classes (int): Passed to SegmentationLoss.
    """

    def __init__(
        self,
        w_cls:           float = 1.0,
        w_loc:           float = 1.0,
        w_seg:           float = 1.0,
        num_seg_classes: int   = 3,
    ):
        super().__init__()
        self.w_cls = w_cls
        self.w_loc = w_loc
        self.w_seg = w_seg

        self.cls_loss = nn.CrossEntropyLoss()
        self.loc_loss = IoULoss(reduction="mean")
        self.seg_loss = SegmentationLoss(num_classes=num_seg_classes,
                                         dice_weight=1.0)

    def forward(
        self,
        outputs:  dict,
        labels:   torch.Tensor,          # [B] long
        bboxes:   torch.Tensor,          # [B, 4] pixel xywh  (-1 sentinel if missing)
        masks:    torch.Tensor,          # [B, H, W] long     (empty if missing)
        image_size: int = 224,
    ) -> dict:
        """
        Args:
            outputs:    dict from MultiTaskPerceptionModel.forward()
            labels:     ground-truth class labels [B]
            bboxes:     ground-truth bboxes in normalised [0,1] xywh;
                        samples with no annotation have bbox == -1.
            masks:      ground-truth trimap masks [B, H, W];
                        may be empty tensor if load_mask=False.
            image_size: used to scale bboxes from [0,1] → pixel space.

        Returns:
            dict with keys 'total', 'cls', 'loc', 'seg'.
        """
        loss_cls = torch.tensor(0.0, device=labels.device)
        loss_loc = torch.tensor(0.0, device=labels.device)
        loss_seg = torch.tensor(0.0, device=labels.device)

        # ---- Classification loss (all samples) ---------------------------
        loss_cls = self.cls_loss(outputs["classification"], labels)

        # ---- Localization loss (only annotated samples) ------------------
        # Samples with no XML annotation have bbox == -1 sentinel
        valid_bbox = (bboxes >= 0).all(dim=1)   # [B] bool
        if valid_bbox.any():
            pred_px = outputs["localization"][valid_bbox]    # [K, 4] pixels
            tgt_px  = bboxes[valid_bbox] * image_size        # [K, 4] pixels
            loss_loc = self.loc_loss(pred_px, tgt_px)

        # ---- Segmentation loss (only samples with mask) ------------------
        if masks.numel() > 0:
            seg_pred = outputs["segmentation"]               # [B, C, H, W]
            # Resize mask to match logit spatial size if needed
            if seg_pred.shape[2:] != masks.shape[1:]:
                masks = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=seg_pred.shape[2:],
                    mode="nearest",
                ).squeeze(1).long()
            loss_seg = self.seg_loss(seg_pred, masks)

        total = (self.w_cls * loss_cls
                 + self.w_loc * loss_loc
                 + self.w_seg * loss_seg)

        return {
            "total": total,
            "cls":   loss_cls,
            "loc":   loss_loc,
            "seg":   loss_seg,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_iou_batch(pred_px: torch.Tensor, tgt_px: torch.Tensor,
                      eps: float = 1e-6) -> float:
    px1 = pred_px[:,0] - pred_px[:,2]*0.5
    py1 = pred_px[:,1] - pred_px[:,3]*0.5
    px2 = pred_px[:,0] + pred_px[:,2]*0.5
    py2 = pred_px[:,1] + pred_px[:,3]*0.5
    tx1 = tgt_px[:,0]  - tgt_px[:,2]*0.5
    ty1 = tgt_px[:,1]  - tgt_px[:,3]*0.5
    tx2 = tgt_px[:,0]  + tgt_px[:,2]*0.5
    ty2 = tgt_px[:,1]  + tgt_px[:,3]*0.5
    iw  = torch.relu(torch.min(px2,tx2) - torch.max(px1,tx1))
    ih  = torch.relu(torch.min(py2,ty2) - torch.max(py1,ty1))
    inter = iw * ih
    union = (px2-px1).clamp(0)*(py2-py1).clamp(0) + \
            (tx2-tx1).clamp(0)*(ty2-ty1).clamp(0) - inter
    return (inter/(union+eps)).mean().item()


@torch.no_grad()
def compute_dice_batch(logits: torch.Tensor, target: torch.Tensor,
                       num_classes: int = 3, eps: float = 1e-6) -> float:
    preds = logits.argmax(1)
    dices = []
    for c in range(num_classes):
        p = (preds==c).float(); t = (target==c).float()
        inter = (p*t).sum()
        denom = p.sum() + t.sum()
        dices.append(((2*inter+eps)/(denom+eps)).item())
    return float(np.mean(dices))


# ---------------------------------------------------------------------------
# Collate — handles all three tasks simultaneously
# ---------------------------------------------------------------------------

def multitask_collate(batch):
    """Stack images; propagate bbox/mask sentinels for unannotated samples."""
    images = torch.stack([s["image"] for s in batch])
    labels = torch.tensor([s["label"] for s in batch])
    bboxes = torch.stack([s["bbox"]  for s in batch])   # sentinel = -1

    # Masks: stack only if all samples have a mask; otherwise return empty
    masks_list = [s["mask"] for s in batch]
    if all(m.numel() > 0 for m in masks_list):
        # Ensure all masks have the same spatial size
        h = masks_list[0].shape[0]
        w = masks_list[0].shape[1]
        masks = torch.stack([
            m if m.shape == (h, w)
            else torch.zeros(h, w, dtype=torch.long)
            for m in masks_list
        ])
    else:
        masks = torch.tensor([])

    return {"image": images, "label": labels,
            "bbox": bboxes, "mask": masks}


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  MultiTaskLoss,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    image_size: int,
    epoch:      int,
) -> dict:
    model.train()
    sums = dict(total=0., cls=0., loc=0., seg=0.,
                acc=0., iou=0., dice=0.)
    n = 0
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        bboxes = batch["bbox" ].to(device, non_blocking=True)
        masks  = batch["mask" ]
        if masks.numel() > 0:
            masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)

        losses = criterion(outputs, labels, bboxes, masks, image_size)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = images.size(0)
        for k in ("total","cls","loc","seg"):
            sums[k] += losses[k].item() * bs

        # Accuracy
        sums["acc"] += (outputs["classification"].argmax(1)==labels
                        ).float().mean().item() * bs

        # mIoU (only annotated)
        valid = (bboxes >= 0).all(1)
        if valid.any():
            sums["iou"] += compute_iou_batch(
                outputs["localization"][valid].detach(),
                bboxes[valid].detach() * image_size,
            ) * valid.sum().item()

        # Dice
        if masks.numel() > 0:
            seg_pred = outputs["segmentation"].detach()
            msk      = masks
            if seg_pred.shape[2:] != msk.shape[1:]:
                msk = F.interpolate(msk.unsqueeze(1).float(),
                                    size=seg_pred.shape[2:],
                                    mode="nearest").squeeze(1).long()
            sums["dice"] += compute_dice_batch(seg_pred, msk) * bs

        n += bs

        if step % 30 == 0:
            print(
                f"  [{epoch}] step {step}/{len(loader)} | "
                f"loss={sums['total']/n:.3f} "
                f"cls={sums['cls']/n:.3f} "
                f"loc={sums['loc']/n:.3f} "
                f"seg={sums['seg']/n:.3f} | "
                f"acc={sums['acc']/n:.3f} "
                f"iou={sums['iou']/n:.3f} "
                f"dice={sums['dice']/n:.3f}"
            )

    elapsed = time.time() - t0
    # return {k: v/n for k,v in sums.items()} | {"time": elapsed}
    return dict({k: v/n for k,v in sums.items()}, time=elapsed)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  MultiTaskLoss,
    device:     torch.device,
    image_size: int,
) -> dict:
    model.eval()
    sums = dict(total=0., cls=0., loc=0., seg=0.,
                acc=0., iou=0., dice=0.)
    all_preds, all_labels = [], []
    n = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        bboxes = batch["bbox" ].to(device, non_blocking=True)
        masks  = batch["mask" ]
        if masks.numel() > 0:
            masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        losses  = criterion(outputs, labels, bboxes, masks, image_size)

        bs = images.size(0)
        for k in ("total","cls","loc","seg"):
            sums[k] += losses[k].item() * bs

        preds = outputs["classification"].argmax(1)
        sums["acc"] += (preds==labels).float().mean().item() * bs
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        valid = (bboxes >= 0).all(1)
        if valid.any():
            sums["iou"] += compute_iou_batch(
                outputs["localization"][valid],
                bboxes[valid] * image_size,
            ) * valid.sum().item()

        if masks.numel() > 0:
            seg_pred = outputs["segmentation"]
            msk      = masks
            if seg_pred.shape[2:] != msk.shape[1:]:
                msk = F.interpolate(msk.unsqueeze(1).float(),
                                    size=seg_pred.shape[2:],
                                    mode="nearest").squeeze(1).long()
            sums["dice"] += compute_dice_batch(seg_pred, msk) * bs

        n += bs

    metrics = {k: v/n for k,v in sums.items()}

    if _SKLEARN:
        yp = torch.cat(all_preds).numpy()
        yt = torch.cat(all_labels).numpy()
        metrics["macro_f1"] = sk_f1(yt, yp, average="macro", zero_division=0)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Device      : {device}")
    print(f"Task-1 ckpt : {args.classifier_ckpt or 'none'}")
    print(f"Task-2 ckpt : {args.localizer_ckpt  or 'none'}")
    print(f"Task-3 ckpt : {args.unet_ckpt        or 'none'}")
    print(f"LR heads    : {args.lr_heads}   backbone: {args.lr_backbone}")
    print(f"Loss weights: cls={args.w_cls} loc={args.w_loc} seg={args.w_seg}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ W&B
    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name,
                   config=vars(args))

    # ------------------------------------------------------------ Transforms
    train_tfm = get_transforms("train", args.image_size)
    val_tfm   = get_transforms("val",   args.image_size)

    # ---------------------------------------------------------------- Dataset
    full_trainval = OxfordIIITPetDataset(
        root=args.data_root, split="train",
        transform=train_tfm,
        load_bbox=True, load_mask=True,
    )
    test_dataset = OxfordIIITPetDataset(
        root=args.data_root, split="test",
        transform=val_tfm,
        load_bbox=True, load_mask=True,
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
            collate_fn   = multitask_collate,
        )

    train_loader = make_loader(train_ds,     shuffle=True)
    val_loader   = make_loader(val_ds,       shuffle=False)
    test_loader  = make_loader(test_dataset, shuffle=False)

    # --------------------------------------------------------------- Model
    # MultiTaskPerceptionModel.__init__ calls gdown internally.
    # We build it pointing at the already-downloaded checkpoint files so
    # gdown finds them on disk and skips the network download.
    clf_path = args.classifier_ckpt # or "checkpoints/task1_best.pth"
    loc_path = args.localizer_ckpt  # or "checkpoints/task2_best.pth"
    unet_path = args.unet_ckpt      # or "checkpoints/task3_best.pth"

    model = MultiTaskPerceptionModel(
        num_breeds      = NUM_BREEDS,
        seg_classes     = NUM_SEG_CLASSES,
        in_channels     = 3,
        classifier_path = clf_path,
        localizer_path  = loc_path,
        unet_path       = unet_path,
        image_size      = args.image_size,
        dropout_p       = args.dropout_p,
    ).to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {total_p:,}  trainable: {trainable_p:,}\n")

    # --------------------------------------------------------- Loss / Optim
    criterion = MultiTaskLoss(
        w_cls=args.w_cls, w_loc=args.w_loc, w_seg=args.w_seg,
        num_seg_classes=NUM_SEG_CLASSES,
    )

    # Three parameter groups with differential LRs:
    #   backbone (shared encoder) — smallest LR to avoid forgetting
    #   heads (cls + loc)         — medium LR
    #   seg decoder               — medium LR
    backbone_params = list(model.encoder.parameters())
    head_params = (
        list(model.cls_head.parameters()) +
        list(model.loc_head.parameters())
    )
    decoder_params = (
        list(model.bridge.parameters())   +
        list(model.up5.parameters())      +
        list(model.up4.parameters())      +
        list(model.up3.parameters())      +
        list(model.up2.parameters())      +
        list(model.up1.parameters())      +
        list(model.seg_head.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params,     "lr": args.lr_heads},
            {"params": decoder_params,  "lr": args.lr_heads},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ------------------------------------------------------------ Resume
    start_epoch = 1
    best_score  = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_score  = ckpt.get("best_score", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_score={best_score:.4f}")

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=200)

    # -------------------------------------------------------- Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = scheduler.get_last_lr()[0] if epoch > 1 else args.lr_heads

        tr  = train_one_epoch(model, train_loader, criterion,
                               optimizer, device, args.image_size, epoch)
        val = evaluate(model, val_loader, criterion,
                       device, args.image_size)
        scheduler.step()

        f1_str = f"  f1={val['macro_f1']:.3f}" if "macro_f1" in val else ""
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"loss={val['total']:.4f} "
            f"cls={val['cls']:.4f} loc={val['loc']:.4f} seg={val['seg']:.4f} | "
            f"acc={val['acc']:.4f}{f1_str} "
            f"iou={val['iou']:.4f} dice={val['dice']:.4f} | "
            f"lr={lr_now:.2e}  ({tr['time']:.1f}s)"
        )

        if use_wandb:
            log = {
                "epoch":           epoch,
                "lr":              lr_now,
                "train/loss":      tr["total"],
                "train/acc":       tr["acc"],
                "train/iou":       tr["iou"],
                "train/dice":      tr["dice"],
                "val/loss":        val["total"],
                "val/loss_cls":    val["cls"],
                "val/loss_loc":    val["loc"],
                "val/loss_seg":    val["seg"],
                "val/acc":         val["acc"],
                "val/iou":         val["iou"],
                "val/dice":        val["dice"],
            }
            if "macro_f1" in val:
                log["val/macro_f1"] = val["macro_f1"]
            wandb.log(log)

        # Save best checkpoint — composite score: equal weight on all metrics
        f1 = val.get("macro_f1", val["acc"])
        score = (f1 + val["iou"] + val["dice"]) / 3.0
        if score > best_score:
            best_score = score
            ckpt_path  = os.path.join(args.ckpt_dir, "task4_best.pth")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "best_score":       best_score,
                "args":             vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved → {ckpt_path}  (score={best_score:.4f})")

    # ---------------------------------------------------- Final test eval
    print("\nEvaluating on held-out test set …")
    test = evaluate(model, test_loader, criterion, device, args.image_size)
    f1_str = f"  macro_f1={test['macro_f1']:.4f}" if "macro_f1" in test else ""
    print(
        f"Test  loss={test['total']:.4f} | "
        f"acc={test['acc']:.4f}{f1_str} | "
        f"iou={test['iou']:.4f} | "
        f"dice={test['dice']:.4f}"
    )

    if use_wandb:
        log = {
            "test/loss":  test["total"],
            "test/acc":   test["acc"],
            "test/iou":   test["iou"],
            "test/dice":  test["dice"],
        }
        if "macro_f1" in test:
            log["test/macro_f1"] = test["macro_f1"]
        wandb.log(log)
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()