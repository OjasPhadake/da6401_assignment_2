"""
train_multitask.py  —  Task 4: Unified Multi-Task Learning Pipeline

CORRECT WORKFLOW
================
Step 1: Train tasks 1–3 individually:
    python train_classification.py  →  checkpoints/task1_best.pth
    python train_localization.py    →  checkpoints/task2_best.pth
    python train_segmentation.py    →  checkpoints/task3_best.pth

Step 2: Train the unified model:
    python train_multitask.py \
        --clf_ckpt  checkpoints/task1_best.pth \
        --loc_ckpt  checkpoints/task2_best.pth \
        --unet_ckpt checkpoints/task3_best.pth
    →  produces  checkpoints/task4_best.pth

Step 3: Upload task4_best.pth to Google Drive.
        Put its Drive ID in models/multitask.py → TASK4_DRIVE_ID.

Step 4: Also upload task1_best.pth, task2_best.pth, task3_best.pth if
        multitask.py needs them for the fallback assembly path.

NOTE: train_multitask.py builds the MultiTaskPerceptionModel directly
from sub-module weights WITHOUT calling gdown (gdown is only for the
autograder's inference path).  We do this by directly loading weights
into the model after construction.
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
from models.segmentation import SegmentationLoss
from losses.iou_loss     import IoULoss

# Import model components directly to avoid gdown during training
from models.vgg11        import VGG11Encoder
from models.layers       import CustomDropout
from models.segmentation import DecoderBlock, _conv_bn_relu
import torch.nn.functional as F

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
    print("[WARN] wandb not installed.")

try:
    from sklearn.metrics import f1_score as sk_f1
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ===========================================================================
# CONFIG
# ===========================================================================
DEFAULT_DATA_ROOT     = os.path.join(ROOT, "data")
DEFAULT_CKPT_DIR      = os.path.join(ROOT, "checkpoints")
DEFAULT_EPOCHS        = 20
DEFAULT_BATCH_SIZE    = 16
DEFAULT_LR_HEADS      = 1e-3
DEFAULT_LR_BACKBONE   = 1e-4
DEFAULT_WEIGHT_DECAY  = 1e-4
DEFAULT_DROPOUT_P     = 0.5
DEFAULT_IMAGE_SIZE    = 224
DEFAULT_VAL_SPLIT     = 0.1
DEFAULT_NUM_WORKERS   = 4
DEFAULT_SEED          = 42
DEFAULT_WANDB_PROJECT = "da6401_a2_task4"
NUM_BREEDS            = 37
NUM_SEG_CLASSES       = 3
W_CLS, W_LOC, W_SEG  = 1.0, 1.0, 1.0
# ===========================================================================


# ---------------------------------------------------------------------------
# Build MultiTaskPerceptionModel WITHOUT gdown (for training only)
# ---------------------------------------------------------------------------

def build_multitask_model(
    clf_ckpt:   str,
    loc_ckpt:   str,
    unet_ckpt:  str,
    image_size: int   = 224,
    dropout_p:  float = 0.5,
    num_breeds: int   = 37,
    seg_classes:int   = 3,
) -> nn.Module:
    """Build and initialise the multitask model from individual task checkpoints.

    This bypasses gdown entirely — it loads weights directly from local paths.
    The resulting model is the same architecture as MultiTaskPerceptionModel
    but constructed inline to avoid the __init__ gdown call during training.
    """
    # Build the model structure (gdown will be called in __init__ but we
    # pass dummy paths that won't be found, relying on the fallback assembly)

    # Instead: build from scratch using the same architecture
    class _MultiTask(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_size = image_size

            self.encoder  = VGG11Encoder(in_channels=3)
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # ← modern replacement
                nn.Flatten(),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(512, num_breeds),
            )
            self.loc_head = nn.Sequential(
                nn.Flatten(),

                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),

                nn.Linear(512, 4),
                nn.Sigmoid(),
            )
            self.bridge   = nn.Sequential(_conv_bn_relu(512, 512),
                                          CustomDropout(p=dropout_p))
            self.up5      = DecoderBlock(512, 512, 512)
            self.up4      = DecoderBlock(512, 512, 256)
            self.up3      = DecoderBlock(256, 256, 128)
            self.up2      = DecoderBlock(128, 128,  64)
            self.up1      = DecoderBlock( 64,  64,  64)
            self.seg_head = nn.Conv2d(64, seg_classes, 1)

        def forward(self, x):
            H, W = x.shape[2], x.shape[3]
            bottleneck, skips = self.encoder(x, return_features=True)
            cls = self.cls_head(bottleneck)
            bbox = self.loc_head(bottleneck) * self.image_size
            d = self.bridge(bottleneck)
            d = self.up5(d, skips["s5"])
            d = self.up4(d, skips["s4"])
            d = self.up3(d, skips["s3"])
            d = self.up2(d, skips["s2"])
            d = self.up1(d, skips["s1"])
            seg = self.seg_head(d)
            if seg.shape[2:] != (H, W):
                seg = F.interpolate(seg, size=(H,W),
                                    mode="bilinear", align_corners=False)
            return {"classification": cls, "localization": bbox,
                    "segmentation": seg}

    model = _MultiTask()

    def _filt(sd, pfx):
        return {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}

    def _safe_load(path):
        if not path or not os.path.isfile(path):
            return None
        try:
            ckpt = torch.load(path, map_location="cpu")
            return ckpt.get("model_state_dict", ckpt)
        except Exception as e:
            print(f"[WARN] Could not load '{path}': {e}")
            return None

    # ---- Load Task-1 classifier (encoder + cls_head) --------------------
    clf_sd = _safe_load(clf_ckpt)
    if clf_sd is not None:
        model.encoder.load_state_dict(_filt(clf_sd, "encoder."), strict=False)
        # head.head.* → cls_head.*
        cls_sd = _filt(clf_sd, "head.head.")
        m, u = model.cls_head.load_state_dict(cls_sd, strict=False)
        print(f"[Build] Task-1 encoder + cls_head loaded  "
              f"(missing={len(m)}, unexpected={len(u)})")
    else:
        print("[WARN] Task-1 checkpoint not found — encoder+cls_head random")

    # ---- Load Task-2 localizer (loc_head, blend encoder) -----------------
    loc_sd = _safe_load(loc_ckpt)
    if loc_sd is not None:
        loc_h = _filt(loc_sd, "regression_head.head.")
        m, u  = model.loc_head.load_state_dict(loc_h, strict=False)
        print(f"[Build] Task-2 loc_head loaded  "
              f"(missing={len(m)}, unexpected={len(u)})")
        if clf_sd is not None:
            # Average encoder from classifier and localizer
            loc_enc = _filt(loc_sd, "encoder.")
            cur_sd  = model.encoder.state_dict()
            for k in cur_sd:
                if k in loc_enc and cur_sd[k].dtype.is_floating_point:
                    cur_sd[k] = (cur_sd[k] + loc_enc[k]) / 2.0
            model.encoder.load_state_dict(cur_sd, strict=False)
    else:
        print("[WARN] Task-2 checkpoint not found — loc_head random")

    # ---- Load Task-3 U-Net (seg decoder, blend encoder) ------------------
    unet_sd = _safe_load(unet_ckpt)
    if unet_sd is not None:
        for attr in ("bridge","up5","up4","up3","up2","up1","seg_head"):
            sub = _filt(unet_sd, f"{attr}.")
            if sub:
                m, u = getattr(model, attr).load_state_dict(sub, strict=False)
        print(f"[Build] Task-3 seg decoder loaded")

        n = sum([clf_sd is not None, loc_sd is not None])
        unet_enc = _filt(unet_sd, "encoder.")
        cur_sd   = model.encoder.state_dict()
        for k in cur_sd:
            if k in unet_enc and cur_sd[k].dtype.is_floating_point:
                cur_sd[k] = (cur_sd[k] * n + unet_enc[k]) / (n + 1)
        model.encoder.load_state_dict(cur_sd, strict=False)
    else:
        print("[WARN] Task-3 checkpoint not found — seg decoder random")

    return model


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="DA6401 A2 Task 4 — Unified multi-task training"
    )
    p.add_argument("--data_root",    default=DEFAULT_DATA_ROOT)
    p.add_argument("--ckpt_dir",     default=DEFAULT_CKPT_DIR)
    p.add_argument("--clf_ckpt",     default=None,
                   help="Path to task1_best.pth (classification checkpoint)")
    p.add_argument("--loc_ckpt",     default=None,
                   help="Path to task2_best.pth (localization checkpoint)")
    p.add_argument("--unet_ckpt",    default=None,
                   help="Path to task3_best.pth (segmentation checkpoint)")
    p.add_argument("--resume",       default=None,
                   help="Resume from task4 checkpoint")
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr_heads",     type=float, default=DEFAULT_LR_HEADS)
    p.add_argument("--lr_backbone",  type=float, default=DEFAULT_LR_BACKBONE)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--dropout_p",    type=float, default=DEFAULT_DROPOUT_P)
    p.add_argument("--image_size",   type=int,   default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--val_split",    type=float, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--num_workers",  type=int,   default=DEFAULT_NUM_WORKERS)
    p.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    p.add_argument("--w_cls",        type=float, default=W_CLS)
    p.add_argument("--w_loc",        type=float, default=W_LOC)
    p.add_argument("--w_seg",        type=float, default=W_SEG)
    p.add_argument("--wandb_project",default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--run_name",     default=None)
    p.add_argument("--no_wandb",     action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_loc=1.0, w_seg=1.0, num_seg_classes=3):
        super().__init__()
        self.w_cls = w_cls; self.w_loc = w_loc; self.w_seg = w_seg
        self.cls_loss = nn.CrossEntropyLoss()
        self.loc_loss = IoULoss(reduction="mean")
        self.seg_loss = SegmentationLoss(num_classes=num_seg_classes,
                                         dice_weight=1.0)

    def forward(self, outputs, labels, bboxes, masks, image_size=224):
        device = labels.device
        l_cls = self.cls_loss(outputs["classification"], labels)
        l_loc = torch.tensor(0., device=device)
        l_seg = torch.tensor(0., device=device)

        valid = (bboxes >= 0).all(1)
        if valid.any():
            l_loc = self.loc_loss(outputs["localization"][valid],
                                  bboxes[valid] * image_size)

        if masks.numel() > 0:
            seg_pred = outputs["segmentation"]
            if seg_pred.shape[2:] != masks.shape[1:]:
                masks = F.interpolate(masks.unsqueeze(1).float(),
                                      size=seg_pred.shape[2:],
                                      mode="nearest").squeeze(1).long()
            l_seg = self.seg_loss(seg_pred, masks)

        total = self.w_cls*l_cls + self.w_loc*l_loc + self.w_seg*l_seg
        return {"total": total, "cls": l_cls, "loc": l_loc, "seg": l_seg}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _iou_batch(p, t, eps=1e-6):
    px1=p[:,0]-p[:,2]*.5; py1=p[:,1]-p[:,3]*.5
    px2=p[:,0]+p[:,2]*.5; py2=p[:,1]+p[:,3]*.5
    tx1=t[:,0]-t[:,2]*.5; ty1=t[:,1]-t[:,3]*.5
    tx2=t[:,0]+t[:,2]*.5; ty2=t[:,1]+t[:,3]*.5
    iw=torch.relu(torch.min(px2,tx2)-torch.max(px1,tx1))
    ih=torch.relu(torch.min(py2,ty2)-torch.max(py1,ty1))
    inter=iw*ih
    union=(px2-px1).clamp(0)*(py2-py1).clamp(0)+(tx2-tx1).clamp(0)*(ty2-ty1).clamp(0)-inter
    return (inter/(union+eps)).mean().item()

@torch.no_grad()
def _dice_batch(logits, target, nc=3, eps=1e-6):
    preds=logits.argmax(1)
    dices=[]
    for c in range(nc):
        p=(preds==c).float(); t=(target==c).float()
        dices.append(((2*(p*t).sum()+eps)/(p.sum()+t.sum()+eps)).item())
    return float(np.mean(dices))


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def mt_collate(batch):
    images = torch.stack([s["image"] for s in batch])
    labels = torch.tensor([s["label"] for s in batch])
    bboxes = torch.stack([s["bbox"] for s in batch])
    ml = [s["mask"] for s in batch]
    if all(m.numel() > 0 for m in ml):
        h, w = ml[0].shape
        masks = torch.stack([
            m if m.shape == (h,w) else torch.zeros(h,w,dtype=torch.long)
            for m in ml
        ])
    else:
        masks = torch.tensor([])
    return {"image":images,"label":labels,"bbox":bboxes,"mask":masks}


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, image_size, epoch):
    model.train()
    sums = dict(total=0.,cls=0.,loc=0.,seg=0.,acc=0.,iou=0.,dice=0.)
    n = 0; t0 = time.time()

    for step, batch in enumerate(loader, 1):
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        bboxes = batch["bbox" ].to(device, non_blocking=True)
        masks  = batch["mask" ]
        if masks.numel() > 0:
            masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(imgs)
        losses = criterion(out, labels, bboxes, masks, image_size)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = imgs.size(0)
        for k in sums: sums[k] = sums.get(k, 0)
        for k in ("total","cls","loc","seg"):
            sums[k] += losses[k].item() * bs
        sums["acc"] += (out["classification"].argmax(1)==labels
                        ).float().mean().item() * bs
        valid = (bboxes>=0).all(1)
        if valid.any():
            sums["iou"] += _iou_batch(out["localization"][valid].detach(),
                                      bboxes[valid].detach()*image_size
                                      ) * valid.sum().item()
        if masks.numel() > 0:
            seg = out["segmentation"].detach(); msk = masks
            if seg.shape[2:] != msk.shape[1:]:
                msk = F.interpolate(msk.unsqueeze(1).float(),
                                    size=seg.shape[2:],
                                    mode="nearest").squeeze(1).long()
            sums["dice"] += _dice_batch(seg, msk) * bs
        n += bs

        if step % 30 == 0:
            print(f"  [{epoch}] {step}/{len(loader)} | "
                  f"loss={sums['total']/n:.3f} cls={sums['cls']/n:.3f} "
                  f"loc={sums['loc']/n:.3f} seg={sums['seg']/n:.3f} | "
                  f"acc={sums['acc']/n:.3f} iou={sums['iou']/n:.3f} "
                  f"dice={sums['dice']/n:.3f}")

    r = {k: v/n for k,v in sums.items()}
    r["time"] = time.time()-t0
    return r


@torch.no_grad()
def evaluate(model, loader, criterion, device, image_size):
    model.eval()
    sums = dict(total=0.,cls=0.,loc=0.,seg=0.,acc=0.,iou=0.,dice=0.)
    all_p, all_l = [], []
    n = 0

    for batch in loader:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        bboxes = batch["bbox" ].to(device, non_blocking=True)
        masks  = batch["mask" ]
        if masks.numel() > 0:
            masks = masks.to(device, non_blocking=True)

        out    = model(imgs)
        losses = criterion(out, labels, bboxes, masks, image_size)
        bs     = imgs.size(0)
        for k in ("total","cls","loc","seg"):
            sums[k] += losses[k].item() * bs
        preds = out["classification"].argmax(1)
        sums["acc"] += (preds==labels).float().mean().item() * bs
        all_p.append(preds.cpu()); all_l.append(labels.cpu())
        valid = (bboxes>=0).all(1)
        if valid.any():
            sums["iou"] += _iou_batch(out["localization"][valid],
                                      bboxes[valid]*image_size
                                      ) * valid.sum().item()
        if masks.numel() > 0:
            seg = out["segmentation"]; msk = masks
            if seg.shape[2:] != msk.shape[1:]:
                msk = F.interpolate(msk.unsqueeze(1).float(),
                                    size=seg.shape[2:],
                                    mode="nearest").squeeze(1).long()
            sums["dice"] += _dice_batch(seg, msk) * bs
        n += bs

    metrics = {k: v/n for k,v in sums.items()}
    if _SKLEARN:
        metrics["macro_f1"] = sk_f1(
            torch.cat(all_l).numpy(), torch.cat(all_p).numpy(),
            average="macro", zero_division=0
        )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Device      : {device}")
    print(f"clf_ckpt    : {args.clf_ckpt  or 'none (random)'}")
    print(f"loc_ckpt    : {args.loc_ckpt  or 'none (random)'}")
    print(f"unet_ckpt   : {args.unet_ckpt or 'none (random)'}")
    print(f"LR heads    : {args.lr_heads}   backbone: {args.lr_backbone}")
    print(f"Loss weights: cls={args.w_cls} loc={args.w_loc} seg={args.w_seg}")
    print(f"{'='*60}\n")

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name,
                   config=vars(args))

    # ---- Dataset ----
    train_tfm = get_transforms("train", args.image_size)
    val_tfm   = get_transforms("val",   args.image_size)

    full_tv = OxfordIIITPetDataset(args.data_root, "train", train_tfm,
                                    load_bbox=True, load_mask=True)
    test_ds = OxfordIIITPetDataset(args.data_root, "test",  val_tfm,
                                    load_bbox=True, load_mask=True)
    n_val   = int(len(full_tv) * args.val_split)
    tr_ds, val_ds = random_split(
        full_tv, [len(full_tv)-n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train:{len(tr_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")

    mk = lambda ds, sh: DataLoader(ds, args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers,
                                   pin_memory=(device.type=="cuda"),
                                   drop_last=sh, collate_fn=mt_collate)
    tr_ldr = mk(tr_ds, True); val_ldr = mk(val_ds, False)
    test_ldr = mk(test_ds, False)

    # ---- Model (no gdown during training) ----
    model = build_multitask_model(
        clf_ckpt=args.clf_ckpt, loc_ckpt=args.loc_ckpt,
        unet_ckpt=args.unet_ckpt, image_size=args.image_size,
        dropout_p=args.dropout_p, num_breeds=NUM_BREEDS,
        seg_classes=NUM_SEG_CLASSES,
    ).to(device)

    tp = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {tp:,}\n")

    # ---- Optimiser with 3 param groups ----
    criterion = MultiTaskLoss(args.w_cls, args.w_loc, args.w_seg,
                              NUM_SEG_CLASSES)
    backbone_p = list(model.encoder.parameters())
    head_p     = (list(model.cls_head.parameters()) +
                  list(model.loc_head.parameters()))
    decoder_p  = (list(model.bridge.parameters())   +
                  list(model.up5.parameters())       +
                  list(model.up4.parameters())       +
                  list(model.up3.parameters())       +
                  list(model.up2.parameters())       +
                  list(model.up1.parameters())       +
                  list(model.seg_head.parameters()))

    optimizer = torch.optim.AdamW(
        [{"params": backbone_p, "lr": args.lr_backbone},
         {"params": head_p,     "lr": args.lr_heads},
         {"params": decoder_p,  "lr": args.lr_heads}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ---- Resume ----
    start_epoch = 1; best_score = 0.
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_score  = ckpt.get("best_score", 0.)
        print(f"Resumed epoch {start_epoch}, best={best_score:.4f}")

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=200)

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs+1):
        lr_now = scheduler.get_last_lr()[0] if epoch > 1 else args.lr_heads
        tr  = train_one_epoch(model, tr_ldr, criterion, optimizer,
                               device, args.image_size, epoch)
        val = evaluate(model, val_ldr, criterion, device, args.image_size)
        scheduler.step()

        f1s = f" f1={val['macro_f1']:.4f}" if "macro_f1" in val else ""
        print(f"[{epoch:03d}/{args.epochs}] "
              f"loss={val['total']:.4f} cls={val['cls']:.4f} "
              f"loc={val['loc']:.4f} seg={val['seg']:.4f} | "
              f"acc={val['acc']:.4f}{f1s} "
              f"iou={val['iou']:.4f} dice={val['dice']:.4f} | "
              f"lr={lr_now:.2e} ({tr['time']:.1f}s)")

        if use_wandb:
            log = {"epoch":epoch,"lr":lr_now,
                   "train/loss":tr["total"],"train/acc":tr["acc"],
                   "train/iou":tr["iou"],"train/dice":tr["dice"],
                   "val/loss":val["total"],"val/cls":val["cls"],
                   "val/loc":val["loc"],"val/seg":val["seg"],
                   "val/acc":val["acc"],"val/iou":val["iou"],
                   "val/dice":val["dice"]}
            if "macro_f1" in val: log["val/macro_f1"]=val["macro_f1"]
            wandb.log(log)

        f1  = val.get("macro_f1", val["acc"])
        score = (f1 + val["iou"] + val["dice"]) / 3.0
        if score > best_score:
            best_score = score
            ckpt_path  = os.path.join(args.ckpt_dir, "task4_best.pth")
            torch.save({"epoch":epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state":  optimizer.state_dict(),
                        "best_score": best_score,
                        "args": vars(args)}, ckpt_path)
            print(f"  ✓ Saved → {ckpt_path}  (score={best_score:.4f})")

    # ---- Final test ----
    print("\nTest set evaluation …")
    test = evaluate(model, test_ldr, criterion, device, args.image_size)
    f1s  = f"  f1={test['macro_f1']:.4f}" if "macro_f1" in test else ""
    print(f"Test loss={test['total']:.4f} | "
          f"acc={test['acc']:.4f}{f1s} | "
          f"iou={test['iou']:.4f} | dice={test['dice']:.4f}")

    if use_wandb:
        log = {"test/loss":test["total"],"test/acc":test["acc"],
               "test/iou":test["iou"],"test/dice":test["dice"]}
        if "macro_f1" in test: log["test/macro_f1"]=test["macro_f1"]
        wandb.log(log); wandb.finish()

    print("\nDone. Now:")
    print(f"  1. Upload checkpoints/task4_best.pth to Google Drive")
    print(f"  2. Set TASK4_DRIVE_ID in models/multitask.py")
    print(f"  3. Submit to Gradescope")


if __name__ == "__main__":
    main()