"""
verify_dataset.py — Run this BEFORE training to catch data issues early.

Usage (from da6401_assignment_2/):
    python verify_dataset.py

Checks:
    1. trainval.txt and test.txt exist and are parseable
    2. A sample of images can be opened
    3. Bbox XMLs parse correctly
    4. Trimap masks load and have the right values
    5. DataLoader can produce a batch without errors
    6. Class distribution across train/test splits
"""

import os
import sys
from pathlib import Path
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATA_ROOT = os.path.join(ROOT, "data")

# ---- 1. Check files exist --------------------------------------------------
print("=" * 60)
print("1. Checking required files …")

required = [
    "annotations/trainval.txt",
    "annotations/test.txt",
    "annotations/trimaps",
    "images",
]
all_ok = True
for rel in required:
    p = Path(DATA_ROOT) / rel
    if p.exists():
        print(f"   ✓  {rel}")
    else:
        print(f"   ✗  MISSING: {rel}")
        all_ok = False

if not all_ok:
    print("\n[ERROR] Fix the missing paths above before continuing.")
    sys.exit(1)

# ---- 2. Parse split files --------------------------------------------------
print("\n2. Parsing split files …")
from data.pets_dataset import OxfordIIITPetDataset, get_transforms

train_ds_raw = OxfordIIITPetDataset(
    root=DATA_ROOT, split="train", transform=None,
    load_bbox=False, load_mask=False,
)
test_ds_raw = OxfordIIITPetDataset(
    root=DATA_ROOT, split="test", transform=None,
    load_bbox=False, load_mask=False,
)

print(f"   trainval.txt: {len(train_ds_raw)} samples")
print(f"   test.txt    : {len(test_ds_raw)} samples")

# ---- 3. Class distribution -------------------------------------------------
print("\n3. Class distribution (trainval) …")
label_counts = Counter(lbl for _, lbl in train_ds_raw.samples)
print(f"   Unique classes : {len(label_counts)}")
min_cls = min(label_counts, key=label_counts.get)
max_cls = max(label_counts, key=label_counts.get)
print(f"   Min samples/class: {label_counts[min_cls]} (class {min_cls})")
print(f"   Max samples/class: {label_counts[max_cls]} (class {max_cls})")

# ---- 4. Sample a few images ------------------------------------------------
print("\n4. Opening 5 sample images …")
import random
random.seed(0)
indices = random.sample(range(len(train_ds_raw)), min(5, len(train_ds_raw)))
for i in indices:
    batch = train_ds_raw[i]
    img   = batch["image"]   # PIL image (no transform applied)
    lbl   = batch["label"]
    print(f"   [{i:04d}] label={lbl}  image type={type(img).__name__}  "
          f"size={img.size if hasattr(img, 'size') else img.shape}")

# ---- 5. Check bounding boxes -----------------------------------------------
print("\n5. Checking bounding boxes (first 5 samples with XMLs) …")
bbox_ds = OxfordIIITPetDataset(
    root=DATA_ROOT, split="train", transform=None,
    load_bbox=True, load_mask=False,
)
found = 0
for i in range(min(200, len(bbox_ds))):
    b = bbox_ds[i]["bbox"]
    if (b >= 0).all():
        print(f"   [{i:04d}] bbox={b.tolist()} (normalised [0,1])")
        found += 1
    if found >= 5:
        break
if found == 0:
    print("   [WARN] No XML annotation files found — check data/annotations/xmls/")

# ---- 6. Check trimap masks -------------------------------------------------
print("\n6. Checking trimap masks …")
import torch, numpy as np
mask_ds = OxfordIIITPetDataset(
    root=DATA_ROOT, split="train", transform=None,
    load_bbox=False, load_mask=True,
)
found = 0
for i in range(min(200, len(mask_ds))):
    m = mask_ds[i]["mask"]
    if m.numel() > 0:
        vals = m.unique().tolist()
        print(f"   [{i:04d}] mask shape={tuple(m.shape)}  unique values={vals}")
        assert set(vals).issubset({0, 1, 2}), \
            f"Unexpected mask values: {vals} (expect subset of {{0,1,2}})"
        found += 1
    if found >= 3:
        break
if found == 0:
    print("   [WARN] No trimaps found — check data/annotations/trimaps/")

# ---- 7. DataLoader batch test ----------------------------------------------
print("\n7. Testing DataLoader (1 batch) …")
from torch.utils.data import DataLoader

tfm = get_transforms("train", image_size=224)
loader_ds = OxfordIIITPetDataset(
    root=DATA_ROOT, split="train", transform=tfm,
    load_bbox=False, load_mask=False,
)
loader = DataLoader(loader_ds, batch_size=8, shuffle=True,
                    num_workers=0)   # num_workers=0 for quick check
batch = next(iter(loader))
img_t = batch["image"]
lbl_t = batch["label"]
print(f"   image batch shape : {tuple(img_t.shape)}")   # expect [8, 3, 224, 224]
print(f"   label batch shape : {tuple(lbl_t.shape) if isinstance(lbl_t, torch.Tensor) else lbl_t}")
print(f"   pixel value range : [{img_t.min():.2f}, {img_t.max():.2f}]")
assert img_t.shape == (8, 3, 224, 224), \
    f"Unexpected shape: {img_t.shape}"

print("\n" + "=" * 60)
print("All checks passed ✓  —  You are ready to run train_classification.py")
print("=" * 60)