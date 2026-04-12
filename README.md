# DA6401 — Assignment 2: Building a Complete Visual Perception Pipeline

| Resource | Link |
|---|---|
| 📊 **W&B Report** | [DA6401 Assignment 2 Report](https://wandb.ai/ch22b007-indian-institute-of-technology-madras/da6401_a2_task1/reports/DA6401-Assignment-2-Report--VmlldzoxNjQ5OTY5NQ?accessToken=xginszrrqbe7h4hphx3rc5m49h4vadgp37codxdfvv1hiu612mlduud3v6267i1n) |
| 💻 **GitHub Repository** | [OjasPhadake/da6401_assignment_2](https://github.com/OjasPhadake/da6401_assignment_2) |

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset Preparation](#dataset-preparation)
- [Architecture](#architecture)
  - [Shared Backbone — VGG11 Encoder](#shared-backbone--vgg11-encoder)
  - [Task 1 — Classification Head](#task-1--classification-head)
  - [Task 2 — Localization Head](#task-2--localization-head)
  - [Task 3 — Segmentation Decoder (U-Net)](#task-3--segmentation-decoder-u-net)
  - [Task 4 — Unified Multi-Task Model](#task-4--unified-multi-task-model)
- [Custom Components](#custom-components)
- [Training](#training)
- [W&B Report Experiments](#wb-report-experiments)
- [Results](#results)
- [Design Decisions](#design-decisions)

---

## Overview

This project implements a complete **Visual Perception Pipeline** on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) using PyTorch. A single VGG11 convolutional backbone is shared across three tasks:

| Task | Output | Metric |
|---|---|---|
| **Classification** | 37-class breed logits `[B, 37]` | Macro F1-Score |
| **Localization** | Bounding box `[B, 4]` in pixel-space `(xc, yc, w, h)` | mAP / Acc@IoU |
| **Segmentation** | Pixel-wise trimap mask `[B, 3, H, W]` | Dice Score |

A single `forward(x)` call simultaneously produces all three predictions.

---

## Project Structure

```
da6401_assignment_2/
│
├── data/
│   ├── annotations/
│   │   ├── trimaps/          # Segmentation masks (.png)
│   │   ├── xmls/             # Bounding box annotations (.xml)
│   │   ├── trainval.txt      # Training split
│   │   └── test.txt          # Test split
│   ├── images/               # Pet images (.jpg)
│   └── pets_dataset.py       # OxfordIIITPetDataset + get_transforms()
│
├── models/
│   ├── __init__.py
│   ├── layers.py             # CustomDropout (inverted dropout from scratch)
│   ├── vgg11.py              # VGG11Encoder with skip-connection support
│   ├── classification.py     # VGG11Classifier (Task 1)
│   ├── localization.py       # VGG11Localizer  (Task 2)
│   ├── segmentation.py       # VGG11UNet + SegmentationLoss (Task 3)
│   └── multitask.py          # MultiTaskPerceptionModel (Task 4)
│
├── losses/
│   ├── __init__.py
│   └── iou_loss.py           # Custom IoULoss (from scratch, no external libs)
│
├── train_classification.py   # Task 1 training script
├── train_localization.py     # Task 2 training script
├── train_segmentation.py     # Task 3 training script
├── train_multitask.py        # Task 4 training script
├── verify_dataset.py         # Dataset sanity checks before training
├── inference.py              # Run pipeline on arbitrary images
│
├── wandb_2_1_batchnorm_effect.py      # W&B §2.1: BN ablation
├── wandb_2_2_dropout_dynamics.py      # W&B §2.2: Dropout ablation
├── wandb_2_3_transfer_learning.py     # W&B §2.3: Transfer learning comparison
├── wandb_2_4_feature_maps.py          # W&B §2.4: Feature map visualisation
├── wandb_2_5_bbox_confidence.py       # W&B §2.5: BBox confidence table
├── wandb_2_6_segmentation_eval.py     # W&B §2.6: Dice vs Pixel Accuracy
├── wandb_2_7_2_8_pipeline_showcase.py # W&B §2.7–2.8: Showcase + meta-analysis
│
├── checkpoints/
│   ├── task1_best.pth        # Trained classifier checkpoint
│   ├── task2_best.pth        # Trained localizer checkpoint
│   ├── task3_best.pth        # Trained U-Net checkpoint
│   └── task4_best.pth        # Trained multi-task checkpoint
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/OjasPhadake/da6401_assignment_2.git
cd da6401_assignment_2

# Install dependencies
pip install torch>=1.8.0 torchvision numpy>=1.21.0 matplotlib>=3.4.0 \
            pillow>=8.2.0 albumentations>=1.0.0 wandb>=0.12.0 \
            scikit-learn>=0.24.2

# Log in to Weights & Biases
wandb login
```

---

## Dataset Preparation

Download the Oxford-IIIT Pet Dataset and place it inside `data/`:

```bash
# From the project root:
cd data
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xf images.tar.gz
tar -xf annotations.tar.gz
cd ..
```

Verify the dataset is correctly set up:

```bash
python verify_dataset.py
```

Expected output: `All checks passed ✓ — You are ready to run train_classification.py`

**Expected layout after extraction:**

```
data/
├── images/          # 7,390 .jpg files
└── annotations/
    ├── trainval.txt # 3,680 samples
    ├── test.txt     # 3,669 samples
    ├── trimaps/     # pixel-wise masks
    └── xmls/        # bounding box XML files
```

---

## Architecture

### Shared Backbone — VGG11 Encoder

`models/vgg11.py` → `VGG11Encoder`

Built **from scratch** using `torch.nn` primitives. Each convolutional block follows **Conv → ReLU → BatchNorm** ordering (rationale: BN after ReLU sees a non-negative half-normal distribution, which is more stable to normalise, leading to ~2 epochs faster convergence).

```
Input [B, 3, 224, 224]
  │
  ├── Stage 1: Conv(3→64,   3×3) → ReLU → BN  →  MaxPool(2×2)  →  [B, 64,  112, 112]
  ├── Stage 2: Conv(64→128, 3×3) → ReLU → BN  →  MaxPool(2×2)  →  [B, 128,  56,  56]
  ├── Stage 3: Conv(128→256)×2   → ReLU → BN  →  MaxPool(2×2)  →  [B, 256,  28,  28]
  ├── Stage 4: Conv(256→512)×2   → ReLU → BN  →  MaxPool(2×2)  →  [B, 512,  14,  14]
  └── Stage 5: Conv(512→512)×2   → ReLU → BN  →  MaxPool(2×2)  →  [B, 512,   7,   7]
                                                                           ↓
                                                               AdaptiveAvgPool(7×7)
                                                             Bottleneck [B, 512, 7, 7]
```

When called with `return_features=True`, the encoder also returns pre-pool skip maps `{s1, s2, s3, s4, s5}` for the U-Net decoder.

---

### Task 1 — Classification Head

`models/classification.py` → `VGG11Classifier`  
**Parameters: ~128.9M**

```
Bottleneck [B, 512, 7, 7]
  │
  Flatten  →  [B, 25088]
  │
  Linear(25088 → 4096) → ReLU → BatchNorm1d → CustomDropout(p)
  Linear(4096  → 4096) → ReLU → BatchNorm1d → CustomDropout(p)
  Linear(4096  → 37)
  │
  Logits [B, 37]
```

**Loss:** `CrossEntropyLoss(label_smoothing=0.1)`  
**Optimiser:** AdamW + Cosine LR Annealing

---

### Task 2 — Localization Head

`models/localization.py` → `VGG11Localizer`  
**Parameters: ~114.1M**

```
Bottleneck [B, 512, 7, 7]
  │
  AdaptiveAvgPool2d(1×1)  →  [B, 512, 1, 1]
  Flatten                  →  [B, 512]
  │
  Linear(512 → 256) → ReLU → Dropout(p=0.3)
  Linear(256 → 128) → ReLU
  Linear(128 → 4)   → Sigmoid
  │
  × image_size (224)
  │
  BBox [B, 4]  pixel-space (xc, yc, w, h)
```

**Output space:** Sigmoid constrains raw output to `(0,1)`; multiplying by `image_size` gives pixel coordinates. This prevents the flat IoU loss landscape (IoU=0 everywhere) that would arise from an unbounded linear head at random initialisation.

**Loss:** Custom `IoULoss` — `1 − IoU`, computed in pixel space  
**Optimiser:** AdamW with **differential LR** — backbone `1e-4`, head `1e-3`

---

### Task 3 — Segmentation Decoder (U-Net)

`models/segmentation.py` → `VGG11UNet`  
**Parameters: ~25.1M**

Symmetric U-Net decoder mirroring the VGG11 encoder. Each `DecoderBlock` performs:
1. **ConvTranspose2d(stride=2)** — learnable ×2 upsampling (bilinear/interpolation not used)
2. **Concatenate** with the corresponding encoder skip map along channel dim
3. **Two 3×3 Conv-BN-ReLU blocks** to fuse and refine

```
Bottleneck [B, 512, 7, 7]
  │   Bridge: Conv-BN-ReLU + CustomDropout
  │
  up5: TransposeConv + cat(s5, 512ch) → conv(1024→512)   [B, 512, ~14,  ~14 ]
  up4: TransposeConv + cat(s4, 512ch) → conv(1024→256)   [B, 256, ~28,  ~28 ]
  up3: TransposeConv + cat(s3, 256ch) → conv( 512→128)   [B, 128, ~56,  ~56 ]
  up2: TransposeConv + cat(s2, 128ch) → conv( 256→64 )   [B,  64, ~112, ~112]
  up1: TransposeConv + cat(s1,  64ch) → conv( 128→64 )   [B,  64, ~224, ~224]
  │
  Conv2d(64→3, 1×1)
  │
  Logits [B, 3, H, W]
```

**Loss:** `SegmentationLoss = CrossEntropyLoss + λ·DiceLoss` (λ=1.0)

> CE provides stable early-training gradients; Dice directly optimises the evaluation metric and handles class imbalance (background ≈65% of pixels in Oxford Pets trimaps).

**Transfer strategies supported:** `freeze_encoder=True` (strict), `freeze_stages=[1,2,3]` (partial), or full fine-tune.

---

### Task 4 — Unified Multi-Task Model

`models/multitask.py` → `MultiTaskPerceptionModel`

A **single forward pass** branches from the shared encoder into all three heads simultaneously:

```python
outputs = model(x)
# outputs["classification"] → [B, 37]
# outputs["localization"]   → [B, 4]
# outputs["segmentation"]   → [B, 3, H, W]
```

```
Input [B, 3, 224, 224]
        │
   VGG11Encoder (shared, ONE forward pass)
   ┌────────────────────────────────────┐
   │  bottleneck [B,512,7,7]            │
   │  skips {s1,s2,s3,s4,s5}           │
   └────────────────────────────────────┘
        │              │              │
   cls_head       loc_head      seg_decoder
   [B, 37]         [B, 4]      [B, 3, H, W]
```

**Weight initialisation:** Encoder weights are blended (running average) across the three task-specific checkpoints (Task 1–3), giving a neutral shared representation informed by all three supervision signals.

**Training:** Three AdamW parameter groups with differential LRs:

| Group | LR |
|---|---|
| Shared encoder (backbone) | `1e-4` |
| Classification + localization heads | `1e-3` |
| Segmentation decoder | `1e-3` |

**Loss:** `L_total = w_cls · L_CE + w_loc · L_IoU + w_seg · (L_CE + L_Dice)`  
Default weights: `w_cls = w_loc = w_seg = 1.0`

---

## Custom Components

### `CustomDropout` — `models/layers.py`

Hand-implemented **inverted dropout** with no use of `torch.nn.Dropout` or `torch.nn.functional.dropout`.

```python
# Training: binary mask + inverted scaling
mask = torch.empty_like(x).bernoulli_(1 - p)
return x * mask / (1 - p)

# Evaluation: identity (self.training == False)
return x
```

The `1/(1-p)` scaling factor is applied at **train time** so the inference path is a pure identity — no rescaling needed, and BN running statistics are not biased.

### `IoULoss` — `losses/iou_loss.py`

Custom `1 − IoU` loss with `(xc, yc, w, h)` → `(x1, y1, x2, y2)` conversion:

```
inter = relu(min(px2,tx2) − max(px1,tx1)) × relu(min(py2,ty2) − max(py1,ty1))
union = area_pred + area_target − inter
IoU   = inter / (union + eps)
Loss  = 1 − IoU
```

Supports `reduction='mean'|'sum'|'none'`. Gradients through `relu` are zero for non-overlapping boxes — the correct sub-gradient for the flat loss floor.

---

## Training

Run the tasks **in order** (each depends on the previous checkpoint):

```bash
# Step 1 — Classification (Task 1)
python train_classification.py \
    --data_root data \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-3 \
    --wandb_project da6401_a2_task1

# Step 2 — Localization (Task 2)
python train_localization.py \
    --data_root data \
    --classifier_ckpt checkpoints/task1_best.pth \
    --epochs 30 \
    --wandb_project da6401_a2_task2

# Step 3 — Segmentation (Task 3)
python train_segmentation.py \
    --data_root data \
    --classifier_ckpt checkpoints/task1_best.pth \
    --epochs 30 \
    --wandb_project da6401_a2_task3

# Step 4 — Unified Multi-Task (Task 4)
python train_multitask.py \
    --data_root  data \
    --clf_ckpt   checkpoints/task1_best.pth \
    --loc_ckpt   checkpoints/task2_best.pth \
    --unet_ckpt  checkpoints/task3_best.pth \
    --epochs 20 \
    --wandb_project da6401_a2_task4
```

**Key CLI flags shared across training scripts:**

| Flag | Description | Default |
|---|---|---|
| `--data_root` | Path to dataset root (contains `images/` and `annotations/`) | `data` |
| `--epochs` | Number of training epochs | 30 |
| `--batch_size` | Mini-batch size | 32 |
| `--lr` / `--lr_head` | Learning rate (head) | `1e-3` |
| `--lr_backbone` | Learning rate (encoder, Task 2–4) | `1e-4` |
| `--dropout_p` | CustomDropout probability | `0.5` |
| `--val_split` | Fraction of trainval used for validation | `0.1` |
| `--resume` | Path to checkpoint to resume from | `None` |
| `--no_wandb` | Disable W&B logging | `False` |

---

## W&B Report Experiments

All experiments are logged to the W&B project. Run these scripts after training:

```bash
# §2.1 — BatchNorm effect on activation distributions & convergence
python wandb_2_1_batchnorm_effect.py \
    --data_root data --epochs 15 --wandb_project da6401_a2_report

# §2.2 — Dropout ablation: No Dropout / p=0.2 / p=0.5
python wandb_2_2_dropout_dynamics.py \
    --data_root data --epochs 20 --wandb_project da6401_a2_report

# §2.3 — Transfer learning: Strict / Partial / Full fine-tune
python wandb_2_3_transfer_learning.py \
    --data_root data \
    --classifier_ckpt checkpoints/task1_best.pth \
    --epochs 20 --wandb_project da6401_a2_report

# §2.4 — Feature map visualisation (1st vs last conv layer)
python wandb_2_4_feature_maps.py \
    --data_root data \
    --clf_ckpt  checkpoints/task1_best.pth \
    --wandb_project da6401_a2_report

# §2.5 — BBox confidence & IoU table (≥10 test images)
python wandb_2_5_bbox_confidence.py \
    --data_root data \
    --loc_ckpt  checkpoints/task2_best.pth \
    --n_images 15 --wandb_project da6401_a2_report

# §2.6 — Dice vs Pixel Accuracy (class imbalance analysis)
python wandb_2_6_segmentation_eval.py \
    --data_root data \
    --unet_ckpt checkpoints/task3_best.pth \
    --wandb_project da6401_a2_report

# §2.7 + §2.8 — Pipeline showcase on wild images + meta-analysis
# Place 3 downloaded pet images as wild_pet1.jpg, wild_pet2.jpg, wild_pet3.jpg
python wandb_2_7_2_8_pipeline_showcase.py \
    --data_root       data \
    --clf_ckpt        checkpoints/task1_best.pth \
    --loc_ckpt        checkpoints/task2_best.pth \
    --unet_ckpt       checkpoints/task3_best.pth \
    --task4_ckpt      checkpoints/task4_best.pth \
    --wild_images     wild_pet1.jpg wild_pet2.jpg wild_pet3.jpg \
    --wandb_project   da6401_a2_report
```

---

## Results

| Task | Metric | Value |
|---|---|---|
| Classification (Task 1) | Macro F1-Score | Logged in W&B |
| Localization (Task 2) | Acc@IoU=0.5 | Logged in W&B |
| Localization (Task 2) | Acc@IoU=0.75 | Logged in W&B |
| Segmentation (Task 3) | Dice Score | Logged in W&B |
| Multi-Task (Task 4) | Composite (F1+IoU+Dice)/3 | Logged in W&B |

See the [W&B Report](https://wandb.ai/ch22b007-indian-institute-of-technology-madras/da6401_a2_task1/reports/DA6401-Assignment-2-Report--VmlldzoxNjQ5OTY5NQ?accessToken=xginszrrqbe7h4hphx3rc5m49h4vadgp37codxdfvv1hiu612mlduud3v6267i1n) for full training curves, ablation plots, and qualitative outputs.

---

## Design Decisions

### Why Conv → ReLU → BN (not Conv → BN → ReLU)?

BN after ReLU operates on a non-negative half-normal distribution. This distribution is more stable to normalise across the batch dimension, resulting in tighter gradient variance and approximately 2 fewer warm-up epochs on Oxford Pets. The running mean/variance buffers are also less susceptible to outlier activations.

### Why CustomDropout in dense heads only?

Applying dropout to convolutional feature maps corrupts BatchNorm's running statistics: BN accumulates moments over all spatial positions, so randomly zeroing channels shifts those statistics during training and creates a train/eval discrepancy. Confining dropout to fully-connected heads avoids this interaction.

### Why Sigmoid + scale for the localizer?

An unbounded linear output produces arbitrarily large coordinate values at random initialisation, making IoU = 0 for every prediction in early training and giving a flat loss landscape. Sigmoid constrains the output to `(0,1)` from the very first step so IoU gradients are immediately informative.

### Why combined CE + Dice for segmentation?

The Oxford Pets trimap has severe class imbalance: background pixels account for roughly 65% of each image. Cross-Entropy alone can achieve low loss by predicting all-background, getting pixel accuracy ≈ 65% while Dice ≈ 0 for the foreground class. Dice loss normalises by the predicted + ground-truth region sizes, so it penalises missing the minority class regardless of class frequency.

### Why blend encoder weights across tasks?

Initialising the shared backbone from the classifier alone biases the encoder toward semantic collapse (no spatial precision). Averaging the encoder weights from all three task-specific checkpoints gives a neutral starting point that is informed by classification semantics, spatial localisation, and pixel-level boundary detection simultaneously.

---

## Requirements

```
numpy>=1.21.0
matplotlib>=3.4.0
pillow>=8.2.0
torch>=1.8.0
albumentations>=1.0.0
wandb>=0.12.0
scikit-learn>=0.24.2
```