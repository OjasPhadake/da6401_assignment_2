"""
wandb_2_1_batchnorm_effect.py  —  W&B Report Section 2.1
=========================================================
Trains VGG11 WITH and WITHOUT BatchNorm, then:
  - Plots activation distributions of the 3rd conv layer on a fixed input
  - Logs training/val loss curves for both runs
  - Logs a W&B Table comparing max-stable LR and convergence epoch

Run from da6401_assignment_2/:
    python wandb_2_1_batchnorm_effect.py \
        --data_root data \
        --epochs 15 \
        --wandb_project da6401_a2_report

The "3rd convolutional layer" in VGG11 = first conv inside stage3
(channels 128→256), which is the 3rd distinct Conv2d in the network.
"""

import argparse
import os
import sys
import random

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
from data.pets_dataset    import OxfordIIITPetDataset, get_transforms
from models.layers        import CustomDropout
from models.vgg11         import VGG11Encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _conv_relu(in_ch, out_ch):
    """Conv block WITHOUT BatchNorm (for ablation)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )


def _conv_bn_relu(in_ch, out_ch):
    """Conv block WITH BatchNorm (standard)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )


# ---------------------------------------------------------------------------
# Build VGG11 encoder with or without BN
# ---------------------------------------------------------------------------

class VGG11EncoderNoBN(nn.Module):
    """VGG11 encoder WITHOUT BatchNorm — ablation for §2.1."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.stage1 = nn.Sequential(_conv_relu(in_channels, 64))
        self.pool1  = nn.MaxPool2d(2, 2)
        self.stage2 = nn.Sequential(_conv_relu(64, 128))
        self.pool2  = nn.MaxPool2d(2, 2)
        self.stage3 = nn.Sequential(_conv_relu(128, 256), _conv_relu(256, 256))
        self.pool3  = nn.MaxPool2d(2, 2)
        self.stage4 = nn.Sequential(_conv_relu(256, 512), _conv_relu(512, 512))
        self.pool4  = nn.MaxPool2d(2, 2)
        self.stage5 = nn.Sequential(_conv_relu(512, 512), _conv_relu(512, 512))
        self.pool5  = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.pool4(self.stage4(x))
        x = self.pool5(self.stage5(x))
        return self.adaptive_pool(x)


class VGG11Classifier(nn.Module):
    """Full VGG11 classifier with switchable BN."""

    def __init__(self, use_bn=True, num_classes=37, dropout_p=0.5):
        super().__init__()
        self.encoder = VGG11Encoder() if use_bn else VGG11EncoderNoBN()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Activation hook — captures output of the 3rd conv layer
# (stage3[0][0] = first Conv2d inside stage3 = 3rd Conv2d in the network)
# ---------------------------------------------------------------------------

class ActivationCapture:
    def __init__(self):
        self.activations = None

    def hook(self, module, input, output):
        self.activations = output.detach().cpu()

    def register(self, model):
        # stage3 is a Sequential of two _conv_bn_relu blocks
        # stage3[0] is the first block; stage3[0][0] is the Conv2d
        target = model.encoder.stage3[0][0]   # 3rd Conv2d in VGG11
        return target.register_forward_hook(self.hook)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

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
        loss = crit(model(imgs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot_loss += loss.item() * imgs.size(0)
        correct  += (model(imgs).argmax(1) == labels).sum().item()
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
        logits  = model(imgs)
        tot_loss += crit(logits, labels).item() * imgs.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    return tot_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(use_bn, args, fixed_batch, device, run_name):
    """Train one model variant and return per-epoch metrics."""
    model = VGG11Classifier(use_bn=use_bn, num_classes=37,
                             dropout_p=0.5).to(device)

    # Log to W&B
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={"use_bn": use_bn, "lr": args.lr,
                "epochs": args.epochs, "batch_size": args.batch_size},
        reinit=True,
    )

    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6
    )

    # Register activation hook on 3rd conv layer
    capture = ActivationCapture()
    handle  = capture.register(model)

    # Capture activations on fixed batch BEFORE training
    model.eval()
    with torch.no_grad():
        model(fixed_batch.to(device))
    acts_before = capture.activations.float().numpy()

    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, args._train_loader,
                                      crit, opt, device)
        val_loss, val_acc = eval_epoch(model, args._val_loader, crit, device)
        sched.step()

        wandb.log({"epoch": epoch,
                   "train/loss": tr_loss, "train/acc": tr_acc,
                   "val/loss":   val_loss, "val/acc":   val_acc,
                   "lr": sched.get_last_lr()[0]})

        history.append({"epoch": epoch,
                        "train_loss": tr_loss, "val_loss": val_loss,
                        "train_acc":  tr_acc,  "val_acc":  val_acc})

        print(f"  [{run_name}] E{epoch:02d} "
              f"tr={tr_loss:.4f}/{tr_acc:.3f}  "
              f"val={val_loss:.4f}/{val_acc:.3f}")

    # Capture activations AFTER training
    model.eval()
    with torch.no_grad():
        model(fixed_batch.to(device))
    acts_after = capture.activations.float().numpy()
    handle.remove()

    # ---- Plot activation distributions ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, acts, title in zip(
        axes,
        [acts_before, acts_after],
        ["Before Training", "After Training"]
    ):
        flat = acts.flatten()
        ax.hist(flat, bins=80, color="#4C72B0", alpha=0.8, density=True)
        ax.axvline(flat.mean(), color="red",    linestyle="--",
                   label=f"mean={flat.mean():.3f}")
        ax.axvline(flat.std(),  color="orange", linestyle=":",
                   label=f"std={flat.std():.3f}")
        ax.set_title(f"{title}\n{'With BN' if use_bn else 'No BN'}")
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle(f"3rd Conv Layer Activations — "
                 f"{'With BatchNorm' if use_bn else 'Without BatchNorm'}",
                 fontsize=13)
    plt.tight_layout()

    wandb.log({"activation_distribution": wandb.Image(fig)})
    plt.savefig(f"act_dist_{'bn' if use_bn else 'nobn'}.png", dpi=120)
    plt.close()

    wandb.finish()
    return history, {"mean_before": float(acts_before.mean()),
                     "std_before":  float(acts_before.std()),
                     "mean_after":  float(acts_after.mean()),
                     "std_after":   float(acts_after.std())}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     default="data")
    p.add_argument("--epochs",        type=int,   default=15)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--val_split",     type=float, default=0.1)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--wandb_project", default="da6401_a2_report")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build datasets
    train_tfm = get_transforms("train", 224)
    val_tfm   = get_transforms("val",   224)
    full_tv   = OxfordIIITPetDataset(args.data_root, "train", train_tfm,
                                     load_bbox=False, load_mask=False)
    n_val     = int(len(full_tv) * args.val_split)
    tr_ds, val_ds = random_split(
        full_tv, [len(full_tv)-n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    args._train_loader = DataLoader(
        tr_ds,  args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    args._val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Fixed batch for activation comparison (same images for both models)
    fixed_batch_list = []
    for i, b in enumerate(args._val_loader):
        fixed_batch_list.append(b["image"][:8])
        break
    fixed_batch = fixed_batch_list[0]   # [8, 3, 224, 224]
    print(f"Fixed batch shape: {fixed_batch.shape}")

    # Run WITH BatchNorm
    print("\n=== Run 1: WITH BatchNorm ===")
    hist_bn,  stats_bn  = run_experiment(
        use_bn=True,  args=args, fixed_batch=fixed_batch,
        device=device, run_name="vgg11_with_bn"
    )

    # Run WITHOUT BatchNorm
    print("\n=== Run 2: WITHOUT BatchNorm ===")
    hist_nobn, stats_nobn = run_experiment(
        use_bn=False, args=args, fixed_batch=fixed_batch,
        device=device, run_name="vgg11_no_bn"
    )

    # Summary comparison plot
    wandb.init(project=args.wandb_project, name="section_2_1_summary",
               reinit=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    ax.plot([h["epoch"]      for h in hist_bn],
            [h["train_loss"] for h in hist_bn],
            "b-", label="BN train")
    ax.plot([h["epoch"]      for h in hist_bn],
            [h["val_loss"]   for h in hist_bn],
            "b--", label="BN val")
    ax.plot([h["epoch"]      for h in hist_nobn],
            [h["train_loss"] for h in hist_nobn],
            "r-", label="No-BN train")
    ax.plot([h["epoch"]      for h in hist_nobn],
            [h["val_loss"]   for h in hist_nobn],
            "r--", label="No-BN val")
    ax.set_title("Loss Curves: BN vs No-BN")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # Accuracy curves
    ax = axes[1]
    ax.plot([h["epoch"]     for h in hist_bn],
            [h["val_acc"]   for h in hist_bn],
            "b-o", label="BN val_acc")
    ax.plot([h["epoch"]     for h in hist_nobn],
            [h["val_acc"]   for h in hist_nobn],
            "r-s", label="No-BN val_acc")
    ax.set_title("Validation Accuracy: BN vs No-BN")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    wandb.log({"summary_curves": wandb.Image(fig)})
    plt.savefig("summary_bn_comparison.png", dpi=120)
    plt.close()

    # Summary table
    tbl = wandb.Table(columns=["Run", "Act Mean (after)", "Act Std (after)",
                                "Best Val Acc", "Epoch of Best Val Acc"])
    best_bn    = max(hist_bn,    key=lambda h: h["val_acc"])
    best_nobn  = max(hist_nobn,  key=lambda h: h["val_acc"])
    tbl.add_data("With BN",    round(stats_bn["mean_after"],  4),
                 round(stats_bn["std_after"],   4),
                 round(best_bn["val_acc"],  4), best_bn["epoch"])
    tbl.add_data("Without BN", round(stats_nobn["mean_after"], 4),
                 round(stats_nobn["std_after"],  4),
                 round(best_nobn["val_acc"], 4), best_nobn["epoch"])
    wandb.log({"bn_comparison_table": tbl})
    wandb.finish()

    print("\n=== Section 2.1 complete ===")
    print(f"BN    best val acc: {best_bn['val_acc']:.4f}  "
          f"at epoch {best_bn['epoch']}")
    print(f"No-BN best val acc: {best_nobn['val_acc']:.4f}  "
          f"at epoch {best_nobn['epoch']}")


if __name__ == "__main__":
    main()