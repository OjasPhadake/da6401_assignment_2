"""Unified multi-task model
"""

import warnings
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg11        import VGG11Encoder
from .layers       import CustomDropout
from .segmentation import DecoderBlock, _conv_bn_relu


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    Architecture
    ============
    One VGG11Encoder backbone is shared across all three task heads.
    A single forward(x) call simultaneously produces:

        classification → [B, num_breeds]          37-class logits
        localization   → [B, 4]                   pixel-space (xc,yc,w,h)
        segmentation   → [B, seg_classes, H, W]   per-pixel logits

    The encoder runs ONCE and its output is routed to three independent heads.

    Shared encoder strategy
    -----------------------
    We initialise the shared backbone by loading weights from the best
    available task-specific checkpoint (Task 1 classifier is preferred, as
    classification provides the richest semantic gradients; Task 3 U-Net
    encoder is second choice, Task 2 localizer is the fallback).

    During multi-task fine-tuning the shared encoder is updated by the sum
    of gradients from all three loss terms.  For Oxford Pets this works well
    because the three tasks are co-located — the same animal body is the
    subject of all three predictions.  Empirically, gradient interference is
    mild: the classification and segmentation gradients both push the encoder
    toward sharper object-vs-background representations.

    Weight loading
    --------------
    __init__ accepts three checkpoint paths produced by the task-specific
    training scripts.  For each checkpoint it:
        1. Downloads the file from Google Drive via gdown (if not already
           present on disk).
        2. Extracts only the relevant sub-module weights.
        3. Copies them into the unified model.

    If a checkpoint is missing or corrupted a warning is printed and that
    component is left with its random initialisation — the model can still
    be trained.

    Decoder key alignment
    ---------------------
    VGG11UNet stores its decoder as:
        bridge.*, up5.*, up4.*, up3.*, up2.*, up1.*, seg_head.*
    MultiTaskPerceptionModel uses identical attribute names and DecoderBlock
    definitions, so state_dicts transfer directly with strict=False.

    Args:
        num_breeds      (int):   Classification output classes. Default 37.
        seg_classes     (int):   Segmentation output classes. Default 3.
        in_channels     (int):   Input image channels. Default 3.
        classifier_path (str):   Path / download destination for Task-1 ckpt.
        localizer_path  (str):   Path / download destination for Task-2 ckpt.
        unet_path       (str):   Path / download destination for Task-3 ckpt.
        image_size      (int):   Square input size (pixels). Default 224.
        dropout_p       (float): CustomDropout probability in all heads.
    """

    def __init__(
        self,
        num_breeds:      int   = 37,
        seg_classes:     int   = 3,
        in_channels:     int   = 3,
        classifier_path: str   = "classifier.pth",
        localizer_path:  str   = "localizer.pth",
        unet_path:       str   = "unet.pth",
        image_size:      int   = 224,
        dropout_p:       float = 0.5,
    ):
        """
        Initialize the shared backbone/heads using these trained weights.

        Args:
            num_breeds:      Number of output classes for classification head.
            seg_classes:     Number of output classes for segmentation head.
            in_channels:     Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path:  Path to trained localizer weights.
            unet_path:       Path to trained unet weights.
        """
        # ------------------------------------------------------------------
        # Download checkpoints from Google Drive.
        # Replace the placeholder IDs with your actual Drive file IDs before
        # submission.  gdown is idempotent: if the file already exists on
        # disk it is not re-downloaded.
        # ------------------------------------------------------------------
        import gdown
        gdown.download(id="1K3XEU26fbmLZcL8b215R2AAEWqGpdRR8", 
                       output=classifier_path, quiet=False)
        gdown.download(id="1PTe_K_CjgcMLkOpU-GnJXm8ZQ35BEUx6",
                       output=localizer_path,  quiet=False)
        gdown.download(id="1P8g73QUmIAt6yXJjyeg5Zy7Ux1qVxnLR",
                       output=unet_path,       quiet=False)

        super().__init__()

        self.image_size  = image_size
        self.num_breeds  = num_breeds
        self.seg_classes = seg_classes

        # ==================================================================
        # Shared backbone
        # ==================================================================
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ==================================================================
        # Head 1 — Classification
        # Mirrors ClassificationHead in models/classification.py exactly.
        # Input:  bottleneck [B, 512, 7, 7]
        # Output: logits     [B, num_breeds]
        # ==================================================================
        self.cls_head = nn.Sequential(
            nn.Flatten(),                                   # [B, 25088]

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_breeds),
        )

        # ==================================================================
        # Head 2 — Localization
        # Mirrors LocalizationHead in models/localization.py exactly.
        # Input:  bottleneck [B, 512, 7, 7]
        # Output: bbox       [B, 4]  in pixel coords (Sigmoid × image_size)
        # ==================================================================
        self.loc_head = nn.Sequential(
            nn.Flatten(),                                   # [B, 25088]

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 4),
            nn.Sigmoid(),                                   # → (0, 1)
        )

        # ==================================================================
        # Head 3 — Segmentation decoder
        # Mirrors the decoder in VGG11UNet (models/segmentation.py) exactly:
        #   - Same DecoderBlock class (imported from segmentation.py)
        #   - Same attribute names:  bridge, up5, up4, up3, up2, up1, seg_head
        # This means Task-3 checkpoint weights transfer with zero key remapping.
        #
        # Input:  bottleneck [B,512,7,7] + skip maps from encoder
        # Output: logits     [B, seg_classes, H, W]
        # ==================================================================
        self.bridge = nn.Sequential(
            _conv_bn_relu(512, 512),
            CustomDropout(p=dropout_p),
        )
        self.up5      = DecoderBlock(in_ch=512, skip_ch=512, out_ch=512)
        self.up4      = DecoderBlock(in_ch=512, skip_ch=512, out_ch=256)
        self.up3      = DecoderBlock(in_ch=256, skip_ch=256, out_ch=128)
        self.up2      = DecoderBlock(in_ch=128, skip_ch=128, out_ch=64)
        self.up1      = DecoderBlock(in_ch=64,  skip_ch=64,  out_ch=64)
        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

        # ==================================================================
        # Initialise all newly created weights before loading pretrained ones
        # ==================================================================
        self._init_heads()

        # ==================================================================
        # Load pretrained weights from the three task checkpoints
        # ==================================================================
        self._load_pretrained(
            classifier_path, localizer_path, unet_path,
            num_breeds, seg_classes, in_channels, dropout_p,
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_heads(self) -> None:
        """Xavier / Kaiming init for all head parameters."""
        for m in list(self.cls_head.modules()) + \
                 list(self.loc_head.modules()) + \
                 list(self.bridge.modules())   + \
                 list(self.up5.modules())      + \
                 list(self.up4.modules())      + \
                 list(self.up3.modules())      + \
                 list(self.up2.modules())      + \
                 list(self.up1.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # seg_head
        nn.init.xavier_uniform_(self.seg_head.weight)
        nn.init.zeros_(self.seg_head.bias)

    # ------------------------------------------------------------------
    # Checkpoint loading helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_load(path: str):
        """Load checkpoint; return state_dict or None on failure."""
        try:
            ckpt = torch.load(path, map_location="cpu")
            return ckpt.get("model_state_dict", ckpt)
        except Exception as exc:
            warnings.warn(
                f"Could not load '{path}': {exc}. "
                "Using random initialisation for this component."
            )
            return None

    @staticmethod
    def _filter_keys(sd: dict, prefix: str, strip: bool = True) -> dict:
        """Return sub-dict whose keys start with `prefix`.

        If strip=True, remove the prefix from the returned keys so the
        sub-dict can be loaded directly into the target sub-module.
        """
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):] if strip else k
                out[new_key] = v
        return out

    def _load_pretrained(
        self,
        clf_path:    str,
        loc_path:    str,
        unet_path:   str,
        num_breeds:  int,
        seg_classes: int,
        in_channels: int,
        dropout_p:   float,
    ) -> None:
        """
        Copy weights from three independently trained task checkpoints.

        Loading order and encoder blending strategy
        --------------------------------------------
        1. Load Task-1 classifier  → initialise shared encoder + cls_head.
        2. Load Task-2 localizer   → copy loc_head; blend encoder (mean of
                                     classifier and localizer encoder weights).
        3. Load Task-3 U-Net       → copy decoder; blend encoder (mean of all
                                     three encoder weight sets).

        Blending the encoder from all three tasks gives a neutral starting
        point that is informed by all three supervision signals, rather than
        biasing the shared backbone toward any one task.
        """
        clf_sd  = self._safe_load(clf_path)
        loc_sd  = self._safe_load(loc_path)
        unet_sd = self._safe_load(unet_path)

        loaded_sources = 0   # count of successful encoder loads

        # ---- Task 1: classifier ------------------------------------------
        if clf_sd is not None:
            # Encoder
            enc_sd = self._filter_keys(clf_sd, "encoder.")
            self.encoder.load_state_dict(enc_sd, strict=False)
            loaded_sources = 1

            # Classification head
            # Saved as:  head.head.<idx>.<param>
            # Target  :  cls_head.<idx>.<param>
            cls_sd = self._filter_keys(clf_sd, "head.head.")
            m, u = self.cls_head.load_state_dict(cls_sd, strict=False)
            if m: print(f"[MTLoad] cls_head missing  : {m}")
            if u: print(f"[MTLoad] cls_head unexpected: {u}")
            print(f"[MTLoad] Loaded cls_head from '{clf_path}'")

        # ---- Task 2: localizer -------------------------------------------
        if loc_sd is not None:
            # Localization head
            # Saved as:  regression_head.head.<idx>.<param>
            # Target  :  loc_head.<idx>.<param>
            loc_h_sd = self._filter_keys(loc_sd, "regression_head.head.")
            m, u = self.loc_head.load_state_dict(loc_h_sd, strict=False)
            if m: print(f"[MTLoad] loc_head missing  : {m}")
            if u: print(f"[MTLoad] loc_head unexpected: {u}")
            print(f"[MTLoad] Loaded loc_head from '{loc_path}'")

            # Blend encoder: running mean of all loaded sources
            loc_enc_sd = self._filter_keys(loc_sd, "encoder.")
            self._blend_encoder(loc_enc_sd, loaded_sources)
            loaded_sources += 1

        # ---- Task 3: U-Net -----------------------------------------------
        if unet_sd is not None:
            # Decoder components — attribute names are IDENTICAL between
            # VGG11UNet and MultiTaskPerceptionModel, so keys transfer as-is.
            decoder_attrs = ("bridge", "up5", "up4", "up3", "up2", "up1",
                             "seg_head")
            for attr in decoder_attrs:
                sub_sd = self._filter_keys(unet_sd, f"{attr}.")
                if sub_sd:
                    m, u = getattr(self, attr).load_state_dict(
                        sub_sd, strict=False
                    )
                    if m: print(f"[MTLoad] {attr} missing  : {m}")
            print(f"[MTLoad] Loaded segmentation decoder from '{unet_path}'")

            # Blend encoder
            unet_enc_sd = self._filter_keys(unet_sd, "encoder.")
            self._blend_encoder(unet_enc_sd, loaded_sources)
            loaded_sources += 1

        if loaded_sources == 0:
            print("[MTLoad] No checkpoints loaded — using random initialisation.")
        else:
            print(f"[MTLoad] Encoder blended from {loaded_sources} source(s).")

    def _blend_encoder(self, new_enc_sd: dict, n_existing: int) -> None:
        """Running average blend: (current × n + new) / (n + 1)."""
        if not new_enc_sd:
            return
        current_sd = self.encoder.state_dict()
        blended = {}
        for k, v in current_sd.items():
            if k in new_enc_sd and v.dtype.is_floating_point:
                blended[k] = (v * n_existing + new_enc_sd[k]) / (n_existing + 1)
            else:
                blended[k] = v
        self.encoder.load_state_dict(blended, strict=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task model.

        A SINGLE encoder forward pass feeds all three task heads simultaneously.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization':   [B, 4] bounding box tensor in pixel space
                                       (x_center, y_center, width, height).
            - 'segmentation':   [B, seg_classes, H, W] segmentation logits.
        """
        H_in, W_in = x.shape[2], x.shape[3]

        # ---- Single shared encoder pass ----------------------------------
        # return_features=True gives us:
        #   bottleneck [B,512,7,7] — for cls + loc heads
        #   skips dict             — for segmentation decoder
        bottleneck, skips = self.encoder(x, return_features=True)

        # ---- Head 1: Classification --------------------------------------
        # bottleneck → Flatten → FC → logits
        cls_logits = self.cls_head(bottleneck)            # [B, num_breeds]

        # ---- Head 2: Localization ----------------------------------------
        # bottleneck → Flatten → FC → Sigmoid → scale to pixels
        norm_boxes = self.loc_head(bottleneck)            # [B, 4] ∈ (0,1)
        bbox = norm_boxes * self.image_size               # [B, 4] pixels

        # ---- Head 3: Segmentation ----------------------------------------
        # bottleneck → bridge → decoder blocks with skip connections
        d = self.bridge(bottleneck)
        d = self.up5(d, skips["s5"])                      # [B,512,~H/16,~W/16]
        d = self.up4(d, skips["s4"])                      # [B,256,~H/8, ~W/8 ]
        d = self.up3(d, skips["s3"])                      # [B,128,~H/4, ~W/4 ]
        d = self.up2(d, skips["s2"])                      # [B, 64,~H/2, ~W/2 ]
        d = self.up1(d, skips["s1"])                      # [B, 64,~H,   ~W   ]
        seg_logits = self.seg_head(d)                     # [B, C, ~H, ~W]

        # Guarantee seg output exactly matches input spatial resolution.
        # (Handles ±1 pixel drift from AdaptiveAvgPool → ConvTranspose2d)
        if seg_logits.shape[2:] != (H_in, W_in):
            seg_logits = F.interpolate(
                seg_logits, size=(H_in, W_in),
                mode="bilinear", align_corners=False,
            )

        return {
            "classification": cls_logits,   # [B, num_breeds]
            "localization":   bbox,          # [B, 4]  pixel xywh
            "segmentation":   seg_logits,    # [B, seg_classes, H, W]
        }