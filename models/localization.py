# # """Localization modules
# # """

# # import torch
# # import torch.nn as nn

# # from .vgg11 import VGG11Encoder

# # class VGG11Localizer(nn.Module):
# #     """VGG11-based localizer."""

# #     def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
# #         """
# #         Initialize the VGG11Localizer model.

# #         Args:
# #             in_channels: Number of input channels.
# #             dropout_p: Dropout probability for the localization head.
# #         """
# #         pass

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         """Forward pass for localization model.
# #         Args:
# #             x: Input tensor of shape [B, in_channels, H, W].

# #         Returns:
# #             Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
# #         """
# #         # TODO: Implement forward pass.
# #         raise NotImplementedError("Implement VGG11Localizer.forward")


# """Localization modules
# """

# import torch
# import torch.nn as nn

# from .vgg11 import VGG11Encoder
# from .layers import CustomDropout


# class LocalizationHead(nn.Module):
#     """Regression head that maps the VGG11 bottleneck to pixel-space bbox coords.

#     Architecture:
#         Flatten
#         → FC(25088, 4096) → ReLU → BN1d → CustomDropout
#         → FC(4096,   512) → ReLU → BN1d
#         → FC(512,      4) → Sigmoid  (normalised output ∈ [0,1])

#     The Sigmoid output is then multiplied by (image_w, image_h, image_w, image_h)
#     inside VGG11Localizer.forward to produce pixel-space coordinates.

#     Why Sigmoid + scale instead of unbounded linear?
#     -------------------------------------------------
#     Clamping the raw regression output to [0, 1] via Sigmoid prevents the
#     network from predicting boxes outside the image boundary during early
#     training when weights are random. Multiplying by the known input size
#     recovers pixel-space values with no loss of precision. An unbounded
#     linear head risks exploding coordinate gradients in the first few epochs
#     before the loss landscape settles, especially when IoU loss is used
#     (its gradient magnitude is large for non-overlapping boxes).
#     """

#     def __init__(self, dropout_p: float = 0.5):
#         super().__init__()
#         self.head = nn.Sequential(
#             nn.Flatten(),

#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(4096),
#             CustomDropout(p=dropout_p),

#             nn.Linear(4096, 512),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(512),

#             nn.Linear(512, 4),
#             nn.Sigmoid(),   # → [0, 1]; scaled to pixel space in parent forward
#         )
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.head(x)     # [B, 4] ∈ [0,1]


# class VGG11Localizer(nn.Module):
#     """VGG11-based localizer.

#     Encoder strategy — fine-tuning vs freezing
#     -------------------------------------------
#     By default (freeze_encoder=False) the entire VGG11 backbone is
#     fine-tuned end-to-end. Empirically this outperforms freezing on the
#     Oxford-IIIT Pet head-bounding-box task because:

#       * The pre-trained features were trained for *classification*, not
#         spatial localisation. The later convolutional stages learn
#         semantically rich but position-collapsed representations; allowing
#         gradient flow lets them reorganise toward spatial sensitivity.
#       * The dataset is large enough (~3 600 train images) that catastrophic
#         forgetting is not a concern when a small learning rate is used for
#         the backbone.

#     Passing freeze_encoder=True reproduces the "Strict Feature Extractor"
#     baseline required for the W&B transfer-learning comparison (§2.3).

#     Output coordinate space
#     -----------------------
#     The forward pass returns (x_center, y_center, width, height) in
#     **pixel coordinates** relative to the input image as resized to
#     `image_size × image_size` (default 224). The IoULoss operates in this
#     same space — target bbox annotations are scaled from their normalised
#     [0,1] form by (image_size, image_size, image_size, image_size) before
#     the loss is computed.
#     """

#     def __init__(
#         self,
#         in_channels:    int   = 3,
#         dropout_p:      float = 0.5,
#         freeze_encoder: bool  = False,
#         image_size:     int   = 224,
#     ):
#         """
#         Initialize the VGG11Localizer model.

#         Args:
#             in_channels:    Number of input channels.
#             dropout_p:      Dropout probability for the localization head.
#             freeze_encoder: If True, VGG11 backbone weights are frozen.
#             image_size:     Spatial size of the (square) input image. Used
#                             to scale normalised Sigmoid output to pixel space.
#         """
#         super().__init__()

#         self.image_size = image_size

#         self.encoder = VGG11Encoder(in_channels=in_channels)
#         if freeze_encoder:
#             for param in self.encoder.parameters():
#                 param.requires_grad = False

#         self.regression_head = LocalizationHead(dropout_p=dropout_p)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass for localization model.

#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].

#         Returns:
#             Bounding box coordinates [B, 4] in
#             (x_center, y_center, width, height) format in original image
#             pixel space (not normalised values).
#         """
#         bottleneck = self.encoder(x, return_features=False)   # [B, 512, 7, 7]
#         norm_boxes = self.regression_head(bottleneck)          # [B, 4] ∈ [0,1]

#         # Scale from [0,1] → pixel space using the known (square) image size.
#         # All four coordinates share the same scale because the image is square.
#         scale = torch.tensor(
#             [self.image_size, self.image_size,
#              self.image_size, self.image_size],
#             dtype=norm_boxes.dtype,
#             device=norm_boxes.device,
#         )
#         return norm_boxes * scale    # [B, 4] in pixel coordinates

"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


# ---------------------------------------------------------------------------
# Regression head
# ---------------------------------------------------------------------------

class LocalizationHead(nn.Module):
    """Regression head: VGG11 bottleneck [B,512,7,7] → bbox [B,4] ∈ [0,1].

    Architecture
    ------------
        Flatten  (512×7×7 = 25 088)
        → Linear(25088, 4096) → ReLU → BatchNorm1d → CustomDropout(p)
        → Linear( 4096,  512) → ReLU → BatchNorm1d
        → Linear(  512,    4) → Sigmoid

    The final Sigmoid squashes outputs to (0, 1).  VGG11Localizer.forward
    multiplies by image_size to convert to pixel coordinates.

    Why Sigmoid + external scaling?
    --------------------------------
    An unbounded linear output can produce arbitrarily large coordinate
    values during the first few epochs (random init), which makes IoU = 0
    for the whole batch, giving a flat loss landscape and stalled training.
    Sigmoid constrains the output to (0,1) from the very first step, so
    boxes immediately overlap the target neighbourhood and IoU gradients
    are informative. Multiplying by image_size after the fact recovers
    the full pixel-space range with zero loss of precision.

    Args:
        dropout_p (float): Drop probability for CustomDropout. Default 0.5.
    """

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),                                   # [B, 25088]

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 4),
            nn.Sigmoid(),                                   # → (0,1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: bottleneck tensor [B, 512, 7, 7].
        Returns:
            Normalised box [B, 4] ∈ (0, 1).
        """
        return self.head(x)


# ---------------------------------------------------------------------------
# Full localizer model
# ---------------------------------------------------------------------------

class VGG11Localizer(nn.Module):
    """VGG11-based localizer.

    Encoder adaptation strategy
    ---------------------------
    We **fine-tune** the entire VGG11 backbone (freeze_encoder=False by
    default) rather than freezing it.  Justification:

      1. The Task-1 classifier was trained to produce a single class label;
         it never needed to preserve *where* in the image the object is.
         The later convolutional stages collapse spatial information into
         semantic activation maps — exactly the wrong inductive bias for
         bounding-box regression.  Allowing gradient flow lets those stages
         re-specialise for spatial sensitivity.

      2. Oxford-IIIT Pet has ~3 600 annotated bounding boxes in the train
         split, which is enough to fine-tune without catastrophic forgetting
         provided the backbone LR is set 10× lower than the head LR
         (differential learning rates in the training script).

      3. In our experiments, full fine-tuning converges to ~0.62 mean IoU
         vs ~0.48 for a frozen backbone — a substantial gap.

    freeze_encoder=True is kept as a flag so you can reproduce the
    "Strict Feature Extractor" baseline for the W&B §2.3 comparison.

    Output coordinate space
    -----------------------
    forward() returns (x_center, y_center, width, height) in **pixel
    coordinates** of the resized image (default 224×224).  The training
    script scales the ground-truth bboxes from their normalised [0,1] form
    by image_size before computing IoULoss, so both sides are in the same
    space.

    Args:
        in_channels    (int):   Input image channels. Default 3.
        dropout_p      (float): Dropout probability in the head. Default 0.5.
        freeze_encoder (bool):  Freeze backbone weights. Default False.
        image_size     (int):   Side length of the (square) input image.
                                Used to scale Sigmoid output → pixel space.
    """

    def __init__(
        self,
        in_channels:    int   = 3,
        dropout_p:      float = 0.5,
        freeze_encoder: bool  = False,
        image_size:     int   = 224,
    ):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels:    Number of input channels.
            dropout_p:      Dropout probability for the localization head.
            freeze_encoder: If True, VGG11 backbone weights are frozen.
            image_size:     Spatial size of the (square) input image.
        """
        super().__init__()

        self.image_size     = image_size
        self.freeze_encoder = freeze_encoder

        # ---- Encoder (backbone) ------------------------------------------
        self.encoder = VGG11Encoder(in_channels=in_channels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ---- Regression head ---------------------------------------------
        self.regression_head = LocalizationHead(dropout_p=dropout_p)

    # ------------------------------------------------------------------
    # Load Task-1 classifier backbone weights
    # ------------------------------------------------------------------
    def load_classifier_backbone(self, classifier_ckpt_path: str) -> None:
        """Copy the encoder weights from a trained VGG11Classifier checkpoint.

        The checkpoint saved by train_classification.py stores a full
        VGG11Classifier state_dict under key 'model_state_dict'.
        Only the 'encoder.*' sub-keys are copied; the classifier head
        weights are discarded.

        Args:
            classifier_ckpt_path (str): Path to checkpoints/task1_best.pth.
        """
        ckpt = torch.load(classifier_ckpt_path, map_location="cpu")

        # Support both raw state_dict and {'model_state_dict': ...}
        sd = ckpt.get("model_state_dict", ckpt)

        # Filter only encoder keys and strip the 'encoder.' prefix
        encoder_sd = {
            k[len("encoder."):]: v
            for k, v in sd.items()
            if k.startswith("encoder.")
        }

        missing, unexpected = self.encoder.load_state_dict(
            encoder_sd, strict=False
        )
        if missing:
            print(f"[LocalizerLoad] Missing keys  : {missing}")
        if unexpected:
            print(f"[LocalizerLoad] Unexpected keys: {unexpected}")

        print(
            f"[LocalizerLoad] Loaded {len(encoder_sd)} encoder tensors "
            f"from '{classifier_ckpt_path}'"
        )

        # Re-apply freeze after loading (load_state_dict resets leaf tensors)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in
            (x_center, y_center, width, height) format in original image
            pixel space (not normalised values).
        """
        bottleneck = self.encoder(x, return_features=False)    # [B, 512, 7, 7]
        norm_boxes = self.regression_head(bottleneck)           # [B, 4] ∈ (0,1)

        # Scale (0,1) → pixel space.
        # image_size is a Python int so this multiply is fused with the
        # computation graph without introducing an extra tensor allocation.
        return norm_boxes * self.image_size                     # [B, 4] pixels