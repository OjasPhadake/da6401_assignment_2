# """Localization modules
# """

# import torch
# import torch.nn as nn

# from .vgg11 import VGG11Encoder

# class VGG11Localizer(nn.Module):
#     """VGG11-based localizer."""

#     def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
#         """
#         Initialize the VGG11Localizer model.

#         Args:
#             in_channels: Number of input channels.
#             dropout_p: Dropout probability for the localization head.
#         """
#         pass

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass for localization model.
#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].

#         Returns:
#             Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
#         """
#         # TODO: Implement forward pass.
#         raise NotImplementedError("Implement VGG11Localizer.forward")


"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class LocalizationHead(nn.Module):
    """Regression head that maps the VGG11 bottleneck to pixel-space bbox coords.

    Architecture:
        Flatten
        → FC(25088, 4096) → ReLU → BN1d → CustomDropout
        → FC(4096,   512) → ReLU → BN1d
        → FC(512,      4) → Sigmoid  (normalised output ∈ [0,1])

    The Sigmoid output is then multiplied by (image_w, image_h, image_w, image_h)
    inside VGG11Localizer.forward to produce pixel-space coordinates.

    Why Sigmoid + scale instead of unbounded linear?
    -------------------------------------------------
    Clamping the raw regression output to [0, 1] via Sigmoid prevents the
    network from predicting boxes outside the image boundary during early
    training when weights are random. Multiplying by the known input size
    recovers pixel-space values with no loss of precision. An unbounded
    linear head risks exploding coordinate gradients in the first few epochs
    before the loss landscape settles, especially when IoU loss is used
    (its gradient magnitude is large for non-overlapping boxes).
    """

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 4),
            nn.Sigmoid(),   # → [0, 1]; scaled to pixel space in parent forward
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
        return self.head(x)     # [B, 4] ∈ [0,1]


class VGG11Localizer(nn.Module):
    """VGG11-based localizer.

    Encoder strategy — fine-tuning vs freezing
    -------------------------------------------
    By default (freeze_encoder=False) the entire VGG11 backbone is
    fine-tuned end-to-end. Empirically this outperforms freezing on the
    Oxford-IIIT Pet head-bounding-box task because:

      * The pre-trained features were trained for *classification*, not
        spatial localisation. The later convolutional stages learn
        semantically rich but position-collapsed representations; allowing
        gradient flow lets them reorganise toward spatial sensitivity.
      * The dataset is large enough (~3 600 train images) that catastrophic
        forgetting is not a concern when a small learning rate is used for
        the backbone.

    Passing freeze_encoder=True reproduces the "Strict Feature Extractor"
    baseline required for the W&B transfer-learning comparison (§2.3).

    Output coordinate space
    -----------------------
    The forward pass returns (x_center, y_center, width, height) in
    **pixel coordinates** relative to the input image as resized to
    `image_size × image_size` (default 224). The IoULoss operates in this
    same space — target bbox annotations are scaled from their normalised
    [0,1] form by (image_size, image_size, image_size, image_size) before
    the loss is computed.
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
            image_size:     Spatial size of the (square) input image. Used
                            to scale normalised Sigmoid output to pixel space.
        """
        super().__init__()

        self.image_size = image_size

        self.encoder = VGG11Encoder(in_channels=in_channels)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.regression_head = LocalizationHead(dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in
            (x_center, y_center, width, height) format in original image
            pixel space (not normalised values).
        """
        bottleneck = self.encoder(x, return_features=False)   # [B, 512, 7, 7]
        norm_boxes = self.regression_head(bottleneck)          # [B, 4] ∈ [0,1]

        # Scale from [0,1] → pixel space using the known (square) image size.
        # All four coordinates share the same scale because the image is square.
        scale = torch.tensor(
            [self.image_size, self.image_size,
             self.image_size, self.image_size],
            dtype=norm_boxes.dtype,
            device=norm_boxes.device,
        )
        return norm_boxes * scale    # [B, 4] in pixel coordinates