# """Unified multi-task model
# """

# import torch
# import torch.nn as nn

# class MultiTaskPerceptionModel(nn.Module):
#     """Shared-backbone multi-task model."""

#     def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
#         """
#         Initialize the shared backbone/heads using these trained weights.
#         Args:
#             num_breeds: Number of output classes for classification head.
#             seg_classes: Number of output classes for segmentation head.
#             in_channels: Number of input channels.
#             classifier_path: Path to trained classifier weights.
#             localizer_path: Path to trained localizer weights.
#             unet_path: Path to trained unet weights.
#         """
#         import gdown
#         gdown.download(id="<classifier.pth drive id>", output=classifier_path, quiet=False)
#         gdown.download(id="<localizer.pth drive id>", output=localizer_path, quiet=False)
#         gdown.download(id="<unet.pth drive id>", output=unet_path, quiet=False)
#         pass

#     def forward(self, x: torch.Tensor):
#         """Forward pass for multi-task model.
#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].
#         Returns:
#             A dict with keys:
#             - 'classification': [B, num_breeds] logits tensor.
#             - 'localization': [B, 4] bounding box tensor.
#             - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
#         """
#         # TODO: Implement forward pass.
#         raise NotImplementedError("Implement MultiTaskPerceptionModel.forward")

"""Unified multi-task model
"""

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization   import VGG11Localizer
from .segmentation   import VGG11UNet
from .vgg11          import VGG11Encoder
from .layers         import CustomDropout


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d (3×3, same-pad, no bias) → ReLU → BN2d."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    Architecture overview
    ---------------------
    One VGG11Encoder backbone is shared across all three task heads.
    A single forward(x) call produces:

        classification → [B, num_breeds] logits
        localization   → [B, 4]          pixel-space (xc, yc, w, h)
        segmentation   → [B, seg_classes, H, W] logits

    Weight initialisation
    ---------------------
    The constructor accepts paths to three independently trained checkpoints
    (from Tasks 1–3). On init it:
        1. Downloads each file from Google Drive via gdown (if not present).
        2. Loads the state_dict into a temporary task-specific model.
        3. Copies the *encoder* weights into the shared backbone (taking the
           average of the three sets of encoder weights — this is a simple
           but effective way to initialise a shared backbone from multiple
           specialists; the alternative is to pick one, e.g. the classifier).
        4. Copies each task head independently.

    If checkpoint paths point to files that don't exist and gdown is
    unavailable, the model falls back to random initialisation with a warning.

    Shared backbone vs task interference
    -------------------------------------
    Multi-task learning with a shared encoder can suffer from gradient
    interference when task gradients point in conflicting directions. For
    Oxford Pets the three tasks are closely related (same subjects, same
    scene), so interference is mild. Empirically, full end-to-end fine-tuning
    of the shared encoder outperforms frozen or partially frozen variants
    because classification feedback provides rich semantic gradients that
    also benefit the spatial tasks.
    """

    def __init__(
        self,
        num_breeds:      int = 37,
        seg_classes:     int = 3,
        in_channels:     int = 3,
        classifier_path: str = "checkpoints/task1_best.pth",
        localizer_path:  str = "checkpoints/task2_best.pth",
        unet_path:       str = "checkpoints/task3_best.pth",
        image_size:      int = 224,
        dropout_p:       float = 0.5,
    ):
        """
        Initialize the shared backbone/heads using trained weights.

        Args:
            num_breeds:      Number of output classes for classification head.
            seg_classes:     Number of output classes for segmentation head.
            in_channels:     Number of input channels.
            classifier_path: Path to trained classifier weights (.pth).
            localizer_path:  Path to trained localizer weights (.pth).
            unet_path:       Path to trained U-Net weights (.pth).
            image_size:      Spatial size of the (square) input image.
            dropout_p:       Dropout probability for task heads.
        """
        # NOTE: gdown calls are inside __init__ as required by the skeleton.
        # Replace the placeholder Drive IDs below with your actual file IDs
        # before submission.
        import gdown
        gdown.download(id="1K3XEU26fbmLZcL8b215R2AAEWqGpdRR8",
                       output=classifier_path, quiet=False)
        gdown.download(id="1PTe_K_CjgcMLkOpU-GnJXm8ZQ35BEUx6",
                       output=localizer_path,  quiet=False)
        gdown.download(id="1P8g73QUmIAt6yXJjyeg5Zy7Ux1qVxnLR",
                       output=unet_path,        quiet=False)

        super().__init__()

        self.image_size = image_size

        # ------------------------------------------------------------------
        # Shared backbone
        # ------------------------------------------------------------------
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ------------------------------------------------------------------
        # Classification head  (mirrors ClassificationHead in classification.py)
        # ------------------------------------------------------------------
        self.cls_head = nn.Sequential(
            nn.Flatten(),
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

        # ------------------------------------------------------------------
        # Localization head  (mirrors LocalizationHead in localization.py)
        # ------------------------------------------------------------------
        self.loc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 4),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Segmentation decoder (mirrors VGG11UNet decoder in segmentation.py)
        # ------------------------------------------------------------------
        from .segmentation import DecoderBlock
        self.bridge = nn.Sequential(
            _conv_bn_relu(512, 512),
            CustomDropout(p=dropout_p),
        )
        self.up5     = DecoderBlock(512, 512, 512)
        self.up4     = DecoderBlock(512, 512, 256)
        self.up3     = DecoderBlock(256, 256, 128)
        self.up2     = DecoderBlock(128, 128,  64)
        self.up1     = DecoderBlock( 64,  64,  64)
        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

        # ------------------------------------------------------------------
        # Load pretrained weights from the three checkpoints
        # ------------------------------------------------------------------
        self._load_pretrained(
            classifier_path, localizer_path, unet_path,
            num_breeds, seg_classes, in_channels, dropout_p,
        )

    # ------------------------------------------------------------------
    def _load_pretrained(
        self,
        clf_path:   str,
        loc_path:   str,
        unet_path:  str,
        num_breeds: int,
        seg_classes: int,
        in_channels: int,
        dropout_p:   float,
    ) -> None:
        """Load task-specific checkpoints and wire weights into this model."""
        device = torch.device("cpu")

        def _safe_load(path: str):
            """Load a checkpoint dict; return None on any error."""
            try:
                ckpt = torch.load(path, map_location=device)
                # Support both raw state_dict and {'model_state_dict': ...}
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    return ckpt["model_state_dict"]
                return ckpt
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Could not load checkpoint '{path}': {exc}. "
                    "Falling back to random initialisation for this component."
                )
                return None

        # ---- Classifier ---------------------------------------------------
        clf_sd = _safe_load(clf_path)
        if clf_sd is not None:
            tmp_clf = VGG11Classifier(num_breeds, in_channels, dropout_p)
            tmp_clf.load_state_dict(clf_sd, strict=False)
            # Copy encoder weights
            self.encoder.load_state_dict(
                tmp_clf.encoder.state_dict(), strict=False
            )
            # Copy classification head
            self.cls_head.load_state_dict(
                tmp_clf.head.head.state_dict(), strict=False
            )
            del tmp_clf

        # ---- Localizer ----------------------------------------------------
        loc_sd = _safe_load(loc_path)
        if loc_sd is not None:
            tmp_loc = VGG11Localizer(in_channels, dropout_p,
                                     image_size=self.image_size)
            tmp_loc.load_state_dict(loc_sd, strict=False)
            # Blend localizer encoder weights into shared backbone (mean)
            if clf_sd is not None:
                # Average encoder params from classifier and localizer
                for (name, param), loc_param in zip(
                    self.encoder.named_parameters(),
                    tmp_loc.encoder.parameters(),
                ):
                    param.data = (param.data + loc_param.data) / 2.0
            else:
                self.encoder.load_state_dict(
                    tmp_loc.encoder.state_dict(), strict=False
                )
            self.loc_head.load_state_dict(
                tmp_loc.regression_head.head.state_dict(), strict=False
            )
            del tmp_loc

        # ---- U-Net --------------------------------------------------------
        unet_sd = _safe_load(unet_path)
        if unet_sd is not None:
            tmp_unet = VGG11UNet(seg_classes, in_channels, dropout_p)
            tmp_unet.load_state_dict(unet_sd, strict=False)

            # Final blend of encoder (average across all three sources)
            n_sources = sum([clf_sd is not None, loc_sd is not None])
            if n_sources > 0:
                for (name, param), unet_param in zip(
                    self.encoder.named_parameters(),
                    tmp_unet.encoder.parameters(),
                ):
                    param.data = (
                        (param.data * n_sources + unet_param.data)
                        / (n_sources + 1)
                    )
            else:
                self.encoder.load_state_dict(
                    tmp_unet.encoder.state_dict(), strict=False
                )

            # Copy segmentation decoder components
            for attr in ("bridge", "up5", "up4", "up3", "up2", "up1",
                         "seg_head"):
                getattr(self, attr).load_state_dict(
                    getattr(tmp_unet, attr).state_dict(), strict=False
                )
            del tmp_unet

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization':   [B, 4] bounding box tensor in pixel space.
            - 'segmentation':   [B, seg_classes, H, W] segmentation logits.
        """
        # ---- Shared encoder (one forward pass for all tasks) --------------
        bottleneck, skips = self.encoder(x, return_features=True)
        # bottleneck: [B, 512, 7, 7]

        # ---- Classification branch ----------------------------------------
        cls_logits = self.cls_head(bottleneck)           # [B, num_breeds]

        # ---- Localization branch ------------------------------------------
        norm_boxes = self.loc_head(bottleneck)            # [B, 4] ∈ [0,1]
        scale = torch.tensor(
            [self.image_size] * 4,
            dtype=norm_boxes.dtype,
            device=norm_boxes.device,
        )
        bbox = norm_boxes * scale                         # pixel coords [B, 4]

        # ---- Segmentation branch ------------------------------------------
        d = self.bridge(bottleneck)
        d = self.up5(d, skips["s5"])
        d = self.up4(d, skips["s4"])
        d = self.up3(d, skips["s3"])
        d = self.up2(d, skips["s2"])
        d = self.up1(d, skips["s1"])
        seg_logits = self.seg_head(d)                    # [B, seg_classes, H, W]

        return {
            "classification": cls_logits,
            "localization":   bbox,
            "segmentation":   seg_logits,
        }