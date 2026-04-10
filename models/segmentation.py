# """Segmentation model
# """

# import torch
# import torch.nn as nn

# class VGG11UNet(nn.Module):
#     """U-Net style segmentation network.
#     """

#     def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
#         """
#         Initialize the VGG11UNet model.

#         Args:
#             num_classes: Number of output classes.
#             in_channels: Number of input channels.
#             dropout_p: Dropout probability for the segmentation head.
#         """
#         pass

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass for segmentation model.
#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].

#         Returns:
#             Segmentation logits [B, num_classes, H, W].
#         """
#         # TODO: Implement forward pass.
#         raise NotImplementedError("Implement VGG11UNet.forward")
"""Segmentation model
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d (3×3, same-pad, no bias) → ReLU → BN2d."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )


class DecoderBlock(nn.Module):
    """One U-Net decoder stage.

    Steps:
        1. ConvTranspose2d (stride=2) — learnable ×2 upsampling.
        2. Concatenate with the corresponding encoder skip map along ch dim.
        3. Two 3×3 conv-BN-ReLU layers to fuse and refine features.

    Why ConvTranspose2d instead of bilinear + conv?
    -----------------------------------------------
    The assignment explicitly requires *learnable* upsampling via transposed
    convolutions. Unlike fixed bilinear interpolation, ConvTranspose2d learns
    a task-specific upsampling kernel during training, which can better
    recover fine-grained spatial details (e.g. thin pet limbs, fur texture
    boundaries). The trade-off is a small increase in parameter count and
    occasional checkerboard artefacts — the latter is mitigated by following
    each transposed conv with a standard 3×3 conv that smooths the output.

    Args:
        in_ch   (int): Channel count of the upsampled (coarser) feature map.
        skip_ch (int): Channel count of the encoder skip connection.
        out_ch  (int): Channel count after the fusion convs.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Learnable ×2 spatial upsampling
        self.upsample = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=2, stride=2, bias=False
        )
        # After concatenation: in_ch + skip_ch channels → out_ch
        self.conv = nn.Sequential(
            _conv_bn_relu(in_ch + skip_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x    (Tensor): Coarser decoder feature map [B, in_ch, H, W].
            skip (Tensor): Corresponding encoder skip map [B, skip_ch, 2H, 2W].

        Returns:
            Fused tensor [B, out_ch, 2H, 2W].
        """
        x = self.upsample(x)   # [B, in_ch, 2H, 2W]

        # Guard against off-by-one spatial size mismatches (can occur when
        # the input spatial dim is odd).
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)   # [B, in_ch+skip_ch, 2H, 2W]
        return self.conv(x)                # [B, out_ch, 2H, 2W]


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.

    Architecture
    ------------
    Encoder (contracting path)  — VGG11Encoder (Task 1 backbone):
        s1: [B,  64, H,   W  ]   stage-1 pre-pool activations
        s2: [B, 128, H/2, W/2]
        s3: [B, 256, H/4, W/4]
        s4: [B, 512, H/8, W/8]
        s5: [B, 512, H/16,W/16]  ← fed into bottleneck

    Bottleneck: [B, 512, H/32, W/32]  (after pool5 + adaptive_pool → 7×7)

    Decoder (expansive path)  — four DecoderBlocks + final ×2 up:
        up5: TransposeConv(512→512) + cat(s5 512) → conv(1024→512)
        up4: TransposeConv(512→256) + cat(s4 512) → conv(768→256)
        up3: TransposeConv(256→128) + cat(s3 256) → conv(384→128)
        up2: TransposeConv(128→64)  + cat(s2 128) → conv(192→64)
        up1: TransposeConv(64→64)   + cat(s1  64) → conv(128→64)
        head: Conv2d(64, num_classes, 1×1)

    The decoder is *symmetric* to the encoder in the sense that each pooling
    step is mirrored by exactly one TransposeConv upsampling step, and skip
    connections bridge every encoder stage to its mirror decoder stage —
    following the canonical U-Net design (Ronneberger et al., 2015).

    Loss function choice
    --------------------
    Training uses a combination of Cross-Entropy loss and Dice loss:
        L_total = L_CE + λ · L_Dice   (λ = 1.0 by default)

    Cross-Entropy optimises pixel-wise log-likelihood and produces strong
    gradients from the start. Dice loss directly optimises the evaluation
    metric (Dice Coefficient) and is robust to class imbalance — important
    here because background pixels vastly outnumber foreground pixels in the
    Oxford-IIIT Pet trimaps. Using both together empirically outperforms
    either alone on imbalanced segmentation benchmarks.

    Encoder freezing
    ----------------
    The same three transfer-learning strategies used in Task 2 apply here.
    freeze_encoder=True / freeze_stages (partial) / False (full fine-tune).
    """

    def __init__(
        self,
        num_classes:    int   = 3,
        in_channels:    int   = 3,
        dropout_p:      float = 0.5,
        freeze_encoder: bool  = False,
    ):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes:    Number of output segmentation classes (3 for trimap).
            in_channels:    Number of input image channels.
            dropout_p:      Dropout probability used in the bottleneck bridge.
            freeze_encoder: If True, VGG11 backbone weights are frozen.
        """
        super().__init__()

        # ---- Encoder -------------------------------------------------------
        self.encoder = VGG11Encoder(in_channels=in_channels)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ---- Bottleneck bridge --------------------------------------------
        # Applied to the 7×7 adaptive-pooled bottleneck before the decoder.
        # CustomDropout here (not in conv stages) for the same BN-conflict
        # reason described in VGG11Encoder; the bridge has no BN.
        self.bridge = nn.Sequential(
            _conv_bn_relu(512, 512),
            CustomDropout(p=dropout_p),
        )

        # ---- Decoder blocks -----------------------------------------------
        # up5: bottleneck(512) × 2 + skip s5(512) → 512
        self.up5 = DecoderBlock(in_ch=512, skip_ch=512, out_ch=512)
        # up4: 512 × 2 + skip s4(512) → 256
        self.up4 = DecoderBlock(in_ch=512, skip_ch=512, out_ch=256)
        # up3: 256 × 2 + skip s3(256) → 128
        self.up3 = DecoderBlock(in_ch=256, skip_ch=256, out_ch=128)
        # up2: 128 × 2 + skip s2(128) → 64
        self.up2 = DecoderBlock(in_ch=128, skip_ch=128, out_ch=64)
        # up1: 64 × 2 + skip s1(64)  → 64  (restores original resolution)
        self.up1 = DecoderBlock(in_ch=64,  skip_ch=64,  out_ch=64)

        # ---- Segmentation head --------------------------------------------
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                if m not in list(self.encoder.modules()):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m not in list(self.encoder.modules()):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # ---- Encoder ------------------------------------------------------
        # return_features=True gives us the pre-pool skip maps s1…s5 AND the
        # 7×7 adaptive-pooled bottleneck.
        bottleneck, skips = self.encoder(x, return_features=True)
        # bottleneck: [B, 512, 7, 7]
        # skips:  s1[B,64,H,W], s2[B,128,H/2,W/2], s3[B,256,H/4,W/4],
        #         s4[B,512,H/8,W/8], s5[B,512,H/16,W/16]

        # ---- Bridge -------------------------------------------------------
        d = self.bridge(bottleneck)          # [B, 512, 7, 7]

        # ---- Decoder ------------------------------------------------------
        d = self.up5(d,  skips["s5"])        # [B, 512, 14, 14]
        d = self.up4(d,  skips["s4"])        # [B, 256, 28, 28]
        d = self.up3(d,  skips["s3"])        # [B, 128, 56, 56]
        d = self.up2(d,  skips["s2"])        # [B,  64, 112,112]
        d = self.up1(d,  skips["s1"])        # [B,  64, 224,224]

        # ---- Segmentation head --------------------------------------------
        return self.seg_head(d)              # [B, num_classes, H, W]