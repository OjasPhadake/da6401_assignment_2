"""Classification components
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class ClassificationHead(nn.Module):
 
    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # ← modern replacement
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
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
        return self.head(x)
    
class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head    = ClassificationHead(num_classes=num_classes,
                                          dropout_p=dropout_p)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x, return_features=False)  # [B, 512, 7, 7]
        logits   = self.head(features)                     # [B, num_classes]
        return logits
    
    def get_encoder(self) -> VGG11Encoder:
        """Return the shared convolutional backbone."""
        return self.encoder
