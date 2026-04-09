"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout

def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Helper function to create a conv-batchnorm-relu block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True), 
        nn.BatchNorm2d(out_ch),
    )
    
class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """
    super().__init__()
    
    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        
        # Stage 1
        self.stage1 = nn.Sequential(
            _conv_bn_relu(in_channels, 64), 
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/2
        
        # Stage 2
        self.stage2 = nn.Sequential(
            _conv_bn_relu(64, 128),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/4
    
        # Stage 3
        self.stage3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/8
        
        # Stage 4
        self.stage4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 1/16
        
        # Stage 5
        self.stage5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # 1/32
        
        # Bottleneck
        self.adaptive_pool  =  nn.AdaptiveAvgPool2d((7, 7))  # Output fixed size for fully connected layers
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        s1 = self.stage1(x)
        p1 = self.pool1(s1)
        
        s2 = self.stage2(p1)
        p2 = self.pool2(s2) 
        
        s3 = self.stage3(p2)
        p3 = self.pool3(s3)
        
        s4 = self.stage4(p3)
        p4 = self.pool4(s4)
        
        s5 = self.stage5(p4)
        p5 = self.pool5(s5)
        
        # Bottleneck
        bottleneck = self.adaptive_pool(p5)
        
        if return_features:
            features = {
                's1': s1,  # [B, 64, H/2, W/2]
                's2': s2,  # [B, 128, H/4, W/4]
                's3': s3,  # [B, 256, H/8, W/8]
                's4': s4,  # [B, 512, H/16, W/16]
                's5': s5,  # [B, 512, H/32, W/32]
            }
            return bottleneck, features
        else:
            return bottleneck