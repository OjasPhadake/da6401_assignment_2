"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        if not(0 <= p <= 1):
            raise ValueError("Dropout probability must be in the range [0, 1], but got {}".format(p))
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        if not self.training or self.p == 0:
            return x
        
        keep_prob = 1 - self.p
        
        # Create a mask
        mask = torch.empty_like(x).bernoulli_(keep_prob)
        # Scale the output
        output = x * mask / keep_prob
        return output

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)