# """Custom IoU loss 
# """

# import torch
# import torch.nn as nn

# class IoULoss(nn.Module):
#     """IoU loss for bounding box regression.
#     """

#     def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
#         """
#         Initialize the IoULoss module.
#         Args:
#             eps: Small value to avoid division by zero.
#             reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
#         """
#         super().__init__()
#         self.eps = eps
#         self.reduction = reduction
#         # TODO: validate reduction in {"none", "mean", "sum"}.

#     def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
#         """Compute IoU loss between predicted and target bounding boxes.
#         Args:
#             pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
#             target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
#         # TODO: implement IoU loss.
#         raise NotImplementedError("Implement IoULoss.forward")

"""Custom IoU loss
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Computes  loss = 1 - IoU  for each (pred, target) pair in a batch,
    then reduces across the batch dimension.

    Both boxes are expected in (x_center, y_center, width, height) format
    in **pixel space** — the same coordinate space that VGG11Localizer.forward
    returns (i.e. values are NOT normalised to [0,1]; they are scaled by the
    image dimensions before loss computation).

    Mathematical derivation
    -----------------------
    Given a box (xc, yc, w, h):
        x1 = xc - w/2,   x2 = xc + w/2
        y1 = yc - h/2,   y2 = yc + h/2

    Intersection:
        inter_w = max(0, min(x2_pred, x2_tgt) - max(x1_pred, x1_tgt))
        inter_h = max(0, min(y2_pred, y2_tgt) - max(y1_pred, y1_tgt))
        inter   = inter_w * inter_h

    Union:
        area_pred = w_p * h_p
        area_tgt  = w_t * h_t
        union     = area_pred + area_tgt - inter

    IoU  = inter / (union + eps)
    Loss = 1 - IoU  → gradient pushes boxes toward higher overlap.

    Numerical stability
    -------------------
    eps is added to the union denominator only, guarding against zero-area
    boxes. Gradients through torch.relu (used to clamp negative inter dims)
    are zero for non-overlapping boxes — the correct sub-gradient because
    loss = 1 when IoU = 0 (flat floor) and only decreases once boxes overlap.

    Args:
        eps       (float): Denominator stabiliser. Default: 1e-6.
        reduction (str):   'mean' | 'sum' | 'none'. Default: 'mean'.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.

        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output:
                       'none' | 'mean' | 'sum'.
        """
        super().__init__()

        # Validate reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum'; got '{reduction}'"
            )

        self.eps       = eps
        self.reduction = reduction

    def forward(
        self,
        pred_boxes:   torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes:   [B, 4] predicted boxes in
                          (x_center, y_center, width, height) pixel format.
            target_boxes: [B, 4] target boxes in the same format.

        Returns:
            Scalar loss when reduction is 'mean' or 'sum'.
            Per-sample tensor [B] when reduction is 'none'.
        """
        # ---- xywh → xyxy -------------------------------------------------
        # Predicted corners
        px1 = pred_boxes[:, 0] - pred_boxes[:, 2] * 0.5
        py1 = pred_boxes[:, 1] - pred_boxes[:, 3] * 0.5
        px2 = pred_boxes[:, 0] + pred_boxes[:, 2] * 0.5
        py2 = pred_boxes[:, 1] + pred_boxes[:, 3] * 0.5

        # Target corners
        tx1 = target_boxes[:, 0] - target_boxes[:, 2] * 0.5
        ty1 = target_boxes[:, 1] - target_boxes[:, 3] * 0.5
        tx2 = target_boxes[:, 0] + target_boxes[:, 2] * 0.5
        ty2 = target_boxes[:, 1] + target_boxes[:, 3] * 0.5

        # ---- Intersection ------------------------------------------------
        inter_w = torch.relu(torch.min(px2, tx2) - torch.max(px1, tx1))
        inter_h = torch.relu(torch.min(py2, ty2) - torch.max(py1, ty1))
        inter   = inter_w * inter_h                        # [B]

        # ---- Areas -------------------------------------------------------
        area_pred   = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        area_target = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)

        # ---- Union & IoU -------------------------------------------------
        union = area_pred + area_target - inter            # [B]
        iou   = inter / (union + self.eps)                 # [B]  ∈ [0, 1]

        # ---- Loss --------------------------------------------------------
        loss = 1.0 - iou                                   # [B]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss   # 'none'

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction='{self.reduction}'"