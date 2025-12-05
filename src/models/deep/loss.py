"""Custom loss functions for Deep Learning models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (float): Weighting factor for positive class (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (before sigmoid)
            targets: Binary targets (0 or 1)
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # pt is the probability of the true class
        
        # Note: The standard definition uses alpha for the positive class.
        # Here we apply alpha to the loss term. 
        # For multi-label, we can apply alpha to positive examples and (1-alpha) to negative?
        # A common simplification is just alpha * (1-pt)^gamma * BCE.
        # But strictly: alpha_t = alpha if y=1 else (1-alpha).
        
        # Let's implement the alpha-balanced variant correctly for binary/multi-label
        # targets is 0 or 1.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

