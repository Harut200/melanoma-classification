import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss (Lin et al., 2017) for the class imbalance.

    (1 - p_t)**gamma down-weights easy examples; alpha weights the positive
    class. Defaults (0.25, 2.0) follow the RetinaNet paper.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.view_as(logits).to(logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_factor * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss