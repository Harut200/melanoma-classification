import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss (Lin et al., 2017) for the severe class imbalance in
    melanoma detection (~1.76% positive rate).

    BCE spends most of its gradient on the easy negatives it already
    classifies correctly, simply because there are 55x more of them. Focal
    loss down-weights well-classified examples by (1 - p_t) ** gamma so the
    rare, hard positives dominate the gradient instead, and `alpha` gives a
    second, independent knob to upweight the positive class on top of that.

    alpha=0.8 means positives are weighted more heavily than negatives
    (alpha for y=1, (1 - alpha) for y=0) -- tune it down if the model starts
    over-predicting melanoma.
    """
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
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