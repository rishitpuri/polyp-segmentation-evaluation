"""Loss functions.

Dice + BCE is the standard baseline in the polyp segmentation literature
(PraNet, HarDNet-MSEG, Polyp-PVT all use a variant of it) and is kept
identical across architectures so the comparison stays fair.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2, 3)) + self.smooth
        den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth
        dice = 1 - (num / den).mean()
        return self.bce_weight * bce + self.dice_weight * dice
