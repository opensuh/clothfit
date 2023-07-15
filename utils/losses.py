

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


class MAASE(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(MAASE, self).__init__()
        """Mean Average and Squared Error = 0.5*L1Loss + 0.5*L2Loss"""

    def forward(self, output, target):
        l2_loss = torch.mul(F.mse_loss(output, target), 0.5)
        l1_loss = torch.mul(F.l1_loss(output, target), 0.5)
        return torch.add(l1_loss, l2_loss)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, gt, smooth=1.0):
        #flatten label and prediction tensors
        pred = pred.view(-1)
        gt = gt.view(-1)

        intersection = (pred * gt).sum()
        dice = 1 -1 * (2.0*intersection + smooth)/(pred.sum() + gt.sum() + smooth)
        
        return dice   

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)