import torch 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class BinaryDiceCoefficient(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2 * torch.sum(predict * target, dim=1) + 1.
        den = torch.sum(predict * predict + target * target, dim=1) + 1.

        dice = num / den

        return dice.mean()


class DiceCoefficient(nn.Module):
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return "dice"

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceCoefficient()
        predict = torch.argmax(predict, 1)

        dice_coeffs = [] 
        for i in range(1, target.shape[1]):
            pred = (predict == i).type_as(predict)
            dice_coeff = dice(pred, target[:, i])
            dice_coeffs.append(dice_coeff.detach())

        return dice_coeffs

class MultiHeadDiceCoefficient(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.dice = nn.ModuleList([DiceCoefficient() for _ in range(n_heads)])

    def __str__(self):
        return "dice"

    def forward(self, predicts, targets):
        dice_coeffs = [dice(predicts[i], targets[i]) for i, dice in enumerate(self.dice)]
        return dice_coeffs
        
