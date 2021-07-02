import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0]
        predict = predict.view(predict.shape[0], -1)
        target = target.view(target.shape[0], -1)

        num = 2 * torch.sum(torch.mul(predict, target), dim=1) + 1.
        den = torch.sum(predict * predict + target * target, dim=1) + 1.

        loss = 1 - num / den
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.dice = BinaryDiceLoss()

    def forward(self, logits, target):
        n_fgs = target.shape[1] - 1 # no bg
        weight = [1] * n_fgs if self.weight is None else self.weight

        total_loss = 0
        predict = F.softmax(logits, dim=1)

        for i in range(1, target.shape[1]): # no bg
            total_loss += weight[i - 1] * self.dice(predict[:, i], target[:, i])
            
        return total_loss / (sum(weight) + 1e-6)


class MultiHeadDiceLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.dice = nn.ModuleList([DiceLoss(w) for w in weights])
        self.weight = [sum(w) for w in weights]

    def forward(self, logits, targets):
        total_loss = 0

        for i, dice in enumerate(self.dice):
            total_loss += self.weight[i] * dice(logits[i], targets[i])
        
        return total_loss / (sum(self.weight) + 1e-6)


class MultiHeadCELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.ce = nn.ModuleList([nn.CrossEntropyLoss() for _ in weights])
        self.weight = [sum(w) for w in weights]

    def forward(self, logits, targets):
        total_loss = 0

        for i, ce in enumerate(self.ce):
            total_loss += self.weight[i] * ce(logits[i], torch.argmax(targets[i], 1))
        return total_loss / (sum(self.weight) + 1e-6)


class DCandCELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dc = DiceLoss(weights)

    def forward(self, logits, target):
        dc_loss = self.dc(logits, target)
        ce_loss = self.ce(logits, torch.argmax(target, axis=1))
        
        result = (dc_loss + ce_loss) / 2
        return result


class MultiHeadDCandCELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.dc_and_ce = nn.ModuleList([DCandCELoss(w) for w in weights])
        self.weight = [sum(w) for w in weights]

    def forward(self, logits, targets):
        total_loss = 0

        for i, dc_and_ce in enumerate(self.dc_and_ce):
            total_loss += self.weight[i] * dc_and_ce(logits[i], targets[i])
        return total_loss / (sum(self.weight) + 1e-6)