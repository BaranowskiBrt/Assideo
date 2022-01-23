import math

import torch
from torch import nn


class CosFaceLoss(nn.Module):  # Large Margin Cosine Loss
    def __init__(self, scaling, margin):
        super(CosFaceLoss, self).__init__()
        self.scaling = scaling
        self.margin = margin

        self.ce = nn.CrossEntropyLoss()

    def forward(self, cosine, label):
        target_logit = cosine[torch.arange(len(cosine)), label] - self.margin
        cosine[torch.arange(len(cosine)), label] = target_logit
        cosine *= self.scaling

        output = self.ce(cosine, label)
        return output


class ArcFaceLoss(nn.Module):  # Additive Angular Margin Loss
    def __init__(self, scaling, margin):
        super(ArcFaceLoss, self).__init__()
        self.scaling = scaling
        self.margin = margin

        # Parameters for making the function monotonic in the whole angle range
        self.monotonic_thresh = -math.cos(margin)
        self.monotonic_values = math.sin(margin) * margin

        self.ce = nn.CrossEntropyLoss()

    def forward(self, cosine, label):
        target_logits = cosine[torch.arange(len(cosine)), label]
        transformed_logits = torch.cos(torch.acos(target_logits) + self.margin)
        transformed_logits = torch.where(target_logits > self.monotonic_thresh,
                                         transformed_logits,
                                         target_logits - self.monotonic_values)
        cosine[torch.arange(len(cosine)), label] = transformed_logits
        cosine *= self.scaling

        output = self.ce(cosine, label)
        return output


class CustomSoftmax(nn.Module):
    def __init__(self):
        super(CustomSoftmax, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cosine, label):
        output = self.ce(cosine, label)
        return output
