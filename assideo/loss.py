import math

import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        log_proba = self.ce(input, target)
        p = torch.exp(-log_proba)
        loss = (1 - p)**self.gamma * log_proba
        return loss.mean()


class CosFaceLoss(nn.Module):  # Large Margin Cosine Loss
    def __init__(self, scaling, margin, use_focal=True):
        super().__init__()
        self.scaling = scaling
        self.margin = margin

        self.loss = FocalLoss() if use_focal else nn.CrossEntropyLoss()

    def forward(self, cosine, label):
        target_logit = cosine[torch.arange(len(cosine)), label] - self.margin
        cosine[torch.arange(len(cosine)), label] = target_logit
        cosine *= self.scaling

        output = self.loss(cosine, label)
        return output


class ArcFaceLoss(nn.Module):  # Additive Angular Margin Loss
    def __init__(self, scaling, margin, use_focal=True):
        super().__init__()
        self.scaling = scaling
        self.margin = margin

        # Parameters for making the function monotonic in the whole angle range
        self.monotonic_thresh = -math.cos(margin)
        self.monotonic_values = math.sin(margin) * margin

        self.loss = FocalLoss() if use_focal else nn.CrossEntropyLoss()

    def forward(self, cosine, label):
        target_logits = cosine[torch.arange(len(cosine)), label]
        transformed_logits = torch.cos(torch.acos(target_logits) + self.margin)
        transformed_logits = torch.where(target_logits > self.monotonic_thresh,
                                         transformed_logits,
                                         target_logits - self.monotonic_values)
        cosine[torch.arange(len(cosine)), label] = transformed_logits
        cosine *= self.scaling

        output = self.loss(cosine, label)
        return output


class SubcenterArcFaceLoss(nn.Module):  # Additive Angular Margin Loss
    def __init__(self, scaling, margin, subcenter_count, use_focal=True):
        super().__init__()
        self.scaling = scaling
        self.margin = margin

        self.max_pool = torch.nn.MaxPool1d(subcenter_count,
                                           stride=subcenter_count)

        # Parameters for making the function monotonic in the whole angle range
        self.monotonic_thresh = -math.cos(margin)
        self.monotonic_values = math.sin(margin) * margin
        self.loss = FocalLoss() if use_focal else nn.CrossEntropyLoss()

    def forward(self, cosine, label):
        # Cosine tensor has cat_count * subcenter_count values per batch

        # Unsqueeze due to an active bug https://github.com/pytorch/pytorch/issues/51954
        cosine = self.max_pool(cosine.unsqueeze(0)).squeeze(0)

        target_logits = cosine[torch.arange(len(cosine)), label]
        transformed_logits = torch.cos(torch.acos(target_logits) + self.margin)
        transformed_logits = torch.where(target_logits > self.monotonic_thresh,
                                         transformed_logits,
                                         target_logits - self.monotonic_values)
        cosine[torch.arange(len(cosine)), label] = transformed_logits
        cosine *= self.scaling

        output = self.loss(cosine, label)
        return output


class CustomSoftmax(nn.Module):
    def __init__(self, use_focal=True):
        super().__init__()
        self.loss = FocalLoss() if use_focal else nn.CrossEntropyLoss()

    def forward(self, cosine, label):
        output = self.loss(cosine, label)
        return output
