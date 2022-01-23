from functools import partial

import timm
import torch
from torch import nn
from torch.nn import functional as F


class LinearModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # No bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class Backbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.module = timm.create_model(model_name, pretrained=True)
        self.out_features = self.module.classifier.in_features

    def forward(self, x):
        return self.module.forward_features(x)


class GeM(nn.Module):
    def __init__(self, p=2, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.Tensor([p]))
        # self.p = 2
        self.eps = eps

    def forward(self, x):
        return x.clip(min=self.eps).pow(self.p).mean((2, 3)).pow(1. / self.p)


class BaseModel(nn.Module):
    def __init__(self, config, gem_pooling=True):
        super().__init__()
        self.config = config

        self.backbone = Backbone(config.model)
        self.pooling = GeM() if gem_pooling else partial(torch.mean,
                                                         dim=(2, 3))

        self.head = [
            nn.Linear(self.backbone.out_features,
                      self.config.embedding_length,
                      bias=False),
            nn.BatchNorm1d(self.config.embedding_length),
            torch.nn.PReLU(),
            torch.nn.Dropout(p=self.config.get('dropout_proba', 0)),
        ]
        self.head = self.create_head()

    def create_head(self):
        return nn.Sequential(*self.head)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x)  # Global Average Pooling
        x = self.head(x)

        return x


class TrainingModel(BaseModel):
    def __init__(self, config, cat_count):
        self.cat_count = cat_count
        super().__init__(config=config)

    def create_head(self):
        return nn.Sequential(
            *self.head,
            LinearModule(self.config.embedding_length, self.cat_count))
