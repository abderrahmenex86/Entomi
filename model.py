import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)


class PestClassifier(nn.Module):
    """Flat 102-class classifier used by train.py."""

    def __init__(self, num_species=102, pretrained=True):
        super().__init__()

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v3_large(weights=weights)

        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_species)

    def forward(self, x):
        return self.model(x)


class HierarchicalPestClassifier(nn.Module):
    """Older three-head checkpoint: order, family, species."""

    def __init__(self, num_orders=8, num_families=47, num_species=102, pretrained=False):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = 576
        self.order_head = nn.Linear(in_features, num_orders)
        self.family_head = nn.Linear(in_features, num_families)
        self.species_head = nn.Linear(in_features, num_species)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)

        return {
            "order": self.order_head(pooled),
            "family": self.family_head(pooled),
            "species": self.species_head(pooled),
        }
