"""DenseNet121 multi-label classifier for chest X-ray analysis."""

from typing import List

import torch
import torch.nn as nn
from torchvision import models


class DenseNet121Classifier(nn.Module):
    """DenseNet121 backbone with a custom multi-label classification head."""

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the classifier.

        Args:
            num_classes: Number of disease labels to predict.
            pretrained: Whether to load ImageNet pretrained weights.
            dropout: Dropout probability before the final linear layer.
        """
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: Input batch tensor of shape ``(N, 3, H, W)``.

        Returns:
            Logits tensor of shape ``(N, num_classes)``.
        """
        return self.backbone(x)

    def get_feature_extractor(self) -> nn.Module:
        """Return convolutional feature extractor for Grad-CAM."""
        return self.backbone.features

    def get_target_layer(self, layer_name: str) -> nn.Module:
        """Resolve a nested module by dotted attribute path."""
        module: nn.Module = self
        for part in layer_name.split("."):
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        return module
