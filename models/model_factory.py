"""Model construction and checkpoint loading utilities."""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from configs.config import Config
from models.densenet_classifier import DenseNet121Classifier


def build_model(config: Config) -> nn.Module:
    """Instantiate the configured classification model.

    Args:
        config: Project configuration object.

    Returns:
        Initialized PyTorch model.
    """
    if config.model.backbone.lower() == "densenet121":
        return DenseNet121Classifier(
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout,
        )
    raise ValueError(f"Unsupported backbone: {config.model.backbone}")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config: Config,
    device: torch.device,
) -> nn.Module:
    """Load model weights from a training checkpoint.

    Args:
        checkpoint_path: Path to ``.pth`` checkpoint file.
        config: Project configuration object.
        device: Target device for model parameters.

    Returns:
        Model in evaluation mode with loaded weights.
    """
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, str]:
    """Extract metadata stored inside a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        return {
            "epoch": str(checkpoint.get("epoch", "unknown")),
            "val_loss": str(checkpoint.get("val_loss", "unknown")),
            "model_version": str(checkpoint.get("model_version", "unknown")),
        }
    return {"epoch": "unknown", "val_loss": "unknown", "model_version": "unknown"}
