"""Unit tests for DenseNet121 classifier."""

import torch

from configs.config import get_config
from models.model_factory import build_model


def test_model_forward_output_shape() -> None:
    """Model should output logits with shape (batch, num_classes)."""
    config = get_config()
    model = build_model(config)
    batch_size = 2
    inputs = torch.randn(batch_size, 3, config.data.image_size, config.data.image_size)
    outputs = model(inputs)
    assert outputs.shape == (batch_size, config.model.num_classes)


def test_model_output_is_logits_not_probabilities() -> None:
    """Raw outputs should be unbounded logits suitable for BCEWithLogitsLoss."""
    config = get_config()
    model = build_model(config)
    inputs = torch.randn(1, 3, config.data.image_size, config.data.image_size)
    outputs = model(inputs)
    assert outputs.min() < 0.0 or outputs.max() > 1.0
