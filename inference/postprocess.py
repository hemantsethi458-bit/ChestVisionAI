"""Prediction post-processing utilities."""

from typing import Dict, List

import numpy as np
import torch


def format_predictions(
    probabilities: np.ndarray,
    label_names: List[str],
    threshold: float = 0.5,
    top_k: int = 3,
) -> Dict[str, object]:
    """Format model probabilities into structured prediction output.

    Args:
        probabilities: Array of shape ``(num_classes,)`` with values in [0, 1].
        label_names: Ordered disease label names.
        threshold: Binary decision threshold.
        top_k: Number of top predictions to return.

    Returns:
        Dictionary containing positive labels, top-k predictions, and scores.
    """
    positive_labels = [
        {
            "label": label_names[index],
            "confidence": float(probabilities[index]),
            "confidence_pct": round(float(probabilities[index]) * 100.0, 2),
        }
        for index, value in enumerate(probabilities)
        if value >= threshold
    ]

    ranked_indices = np.argsort(probabilities)[::-1][:top_k]
    top_predictions = [
        {
            "label": label_names[index],
            "confidence": float(probabilities[index]),
            "confidence_pct": round(float(probabilities[index]) * 100.0, 2),
        }
        for index in ranked_indices
    ]

    return {
        "positive_labels": positive_labels,
        "top_predictions": top_predictions,
        "all_scores": {
            label_names[index]: float(probabilities[index]) for index in range(len(label_names))
        },
    }


def logits_to_probabilities(logits: torch.Tensor) -> np.ndarray:
    """Convert model logits to numpy probabilities."""
    return torch.sigmoid(logits).detach().cpu().numpy()
