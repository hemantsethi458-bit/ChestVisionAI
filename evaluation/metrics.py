"""Classification metrics for multi-label chest X-ray prediction."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, criterion: nn.Module) -> float:
    """Compute scalar loss value for a batch."""
    return float(criterion(logits, labels).item())


def sigmoid_predictions(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert logits to binary predictions using a probability threshold."""
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    return (probabilities >= threshold).astype(np.int32)


def compute_batch_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute sample-averaged metrics for one batch."""
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    targets = labels.detach().cpu().numpy()
    predictions = (probabilities >= threshold).astype(np.int32)

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, average="samples", zero_division=0)),
        "recall": float(recall_score(targets, predictions, average="samples", zero_division=0)),
        "f1": float(f1_score(targets, predictions, average="samples", zero_division=0)),
    }


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Compute comprehensive multi-label metrics over a full dataset.

    Args:
        y_true: Ground-truth binary matrix ``(N, C)``.
        y_prob: Predicted probabilities ``(N, C)``.
        label_names: Ordered disease label names.
        threshold: Decision threshold for binary predictions.

    Returns:
        Dictionary with global and per-label metrics.
    """
    y_pred = (y_prob >= threshold).astype(np.int32)

    per_label_auc = {}
    for index, label in enumerate(label_names):
        try:
            if len(np.unique(y_true[:, index])) > 1:
                per_label_auc[label] = float(roc_auc_score(y_true[:, index], y_prob[:, index]))
            else:
                per_label_auc[label] = float("nan")
        except ValueError:
            per_label_auc[label] = float("nan")

    valid_aucs = [value for value in per_label_auc.values() if not np.isnan(value)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="samples", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="samples", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "macro_auc": macro_auc,
        "per_label_auc": per_label_auc,
        "classification_report": report,
    }
