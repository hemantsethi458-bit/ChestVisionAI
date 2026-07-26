"""Metric visualization helpers."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def save_roc_summary(
    per_label_auc: Dict[str, float],
    output_path: Path,
) -> None:
    """Save a bar chart summarizing per-label ROC-AUC scores."""
    labels = list(per_label_auc.keys())
    values = [per_label_auc[label] for label in labels]

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(labels, values, color="#2563eb")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("ROC-AUC")
    axis.set_title("Per-Label ROC-AUC")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_metrics_summary(metrics: Dict[str, float], output_path: Path) -> None:
    """Save a bar chart for global classification metrics."""
    keys = ["accuracy", "precision", "recall", "f1", "macro_auc"]
    labels = [key.replace("_", " ").title() for key in keys]
    values = [metrics.get(key, 0.0) for key in keys]

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, values, color="#059669")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Evaluation Metrics Summary")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
