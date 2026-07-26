"""Confusion matrix utilities for multi-label classification."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix


def compute_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute per-label confusion matrices.

    Args:
        y_true: Ground-truth binary matrix ``(N, C)``.
        y_pred: Predicted binary matrix ``(N, C)``.

    Returns:
        Array with shape ``(C, 2, 2)``.
    """
    return multilabel_confusion_matrix(y_true, y_pred)


def save_confusion_matrix_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    output_path: Path,
) -> None:
    """Save a grid visualization of per-label confusion matrices."""
    matrices = compute_confusion_matrices(y_true, y_pred)
    count = len(label_names)
    columns = 4
    rows = int(np.ceil(count / columns))

    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3.5))
    axes_array = np.array(axes).reshape(-1)

    for index, label in enumerate(label_names):
        axis = axes_array[index]
        matrix = matrices[index]
        im = axis.imshow(matrix, cmap="Blues")
        axis.set_title(label, fontsize=9)
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xticklabels(["Pred 0", "Pred 1"])
        axis.set_yticklabels(["True 0", "True 1"])
        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
        figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    for index in range(count, len(axes_array)):
        axes_array[index].axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
