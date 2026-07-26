"""Unit tests for multi-label metrics."""

import numpy as np

from configs.constants import DISEASE_LABELS
from evaluation.metrics import compute_multilabel_metrics


def test_compute_multilabel_metrics_keys() -> None:
    """Metrics dictionary should include expected summary keys."""
    y_true = np.zeros((8, len(DISEASE_LABELS)), dtype=np.int32)
    y_true[:4, 0] = 1
    y_prob = np.random.rand(8, len(DISEASE_LABELS)).astype(np.float32)
    metrics = compute_multilabel_metrics(y_true, y_prob, DISEASE_LABELS, threshold=0.5)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "macro_auc" in metrics
    assert len(metrics["per_label_auc"]) == len(DISEASE_LABELS)
