"""Unit tests for label parsing and post-processing."""

import numpy as np

from configs.constants import DISEASE_LABELS
from datasets.splitter import labels_to_vector
from inference.postprocess import format_predictions


def test_labels_to_vector_multi_hot() -> None:
    """Pipe-separated NIH labels should produce a correct multi-hot vector."""
    vector = labels_to_vector("Cardiomegaly|Effusion")
    cardiomegaly_index = DISEASE_LABELS.index("Cardiomegaly")
    effusion_index = DISEASE_LABELS.index("Effusion")
    assert vector[cardiomegaly_index] == 1.0
    assert vector[effusion_index] == 1.0
    assert vector.sum() == 2.0


def test_labels_to_vector_no_finding() -> None:
    """No Finding should map to an all-zero label vector."""
    vector = labels_to_vector("No Finding")
    assert vector.sum() == 0.0


def test_format_predictions_top_k() -> None:
    """Post-processing should return the requested number of top predictions."""
    probabilities = np.linspace(0.0, 1.0, len(DISEASE_LABELS), dtype=np.float32)
    formatted = format_predictions(probabilities, DISEASE_LABELS, threshold=0.9, top_k=3)
    assert len(formatted["top_predictions"]) == 3
