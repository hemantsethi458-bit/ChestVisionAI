"""Unit tests for configuration loading."""

from configs.config import get_config
from configs.constants import DISEASE_LABELS, NUM_CLASSES


def test_config_num_classes_matches_labels() -> None:
    """Configured class count should match NIH label list length."""
    config = get_config()
    assert config.model.num_classes == NUM_CLASSES
    assert len(config.disease_labels) == NUM_CLASSES
    assert config.disease_labels[0] == DISEASE_LABELS[0]


def test_config_paths_are_absolute() -> None:
    """Project paths should resolve to absolute filesystem paths."""
    config = get_config()
    assert config.paths.project_root.is_absolute()
    assert config.paths.weights_dir.is_absolute()
