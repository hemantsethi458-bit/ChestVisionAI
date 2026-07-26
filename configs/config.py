"""Central configuration for ChestVision AI.

All paths and hyperparameters are defined here. Override via environment
variables or a local ``.env`` file without modifying source code.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

from configs.constants import DISEASE_LABELS, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _env_path(key: str, default: str) -> Path:
    """Resolve a path from an environment variable or default."""
    value = os.getenv(key, default)
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class DataConfig:
    """Dataset paths and loading parameters."""

    data_root: Path = field(default_factory=lambda: _env_path("CHESTVISION_DATA_ROOT", "data"))
    csv_filename: str = "Data_Entry_2017.csv"
    images_subdir: str = "images"
    image_size: int = 224
    num_workers: int = int(os.getenv("CHESTVISION_NUM_WORKERS", "4"))
    pin_memory: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    @property
    def csv_path(self) -> Path:
        """Return absolute path to the NIH metadata CSV."""
        return self.data_root / self.csv_filename

    @property
    def images_dir(self) -> Path:
        """Return absolute path to the NIH image directory."""
        return self.data_root / self.images_subdir


@dataclass
class ModelConfig:
    """Model architecture settings."""

    backbone: str = "densenet121"
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    dropout: float = 0.5
    gradcam_target_layer: str = "backbone.features.denseblock4.denselayer16.conv2"


@dataclass
class TrainingConfig:
    """Training hyperparameters and runtime settings."""

    batch_size: int = int(os.getenv("CHESTVISION_BATCH_SIZE", "32"))
    num_epochs: int = int(os.getenv("CHESTVISION_EPOCHS", "50"))
    learning_rate: float = float(os.getenv("CHESTVISION_LR", "1e-4"))
    weight_decay: float = 1e-5
    use_amp: bool = True
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 7
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-7
    log_interval: int = 50
    device: str = os.getenv("CHESTVISION_DEVICE", "auto")


@dataclass
class InferenceConfig:
    """Inference and post-processing settings."""

    threshold: float = float(os.getenv("CHESTVISION_THRESHOLD", "0.5"))
    top_k: int = 3
    history_db_path: Path = field(
        default_factory=lambda: _env_path("CHESTVISION_HISTORY_DB", "logs/prediction_history.json")
    )


@dataclass
class PathsConfig:
    """Project artifact directories."""

    project_root: Path = PROJECT_ROOT
    weights_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "weights")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports" / "output")
    tensorboard_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs" / "tensorboard")

    @property
    def best_model_path(self) -> Path:
        """Path to the best saved checkpoint."""
        return self.weights_dir / "best_model.pth"

    @property
    def latest_model_path(self) -> Path:
        """Path to the most recent checkpoint."""
        return self.weights_dir / "latest_model.pth"


@dataclass
class Config:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    disease_labels: Tuple[str, ...] = tuple(DISEASE_LABELS)
    imagenet_mean: Tuple[float, float, float] = IMAGENET_MEAN
    imagenet_std: Tuple[float, float, float] = IMAGENET_STD


def get_config() -> Config:
    """Return a fresh configuration instance."""
    return Config()


# Module-level default used across the project.
config = get_config()
