"""Albumentations transform pipelines for training and validation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import Config


def get_train_transforms(config: Config) -> A.Compose:
    """Build augmentation pipeline for training.

    Augmentations are conservative for medical imaging to preserve pathology cues.

    Args:
        config: Project configuration object.

    Returns:
        Albumentations compose transform for training.
    """
    size = config.data.image_size
    mean = config.imagenet_mean
    std = config.imagenet_std

    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=0, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_val_transforms(config: Config) -> A.Compose:
    """Build deterministic transform pipeline for validation and inference.

    Args:
        config: Project configuration object.

    Returns:
        Albumentations compose transform for evaluation/inference.
    """
    size = config.data.image_size
    mean = config.imagenet_mean
    std = config.imagenet_std

    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(config: Config) -> A.Compose:
    """Alias for validation transforms used during inference."""
    return get_val_transforms(config)
