"""Image preprocessing utilities for chest X-ray inputs."""

from typing import Tuple

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk in RGB uint8 format.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        RGB image array with shape ``(H, W, 3)`` and dtype ``uint8``.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image.astype(np.uint8)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an image to a square resolution using bilinear interpolation.

    Args:
        image: Input RGB image ``(H, W, 3)``.
        size: Target height and width.

    Returns:
        Resized image ``(size, size, 3)``.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> np.ndarray:
    """Apply per-channel normalization using ImageNet statistics.

    Args:
        image: Float or uint8 RGB image ``(H, W, 3)``.
        mean: Per-channel mean values.
        std: Per-channel standard deviation values.

    Returns:
        Normalized float32 image ``(H, W, 3)``.
    """
    image_float = image.astype(np.float32) / 255.0
    mean_array = np.array(mean, dtype=np.float32)
    std_array = np.array(std, dtype=np.float32)
    return (image_float - mean_array) / std_array
