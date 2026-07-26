"""Overlay Grad-CAM heatmaps on chest X-ray images."""

from typing import Tuple

import cv2
import numpy as np


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a heatmap onto the original X-ray image.

    Args:
        original_image: RGB image ``(H, W, 3)`` in uint8 format.
        heatmap: RGB heatmap ``(H, W, 3)`` in uint8 format.
        alpha: Blending weight for the heatmap layer.

    Returns:
        Blended RGB overlay image.
    """
    resized_heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    overlay = cv2.addWeighted(original_image, 1.0 - alpha, resized_heatmap, alpha, 0)
    return overlay


def save_overlay_image(image: np.ndarray, output_path: str) -> None:
    """Save an RGB overlay image to disk."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)


def create_comparison_panel(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    """Create a horizontal panel with original, heatmap, and overlay views."""
    resized_heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    resized_overlay = cv2.resize(overlay, (original_image.shape[1], original_image.shape[0]))
    return np.hstack([original_image, resized_heatmap, resized_overlay])
