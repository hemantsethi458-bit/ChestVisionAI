"""Grad-CAM heatmap generation for DenseNet121 predictions."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from configs.config import Config
from datasets.preprocessing import load_image, resize_image


class GradCAMGenerator:
    """Generate class-specific Grad-CAM heatmaps for chest X-ray predictions."""

    def __init__(self, model: nn.Module, config: Config, device: torch.device) -> None:
        """Initialize Grad-CAM with target layer from configuration."""
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        target_layer = self._resolve_target_layer()
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def _resolve_target_layer(self) -> nn.Module:
        """Resolve the convolutional layer used for Grad-CAM."""
        layer_path = self.config.model.gradcam_target_layer
        module: nn.Module = self.model
        for part in layer_path.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    def generate(
        self,
        image_path: str,
        target_class_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Grad-CAM heatmap for one or more target classes.

        Args:
            image_path: Path to the input chest X-ray image.
            target_class_indices: Optional list of class indices. If omitted,
                the highest predicted probability class is used.

        Returns:
            Tuple of ``(rgb_image, heatmap)`` where heatmap is uint8 ``(H, W, 3)``.
        """
        rgb_image = load_image(image_path)
        resized = resize_image(rgb_image, self.config.data.image_size)
        input_tensor = self._preprocess(resized)

        if target_class_indices is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.sigmoid(logits)[0]
                target_class_indices = [int(torch.argmax(probabilities).item())]

        targets = [ClassifierOutputTarget(index) for index in target_class_indices]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        cam_map = grayscale_cam[0]
        heatmap = self._apply_colormap(cam_map)
        return rgb_image, heatmap

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert RGB image to normalized model input tensor."""
        mean = np.array(self.config.imagenet_mean, dtype=np.float32)
        std = np.array(self.config.imagenet_std, dtype=np.float32)
        normalized = (image.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _apply_colormap(grayscale_cam: np.ndarray) -> np.ndarray:
        """Convert grayscale CAM to an OpenCV JET colormap."""
        cam_uint8 = np.uint8(255 * grayscale_cam)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
