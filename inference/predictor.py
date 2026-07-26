"""Single-image inference pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from configs.config import Config
from configs.constants import MODEL_VERSION
from datasets.transforms import get_inference_transforms
from gradcam.gradcam import GradCAMGenerator
from gradcam.overlay import create_comparison_panel, overlay_heatmap, save_overlay_image
from inference.history import PredictionHistory
from inference.postprocess import format_predictions, logits_to_probabilities
from models.model_factory import get_checkpoint_metadata, load_model_from_checkpoint
from reports.pdf_generator import generate_pdf_report
from utils.device import get_device
from utils.logger import setup_logger
from utils.paths import ensure_dir

logger = setup_logger(__name__)


@dataclass
class PredictionResult:
    """Structured output from a full inference request."""

    patient_id: str
    image_path: str
    image_name: str
    predictions: Dict[str, object]
    heatmap_path: Optional[str]
    overlay_path: Optional[str]
    report_path: Optional[str]
    model_version: str
    history_id: Optional[str]


class ChestXrayPredictor:
    """End-to-end inference service for chest X-ray analysis."""

    def __init__(self, config: Config, checkpoint_path: Optional[Path] = None) -> None:
        """Load model and supporting services."""
        self.config = config
        self.device = get_device(config.training.device)
        checkpoint = checkpoint_path or config.paths.best_model_path
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint}. Train the model first."
            )

        self.model = load_model_from_checkpoint(checkpoint, config, self.device)
        self.metadata = get_checkpoint_metadata(checkpoint)
        self.transforms = get_inference_transforms(config)
        self.gradcam = GradCAMGenerator(self.model, config, self.device)
        self.history = PredictionHistory(config.inference.history_db_path)
        ensure_dir(config.paths.reports_dir)

    def predict(
        self,
        image_path: str,
        patient_id: str = "UNKNOWN",
        generate_heatmap: bool = True,
        generate_report: bool = True,
        save_artifacts: bool = True,
    ) -> PredictionResult:
        """Run inference on a single chest X-ray image.

        Args:
            image_path: Path to PNG/JPEG chest X-ray image.
            patient_id: Patient identifier for reporting/history.
            generate_heatmap: Whether to compute Grad-CAM heatmap.
            generate_report: Whether to generate a PDF report.
            save_artifacts: Whether to persist heatmap/report files.

        Returns:
            PredictionResult with predictions and artifact paths.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_tensor = self._preprocess_image(path)
        logits = self._forward(image_tensor)
        probabilities = logits_to_probabilities(logits)[0]

        formatted = format_predictions(
            probabilities=probabilities,
            label_names=list(self.config.disease_labels),
            threshold=self.config.inference.threshold,
            top_k=self.config.inference.top_k,
        )

        heatmap_path = None
        overlay_path = None
        report_path = None

        if generate_heatmap:
            original, heatmap = self.gradcam.generate(str(path))
            overlay = overlay_heatmap(original, heatmap)
            if save_artifacts:
                artifact_dir = ensure_dir(self.config.paths.reports_dir / path.stem)
                heatmap_path = str(artifact_dir / "heatmap.png")
                overlay_path = str(artifact_dir / "overlay.png")
                panel_path = str(artifact_dir / "comparison.png")
                save_overlay_image(heatmap, heatmap_path)
                save_overlay_image(overlay, overlay_path)
                save_overlay_image(create_comparison_panel(original, heatmap, overlay), panel_path)

        if generate_report:
            report_path = str(
                ensure_dir(self.config.paths.reports_dir) / f"{path.stem}_{patient_id}_report.pdf"
            )
            generate_pdf_report(
                output_path=report_path,
                patient_id=patient_id,
                predictions=formatted,
                overlay_image_path=overlay_path,
                model_version=self.metadata.get("model_version", MODEL_VERSION),
            )

        history_id = None
        if save_artifacts:
            record = self.history.add_record(
                patient_id=patient_id,
                image_name=path.name,
                predictions=formatted,
                model_version=self.metadata.get("model_version", MODEL_VERSION),
                report_path=report_path,
                heatmap_path=overlay_path or heatmap_path,
            )
            history_id = record["id"]

        logger.info("Prediction complete for %s (patient=%s)", path.name, patient_id)
        return PredictionResult(
            patient_id=patient_id,
            image_path=str(path),
            image_name=path.name,
            predictions=formatted,
            heatmap_path=heatmap_path,
            overlay_path=overlay_path,
            report_path=report_path,
            model_version=self.metadata.get("model_version", MODEL_VERSION),
            history_id=history_id,
        )

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        """Load and transform one image for model input."""
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to read image: {path}")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image)
        return transformed["image"].unsqueeze(0).to(self.device)

    def _forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run model forward pass."""
        with torch.no_grad():
            return self.model(image_tensor)
