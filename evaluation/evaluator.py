"""Model evaluation over validation and test splits."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from evaluation.confusion import save_confusion_matrix_grid
from evaluation.metrics import compute_multilabel_metrics, sigmoid_predictions
from evaluation.visualize import save_metrics_summary, save_roc_summary
from utils.io import write_json
from utils.logger import setup_logger
from utils.paths import ensure_dir

logger = setup_logger(__name__)


class Evaluator:
    """Run inference over a DataLoader and compute multi-label metrics."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Config,
    ) -> None:
        """Initialize evaluator."""
        self.model = model.to(device).eval()
        self.device = device
        self.config = config

    @torch.no_grad()
    def collect_predictions(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Run model over a DataLoader and gather probabilities and labels."""
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_metadata: List[dict] = []

        for images, labels, metadata in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probabilities)
            all_labels.append(labels.numpy())
            all_metadata.extend(metadata)

        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        return y_true, y_prob, all_metadata

    def evaluate(self, loader: DataLoader, split_name: str) -> Dict[str, object]:
        """Evaluate model and persist metric artifacts."""
        y_true, y_prob, _ = self.collect_predictions(loader)
        threshold = self.config.inference.threshold
        metrics = compute_multilabel_metrics(
            y_true=y_true,
            y_prob=y_prob,
            label_names=list(self.config.disease_labels),
            threshold=threshold,
        )
        y_pred = sigmoid_predictions(torch.from_numpy(y_prob), threshold=threshold)

        output_dir = ensure_dir(self.config.paths.logs_dir / "evaluation" / split_name)
        save_confusion_matrix_grid(
            y_true=y_true,
            y_pred=y_pred,
            label_names=list(self.config.disease_labels),
            output_path=output_dir / "confusion_matrices.png",
        )
        save_roc_summary(metrics["per_label_auc"], output_dir / "roc_auc.png")
        save_metrics_summary(metrics, output_dir / "metrics_summary.png")
        write_json(output_dir / "metrics.json", metrics)

        logger.info(
            "%s metrics -> accuracy=%.4f f1=%.4f macro_auc=%.4f",
            split_name,
            metrics["accuracy"],
            metrics["f1"],
            metrics["macro_auc"],
        )
        return metrics
