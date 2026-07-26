"""Training loop with AMP, TensorBoard, checkpointing, and early stopping."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.config import Config
from evaluation.metrics import compute_batch_metrics
from training.checkpoint import save_checkpoint
from training.early_stopping import EarlyStopping
from training.scheduler import create_scheduler
from utils.logger import setup_logger
from utils.paths import ensure_dir

logger = setup_logger(__name__)


@dataclass
class EpochMetrics:
    """Aggregated metrics for one training or validation epoch."""

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class Trainer:
    """Orchestrates model training, validation, and artifact persistence."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Initialize trainer dependencies."""
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = create_scheduler(self.optimizer, config)
        self.early_stopping = EarlyStopping(patience=config.training.early_stopping_patience)
        self.scaler = GradScaler(enabled=config.training.use_amp and device.type == "cuda")

        ensure_dir(config.paths.logs_dir)
        ensure_dir(config.paths.tensorboard_dir)
        ensure_dir(config.paths.weights_dir)
        self.writer = SummaryWriter(log_dir=str(config.paths.tensorboard_dir))
        self.log_file = config.paths.logs_dir / "training.log"
        self.logger = setup_logger("training.trainer", log_file=self.log_file)

        self.best_val_loss = float("inf")

    def _run_epoch(self, loader: DataLoader, train: bool) -> EpochMetrics:
        """Execute one epoch over the provided DataLoader."""
        self.model.train(mode=train)
        losses: List[float] = []
        accuracies: List[float] = []
        precisions: List[float] = []
        recalls: List[float] = []
        f1_scores: List[float] = []

        progress = tqdm(loader, desc="Train" if train else "Val", leave=False)
        for batch_index, (images, labels, _) in enumerate(progress):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            batch_metrics = compute_batch_metrics(logits.detach(), labels)
            losses.append(loss.item())
            accuracies.append(batch_metrics["accuracy"])
            precisions.append(batch_metrics["precision"])
            recalls.append(batch_metrics["recall"])
            f1_scores.append(batch_metrics["f1"])

            if train and (batch_index + 1) % self.config.training.log_interval == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")

        return EpochMetrics(
            loss=float(sum(losses) / max(len(losses), 1)),
            accuracy=float(sum(accuracies) / max(len(accuracies), 1)),
            precision=float(sum(precisions) / max(len(precisions), 1)),
            recall=float(sum(recalls) / max(len(recalls), 1)),
            f1=float(sum(f1_scores) / max(len(f1_scores), 1)),
        )

    def _log_epoch(self, epoch: int, train_metrics: EpochMetrics, val_metrics: EpochMetrics) -> None:
        """Write epoch metrics to logger and TensorBoard."""
        self.logger.info(
            "Epoch %03d | train_loss=%.4f val_loss=%.4f val_f1=%.4f",
            epoch,
            train_metrics.loss,
            val_metrics.loss,
            val_metrics.f1,
        )
        for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
            self.writer.add_scalar(f"Loss/{prefix}", metrics.loss, epoch)
            self.writer.add_scalar(f"Accuracy/{prefix}", metrics.accuracy, epoch)
            self.writer.add_scalar(f"Precision/{prefix}", metrics.precision, epoch)
            self.writer.add_scalar(f"Recall/{prefix}", metrics.recall, epoch)
            self.writer.add_scalar(f"F1/{prefix}", metrics.f1, epoch)

        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("LearningRate", current_lr, epoch)

    def train(self, num_epochs: int) -> Dict[str, float]:
        """Run the full training loop.

        Args:
            num_epochs: Maximum number of epochs to train.

        Returns:
            Dictionary with best validation loss and final epoch count.
        """
        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step(val_metrics.loss)
            self._log_epoch(epoch, train_metrics, val_metrics)

            save_checkpoint(
                path=self.config.paths.latest_model_path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                val_loss=val_metrics.loss,
                extra={"train_loss": train_metrics.loss, "val_f1": val_metrics.f1},
            )

            if val_metrics.loss < self.best_val_loss:
                self.best_val_loss = val_metrics.loss
                save_checkpoint(
                    path=self.config.paths.best_model_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    val_loss=val_metrics.loss,
                    extra={"train_loss": train_metrics.loss, "val_f1": val_metrics.f1},
                )
                self.logger.info("Saved new best model at epoch %d", epoch)

            if self.early_stopping.step(val_metrics.loss):
                self.logger.info("Early stopping triggered at epoch %d", epoch)
                break

        self.writer.close()
        return {"best_val_loss": self.best_val_loss, "epochs_trained": epoch}
