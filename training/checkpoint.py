"""Checkpoint save and load utilities."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from configs.constants import MODEL_VERSION
from utils.paths import ensure_dir


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist model, optimizer, and training state to disk.

    Args:
        path: Destination checkpoint path.
        model: Trained PyTorch model.
        optimizer: Optimizer instance.
        epoch: Current epoch number.
        val_loss: Validation loss at checkpoint time.
        extra: Optional additional metadata to store.
    """
    ensure_dir(path.parent)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_version": MODEL_VERSION,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load a checkpoint into model and optionally optimizer.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        device: Device used for map_location.

    Returns:
        Checkpoint metadata dictionary.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
