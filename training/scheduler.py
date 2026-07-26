"""Learning rate scheduler factory."""

import torch


def create_scheduler(optimizer: torch.optim.Optimizer, config) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Create ReduceLROnPlateau scheduler from training config.

    Args:
        optimizer: Optimizer whose learning rate will be adjusted.
        config: Project configuration object.

    Returns:
        Configured learning rate scheduler.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        min_lr=config.training.min_learning_rate,
        verbose=True,
    )
