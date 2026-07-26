"""Early stopping utility for training loops."""

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    patience: int = 7
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        """Initialize internal counters."""
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Update early stopping state with the latest validation loss.

        Args:
            val_loss: Validation loss for the current epoch.

        Returns:
            True if training should stop, otherwise False.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False
