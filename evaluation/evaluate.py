"""CLI entry point for model evaluation."""

import argparse
from pathlib import Path

from configs.config import get_config
from datasets.dataloaders import create_dataloaders
from evaluation.evaluator import Evaluator
from models.model_factory import load_model_from_checkpoint
from utils.device import get_device
from utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate ChestVision AI model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (defaults to best_model.pth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "both"],
        default="both",
        help="Dataset split to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate model on selected dataset splits."""
    args = parse_args()
    config = get_config()
    device = get_device(config.training.device)

    if args.checkpoint is None:
        checkpoint_path = config.paths.best_model_path
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = config.paths.project_root / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model_from_checkpoint(checkpoint_path, config, device)
    dataloaders = create_dataloaders(config)
    evaluator = Evaluator(model=model, device=device, config=config)

    if args.split in {"val", "both"}:
        evaluator.evaluate(dataloaders.val, split_name="validation")
    if args.split in {"test", "both"}:
        evaluator.evaluate(dataloaders.test, split_name="test")


if __name__ == "__main__":
    main()
