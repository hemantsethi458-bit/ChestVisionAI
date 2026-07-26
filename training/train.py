"""CLI entry point for model training."""

import argparse

from configs.config import get_config
from datasets.dataloaders import create_dataloaders
from models.model_factory import build_model
from training.trainer import Trainer
from utils.device import get_device
from utils.logger import setup_logger
from utils.seed import set_seed

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train ChestVision AI DenseNet121 model")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device preference")
    return parser.parse_args()


def main() -> None:
    """Run end-to-end training pipeline."""
    args = parse_args()
    config = get_config()

    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.device is not None:
        config.training.device = args.device

    set_seed(config.data.random_seed)
    device = get_device(config.training.device)
    logger.info("Using device: %s", device)

    dataloaders = create_dataloaders(config)
    logger.info(
        "Dataset split -> train: %d, val: %d, test: %d",
        len(dataloaders.train.dataset),
        len(dataloaders.val.dataset),
        len(dataloaders.test.dataset),
    )

    model = build_model(config)
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        train_loader=dataloaders.train,
        val_loader=dataloaders.val,
    )
    results = trainer.train(num_epochs=config.training.num_epochs)
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    main()
