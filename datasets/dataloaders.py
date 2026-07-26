"""DataLoader factory for NIH ChestX-ray14."""

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

from configs.config import Config
from datasets.nih_dataset import build_dataset_from_config, load_nih_dataframe
from datasets.splitter import DatasetSplit, split_by_patient
from datasets.transforms import get_train_transforms, get_val_transforms


@dataclass
class DataLoaders:
    """Container for train, validation, and test DataLoaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    split: DatasetSplit


def _collate_batch(batch):
    """Collate dataset tuples into batched tensors and metadata lists."""
    images, labels, metadata = zip(*batch)
    return torch.stack(images), torch.stack(labels), list(metadata)


def _build_loader(dataset: Dataset, config: Config, shuffle: bool) -> DataLoader:
    """Create a DataLoader with project defaults."""
    return DataLoader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        drop_last=shuffle,
        collate_fn=_collate_batch,
    )


def create_dataloaders(config: Config) -> DataLoaders:
    """Create train, validation, and test DataLoaders using patient-level splits.

    Args:
        config: Project configuration object.

    Returns:
        DataLoaders container with all three splits.
    """
    dataframe = load_nih_dataframe(config.data.csv_path)
    split = split_by_patient(
        dataframe=dataframe,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.data.random_seed,
    )

    train_dataset = build_dataset_from_config(
        config=config,
        indices=split.train_indices,
        transform=get_train_transforms(config),
    )
    val_dataset = build_dataset_from_config(
        config=config,
        indices=split.val_indices,
        transform=get_val_transforms(config),
    )
    test_dataset = build_dataset_from_config(
        config=config,
        indices=split.test_indices,
        transform=get_val_transforms(config),
    )

    return DataLoaders(
        train=_build_loader(train_dataset, config, shuffle=True),
        val=_build_loader(val_dataset, config, shuffle=False),
        test=_build_loader(test_dataset, config, shuffle=False),
        split=split,
    )
