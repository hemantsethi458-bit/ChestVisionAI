"""PyTorch Dataset for NIH ChestX-ray14."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from configs.config import Config
from configs.constants import CSV_IMAGE_COLUMN, CSV_LABELS_COLUMN, CSV_PATIENT_COLUMN
from datasets.preprocessing import load_image
from datasets.splitter import labels_to_vector
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ChestXrayDataset(Dataset):
    """Multi-label chest X-ray dataset backed by NIH ChestX-ray14 metadata."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        transform: Optional[A.Compose] = None,
        label_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataframe: Filtered NIH metadata dataframe.
            image_dir: Root directory containing X-ray PNG files.
            transform: Albumentations pipeline applied to each image.
            label_names: Ordered list of disease labels.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_names = label_names or []
        self._image_path_cache = self._build_image_index()

    def _build_image_index(self) -> Dict[str, Path]:
        """Build a filename-to-path index supporting nested NIH folders."""
        index: Dict[str, Path] = {}
        if not self.image_dir.exists():
            logger.warning("Image directory does not exist: %s", self.image_dir)
            return index

        for path in self.image_dir.rglob("*.png"):
            index[path.name] = path
        return index

    def _resolve_image_path(self, filename: str) -> Path:
        """Resolve image path from cache or direct lookup."""
        if filename in self._image_path_cache:
            return self._image_path_cache[filename]

        direct_path = self.image_dir / filename
        if direct_path.exists():
            return direct_path

        matches = list(self.image_dir.rglob(filename))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Image '{filename}' not found under {self.image_dir}. "
            "Ensure NIH ChestX-ray14 images are extracted correctly."
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        """Load one image-label pair.

        Returns:
            Tuple of ``(image_tensor, label_tensor, metadata)``.
            Image tensor shape: ``(3, H, W)``.
            Label tensor shape: ``(num_classes,)``.
        """
        row = self.dataframe.iloc[index]
        filename = str(row[CSV_IMAGE_COLUMN])
        label_string = str(row[CSV_LABELS_COLUMN])
        image_path = self._resolve_image_path(filename)

        image = load_image(str(image_path))
        if self.transform is not None:
            transformed = self.transform(image=image)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label_vector = labels_to_vector(label_string)
        label_tensor = torch.from_numpy(label_vector)

        metadata = {
            "image_name": filename,
            "patient_id": str(row[CSV_PATIENT_COLUMN]),
            "finding_labels": label_string,
            "image_path": str(image_path),
        }
        return image_tensor, label_tensor, metadata


def load_nih_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load and validate the NIH metadata CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"NIH metadata CSV not found at {csv_path}. "
            "Download ChestX-ray14 and update configs/config.py."
        )

    dataframe = pd.read_csv(csv_path)
    required_columns = {CSV_IMAGE_COLUMN, CSV_LABELS_COLUMN, CSV_PATIENT_COLUMN}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    return dataframe


def build_dataset_from_config(
    config: Config,
    indices: List[int],
    transform: Optional[A.Compose],
) -> ChestXrayDataset:
    """Factory helper to construct a dataset slice from configuration."""
    dataframe = load_nih_dataframe(config.data.csv_path)
    subset = dataframe.iloc[indices].copy()
    return ChestXrayDataset(
        dataframe=subset,
        image_dir=config.data.images_dir,
        transform=transform,
        label_names=list(config.disease_labels),
    )
