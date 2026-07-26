"""Patient-level dataset splitting utilities."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.constants import CSV_PATIENT_COLUMN, DISEASE_LABELS, NO_FINDING_LABEL


@dataclass
class DatasetSplit:
    """Container for patient-level train/validation/test indices."""

    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]


def _build_patient_label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-label presence per patient for stratification."""
    patient_labels: Dict[int, np.ndarray] = {}

    for _, row in df.iterrows():
        patient_id = int(row[CSV_PATIENT_COLUMN])
        labels = _parse_finding_labels(str(row["Finding Labels"]))
        vector = patient_labels.get(patient_id)
        if vector is None:
            patient_labels[patient_id] = labels.copy()
        else:
            patient_labels[patient_id] = np.maximum(vector, labels)

    patient_df = pd.DataFrame(
        {
            "patient_id": list(patient_labels.keys()),
            "label_sum": [labels.sum() for labels in patient_labels.values()],
        }
    )
    return patient_df


def _parse_finding_labels(label_string: str) -> np.ndarray:
    """Convert NIH pipe-separated label string to a multi-hot vector."""
    vector = np.zeros(len(DISEASE_LABELS), dtype=np.float32)
    if label_string.strip() == NO_FINDING_LABEL:
        return vector

    findings = [item.strip() for item in label_string.split("|")]
    label_to_index = {label: index for index, label in enumerate(DISEASE_LABELS)}
    for finding in findings:
        if finding in label_to_index:
            vector[label_to_index[finding]] = 1.0
    return vector


def split_by_patient(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> DatasetSplit:
    """Split dataset indices at the patient level to prevent data leakage.

    Args:
        df: NIH metadata dataframe containing patient and label columns.
        train_ratio: Fraction of patients assigned to training.
        val_ratio: Fraction of patients assigned to validation.
        test_ratio: Fraction of patients assigned to testing.
        random_seed: Random seed for reproducibility.

    Returns:
        DatasetSplit with row indices for each partition.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    patient_df = _build_patient_label_matrix(df)
    stratify_labels = patient_df["label_sum"].clip(upper=5)

    train_patients, temp_patients = train_test_split(
        patient_df["patient_id"],
        test_size=(1.0 - train_ratio),
        random_state=random_seed,
        stratify=stratify_labels,
    )

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    temp_patient_df = patient_df[patient_df["patient_id"].isin(temp_patients)]
    val_patients, test_patients = train_test_split(
        temp_patient_df["patient_id"],
        test_size=(1.0 - relative_val_ratio),
        random_state=random_seed,
        stratify=temp_patient_df["label_sum"].clip(upper=5),
    )

    train_set = set(train_patients.tolist())
    val_set = set(val_patients.tolist())
    test_set = set(test_patients.tolist())

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for index, row in df.iterrows():
        patient_id = int(row[CSV_PATIENT_COLUMN])
        if patient_id in train_set:
            train_indices.append(index)
        elif patient_id in val_set:
            val_indices.append(index)
        elif patient_id in test_set:
            test_indices.append(index)

    return DatasetSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


def labels_to_vector(label_string: str) -> np.ndarray:
    """Public helper to convert NIH label strings to multi-hot vectors."""
    return _parse_finding_labels(label_string)
