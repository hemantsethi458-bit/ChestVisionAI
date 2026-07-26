"""Unit tests for patient-level dataset splitting."""

import pandas as pd

from datasets.splitter import split_by_patient


def test_split_by_patient_preserves_all_rows() -> None:
    """All rows should be assigned to exactly one split."""
    dataframe = pd.DataFrame(
        {
            "Image Index": [f"{i:08d}_000.png" for i in range(20)],
            "Finding Labels": ["No Finding"] * 10 + ["Cardiomegaly"] * 10,
            "Patient ID": list(range(10)) * 2,
        }
    )
    split = split_by_patient(
        df=dataframe,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )
    total = len(split.train_indices) + len(split.val_indices) + len(split.test_indices)
    assert total == len(dataframe)
