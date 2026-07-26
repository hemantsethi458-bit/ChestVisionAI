"""Disease labels and normalization constants for NIH ChestX-ray14."""

from typing import Final, List, Tuple

# NIH ChestX-ray14 pathology labels (multi-label; excludes "No Finding").
DISEASE_LABELS: Final[List[str]] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES: Final[int] = len(DISEASE_LABELS)

# ImageNet statistics used for DenseNet121 pretrained backbone.
IMAGENET_MEAN: Final[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[Tuple[float, float, float]] = (0.229, 0.224, 0.225)

# NIH CSV column names.
CSV_IMAGE_COLUMN: Final[str] = "Image Index"
CSV_LABELS_COLUMN: Final[str] = "Finding Labels"
CSV_PATIENT_COLUMN: Final[str] = "Patient ID"
CSV_AGE_COLUMN: Final[str] = "Patient Age"
CSV_GENDER_COLUMN: Final[str] = "Patient Gender"
CSV_VIEW_COLUMN: Final[str] = "View Position"

NO_FINDING_LABEL: Final[str] = "No Finding"

MODEL_VERSION: Final[str] = "ChestVisionAI-DenseNet121-v1.0.0"
