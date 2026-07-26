# ChestVision AI — System Architecture

> Phase 1 design document. Implementation follows in Phases 2–10.

---

## 1. Design Philosophy

ChestVision AI follows **clean architecture** principles:

| Principle | Implementation |
|-----------|----------------|
| Separation of concerns | Each package owns one responsibility |
| Dependency direction | Outer layers depend on inner layers, never reverse |
| Configuration over hard-coding | All paths and hyperparameters in `configs/` |
| Testability | Business logic isolated from I/O and UI |
| Production readiness | Logging, checkpointing, reproducibility built-in |

```
┌─────────────────────────────────────────────────────────────┐
│                    streamlit_app/ (UI)                      │
├─────────────────────────────────────────────────────────────┤
│  reports/  │  gradcam/  │  inference/  │  evaluation/      │
├─────────────────────────────────────────────────────────────┤
│              training/  │  models/  │  datasets/            │
├─────────────────────────────────────────────────────────────┤
│                    configs/  │  utils/                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Folder & Module Reference

### `configs/`

**Purpose:** Single source of truth for all configurable values.

| Module (planned) | Responsibility |
|------------------|----------------|
| `config.py` | Paths, image size, batch size, learning rate, disease labels |
| `constants.py` | NIH ChestX-ray14 label names, normalization stats |

**Why it exists:** Eliminates magic numbers and hard-coded paths across 10+ modules. Changing dataset location requires editing one file.

**Interactions:** Imported by `datasets/`, `training/`, `inference/`, `streamlit_app/`.

---

### `data/`

**Purpose:** Local mount point for the NIH ChestX-ray14 dataset.

**Contents (user-provided, gitignored):**
```
data/
├── images/              # 112,120 PNG chest X-rays
└── Data_Entry_2017.csv  # Labels & metadata
```

**Why it exists:** Keeps large binary data out of version control while providing a predictable local path referenced by `configs/config.py`.

---

### `datasets/`

**Purpose:** Data ingestion, preprocessing, augmentation, and PyTorch DataLoader creation.

| Module (planned) | Responsibility |
|------------------|----------------|
| `nih_dataset.py` | `ChestXrayDataset` — reads CSV, loads images, returns tensors |
| `transforms.py` | Albumentations pipelines (train vs val) |
| `preprocessing.py` | Resize, normalize, grayscale→RGB conversion |
| `dataloaders.py` | Factory for train/val/test DataLoaders with splitting |
| `splitter.py` | Patient-level stratified split (avoid data leakage) |

**Key design decision:** Patient-level splitting. NIH dataset has multiple images per patient; random image-level splits leak patient data between train/val and inflate metrics.

**Tensor shapes:**
- Input image: `(3, 224, 224)` — RGB, ImageNet-normalized
- Label vector: `(14,)` — multi-hot binary vector

---

### `models/`

**Purpose:** Neural network architecture definitions.

| Module (planned) | Responsibility |
|------------------|----------------|
| `densenet_classifier.py` | DenseNet121 backbone + custom classifier head |
| `model_factory.py` | Build/load models from config or checkpoint |

**Architecture (Phase 3):**
```
Input (3, 224, 224)
    ↓
DenseNet121 features (pretrained ImageNet)
    ↓
Global Average Pooling → (1024,)
    ↓
Dropout(0.5)
    ↓
Linear(1024 → 14) → logits (14,)
    ↓
Sigmoid (at inference) → probabilities
```

**Loss:** `BCEWithLogitsLoss` — numerically stable binary cross-entropy for multi-label.

---

### `training/`

**Purpose:** Full training orchestration.

| Module (planned) | Responsibility |
|------------------|----------------|
| `trainer.py` | Main training loop with AMP, gradient scaling |
| `scheduler.py` | Learning rate scheduling (ReduceLROnPlateau) |
| `early_stopping.py` | Patience-based stopping on val loss |
| `checkpoint.py` | Save/load best and latest checkpoints |
| `train.py` | CLI entry point |

**Features:** Mixed precision (AMP), TensorBoard logging, reproducible seeds, GPU auto-detection.

---

### `evaluation/`

**Purpose:** Post-training and validation metrics.

| Module (planned) | Responsibility |
|------------------|----------------|
| `metrics.py` | Accuracy, precision, recall, F1, ROC-AUC |
| `confusion.py` | Per-label confusion matrices |
| `evaluator.py` | Run evaluation on val/test set |
| `visualize.py` | ROC curves, metric bar charts |

---

### `inference/`

**Purpose:** Single-image prediction pipeline for deployment.

| Module (planned) | Responsibility |
|------------------|----------------|
| `predictor.py` | Load model, preprocess image, return predictions |
| `postprocess.py` | Top-k selection, confidence formatting |
| `history.py` | SQLite/JSON prediction history store |
| `predict.py` | CLI entry point |

---

### `gradcam/`

**Purpose:** Model explainability via Grad-CAM heatmaps.

| Module (planned) | Responsibility |
|------------------|----------------|
| `gradcam.py` | Compute gradients w.r.t. target class |
| `overlay.py` | Superimpose heatmap on original X-ray |

**Target layer:** DenseNet121 `features.denseblock4.denselayer16.conv2` — last conv layer before classifier.

---

### `reports/`

**Purpose:** Professional PDF medical report generation.

| Module (planned) | Responsibility |
|------------------|----------------|
| `pdf_generator.py` | ReportLab-based PDF builder |
| `templates.py` | Layout, fonts, sections |

**Report sections:** Patient ID, date, predictions table, confidence bars, Grad-CAM image, model version, disclaimer.

---

### `streamlit_app/`

**Purpose:** Interactive web dashboard.

| Module (planned) | Responsibility |
|------------------|----------------|
| `app.py` | Main entry, sidebar navigation |
| `pages/predict.py` | Upload & predict |
| `pages/history.py` | Past predictions |
| `pages/model_info.py` | Architecture & dataset info |
| `pages/performance.py` | Training metrics viewer |
| `pages/settings.py` | Threshold & path configuration |
| `components/` | Reusable UI widgets |

---

### `utils/`

**Purpose:** Cross-cutting utilities shared by all packages.

| Module (planned) | Responsibility |
|------------------|----------------|
| `logger.py` | Structured logging setup |
| `seed.py` | Reproducibility (torch, numpy, random) |
| `device.py` | GPU/CPU auto-selection |
| `io.py` | Safe file read/write helpers |
| `paths.py` | Resolve project-relative paths |

---

### `logs/` & `weights/`

| Directory | Purpose |
|-----------|---------|
| `logs/` | Text logs + TensorBoard event files |
| `weights/` | `.pth` checkpoints (`best_model.pth`, `latest.pth`) |

Both are gitignored except `.gitkeep` placeholders.

---

### `tests/`

**Purpose:** pytest-based unit and integration tests.

| Module (planned) | Tests |
|------------------|-------|
| `test_dataset.py` | Dataset loading, transforms, splits |
| `test_model.py` | Forward pass shapes, output range |
| `test_metrics.py` | Metric computation correctness |
| `test_inference.py` | End-to-end prediction pipeline |

---

### `docs/`

Additional documentation, diagrams, and API references (expanded in Phase 10).

---

## 3. Technology Stack Rationale

| Library | Role | Why Selected |
|---------|------|--------------|
| **Python 3.11** | Runtime | Stable, performant, wide ML ecosystem support |
| **PyTorch** | Deep learning | Dynamic graphs, medical AI community standard, excellent debugging |
| **Torchvision** | Pretrained models | Official DenseNet121 with ImageNet weights |
| **OpenCV** | Image I/O | Fast PNG/JPEG loading, color space conversion, resize |
| **Albumentations** | Augmentation | Medical-imaging-friendly transforms, fast, composable pipelines |
| **NumPy** | Numerical ops | Foundation for all array operations |
| **Pandas** | CSV parsing | NIH metadata (`Data_Entry_2017.csv`) handling |
| **Scikit-Learn** | Metrics & splits | Industry-standard classification metrics, stratified splitting |
| **Matplotlib** | Plotting | Confusion matrices, ROC curves, metric charts |
| **TensorBoard** | Experiment tracking | Real-time loss/metric curves during training |
| **grad-cam** | Explainability | Battle-tested Grad-CAM for PyTorch models |
| **Streamlit** | Dashboard | Rapid professional UI without frontend framework overhead |
| **ReportLab** | PDF generation | Programmatic PDF with precise medical report layout |
| **PyYAML** | Config files | Optional YAML overrides for hyperparameters |
| **pytest** | Testing | Standard Python test framework |

### Optional: Docker

Containerization (Phase 10+) for reproducible deployment:
- Base image: `python:3.11-slim` + CUDA runtime
- Mount `data/` and `weights/` as volumes
- Expose Streamlit on port 8501

---

## 4. Model Selection: DenseNet121 vs Alternatives

| Model | Pros | Cons | Verdict |
|-------|------|------|---------|
| **DenseNet121** ✅ | Strong on ChestX-ray14 literature; dense connections preserve fine features; 8M params — trains on consumer GPU | Slightly slower inference than MobileNet | **Recommended** |
| EfficientNet-B0 | Better accuracy/param ratio; faster | Less established on ChestX-ray14; different input normalization |
| ResNet50 | Simple, well-understood | Lower AUC reported on ChestX-ray14 vs DenseNet |
| ViT-B/16 | State-of-the-art on some benchmarks | Needs more data; 224×224 may lose detail; heavier compute |

**Recommendation:** DenseNet121 with ImageNet pretraining. It is the most cited backbone on NIH ChestX-ray14, balances accuracy and GPU memory (~4 GB VRAM at batch 32), and integrates cleanly with Grad-CAM via its final convolutional block.

---

## 5. Data Flow

```
NIH CSV + Images
       │
       ▼
  datasets/ (load, augment, split)
       │
       ▼
  models/ (DenseNet121 forward)
       │
       ├──► training/ (optimize weights)
       │         │
       │         ▼
       │    weights/best_model.pth
       │
       ├──► evaluation/ (metrics)
       │
       └──► inference/ (predict)
                 │
                 ├──► gradcam/ (heatmap)
                 ├──► reports/ (PDF)
                 └──► streamlit_app/ (UI)
```

---

## 6. NIH ChestX-ray14 Labels

14 disease classes (multi-label — patient can have multiple):

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

Plus "No Finding" (handled as all-zero label vector or separate logic in evaluation).

---

## 7. Configuration Strategy

All magic numbers live in `configs/config.py`:

```python
# Illustrative — implemented in Phase 2+
DATA_ROOT = "data/"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 14
THRESHOLD = 0.5
```

Environment variable override supported via `python-dotenv` for deployment flexibility.

---

## 8. Phase Roadmap

See [README.md](../README.md#development-phases) for the full phased development plan.
