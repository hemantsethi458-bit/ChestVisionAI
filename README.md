# ChestVision AI

**Production-quality multi-label chest X-ray disease classification with explainability, PDF reporting, and a Streamlit dashboard.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Disclaimer:** ChestVision AI is for **research and educational purposes only**. It is not a medical device and must not be used for clinical diagnosis without proper validation and regulatory approval.

---

## Overview

ChestVision AI analyzes chest X-ray images and predicts **14 thoracic pathologies** from the [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) dataset using **DenseNet121 transfer learning** with multi-label classification.

### Features

| Feature | Description |
|---------|-------------|
| Multi-label prediction | 14 independent disease probabilities from one X-ray |
| DenseNet121 transfer learning | ImageNet-pretrained backbone with custom classifier head |
| Mixed-precision training | AMP, TensorBoard, early stopping, checkpointing |
| Comprehensive evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrices |
| Grad-CAM explainability | Visual heatmaps overlaid on original X-rays |
| PDF medical reports | Patient ID, predictions, confidence, heatmap, model version |
| Prediction history | JSON-backed inference history store |
| Streamlit dashboard | Prediction, history, model info, performance, settings pages |
| Docker support | Containerized deployment for the dashboard |
| Retraining support | CLI-driven training with configurable hyperparameters |

---

## Project Structure

```
ChestVisionAI/
├── configs/              # config.py, constants.py, default.yaml
├── data/                 # Local NIH dataset mount (gitignored)
├── datasets/             # Dataset, transforms, splits, DataLoaders
├── models/               # DenseNet121 classifier + model factory
├── training/             # Trainer, AMP, schedulers, early stopping
├── evaluation/           # Metrics, evaluator, visualization
├── inference/            # Predictor, history, CLI inference
├── gradcam/              # Grad-CAM generation and overlay
├── reports/              # PDF report generation (ReportLab)
├── streamlit_app/        # Professional dashboard UI
├── utils/                # Logging, seeds, device, I/O, paths
├── logs/                 # Training logs + TensorBoard
├── weights/              # Model checkpoints
├── tests/                # pytest unit tests
├── docs/                 # Architecture documentation
├── Dockerfile
├── requirements.txt
└── README.md
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design decisions.

---

## Requirements

- **Python 3.11**
- **CUDA GPU** (recommended for training; CPU works for inference)
- **~45 GB disk space** for NIH ChestX-ray14 (not included in repo)
- **Git** (optional, for version control)

---

## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/<your-username>/ChestVisionAI.git
cd ChestVisionAI

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA (recommended for training)

Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and install the CUDA build matching your GPU driver.

Example (CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Configure environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
```

Edit `.env` or set environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHESTVISION_DATA_ROOT` | `data` | Path to NIH dataset root |
| `CHESTVISION_BATCH_SIZE` | `32` | Training batch size |
| `CHESTVISION_EPOCHS` | `50` | Maximum training epochs |
| `CHESTVISION_LR` | `1e-4` | Learning rate |
| `CHESTVISION_THRESHOLD` | `0.5` | Inference decision threshold |
| `CHESTVISION_DEVICE` | `auto` | `auto`, `cuda`, `cpu`, or `mps` |

---

## Dataset Preparation

The NIH ChestX-ray14 dataset is **not included** in this repository.

### Download

1. Download from [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
2. Extract all image archives
3. Place files in the following structure:

```
data/
├── Data_Entry_2017.csv
└── images/
    ├── 00000001_000.png
    ├── 00000001_001.png
    └── ... (112,120 PNG files)
```

Nested NIH folders (e.g., `images/images_001/images/`) are also supported — the dataset loader searches recursively.

### Verify setup

```bash
python -c "from datasets.nih_dataset import load_nih_dataframe; from configs.config import get_config; print(load_nih_dataframe(get_config().data.csv_path).shape)"
```

Expected output: `(112120, 12)` (approximate row count).

---

## Training

### Start training

```bash
python -m training.train
```

### Custom hyperparameters

```bash
python -m training.train --epochs 30 --batch-size 16 --lr 5e-5 --device cuda
```

### Training outputs

| Artifact | Location |
|----------|----------|
| Best model checkpoint | `weights/best_model.pth` |
| Latest checkpoint | `weights/latest_model.pth` |
| Text logs | `logs/training.log` |
| TensorBoard events | `logs/tensorboard/` |

### Monitor with TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

Open `http://localhost:6006` in your browser.

### Training pipeline components

- **BCEWithLogitsLoss** — multi-label binary cross-entropy
- **AdamW optimizer** with weight decay
- **ReduceLROnPlateau** scheduler
- **Early stopping** (patience=7 on validation loss)
- **Mixed precision (AMP)** on CUDA
- **Gradient clipping** (max norm 1.0)
- **Patient-level split** (70% train / 15% val / 15% test)

---

## Evaluation

Evaluate the best checkpoint on validation and test splits:

```bash
python -m evaluation.evaluate
```

Evaluate a specific checkpoint:

```bash
python -m evaluation.evaluate --checkpoint weights/best_model.pth --split test
```

### Metrics computed

| Metric | Description |
|--------|-------------|
| **Accuracy** | Sample-wise label match rate |
| **Precision** | Fraction of predicted positives that are correct |
| **Recall** | Fraction of actual positives detected |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Per-label and macro-averaged area under ROC curve |
| **Confusion Matrix** | Per-label 2×2 confusion matrix grid |

Artifacts saved to `logs/evaluation/validation/` and `logs/evaluation/test/`.

---

## Inference

### Command line

```bash
python -m inference.predict --image path/to/xray.png --patient-id PAT-001
```

Options:
- `--no-heatmap` — skip Grad-CAM generation
- `--no-report` — skip PDF report
- `--checkpoint weights/best_model.pth` — use a specific checkpoint

### Python API

```python
from configs.config import get_config
from inference.predictor import ChestXrayPredictor

predictor = ChestXrayPredictor(config=get_config())
result = predictor.predict(
    image_path="data/images/sample.png",
    patient_id="PAT-001",
    generate_heatmap=True,
    generate_report=True,
)
print(result.predictions["top_predictions"])
```

---

## Streamlit Dashboard

Launch the dashboard:

```bash
streamlit run streamlit_app/app.py
```

Open `http://localhost:8501`.

### Pages

| Page | Description |
|------|-------------|
| **Prediction** | Upload X-ray, run analysis, view top predictions and Grad-CAM |
| **History** | Browse past predictions and download reports |
| **Model Info** | Architecture, dataset, labels, checkpoint status |
| **Performance** | TensorBoard metrics and evaluation charts |
| **Settings** | Adjust threshold, top-k, and training hyperparameters |

---

## PDF Reports

Reports are automatically generated during inference and saved to:

```
reports/output/<image_stem>_<patient_id>_report.pdf
```

Each report includes:
- Patient ID and timestamp
- Model version
- Top-3 predictions with confidence percentages
- Detected conditions above threshold
- Grad-CAM overlay image
- Clinical disclaimer

---

## Grad-CAM Explainability

Grad-CAM highlights image regions that most influenced the model's prediction for a target disease class.

**Target layer:** `backbone.features.denseblock4.denselayer16.conv2` (final convolutional block of DenseNet121)

Outputs saved per inference:
- `heatmap.png` — raw Grad-CAM colormap
- `overlay.png` — heatmap blended onto original X-ray
- `comparison.png` — side-by-side panel

---

## Retraining

To retrain with updated data or hyperparameters:

```bash
# Full retrain from ImageNet pretrained weights
python -m training.train --epochs 50 --batch-size 32 --lr 1e-4

# Fine-tune from existing checkpoint (modify training/train.py or use lower LR)
python -m training.train --epochs 20 --lr 1e-5
```

Update `configs/config.py` or `configs/default.yaml` for persistent configuration changes.

---

## Docker

Build and run the Streamlit dashboard in Docker:

```bash
docker build -t chestvision-ai .
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/logs:/app/logs \
  chestvision-ai
```

Mount volumes for dataset, trained weights, and logs.

---

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Model Architecture

```
Input: (N, 3, 224, 224)  — RGB, ImageNet-normalized
         ↓
DenseNet121 features (ImageNet pretrained)
         ↓
Global Average Pooling → (1024,)
         ↓
Dropout(p=0.5)
         ↓
Linear(1024 → 14) → logits (N, 14)
         ↓
Sigmoid (inference) → probabilities
```

**Why DenseNet121?** Most cited backbone on NIH ChestX-ray14; dense feature reuse preserves fine-grained pathology cues; trains efficiently on consumer GPUs (~4 GB VRAM at batch 32).

---

## Disease Labels (NIH ChestX-ray14)

| # | Label |
|---|-------|
| 1 | Atelectasis |
| 2 | Cardiomegaly |
| 3 | Effusion |
| 4 | Infiltration |
| 5 | Mass |
| 6 | Nodule |
| 7 | Pneumonia |
| 8 | Pneumothorax |
| 9 | Consolidation |
| 10 | Edema |
| 11 | Emphysema |
| 12 | Fibrosis |
| 13 | Pleural_Thickening |
| 14 | Hernia |

---

## Screenshots

| Dashboard | Grad-CAM | PDF Report |
|-----------|----------|------------|
| _Run `streamlit run streamlit_app/app.py`_ | _Generated in `reports/output/`_ | _Generated in `reports/output/`_ |

---

## Future Improvements

- [ ] EfficientNet / Vision Transformer backbone comparison study
- [ ] Model ensemble for improved macro AUC
- [ ] Native DICOM input support
- [ ] FastAPI REST API for hospital PACS integration
- [ ] MLflow / Weights & Biases experiment tracking
- [ ] ONNX / TorchScript export for edge deployment
- [ ] Active learning pipeline for clinician-in-the-loop retraining
- [ ] Class imbalance handling (weighted BCE or focal loss)
- [ ] Test-time augmentation for improved inference robustness

---

## Citation

If you use the NIH ChestX-ray14 dataset, please cite:

```bibtex
@inproceedings{Wang_2017,
  title={ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on
         Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and
          Bagheri, Mohammadhadi and Summers, Ronald M.},
  booktitle={CVPR},
  year={2017}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
