# RetinaGuard — Diabetic Retinopathy Classification

> **CECS 551 | Phase 4 | Spring 2026 | California State University, Long Beach**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://kaggle.com)

Deep learning system for automated diabetic retinopathy (DR) severity grading from retinal fundus images. Trained and evaluated on the [Diabetic Retinopathy 224×224 Gaussian-Filtered Dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered).

---

## Team

| Name | Student ID |
|---|---|
| Dhathresh Prathap Kora | 033926423 |
| Vinay Krishna | 035261172 |
| Jaswanth Maddineni | 035260834 |
| Rohan Sanda | 035270636 |

---

## Problem Statement

Diabetic retinopathy is the leading cause of preventable blindness worldwide. Early automated screening can prevent vision loss in diabetic patients. This project trains and compares 5 deep learning models to classify DR severity into 5 grades (0–4).

---

## Results Summary

| Model | QWK | Accuracy | Macro F1 | AUROC (Referable) |
|---|---|---|---|---|
| EfficientNet-B0 (CE) | **0.8492** | **0.8182** | 0.6276 | 0.9740 |
| EfficientNet-B0 (Weighted CE) | 0.8298 | 0.7600 | 0.6106 | 0.9522 |
| ResNet-50 (Weighted CE) | 0.8137 | 0.6855 | 0.5059 | 0.9754 |
| EfficientNet-B0 (Focal Loss) | 0.6498 | 0.4273 | 0.3325 | 0.9396 |
| Baseline CNN (Weighted CE) | 0.5096 | 0.5909 | 0.3753 | 0.8463 |

> **Primary metric**: Quadratic Weighted Kappa (QWK) — penalises ordinal misclassifications proportionally.

---

## Repository Structure

```
retinaguard/
├── notebooks/
│   └── retinaguard_phase4.ipynb   # Main Kaggle notebook (run this)
├── src/
│   ├── dataset.py                 # RetinopathyDataset class
│   ├── models.py                  # BaselineCNN, EfficientNet-B0, ResNet-50
│   ├── losses.py                  # CE, Weighted CE, FocalLoss
│   ├── train.py                   # Training engine, TTA
│   └── metrics.py                 # Evaluation suite
├── outputs/
│   ├── figures/                   # All saved plots (PNG)
│   ├── tables/                    # CSV results tables
│   ├── checkpoints/               # Best model weights (.pth)
│   └── gradcam/                   # Grad-CAM++ visualisations
├── docs/
│   └── phase3_report.pdf          # Phase 3 submission
├── requirements.txt
└── README.md
```

---

## Setup & Reproduction

### Prerequisites

- Python 3.10+
- CUDA GPU recommended (Kaggle T4 used for experiments)

### 1. Clone the repository

```bash
git clone https://github.com/Vinayk393/RetinaGuard.git
cd retinaguard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset access

1. Go to: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered
2. Accept the dataset terms and download
3. Extract to match this path structure:
   ```
   /kaggle/input/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered/
   ├── gaussian_filtered_images/gaussian_filtered_images/
   │   ├── No_DR/
   │   ├── Mild/
   │   ├── Moderate/
   │   ├── Severe/
   │   └── Proliferate_DR/
   └── train.csv
   ```

### 4. Run the notebook

**On Kaggle (recommended):**
1. Upload `notebooks/retinaguard_phase4.ipynb` to Kaggle
2. Add the dataset (link above)
3. Enable GPU accelerator (T4 × 1 or T4 × 2)
4. Enable Internet access (required for pretrained weights)
5. Run All Cells

**Expected runtime:** ~2.5 hours for all 5 models on T4

**Expected output:**
```
outputs/
  figures/    → 10+ PNG plots
  tables/     → 6 CSV tables
  checkpoints/→ 5 .pth model weights
  gradcam/    → 4 Grad-CAM++ visualisation grids
```

---

## Methods Overview

### Dataset
- **3,662** retinal fundus images, 224×224px, Gaussian-filtered
- **5 classes**: No DR (49.3%), Mild (10.1%), Moderate (27.3%), Severe (5.3%), Proliferative (8.1%)
- **Imbalance ratio**: 9.4× (No DR vs Severe)

### Preprocessing
- CLAHE (Contrast Limited Adaptive Histogram Equalization) on L-channel
- ImageNet normalization (μ=[0.485, 0.456, 0.406])

### Augmentation (training only)
- Random horizontal/vertical flip
- Random rotation ±20°
- ColorJitter (brightness, contrast, saturation)
- Random affine (translate ±5%, scale 95–105%)

### Models
| Model | Params | Source |
|---|---|---|
| Baseline CNN | 456K | From scratch |
| EfficientNet-B0 | 4.0M | timm, ImageNet pretrained |
| ResNet-50 | 23.5M | timm, ImageNet pretrained |

### Training
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (T_max=20, eta_min=1e-6)
- **Mixed precision**: torch.cuda.amp (AMP)
- **Backbone freeze**: epochs 1–3, unfreeze at epoch 4
- **Early stopping**: patience=5 on Val QWK
- **Batch size**: 32

### Evaluation
- 70/15/15 stratified split (seed=42)
- Test-Time Augmentation: 5 passes, averaged softmax
- Metrics: QWK, Accuracy, Macro F1, AUROC (referable DR ≥ grade 2)
- Explainability: Grad-CAM++ on target class

---

## Key Figures

| Figure | Description |
|---|---|
| Fig 1 | Class distribution (bar + pie + table) |
| Fig 2 | Sample retinal images per class |
| Fig 3 | Pixel intensity distributions |
| Fig 4 | Brightness & contrast boxplots |
| Fig 5 | Train/Val/Test split distribution |
| Fig 6 | Augmentation pipeline preview |
| Fig 7 | Final model comparison (4 metrics) |
| Fig 8 | Radar chart comparison |
| Fig 9 | Per-class recall heatmap |
| Fig 10 | Clinical threshold optimization |

---

## Requirements

See `requirements.txt` for full list.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Dataset: Sovit Ratan Rath (Kaggle)
- Backbone weights: timm (Ross Wightman)
- Grad-CAM++: Jacob Gildenblat et al.
# RetinaGuard
