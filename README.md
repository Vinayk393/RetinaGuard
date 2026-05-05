# RetinaGuard — Diabetic Retinopathy Classification

> **CECS 551 | Phase 4 | Spring 2026 | California State University, Long Beach**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)
[![GitHub](https://img.shields.io/badge/GitHub-RetinaGuard-181717?logo=github)](https://github.com/Vinayk393/RetinaGuard)

Deep learning system for **automated diabetic retinopathy (DR) severity grading** from retinal fundus images. Five model configurations are trained, evaluated, and compared on the publicly available [EyePACS-derived Gaussian-Filtered Dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered).

Primary metric: **Quadratic Weighted Kappa (QWK)** — the clinically appropriate metric for ordinal DR grading that penalises misclassifications proportionally to the distance between grades.

---

## Team

| Name | Student ID |
|---|---|
| Dhathresh Prathap Kora | 033926423 |
| Vinay Krishna | 035261172 |
| Jaswanth Maddineni | 035260834 |
| Rohan Sanda | 035270636 |

---

## Key Contributions

- **Unified evaluation framework** — QWK, AUROC, clinical threshold optimisation, and Grad-CAM++ in a single controlled pipeline
- **Controlled loss function ablation** — CE vs Weighted CE vs Focal Loss under 9.4x real-world class imbalance, with practical deployment guidance for each
- **Alpha-scaled Focal Loss fix** — alpha=0.25 prevents near-zero loss magnitude failure that causes training instability in standard implementations
- **Config-driven reproducibility** — every experiment defined by a single YAML file; seed=42, stratified splits saved as CSV
- **Clinical operating point** — threshold optimisation yields 93.3% sensitivity and 93.3% specificity for referable DR detection

---
## Research Gap

Most prior work on diabetic retinopathy classification focuses on accuracy or AUROC, which do not fully capture the ordinal nature of DR severity grading. Few studies evaluate models using Quadratic Weighted Kappa (QWK), which penalizes clinically severe misclassifications more heavily than adjacent errors.

Additionally, existing work often reports single-model performance without systematically comparing loss functions under real-world class imbalance. There is also limited integration of threshold optimization with interpretability techniques such as Grad-CAM to connect model predictions with clinically meaningful decision-making.

RetinaGuard addresses these gaps by combining QWK, AUROC, threshold optimization, per-class recall, and Grad-CAM++ within a single controlled, config-driven evaluation framework.
---

## Quick Start
Outputs are saved in the `outputs/` directory for full reproducibility.

```bash
git clone https://github.com/Vinayk393/RetinaGuard.git
cd RetinaGuard
pip install -r requirements.txt

# Run any experiment — full pipeline in one command
python src/run_experiment.py --config configs/ce.yaml           # Best model (QWK 0.8582)
python src/run_experiment.py --config configs/focal.yaml        # Best minority recall
python src/run_experiment.py --config configs/weighted_ce.yaml  # Highest Severe recall
python src/run_experiment.py --config configs/resnet50.yaml     # Architecture comparison
python src/run_experiment.py --config configs/baseline_cnn.yaml # Lower-bound reference
```

Each command runs the complete pipeline: **data loading -> training -> TTA evaluation -> figures -> Grad-CAM++ -> all outputs saved automatically.**

Override values at runtime without editing the config:
```bash
python src/run_experiment.py --config configs/focal.yaml --epochs 10 --seed 0 --batch_size 16
```

---

## Results Summary

All results on the held-out test set (550 images, 15% stratified split) with **5-fold Test-Time Augmentation (TTA)**.

| Rank | Model | QWK | Accuracy | Macro F1 | AUROC (Referable) |
|---|---|---|---|---|---|
| 1 | EfficientNet-B0 (CE) | **0.8582** | **0.8200** | **0.6528** | **0.9786** |
| 2 | EfficientNet-B0 (Focal Loss) | 0.8282 | 0.7364 | 0.5923 | 0.9752 |
| 3 | ResNet-50 (Weighted CE) | 0.8006 | 0.6927 | 0.5050 | 0.9763 |
| 4 | EfficientNet-B0 (Weighted CE) | 0.7830 | 0.6436 | 0.4955 | 0.9583 |
| 5 | Baseline CNN (Weighted CE) | 0.7182 | 0.6727 | 0.4778 | 0.9385 |

All results correspond to test-set evaluation with 5-fold TTA using the fixed seed (42) configuration described in the report.

> **Clinical threshold**: At threshold = 0.435, EfficientNet-B0 (CE) achieves **93.3% sensitivity** and **93.3% specificity** for referable DR (grade >= 2), exceeding the 80% clinical screening guideline.

### Per-Class Recall — Best Model (EfficientNet-B0, CE)

| Grade | Class | Recall | Notes |
|---|---|---|---|
| 0 | No DR | 0.99 | Near-perfect — majority class (49.3% of data) |
| 1 | Mild | 0.50 | Hard to distinguish from Moderate |
| 2 | Moderate | 0.83 | Strong |
| 3 | Severe | 0.38 | Lowest — only 193 training samples |
| 4 | Proliferative | 0.45 | Adjacent-grade visual similarity after Gaussian filtering |

---

## Error Analysis

The lowest-performing class for the best model is Grade 3 (Severe DR), with recall of 0.38. This is primarily due to two factors: strong class imbalance (only 193 training samples) and visual similarity with adjacent grades, particularly Moderate (Grade 2) and Proliferative (Grade 4).

Most misclassifications are expected to occur between adjacent grades rather than distant classes. While these errors are less severe than misclassifying referable DR as No DR, false negatives in Grade 3 remain clinically significant because they may delay specialist referral.

The Gaussian-filtered dataset may further reduce fine-grained lesion texture, making it harder to distinguish between advanced stages. Grad-CAM++ visualizations are used to verify that the model focuses on clinically relevant retinal regions rather than background artifacts.

---

## Repository Structure

```
RetinaGuard/
│
├── configs/                          # One YAML per experiment
│   ├── ce.yaml                       # EfficientNet-B0 + Cross-Entropy  <- BEST
│   ├── weighted_ce.yaml              # EfficientNet-B0 + Weighted CE
│   ├── focal.yaml                    # EfficientNet-B0 + Focal Loss
│   ├── resnet50.yaml                 # ResNet-50 + Weighted CE
│   └── baseline_cnn.yaml             # Baseline CNN + Weighted CE
│
├── src/                              # Modular source code
│   ├── __init__.py                   # Package exports
│   ├── run_experiment.py             # <- CLI entry point (start here)
│   ├── models.py                     # BaselineCNN, EfficientNet-B0, ResNet-50, build_model()
│   ├── preprocess.py                 # CLAHE, transforms, RetinopathyDataset, DataLoaders
│   ├── train.py                      # FocalLoss (alpha-scaled), training loop, checkpointing
│   ├── evaluate.py                   # compute_metrics, TTA, threshold optimisation
│   ├── interpret.py                  # Grad-CAM++, recall heatmap, interpretation summary
│   └── utils.py                      # set_seed, save_json, logging, output directory management
│
├── notebook/
│   └── p4 RetinaGuard.ipynb          # Kaggle notebook — EDA, figures, exploration
│
├── outputs/                          # Auto-generated by run_experiment.py
│   ├── figures/                      # PNG plots (EDA + evaluation)
│   ├── tables/                       # CSV results (metrics, splits)
│   └── gradcam/                      # Grad-CAM++ visualisation grids
│
├── docs/
│   └── RetinaGuard.pdf               # IEEE-style final report
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Experiment Configuration

Every experiment is fully defined by a single YAML config in `configs/`. This enables the **controlled comparison** that is the core novelty of this project — identical training conditions across all runs, differing only in model architecture and loss function.

```yaml
# configs/focal.yaml — example
experiment_name: "EfficientNet-B0 (Focal Loss)"
model:        "efficientnet_b0"
loss:         "focal"
focal_gamma:  2.0
focal_alpha:  0.25     # CRITICAL: rescales loss to CE magnitude
lr:           0.0001
num_epochs:   20
patience:     5        # early stopping on Val QWK
seed:         42
```

| Config | Model | Loss | Paper QWK |
|---|---|---|---|
| `ce.yaml` | EfficientNet-B0 | Cross-Entropy | **0.8582** <- best |
| `focal.yaml` | EfficientNet-B0 | Focal Loss | 0.8282 |
| `weighted_ce.yaml` | EfficientNet-B0 | Weighted CE | 0.7830 |
| `resnet50.yaml` | ResNet-50 | Weighted CE | 0.8006 |
| `baseline_cnn.yaml` | Baseline CNN | Weighted CE | 0.7182 |

To add a new experiment, copy any config file, change the relevant fields, and run:
```bash
python src/run_experiment.py --config configs/your_new_config.yaml
```


---

## Setup & Reproduction

### Prerequisites

- Python 3.10+
- CUDA GPU recommended (experiments run on Kaggle T4, ~14 min/epoch)
- ~5 GB free disk space for dataset + outputs

### 1. Clone

```bash
git clone https://github.com/Vinayk393/RetinaGuard.git
cd RetinaGuard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset access

1. Go to: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered
2. Accept licence terms and download
3. Extract to match this path (or update `dataset_root` in the config YAML):

```
/kaggle/input/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered/
├── gaussian_filtered_images/
│   └── gaussian_filtered_images/
│       ├── No_DR/          (1805 images)
│       ├── Mild/           (370 images)
│       ├── Moderate/       (999 images)
│       ├── Severe/         (193 images)
│       └── Proliferate_DR/ (295 images)
└── train.csv
```

### 4. Run an experiment

```bash
# Recommended: best model
python src/run_experiment.py --config configs/ce.yaml

# Skip Grad-CAM generation for faster runs
python src/run_experiment.py --config configs/ce.yaml --no_gradcam

# Override config values inline
python src/run_experiment.py --config configs/focal.yaml --epochs 5 --seed 123
```

**Reproducibility guarantee:** seed=42 applied globally (Python, NumPy, PyTorch, CUDA). Stratified splits saved as CSV. All metrics, history, and outputs logged automatically.

---

## Notebook (Optional — Exploration & Visualisation)

`notebook/p4 RetinaGaurd.ipynb` contains the full EDA, augmentation previews, and inline visualisations. It is for **exploration and figure generation**, not the primary execution path. All experiments should be run via `src/run_experiment.py`.

**To run on Kaggle:**
1. Upload `notebook/p4 RetinaGaurd.ipynb` to a new Kaggle notebook
2. Add dataset via **Add Data** -> search `sovitrath diabetic retinopathy gaussian`
3. Set **Accelerator -> GPU T4 x1** and enable **Internet**
4. Click **Run All** (~2.5 hours for all 5 models)

---

## Methods Overview

### Dataset

| Property | Value |
|---|---|
| Total images | 3,662 |
| Image size | 224 x 224 px |
| Source preprocessing | Gaussian-filtered |
| Classes | 5 (No DR -> Proliferative DR) |
| Imbalance ratio | 9.4x (Grade 0 vs Grade 3) |
| Split | 70 / 15 / 15 stratified (seed=42) |

### Preprocessing Pipeline

1. **CLAHE** — on L-channel of LAB colourspace (clip=2.0, tile=8x8)
2. **ImageNet normalisation** — mu=[0.485,0.456,0.406], sigma=[0.229,0.224,0.225]

### Data Augmentation (training only)

| Transform | Parameters |
|---|---|
| Random horizontal flip | p=0.5 |
| Random vertical flip | p=0.3 |
| Random rotation | +/-20 degrees |
| ColorJitter | brightness +/-0.2, contrast +/-0.2, saturation +/-0.1 |
| Random affine | translate +/-5%, scale 95-105% |

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=20, eta_min=1e-6) |
| Mixed precision | torch.cuda.amp (AMP) |
| Gradient clipping | max norm=1.0 |
| Backbone freeze | Epochs 1-3, unfrozen at epoch 4 |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping | Patience=5 on Val QWK |
| Hardware | NVIDIA T4 (Kaggle) |

### Loss Functions

| Loss | Key detail | When to use |
|---|---|---|
| **CE** | Label smoothing epsilon=0.05 | Best overall QWK, priority is grading accuracy |
| **Weighted CE** | Inverse-freq weights: Severe=2.02, Prolif=1.32 | Highest minority recall, cost of missed severe cases is high |
| **Focal Loss** | gamma=2.0, **alpha=0.25** (magnitude-corrected) | Best minority/majority balance for screening deployment |

---

## Key Findings

**CE wins on overall QWK but underserves minority grades:**
- EfficientNet-B0 (CE): QWK=0.8582, Severe recall=0.38
- The high QWK coexists with near-zero Severe detection — accuracy alone hides this

**Focal Loss gives the best clinical trade-off:**
- QWK=0.8282, Mild=0.79, Severe=0.55 — best minority/majority balance of all 5 models
- alpha=0.25 is essential: without it, focal loss magnitudes collapse to ~0.006, causing effectively random predictions

**Weighted CE maximises minority recall at a cost:**
- Severe recall=0.72 (highest), but accuracy drops to 0.644 and prediction stability suffers
- Appropriate when clinical cost of missing a Severe case is extreme

**Architecture efficiency:**
- EfficientNet-B0 (4.0M params) outperforms ResNet-50 (23.5M) on QWK — compound scaling is more efficient for this task
- Baseline CNN (456K, from scratch) confirms 19.4% relative QWK gain from pretrained transfer learning

---

## Deployment Perspective

RetinaGuard is designed as a clinical decision-support tool rather than an autonomous diagnostic system. In a primary care or community screening workflow, the model can be used to flag patients with likely referable diabetic retinopathy (Grade ≥ 2) for further evaluation by an ophthalmologist.

Before real-world deployment, the model must be validated across diverse imaging devices, patient populations, and acquisition conditions. Threshold selection should be adjusted based on clinical priorities, typically favoring higher sensitivity when the cost of missing severe disease outweighs the cost of additional referrals.

---

## Known Limitations

1. **Dataset size** — 3,662 images vs EyePACS (88,000+); limits generalisation
2. **Image-level splitting** — no patient identifiers in dataset; potential data leakage acknowledged
3. **Severe recall** — Grade 3 recall=0.38 across all models; driven by 193 training samples and visual similarity to Grade 2 after Gaussian filtering
4. **No external validation** — generalisation across camera manufacturers, dilation protocols, and imaging environments unconfirmed
5. **Gaussian filtering** — source preprocessing reduces fine-grained lesion texture that helps distinguish adjacent grades

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
grad-cam>=1.4.0
pyyaml>=6.0
opencv-python-headless>=4.7.0
Pillow>=9.0.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Dataset: [Sovit Ratan Rath](https://www.kaggle.com/sovitrath) (Kaggle)
- Pretrained weights: [timm](https://github.com/huggingface/pytorch-image-models) — Ross Wightman
- Grad-CAM++: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Jacob Gildenblat et al.
- Course: CECS 551 — Machine Learning, CSULB, Spring 2026
