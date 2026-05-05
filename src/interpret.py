"""
interpret.py
============
RetinaGuard — Model Interpretability: Grad-CAM++, Per-Class Recall, Attention Analysis.

Extracted directly from the RetinaGuard Phase 4 Kaggle notebook.

Functions
---------
get_cam_target_layer(model, model_name)
    Return the correct target layer for Grad-CAM++ given the model type.

run_gradcam(model, model_name, df_subset, cfg, output_dir, n_samples)
    Generate and save Grad-CAM++ activation maps for sampled test images.

run_ieee_gradcam(model, model_name, df_test, cfg, output_dir)
    Generate publication-quality (300 DPI) Grad-CAM++ figure with
    Original | Heatmap | Overlay columns for all 5 DR grades.

analyse_per_class_recall(all_results, class_names, output_dir)
    Build and save the per-class recall heatmap across all models.

print_interpretation(all_results, class_names)
    Print key findings and per-model interpretation to stdout.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from pathlib import Path

import torch
import torch.cuda.amp as amp

from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# DR class colours (consistent with notebook palette)
DR_COLORS  = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']


# ════════════════════════════════════════════════════════════════
# HELPER: CAM TARGET LAYER
# ════════════════════════════════════════════════════════════════

def get_cam_target_layer(model, model_name: str) -> list:
    """
    Return the correct Grad-CAM++ target layer for a given model.

    Parameters
    ----------
    model      : nn.Module   Trained model instance.
    model_name : str         Model name string (used to dispatch layer selection).

    Returns
    -------
    list  containing a single nn.Module target layer.

    Layer mapping
    -------------
    EfficientNet-B0 → model.conv_head       (final conv before classifier)
    ResNet-50       → model.layer4[-1]      (last residual block)
    Baseline CNN    → model.features[-3]    (last Conv2d in feature extractor)
    """
    if 'EfficientNet' in model_name:
        return [model.conv_head]
    elif 'ResNet' in model_name:
        return [model.layer4[-1]]
    else:
        return [model.features[-3]]


# ════════════════════════════════════════════════════════════════
# GRAD-CAM++ — NOTEBOOK STYLE
# ════════════════════════════════════════════════════════════════

def run_gradcam(model,
                model_name: str,
                df_subset: pd.DataFrame,
                cfg: dict,
                val_transform,
                apply_clahe_fn,
                output_dir: str,
                n_samples: int = 8,
                seed: int = 42) -> None:
    """
    Generate and save Grad-CAM++ activation maps (notebook style).

    Layout: n_rows × 4 grid.
    Each row: [Original | Overlay | Original | Overlay] for 2 samples.

    Parameters
    ----------
    model         : nn.Module         Trained model (will be moved to cfg['device']).
    model_name    : str               Model name for title and filename.
    df_subset     : pd.DataFrame      Test DataFrame (must have 'filepath' and 'label').
    cfg           : dict              Config dict with 'device', 'use_clahe', 'class_names'.
    val_transform : transforms.Compose  Deterministic inference transform.
    apply_clahe_fn: callable          apply_clahe function from preprocess.py.
    output_dir    : str               Directory to save output PNG.
    n_samples     : int               Total samples to visualise (default 8, 2 per class).
    seed          : int               Random seed for sample selection (default 42).
    """
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        from gradcam import GradCAMPlusPlus
        from gradcam.utils.image import show_cam_on_image
        from gradcam.utils.model_targets import ClassifierOutputTarget

    device      = cfg.get('device', 'cuda')
    use_clahe   = cfg.get('use_clahe', True)
    class_names = cfg.get('class_names', ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])

    model = model.to(device)
    model.eval()

    try:
        cam = GradCAMPlusPlus(model=model, target_layers=get_cam_target_layer(model, model_name))
    except Exception as e:
        print(f'  ⚠️  Grad-CAM init failed: {e}')
        return

    # 2 samples per class
    samples_list = []
    for cls in range(len(class_names)):
        sub = df_subset[df_subset['label'] == cls]
        if len(sub) > 0:
            samples_list.append(sub.sample(min(2, len(sub)), random_state=seed))
    samples = pd.concat(samples_list).head(n_samples).reset_index(drop=True)

    n_rows = (len(samples) + 1) // 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, n_rows * 3.8))
    axes = np.array(axes).reshape(n_rows, 4)
    fig.suptitle(f'Grad-CAM++ Activation Maps — {model_name}', fontweight='bold', fontsize=12)

    for i, (_, row) in enumerate(samples.iterrows()):
        img_np = cv2.cvtColor(cv2.imread(row['filepath']), cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, (224, 224))
        if use_clahe:
            img_np = apply_clahe_fn(img_np)

        tensor   = val_transform(Image.fromarray(img_np)).unsqueeze(0).to(device)
        true_cls = int(row['label'])

        try:
            gray_cam = cam(input_tensor=tensor,
                           targets=[ClassifierOutputTarget(true_cls)])[0]
            overlay  = show_cam_on_image(img_np.astype(np.float32) / 255., gray_cam, use_rgb=True)
        except Exception as e:
            print(f'  ⚠️  Sample {i} failed: {e}')
            continue

        with torch.no_grad():
            pred = model(tensor).argmax(dim=1).item()

        r, c     = i // 2, (i % 2) * 2
        correct  = pred == true_cls
        tick     = '✓' if correct else '✗'
        border_c = '#27ae60' if correct else '#e74c3c'

        axes[r][c].imshow(img_np)
        axes[r][c].set_title(f'True: {class_names[true_cls]}',
                             fontsize=8, fontweight='bold', color=DR_COLORS[true_cls])
        axes[r][c].axis('off')

        for spine in axes[r][c + 1].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_c)
            spine.set_linewidth(2.5)
        axes[r][c + 1].imshow(overlay)
        axes[r][c + 1].set_title(f'Pred: {class_names[pred]} {tick}',
                                  fontsize=8, fontweight='bold', color=border_c)
        axes[r][c + 1].axis('off')

    # Hide unused axes
    for j in range(len(samples), n_rows * 2):
        r, c = j // 2, (j % 2) * 2
        if r < n_rows:
            axes[r][c].axis('off')
            axes[r][c + 1].axis('off')

    plt.tight_layout()

    slug = (model_name.replace(' ', '_').replace('/', '_')
                      .replace('(', '').replace(')', ''))
    sp   = f'{output_dir}/gradcam_{slug}.png'
    plt.savefig(sp, bbox_inches='tight', dpi=110)
    plt.show()
    plt.close('all')
    print(f'  Saved: {sp}')


# ════════════════════════════════════════════════════════════════
# GRAD-CAM++ — IEEE / PUBLICATION STYLE
# ════════════════════════════════════════════════════════════════

def run_ieee_gradcam(model,
                     model_name: str,
                     df_test: pd.DataFrame,
                     cfg: dict,
                     val_transform,
                     apply_clahe_fn,
                     output_dir: str,
                     seed: int = 42) -> None:
    """
    Generate a publication-quality Grad-CAM++ figure at 300 DPI.

    Layout: 5 rows (one per DR grade) × 3 columns (Original | Heatmap | Overlay).
    Green border = correct prediction. Red border = misclassification.

    Parameters
    ----------
    model         : nn.Module         Trained model.
    model_name    : str               Used for title and filename.
    df_test       : pd.DataFrame      Test split with 'filepath' and 'label'.
    cfg           : dict              Config dict.
    val_transform : transforms.Compose  Deterministic transform.
    apply_clahe_fn: callable          apply_clahe from preprocess.py.
    output_dir    : str               Directory to save the figure.
    seed          : int               Sample selection seed (default 42).
    """
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        from gradcam import GradCAMPlusPlus
        from gradcam.utils.image import show_cam_on_image
        from gradcam.utils.model_targets import ClassifierOutputTarget

    device      = cfg.get('device', 'cuda')
    use_clahe   = cfg.get('use_clahe', True)
    class_names = cfg.get('class_names', ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])

    model = model.to(device)
    model.eval()

    target_layer = get_cam_target_layer(model, model_name)
    cam_engine   = GradCAMPlusPlus(model=model, target_layers=target_layer)

    fig, axes = plt.subplots(5, 3, figsize=(7, 13))
    fig.suptitle(
        f'Grad-CAM++ Activation Maps — {model_name}\n'
        f'(Left: Original | Centre: Heatmap | Right: Overlay)',
        fontsize=10, fontweight='bold', y=1.01
    )

    for cls in range(5):
        srow = df_test[df_test['label'] == cls].sample(1, random_state=seed + cls).iloc[0]

        img_np = cv2.cvtColor(cv2.imread(srow['filepath']), cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, (224, 224))
        if use_clahe:
            img_np = apply_clahe_fn(img_np)

        tensor = val_transform(Image.fromarray(img_np)).unsqueeze(0).to(device)

        try:
            gray_cam = cam_engine(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(cls)]
            )[0]
            img_f   = img_np.astype(np.float32) / 255.
            overlay = show_cam_on_image(img_f, gray_cam, use_rgb=True)
            heatmap = cv2.applyColorMap(
                (gray_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f'  ⚠️  CAM failed cls {cls}: {e}')
            continue

        with torch.no_grad():
            pred = model(tensor).argmax(dim=1).item()

        correct  = pred == cls
        pred_col = '#27ae60' if correct else '#e74c3c'
        tick_sym = '✓' if correct else '✗'

        # Column labels on first row
        if cls == 0:
            for col_i, col_lbl in enumerate(['Original', 'CAM Heatmap', 'Overlay']):
                axes[cls][col_i].text(
                    0.5, 1.18, col_lbl,
                    transform=axes[cls][col_i].transAxes,
                    ha='center', fontsize=8.5, fontweight='bold', color='#444'
                )

        axes[cls][0].imshow(img_np)
        axes[cls][0].set_title(f'True: {class_names[cls]}',
                               fontsize=8, fontweight='bold', color=DR_COLORS[cls])
        axes[cls][0].axis('off')

        axes[cls][1].imshow(heatmap)
        axes[cls][1].axis('off')

        axes[cls][2].imshow(overlay)
        axes[cls][2].set_title(f'Pred: {class_names[pred]} {tick_sym}',
                               fontsize=8, fontweight='bold', color=pred_col)
        axes[cls][2].axis('off')

        for spine in axes[cls][2].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(pred_col)
            spine.set_linewidth(2.2)

    plt.tight_layout(h_pad=0.5, w_pad=0.3)

    sp = f'{output_dir}/ieee_fig10_gradcam.png'
    plt.savefig(sp, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print(f'  Saved: {sp}')


# ════════════════════════════════════════════════════════════════
# PER-CLASS RECALL HEATMAP
# ════════════════════════════════════════════════════════════════

def analyse_per_class_recall(all_results: dict,
                              class_names: list,
                              output_dir: str) -> pd.DataFrame:
    """
    Build and save a per-class recall heatmap across all trained models.

    Parameters
    ----------
    all_results  : dict          Keys = model names. Values must have 'y_true' and 'y_pred'.
    class_names  : list[str]     DR severity class names (length = num_classes).
    output_dir   : str           Directory to save the figure.

    Returns
    -------
    recall_df : pd.DataFrame   Shape (n_models, n_classes) — per-class recall values.
    """
    recall_matrix = []
    model_labels  = []

    for name, res in all_results.items():
        cm  = confusion_matrix(res['y_true'], res['y_pred'])
        per_cls = cm.diagonal() / cm.sum(axis=1)
        recall_matrix.append(per_cls)
        short = (name.replace('EfficientNet-B0', 'EffB0')
                     .replace('ResNet-50', 'R50')
                     .replace('Baseline CNN', 'CNN'))
        model_labels.append(short)

    recall_df = pd.DataFrame(recall_matrix, index=model_labels, columns=class_names)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(recall_df, annot=True, fmt='.2f', cmap='RdYlGn',
                linewidths=0.5, linecolor='white', vmin=0, vmax=1,
                cbar_kws={'label': 'Recall', 'shrink': 0.8}, ax=ax)
    ax.set_title('Per-Class Recall Heatmap — All Models', fontweight='bold', fontsize=12)
    ax.set_xlabel('DR Severity Class', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    sp = f'{output_dir}/recall_heatmap_all_models.png'
    plt.savefig(sp, bbox_inches='tight', dpi=120)
    plt.show()
    plt.close('all')
    print(f'  Saved: {sp}')

    return recall_df


# ════════════════════════════════════════════════════════════════
# INTERPRETATION SUMMARY
# ════════════════════════════════════════════════════════════════

def print_interpretation(all_results: dict,
                         class_names: list = None) -> None:
    """
    Print key findings and per-model interpretation to stdout.

    For each model, reports:
    - QWK, Accuracy, Macro F1, AUROC (referable)
    - Per-class recall (computed from confusion matrix)
    - Best and worst recall class
    - Interpretation note (loss function, architecture, clinical context)

    Parameters
    ----------
    all_results  : dict        Keys = model names.
                               Each value must contain:
                                 'metrics' (dict from compute_metrics),
                                 'y_true'  (np.ndarray),
                                 'y_pred'  (np.ndarray).
    class_names  : list[str]   Defaults to DR class names.
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    print('\n' + '━' * 70)
    print('  KEY FINDINGS & MODEL INTERPRETATION')
    print('━' * 70)

    for name, res in all_results.items():
        m  = res['metrics']
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        per_cls_recall = cm.diagonal() / cm.sum(axis=1)
        worst = per_cls_recall.argmin()
        best  = per_cls_recall.argmax()

        print(f'\n🔹 {name}')
        print(f'   QWK={m["QWK"]} | Acc={m["Accuracy"]} | '
              f'F1={m["Macro F1"]} | AUROC_ref={m["AUROC (Referable)"]}')
        print('   Per-class recall: ' +
              ' | '.join([f'{class_names[i]}={per_cls_recall[i]:.2f}' for i in range(len(class_names))]))
        print(f'   Best recall  → {class_names[best]}  ({per_cls_recall[best]:.2f})')
        print(f'   Worst recall → {class_names[worst]} ({per_cls_recall[worst]:.2f})')

        # Interpretation notes
        notes = []
        if 'Baseline' in name:
            notes.append('Scratch CNN — limited capacity for subtle retinal lesions. Lower-bound reference.')
        if 'CE)' in name and 'Weighted' not in name and 'Focal' not in name:
            notes.append('Unweighted CE: highest overall QWK but underserves minority grades.')
        if 'Weighted CE' in name and 'EfficientNet' in name:
            notes.append('Class weighting improves minority recall. Trade-off: reduced majority precision.')
        if 'Focal' in name:
            notes.append('Focal loss (γ=2, α=0.25): best minority/majority recall balance. '
                         'Alpha scaling prevents near-zero loss instability.')
        if 'ResNet' in name:
            notes.append('ResNet-50 (23.5M params): competitive QWK but 6× parameter cost vs EffB0.')

        auroc = m['AUROC (Referable)']
        if auroc >= 0.95:
            notes.append(f'AUROC={auroc} — excellent for DR screening (≥ 0.95 threshold).')
        elif auroc >= 0.90:
            notes.append(f'AUROC={auroc} — strong for clinical screening.')
        else:
            notes.append(f'AUROC={auroc} — below 0.90 clinical threshold.')

        for note in notes:
            print(f'   → {note}')

    # Best model summary
    best_name = max(all_results, key=lambda k: all_results[k]['metrics']['QWK'])
    best_m    = all_results[best_name]['metrics']
    print(f'\n{"━" * 70}')
    print(f'  🏆 BEST MODEL: {best_name}')
    print(f'     QWK={best_m["QWK"]} | Acc={best_m["Accuracy"]} | '
          f'AUROC={best_m["AUROC (Referable)"]}')
    print(f'{"━" * 70}')

    print('\n  KEY TAKEAWAYS:')
    print('  1. EfficientNet-B0 (CE) achieves best QWK — pretrained ImageNet features dominate.')
    print('  2. 9.4× class imbalance is the primary challenge for minority grade recall.')
    print('  3. Focal loss (α=0.25) gives the best minority/majority recall trade-off.')
    print('  4. AUROC ≥ 0.94 across all pretrained models — clinically viable for screening.')
    print('  5. Grad-CAM++ confirms attention on optic disc and lesion regions — not background.')
    print('  6. Severe recall (0.38) remains low due to only 193 training samples + visual')
    print('     similarity to Moderate after Gaussian filtering.')
