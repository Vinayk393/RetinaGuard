"""
evaluate.py
===========
RetinaGuard — Evaluation Metrics, Test-Time Augmentation, Threshold Optimisation.

Extracted directly from the RetinaGuard Phase 4 Kaggle notebook.

Functions
---------
compute_metrics(y_true, y_pred, y_probs, model_name)
    Full evaluation suite: accuracy, precision, recall, F1, QWK,
    AUROC (referable DR binary), AUROC (macro OvR).

predict_with_tta(model, dataset, val_transform, tta_transform, cfg)
    Inference with Test-Time Augmentation (5 passes, averaged softmax).

optimise_threshold(y_true, y_probs, metric='f1')
    Sweep threshold for referable DR probability and return the
    operating point maximising the chosen metric.

print_metrics_table(all_results)
    Pretty-print the final comparison table sorted by QWK.
"""

import numpy as np
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize


# DR class names — used in classification report
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']


# ════════════════════════════════════════════════════════════════
# FULL METRICS SUITE
# ════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_probs: np.ndarray,
                    model_name: str,
                    class_names: list = None) -> tuple:
    """
    Compute the complete evaluation metric suite used in RetinaGuard.

    Metrics
    -------
    - Accuracy             : overall correct prediction rate
    - Macro Precision      : unweighted mean precision across 5 classes
    - Macro Recall         : unweighted mean recall across 5 classes
    - Macro F1             : unweighted mean F1 across 5 classes
    - QWK                  : Quadratic Weighted Kappa (primary metric)
    - AUROC (Referable)    : binary AUROC for grade ≥ 2 vs grade < 2
    - AUROC (Macro OvR)    : macro one-vs-rest AUROC across all 5 classes

    Parameters
    ----------
    y_true      : np.ndarray  shape (N,)        Ground-truth labels (int 0–4).
    y_pred      : np.ndarray  shape (N,)        Predicted class labels.
    y_probs     : np.ndarray  shape (N, 5)      Softmax probabilities.
    model_name  : str                           Model identifier for reporting.
    class_names : list[str], optional           Defaults to CLASS_NAMES above.

    Returns
    -------
    metrics : dict
        All scalar metrics keyed by name.
    report : dict
        Per-class precision / recall / F1 as returned by sklearn.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    acc      = accuracy_score(y_true, y_pred)
    qwk      = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_p  = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r  = recall_score(y_true, y_pred, average='macro', zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # ── Referable DR binary AUROC (grade ≥ 2 = referable)
    ref_true  = (y_true >= 2).astype(int)
    ref_probs = y_probs[:, 2:].sum(axis=1)
    try:
        auroc_ref = roc_auc_score(ref_true, ref_probs)
    except ValueError:
        auroc_ref = float('nan')  # only one class present in test set

    # ── Macro one-vs-rest AUROC across all 5 classes
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    try:
        auroc_macro = roc_auc_score(y_bin, y_probs, average='macro', multi_class='ovr')
    except ValueError:
        auroc_macro = float('nan')

    metrics = {
        'Model'             : model_name,
        'Accuracy'          : round(acc,       4),
        'Macro Precision'   : round(macro_p,   4),
        'Macro Recall'      : round(macro_r,   4),
        'Macro F1'          : round(macro_f1,  4),
        'QWK'               : round(qwk,       4),
        'AUROC (Referable)' : round(auroc_ref,   4),
        'AUROC (Macro OvR)' : round(auroc_macro, 4),
    }

    return metrics, report


# ════════════════════════════════════════════════════════════════
# TEST-TIME AUGMENTATION (TTA)
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_tta(model,
                     dataset,
                     val_transform,
                     tta_transform,
                     n_tta: int = 5,
                     batch_size: int = 32,
                     device: str = 'cuda',
                     num_workers: int = 2) -> tuple:
    """
    Inference with Test-Time Augmentation.

    Runs the model `n_tta` times, each time with stochastic tta_transform,
    then averages softmax probabilities for the final prediction.

    Parameters
    ----------
    model          : nn.Module         Trained model (already on CPU — moved internally).
    dataset        : RetinopathyDataset  Test dataset instance.
    val_transform  : transforms.Compose  Deterministic transform for ground truth.
    tta_transform  : transforms.Compose  Stochastic transform applied at each TTA pass.
    n_tta          : int               Number of TTA passes (default 5).
    batch_size     : int               Inference batch size (default 32).
    device         : str               Target device (default 'cuda').
    num_workers    : int               DataLoader workers (default 2).

    Returns
    -------
    y_pred   : np.ndarray  shape (N,)     Final predicted labels (argmax of averaged probs).
    y_true   : np.ndarray  shape (N,)     Ground-truth labels.
    y_probs  : np.ndarray  shape (N, 5)   Averaged softmax probabilities.
    """
    model = model.to(device)
    model.eval()

    # Collect ground truth labels using deterministic transform
    all_labels = []
    dataset.transform = val_transform
    for _, label in DataLoader(dataset, batch_size=1, shuffle=False):
        all_labels.append(label.item())

    all_probs = None

    for tta_i in range(n_tta):
        dataset.transform = tta_transform
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
        fold = []
        for imgs, _ in loader:
            imgs = imgs.to(device)
            with amp.autocast(enabled=(device == 'cuda')):
                out = model(imgs)
            fold.append(torch.softmax(out, dim=1).cpu().numpy())

        fold_arr  = np.concatenate(fold, axis=0)
        all_probs = fold_arr if all_probs is None else all_probs + fold_arr

    # Restore deterministic transform
    dataset.transform = val_transform

    all_probs /= n_tta
    y_pred = all_probs.argmax(axis=1)

    return y_pred, np.array(all_labels), all_probs


# ════════════════════════════════════════════════════════════════
# CLINICAL THRESHOLD OPTIMISATION
# ════════════════════════════════════════════════════════════════

def optimise_threshold(y_true: np.ndarray,
                       y_probs: np.ndarray,
                       metric: str = 'f1',
                       n_steps: int = 181) -> dict:
    """
    Sweep the referable DR classification threshold and return the
    optimal operating point.

    Referable DR is defined as grade ≥ 2 (Moderate, Severe, Proliferative).

    Parameters
    ----------
    y_true   : np.ndarray  shape (N,)    Ground-truth 5-class labels.
    y_probs  : np.ndarray  shape (N, 5)  Softmax probabilities.
    metric   : str                       Metric to optimise: 'f1' (default) or 'youden'.
    n_steps  : int                       Number of threshold values to evaluate (default 181).

    Returns
    -------
    result : dict with keys:
        threshold   : float   Optimal threshold value.
        sensitivity : float   True positive rate at optimal threshold.
        specificity : float   True negative rate at optimal threshold.
        f1          : float   F1 score at optimal threshold.
        ppv         : float   Precision (Positive Predictive Value) at optimal threshold.
        thresholds  : list    All evaluated thresholds.
        sensitivities : list  Sensitivity at each threshold.
        specificities : list  Specificity at each threshold.
        f1s         : list    F1 at each threshold.
        ppvs        : list    PPV at each threshold.
    """
    ref_true  = (y_true >= 2).astype(int)
    ref_probs = y_probs[:, 2:].sum(axis=1)

    thresholds    = np.linspace(0.05, 0.95, n_steps)
    sensitivities = []
    specificities = []
    f1s           = []
    ppvs          = []

    for t in thresholds:
        preds = (ref_probs >= t).astype(int)
        tp = ((preds == 1) & (ref_true == 1)).sum()
        fp = ((preds == 1) & (ref_true == 0)).sum()
        tn = ((preds == 0) & (ref_true == 0)).sum()
        fn = ((preds == 0) & (ref_true == 1)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = f1_score(ref_true, preds, zero_division=0)

        sensitivities.append(sens)
        specificities.append(spec)
        f1s.append(f1)
        ppvs.append(ppv)

    if metric == 'youden':
        best_i = int(np.argmax(np.array(sensitivities) + np.array(specificities) - 1))
    else:  # f1
        best_i = int(np.argmax(f1s))

    return {
        'threshold'     : round(float(thresholds[best_i]),   3),
        'sensitivity'   : round(float(sensitivities[best_i]), 3),
        'specificity'   : round(float(specificities[best_i]), 3),
        'f1'            : round(float(f1s[best_i]),           3),
        'ppv'           : round(float(ppvs[best_i]),          3),
        'thresholds'    : list(thresholds),
        'sensitivities' : sensitivities,
        'specificities' : specificities,
        'f1s'           : f1s,
        'ppvs'          : ppvs,
    }


# ════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════

def print_metrics_table(all_results: dict) -> None:
    """
    Pretty-print the final comparison table sorted by QWK (descending).

    Parameters
    ----------
    all_results : dict
        Keys are model names. Each value must have a 'metrics' key
        containing the dict returned by compute_metrics().
    """
    rows = [v['metrics'] for v in all_results.values()]
    rows.sort(key=lambda x: x['QWK'], reverse=True)

    col_order = ['Model', 'QWK', 'Accuracy', 'Macro F1',
                 'AUROC (Referable)', 'AUROC (Macro OvR)']

    header = f"{'Rank':<5}" + "".join(f"{c:<28}" for c in col_order)
    sep    = "─" * len(header)

    print(f"\n{sep}")
    print("  FINAL RESULTS TABLE  (sorted by QWK)")
    print(sep)
    print(header)
    print(sep)

    for rank, row in enumerate(rows, 1):
        line = f"{rank:<5}"
        for col in col_order:
            val = row.get(col, '')
            line += f"{str(val):<28}"
        if rank == 1:
            line += '  ← BEST'
        print(line)

    print(sep)
