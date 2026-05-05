"""
run_experiment.py
=================
RetinaGuard — End-to-End Experiment Runner.

Runs one complete experiment (train → evaluate → Grad-CAM++ → save outputs)
from a single YAML config file.

Usage
-----
    python src/run_experiment.py --config configs/ce.yaml
    python src/run_experiment.py --config configs/focal.yaml
    python src/run_experiment.py --config configs/weighted_ce.yaml

    # Override individual config values at the command line:
    python src/run_experiment.py --config configs/ce.yaml --epochs 10 --seed 0

Output structure (auto-created)
--------------------------------
    outputs/{experiment_slug}/
        checkpoints/
            {slug}_best.pth
        figures/
            training_curves.png
            confusion_matrix.png
            roc_pr.png
            per_class_metrics.png
        gradcam/
            gradcam_{slug}.png
        logs/
            {slug}_{timestamp}.log
        history.json
        metrics.json
        classification_report.json
        threshold_results.json

Requirements
------------
    pip install -r requirements.txt
    # Dataset must be accessible at the path specified in the config.

Paper reference
---------------
    RetinaGuard: Deep Learning-Based Diabetic Retinopathy Severity
    Classification for Early Vision-Loss Prevention.
    CECS 551, CSULB, Spring 2026.
"""

import os
import sys
import argparse
import yaml
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# Add project root to path so `src` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils      import set_seed, make_output_dirs, save_json, setup_logger, get_device
from src.models     import build_model
from src.preprocess import make_dataloaders, compute_class_weights
from src.train      import get_ce_loss, get_weighted_ce_loss, get_focal_loss, train_model
from src.evaluate   import compute_metrics, predict_with_tta, optimise_threshold
from src.interpret  import run_gradcam, get_cam_target_layer
from src.preprocess import get_transforms


# ════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='RetinaGuard — run one DR classification experiment from a YAML config.'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to experiment YAML config (e.g. configs/ce.yaml)'
    )
    parser.add_argument('--epochs',     type=int,   default=None, help='Override num_epochs')
    parser.add_argument('--lr',         type=float, default=None, help='Override learning rate')
    parser.add_argument('--seed',       type=int,   default=None, help='Override random seed')
    parser.add_argument('--batch_size', type=int,   default=None, help='Override batch size')
    parser.add_argument('--output_dir', type=str,   default=None, help='Override output root dir')
    parser.add_argument('--no_gradcam', action='store_true',      help='Skip Grad-CAM generation')
    return parser.parse_args()


# ════════════════════════════════════════════════════════════════
# CONFIG LOADING
# ════════════════════════════════════════════════════════════════

def load_config(config_path: str, args) -> dict:
    """Load YAML config and apply any CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs     is not None: cfg['num_epochs']  = args.epochs
    if args.lr         is not None: cfg['lr']          = args.lr
    if args.seed       is not None: cfg['seed']        = args.seed
    if args.batch_size is not None: cfg['batch_size']  = args.batch_size
    if args.output_dir is not None: cfg['output_dir']  = args.output_dir

    # Ensure device is set
    cfg['device'] = get_device()

    return cfg


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_and_split_data(cfg: dict, logger) -> tuple:
    """
    Load CSV, build image paths, stratified split → (df_train, df_val, df_test).
    """
    from src.preprocess import apply_clahe  # noqa — used in dataset

    class_dirs = cfg['class_dirs']  # {0: 'No_DR', 1: 'Mild', ...}
    dataset_root = cfg['dataset_root']
    csv_path     = cfg['csv_path']

    df_raw = pd.read_csv(csv_path)

    def build_path(row):
        label  = int(row['diagnosis'])
        folder = class_dirs[str(label)]
        img_id = str(row['id_code']).strip()
        for ext in ['.png', '.jpg', '.jpeg', '']:
            p = Path(dataset_root) / folder / f'{img_id}{ext}'
            if p.exists():
                return str(p)
        return None

    df_raw['filepath'] = df_raw.apply(build_path, axis=1)
    df_raw['label']    = df_raw['diagnosis'].astype(int)
    df = df_raw.dropna(subset=['filepath']).reset_index(drop=True)

    logger.info(f"Dataset: {len(df)} valid images from {csv_path}")

    # Stratified split
    seed = cfg.get('seed', 42)
    X, y = df.index.values, df['label'].values

    sss1 = StratifiedShuffleSplit(n_splits=1,
        test_size=cfg['val_frac'] + cfg['test_frac'], random_state=seed)
    train_idx, temp_idx = next(sss1.split(X, y))

    sss2 = StratifiedShuffleSplit(n_splits=1,
        test_size=cfg['test_frac'] / (cfg['val_frac'] + cfg['test_frac']),
        random_state=seed)
    val_idx_l, test_idx_l = next(sss2.split(temp_idx, y[temp_idx]))
    val_idx  = temp_idx[val_idx_l]
    test_idx = temp_idx[test_idx_l]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx ].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    logger.info(f"Split → Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")

    return df_train, df_val, df_test


# ════════════════════════════════════════════════════════════════
# LOSS FACTORY
# ════════════════════════════════════════════════════════════════

def build_criterion(cfg: dict, class_weights):
    """Dispatch loss function from config string."""
    loss_name    = cfg['loss'].lower()
    label_smooth = cfg.get('label_smooth', 0.05)

    if loss_name == 'ce':
        return get_ce_loss(label_smooth=label_smooth)
    elif loss_name == 'weighted_ce':
        return get_weighted_ce_loss(weights=class_weights, label_smooth=label_smooth)
    elif loss_name == 'focal':
        return get_focal_loss(
            weights=class_weights,
            gamma=cfg.get('focal_gamma', 2.0),
            alpha=cfg.get('focal_alpha', 0.25),
            label_smooth=label_smooth
        )
    else:
        raise ValueError(f"Unknown loss '{loss_name}'. Choose: ce | weighted_ce | focal")


# ════════════════════════════════════════════════════════════════
# PLOTTING HELPERS (standalone — no notebook dependency)
# ════════════════════════════════════════════════════════════════

DR_COLORS  = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']


def save_training_curves(history: dict, model_name: str, save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Training Curves — {model_name}', fontweight='bold')
    epochs = range(1, len(history['train_loss']) + 1)

    for ax, tr_k, vl_k, title, col in [
        (axes[0], 'train_loss', 'val_loss', 'Loss',     '#e74c3c'),
        (axes[1], 'train_acc',  'val_acc',  'Accuracy', '#3498db'),
        (axes[2], 'train_qwk',  'val_qwk',  'QWK',      '#2ecc71'),
    ]:
        ax.plot(epochs, history[tr_k], color=col, lw=2, label='Train')
        ax.plot(epochs, history[vl_k], color=col, lw=2, ls='--', label='Val', alpha=0.7)
        ax.fill_between(epochs, history[tr_k], history[vl_k], alpha=0.08, color=col)
        best_ep = (int(np.argmin(history[vl_k])) if 'loss' in vl_k
                   else int(np.argmax(history[vl_k]))) + 1
        ax.axvline(best_ep, color='grey', ls=':', lw=1, label=f'Best ep={best_ep}')
        ax.set_title(title, fontweight='bold'); ax.set_xlabel('Epoch')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close('all')
    print(f"  Saved: {save_path}")


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list, model_name: str, save_path: str) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Confusion Matrices — {model_name}', fontweight='bold')

    for ax, data, fmt, title, cmap in [
        (axes[0], cm,   'd',    'Raw Counts',        'Blues'),
        (axes[1], cm_n, '.2f',  'Row-Normalised',    'YlOrRd'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor='white', ax=ax,
                    cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close('all')
    print(f"  Saved: {save_path}")


def save_roc_curves(y_true: np.ndarray, y_probs: np.ndarray,
                    class_names: list, model_name: str, save_path: str) -> None:
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'ROC & PR Curves — {model_name}', fontweight='bold')

    # Referable DR binary ROC
    ref_true  = (y_true >= 2).astype(int)
    ref_probs = y_probs[:, 2:].sum(axis=1)
    try:
        fpr_r, tpr_r, _ = roc_curve(ref_true, ref_probs)
        auc_r = roc_auc_score(ref_true, ref_probs)
        axes[0].plot(fpr_r, tpr_r, color='#e74c3c', lw=2, label=f'Referable DR AUC={auc_r:.3f}')
    except ValueError:
        pass

    # Per-class ROC
    for i, (cls_name, c) in enumerate(zip(class_names, DR_COLORS)):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
            axes[1].plot(fpr, tpr, color=c, lw=2, label=f'{cls_name} AUC={auc:.3f}')
        except ValueError:
            pass

    for ax, title in zip(axes, ['Binary ROC (Referable DR)', 'Per-Class ROC']):
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close('all')
    print(f"  Saved: {save_path}")


def save_per_class_metrics(report: dict, class_names: list,
                           model_name: str, save_path: str) -> None:
    rows = {c: report[c] for c in class_names}
    df_m = pd.DataFrame(rows).T[['precision', 'recall', 'f1-score']]

    x, w = np.arange(len(class_names)), 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w,   df_m['precision'], w, label='Precision', color='#3498db', alpha=0.85)
    ax.bar(x,       df_m['recall'],    w, label='Recall',    color='#e74c3c', alpha=0.85)
    ax.bar(x + w,   df_m['f1-score'],  w, label='F1',        color='#2ecc71', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=20)
    ax.set_title(f'Per-Class Metrics — {model_name}', fontweight='bold')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.1); ax.legend(); ax.grid(axis='y', alpha=0.3)
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close('all')
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Load config
    cfg = load_config(args.config, args)

    experiment_name = cfg.get('experiment_name', 'experiment')
    output_root     = cfg.get('output_dir', 'outputs')
    seed            = cfg.get('seed', 42)
    class_names     = cfg.get('class_names', ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
    device          = cfg['device']

    # ── Output dirs
    dirs = make_output_dirs(output_root, experiment_name)
    logger = setup_logger(dirs['logs'], experiment_name)

    logger.info('=' * 65)
    logger.info(f'  RetinaGuard Experiment: {experiment_name}')
    logger.info(f'  Config: {args.config}')
    logger.info(f'  Output: {dirs["root"]}')
    logger.info('=' * 65)
    logger.info(f'  Model      : {cfg["model"]}')
    logger.info(f'  Loss       : {cfg["loss"]}')
    logger.info(f'  LR         : {cfg["lr"]}')
    logger.info(f'  Epochs     : {cfg["num_epochs"]}')
    logger.info(f'  Batch size : {cfg["batch_size"]}')
    logger.info(f'  Seed       : {seed}')
    logger.info(f'  Device     : {device}')
    logger.info('=' * 65)

    # ── Seed
    set_seed(seed)

    # ── Data
    df_train, df_val, df_test = load_and_split_data(cfg, logger)

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(
        df_train, df_val, df_test, cfg
    )

    # ── Class weights (used by weighted CE and focal loss)
    class_weights = compute_class_weights(df_train, cfg['num_classes'], device)
    logger.info('Class weights: ' +
                ', '.join([f'{n}={w:.3f}' for n, w in zip(class_names, class_weights.cpu().numpy())]))

    # ── Model
    model = build_model(
        cfg['model'],
        num_classes=cfg['num_classes'],
        pretrained=cfg.get('pretrained', True)
    )
    logger.info(f'Model built: {cfg["model"]}')

    # ── Loss
    criterion = build_criterion(cfg, class_weights)
    logger.info(f'Loss: {cfg["loss"]}')

    # ── Train
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        model_name=experiment_name,
        ckpt_dir=dirs['checkpoints'],
        device=device,
        lr=cfg.get('lr', 1e-4),
        weight_decay=cfg.get('weight_decay', 1e-4),
        num_epochs=cfg.get('num_epochs', 20),
        patience=cfg.get('patience', 5),
        use_amp=cfg.get('use_amp', True),
    )

    # ── Save history
    history_path = os.path.join(dirs['root'], 'history.json')
    save_json(history, history_path)

    # ── TTA Evaluation
    logger.info(f'Running TTA (n={cfg.get("tta_n", 5)}) on test set...')
    _, val_tf, tta_tf = get_transforms()

    y_pred, y_true, y_probs = predict_with_tta(
        model=model,
        dataset=test_ds,
        val_transform=val_tf,
        tta_transform=tta_tf,
        n_tta=cfg.get('tta_n', 5),
        batch_size=cfg.get('batch_size', 32),
        device=device,
    )

    # ── Metrics
    metrics, report = compute_metrics(y_true, y_pred, y_probs, experiment_name, class_names)

    logger.info('\n' + '─' * 50)
    logger.info('  TEST RESULTS')
    logger.info('─' * 50)
    for k, v in metrics.items():
        if k != 'Model':
            logger.info(f'  {k:<28s}: {v}')
    logger.info('─' * 50)

    # ── Save metrics
    save_json(metrics, os.path.join(dirs['root'], 'metrics.json'))
    save_json(report,  os.path.join(dirs['root'], 'classification_report.json'))

    # ── Threshold optimisation
    thresh_result = optimise_threshold(y_true, y_probs)
    save_json({k: v for k, v in thresh_result.items()
               if not isinstance(v, list)},
              os.path.join(dirs['root'], 'threshold_results.json'))
    logger.info(f'\n  Optimal threshold  : {thresh_result["threshold"]}')
    logger.info(f'  Sensitivity @ opt  : {thresh_result["sensitivity"]}')
    logger.info(f'  Specificity @ opt  : {thresh_result["specificity"]}')
    logger.info(f'  F1 @ opt           : {thresh_result["f1"]}')

    # ── Figures
    save_training_curves(history, experiment_name,
                         os.path.join(dirs['figures'], 'training_curves.png'))
    save_confusion_matrix(y_true, y_pred, class_names, experiment_name,
                          os.path.join(dirs['figures'], 'confusion_matrix.png'))
    save_roc_curves(y_true, y_probs, class_names, experiment_name,
                    os.path.join(dirs['figures'], 'roc_curves.png'))
    save_per_class_metrics(report, class_names, experiment_name,
                           os.path.join(dirs['figures'], 'per_class_metrics.png'))

    # ── Grad-CAM
    if not args.no_gradcam and cfg['model'] in ['efficientnet_b0', 'resnet50']:
        logger.info('Generating Grad-CAM++ visualisations...')
        from src.preprocess import apply_clahe
        run_gradcam(
            model=model,
            model_name=experiment_name,
            df_subset=df_test,
            cfg=cfg,
            val_transform=val_tf,
            apply_clahe_fn=apply_clahe,
            output_dir=dirs['gradcam'],
            n_samples=8,
            seed=seed,
        )

    # ── Final summary
    logger.info('\n' + '=' * 65)
    logger.info(f'  EXPERIMENT COMPLETE: {experiment_name}')
    logger.info(f'  QWK              : {metrics["QWK"]}')
    logger.info(f'  Accuracy         : {metrics["Accuracy"]}')
    logger.info(f'  AUROC (Referable): {metrics["AUROC (Referable)"]}')
    logger.info(f'  Outputs saved to : {dirs["root"]}')
    logger.info('=' * 65)

    return metrics


if __name__ == '__main__':
    main()
