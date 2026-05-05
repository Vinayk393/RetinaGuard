import os
"""
train.py
========
RetinaGuard — Loss Functions, Training Engine, Checkpointing.

Extracted directly from the RetinaGuard Phase 4 Kaggle notebook.

Loss Functions
--------------
get_ce_loss(label_smooth)
    Standard cross-entropy with optional label smoothing.
get_weighted_ce_loss(weights, label_smooth)
    Inverse-frequency class-weighted cross-entropy.
FocalLoss
    Alpha-scaled focal loss (Lin et al., 2017).
    alpha=0.25 prevents the near-zero magnitude failure of unscaled focal.

Training Functions
------------------
train_one_epoch(...)
    Single training pass with AMP, gradient clipping, optional backbone freeze.
evaluate(...)
    Validation / test pass returning loss, accuracy, QWK, predictions, probabilities.
train_model(...)
    Full training loop with CosineAnnealingLR, early stopping, checkpoint saving.
"""

import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
import torch.cuda.amp as amp

from sklearn.metrics import accuracy_score, cohen_kappa_score


# ════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ════════════════════════════════════════════════════════════════

def get_ce_loss(label_smooth: float = 0.05) -> nn.CrossEntropyLoss:
    """
    Standard cross-entropy loss with label smoothing.

    Parameters
    ----------
    label_smooth : float
        Label smoothing factor (default 0.05).
        Reduces overconfidence while preserving learning signal.

    Returns
    -------
    nn.CrossEntropyLoss
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smooth)


def get_weighted_ce_loss(weights: torch.FloatTensor,
                         label_smooth: float = 0.05) -> nn.CrossEntropyLoss:
    """
    Class-weighted cross-entropy loss.

    Upweights minority DR grades to counter the 9.4× class imbalance.
    Weights should be inverse-frequency normalised (see preprocess.compute_class_weights).

    Parameters
    ----------
    weights : torch.FloatTensor
        Per-class weights tensor, shape (num_classes,), on the correct device.
    label_smooth : float
        Label smoothing factor (default 0.05).

    Returns
    -------
    nn.CrossEntropyLoss

    DR dataset class weights
    ------------------------
    No DR=0.22, Mild=1.05, Moderate=0.39, Severe=2.02, Proliferative=1.32
    """
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smooth)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017) with alpha magnitude scaling.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values down-weight easy examples more.
        Default: 2.0 (recommended for class-imbalanced datasets).
    alpha : float
        Magnitude scaling factor. Set to 0.25 to rescale focal loss to
        the same order of magnitude as standard cross-entropy, preventing
        near-zero loss values that cause training instability.
        Default: 0.25.
    weight : torch.FloatTensor, optional
        Per-class weights (same as weighted CE). If provided, class-level
        reweighting is applied inside the CE base.
    label_smoothing : float
        Label smoothing applied to the CE base loss (default 0.05).

    Notes
    -----
    The alpha=0.25 fix is critical. Without it, focal loss magnitudes of
    ~0.006 were observed, causing effectively zero gradient updates and
    near-random predictions after training. See RetinaGuard Phase 4 report.
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 weight: torch.FloatTensor = None,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt     = torch.exp(-ce)
        focal  = self.alpha * ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def get_focal_loss(weights: torch.FloatTensor = None,
                   gamma: float = 2.0,
                   alpha: float = 0.25,
                   label_smooth: float = 0.05) -> FocalLoss:
    """
    Convenience constructor for FocalLoss.

    Parameters
    ----------
    weights : torch.FloatTensor, optional
        Per-class weights (same shape as number of classes).
    gamma : float
        Focusing parameter (default 2.0).
    alpha : float
        Magnitude rescaling factor (default 0.25).
    label_smooth : float
        Label smoothing (default 0.05).

    Returns
    -------
    FocalLoss
    """
    return FocalLoss(gamma=gamma, alpha=alpha, weight=weights, label_smoothing=label_smooth)


# ════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    freeze_backbone: bool = False):
    """
    Run one training epoch.

    Parameters
    ----------
    model       : nn.Module         Pytorch model (already on device).
    loader      : DataLoader        Training DataLoader.
    optimizer   : torch.optim       Optimiser instance.
    criterion   : nn.Module         Loss function.
    scaler      : GradScaler        AMP gradient scaler.
    device      : str               Target device ('cuda' / 'cpu').
    freeze_backbone : bool          If True, freezes all non-head parameters.
                                    Used for epochs 1–3 of pretrained models.

    Returns
    -------
    avg_loss : float   Mean loss per sample.
    accuracy : float   Overall classification accuracy.
    qwk      : float   Quadratic Weighted Kappa.
    """
    model.train()

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not any(k in name for k in ['classifier', 'fc', 'head']):
                p.requires_grad_(False)

    running_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels_b in loader:
        imgs, labels_b = imgs.to(device), labels_b.to(device)
        optimizer.zero_grad()

        with amp.autocast(enabled=(device == 'cuda')):
            out  = model(imgs)
            loss = criterion(out, labels_b)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        all_preds.extend(out.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels_b.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    qwk = (cohen_kappa_score(all_labels, all_preds, weights='quadratic')
           if len(set(all_labels)) > 1 else 0.0)

    return running_loss / len(loader.dataset), acc, qwk


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Run one evaluation pass (validation or test).

    Parameters
    ----------
    model     : nn.Module    Pytorch model (already on device).
    loader    : DataLoader   Validation or test DataLoader.
    criterion : nn.Module    Loss function.
    device    : str          Target device.

    Returns
    -------
    avg_loss  : float
    accuracy  : float
    qwk       : float
    y_pred    : np.ndarray   shape (N,) — argmax predictions
    y_true    : np.ndarray   shape (N,) — ground-truth labels
    y_probs   : np.ndarray   shape (N, num_classes) — softmax probabilities
    """
    model.eval()
    running_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    for imgs, labels_b in loader:
        imgs, labels_b = imgs.to(device), labels_b.to(device)
        with amp.autocast(enabled=(device == 'cuda')):
            out  = model(imgs)
            loss = criterion(out, labels_b)

        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(probs.argmax(axis=1))
        all_labels.extend(labels_b.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    qwk = (cohen_kappa_score(all_labels, all_preds, weights='quadratic')
           if len(set(all_labels)) > 1 else 0.0)

    return (running_loss / len(loader.dataset), acc, qwk,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                model_name: str,
                ckpt_dir: str,
                device: str,
                lr: float = 1e-4,
                weight_decay: float = 1e-4,
                num_epochs: int = 20,
                patience: int = 5,
                use_amp: bool = True):
    """
    Full training loop with CosineAnnealingLR, early stopping, and checkpointing.

    Strategy
    --------
    - Epochs 1–3: backbone frozen (pretrained models only), only head trained.
    - Epoch 4+  : full unfreezing, end-to-end fine-tuning.
    - Scheduler : CosineAnnealingLR (T_max=num_epochs, eta_min=1e-6).
    - Early stop: patience=5 epochs without improvement in Val QWK.
    - Checkpoint: best model saved to {ckpt_dir}/{slug}_best.pth.

    Parameters
    ----------
    model        : nn.Module        Model to train (not yet on device).
    train_loader : DataLoader       Training DataLoader.
    val_loader   : DataLoader       Validation DataLoader.
    criterion    : nn.Module        Loss function.
    model_name   : str              Human-readable model name (used for slug + logging).
    ckpt_dir     : str              Directory to save checkpoint .pth files.
    device       : str              Target device ('cuda' / 'cpu').
    lr           : float            Initial learning rate (default 1e-4).
    weight_decay : float            AdamW weight decay (default 1e-4).
    num_epochs   : int              Max training epochs (default 20).
    patience     : int              Early stopping patience on Val QWK (default 5).
    use_amp      : bool             Enable mixed-precision training (default True).

    Returns
    -------
    model   : nn.Module   Best model (loaded from checkpoint).
    history : dict        Keys: train_loss, val_loss, train_acc, val_acc,
                                train_qwk, val_qwk, lr.
                          Each value is a list of per-epoch scalars.
    """
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=use_amp)

    history = {k: [] for k in
               ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_qwk', 'val_qwk', 'lr']}

    best_qwk, best_weights, patience_ctr = -1.0, None, 0

    # Sanitise model name for filename
    slug      = (model_name.replace(' ', '_').replace('/', '_')
                           .replace('(', '').replace(')', ''))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f'{ckpt_dir}/{slug}_best.pth'

    is_pretrained = any(k in model_name for k in ['EfficientNet', 'ResNet'])

    print(f'\n{"=" * 65}')
    print(f'  Training  : {model_name}')
    print(f'  Optimizer : AdamW | lr={lr} | wd={weight_decay}')
    print(f'  Scheduler : CosineAnnealingLR | T_max={num_epochs}')
    print(f'  Checkpoint: {ckpt_path}')
    print(f'{"=" * 65}')

    for epoch in range(1, num_epochs + 1):
        freeze = is_pretrained and (epoch <= 3)

        if is_pretrained and epoch == 4:
            for p in model.parameters():
                p.requires_grad_(True)
            print('  ↳ Backbone unfrozen at epoch 4')

        t0 = time.time()
        tr_loss, tr_acc, tr_qwk = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, freeze)
        vl_loss, vl_acc, vl_qwk, _, _, _ = evaluate(
            model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Record history
        for k, v in zip(
            ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_qwk', 'val_qwk', 'lr'],
            [tr_loss, vl_loss, tr_acc, vl_acc, tr_qwk, vl_qwk, current_lr]
        ):
            history[k].append(v)

        # Checkpoint on improvement
        marker = ''
        if vl_qwk > best_qwk:
            best_qwk     = vl_qwk
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, ckpt_path)
            patience_ctr = 0
            marker       = '  ✅ BEST'
        else:
            patience_ctr += 1

        print(f'  Ep {epoch:02d}/{num_epochs} | '
              f'Tr[loss={tr_loss:.4f} acc={tr_acc:.3f} qwk={tr_qwk:.3f}] | '
              f'Vl[loss={vl_loss:.4f} acc={vl_acc:.3f} qwk={vl_qwk:.3f}] | '
              f'{time.time() - t0:.1f}s{marker}')

        if patience_ctr >= patience:
            print(f'  ⏹  Early stop at epoch {epoch} — best Val QWK: {best_qwk:.4f}')
            break

    # Restore best weights
    model.load_state_dict(best_weights)
    print(f'\n  Training complete. Best Val QWK: {best_qwk:.4f}')
    print(f'  Checkpoint saved: {ckpt_path}')

    return model, history
