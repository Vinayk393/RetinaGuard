"""
models.py
=========
RetinaGuard — Model Architectures and Factory.

Architectures
-------------
BaselineCNN
    4-block custom CNN trained from scratch. Lower-bound reference.
    456K parameters.

build_efficientnet_b0(num_classes, pretrained)
    EfficientNet-B0 via timm. Compound-scaled CNN.
    4.0M parameters. ImageNet pretrained.

build_resnet50(num_classes, pretrained)
    ResNet-50 via timm. Deep residual network.
    23.5M parameters. ImageNet pretrained.

build_model(model_name, num_classes, pretrained)
    Config-driven factory — dispatches by model name string.

count_params(model)
    Return (total_params, trainable_params).

print_model_summary()
    Print a formatted parameter summary table.
"""

import torch
import torch.nn as nn
import timm


# ════════════════════════════════════════════════════════════════
# BASELINE CNN
# ════════════════════════════════════════════════════════════════

class BaselineCNN(nn.Module):
    """
    Four-block custom CNN trained from scratch.

    Architecture
    ------------
    Block i: Conv2d(in, out, 3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2)
    Filter counts: [32, 64, 128, 256]

    Head: AdaptiveAvgPool2d(1) → Flatten → Linear(256→256) → ReLU →
          Dropout(0.5) → Linear(256→num_classes)

    Parameters: ~456K total.

    Purpose
    -------
    Lower-bound reference to quantify the gain from pretrained transfer learning.
    EfficientNet-B0 achieves 19.4% relative QWK improvement over this baseline.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ════════════════════════════════════════════════════════════════
# EFFICIENTNET-B0
# ════════════════════════════════════════════════════════════════

def build_efficientnet_b0(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNet-B0 via timm.

    EfficientNet uses compound scaling to jointly scale network depth, width,
    and resolution, yielding superior parameter efficiency compared to
    independently scaled architectures (Tan & Le, ICML 2019).

    Parameters
    ----------
    num_classes : int    Output classes (default 5 for DR grading).
    pretrained  : bool   Use ImageNet pretrained weights (default True).
                         Set False for architecture-only param counting.

    Returns
    -------
    nn.Module   EfficientNet-B0 with replaced classifier head.
    """
    return timm.create_model(
        'efficientnet_b0',
        pretrained=pretrained,
        num_classes=num_classes
    )


# ════════════════════════════════════════════════════════════════
# RESNET-50
# ════════════════════════════════════════════════════════════════

def build_resnet50(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet-50 via timm.

    ResNet-50 uses shortcut connections to enable training of very deep
    networks without vanishing gradients (He et al., CVPR 2016).
    Used as an architectural comparison against EfficientNet-B0.

    Parameters
    ----------
    num_classes : int    Output classes (default 5 for DR grading).
    pretrained  : bool   Use ImageNet pretrained weights (default True).

    Returns
    -------
    nn.Module   ResNet-50 with replaced fully connected head.
    """
    return timm.create_model(
        'resnet50',
        pretrained=pretrained,
        num_classes=num_classes
    )


# ════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ════════════════════════════════════════════════════════════════

# Registry mapping config model_name strings → build functions
_MODEL_REGISTRY = {
    'baseline_cnn'      : lambda nc, pt: BaselineCNN(num_classes=nc),
    'efficientnet_b0'   : lambda nc, pt: build_efficientnet_b0(num_classes=nc, pretrained=pt),
    'resnet50'          : lambda nc, pt: build_resnet50(num_classes=nc, pretrained=pt),
}


def build_model(model_name: str,
                num_classes: int = 5,
                pretrained: bool = True) -> nn.Module:
    """
    Config-driven model factory.

    Dispatches to the correct constructor based on a string key,
    enabling fully config-driven experiments without code changes.

    Parameters
    ----------
    model_name  : str   Model identifier. Must be one of:
                            'baseline_cnn'
                            'efficientnet_b0'
                            'resnet50'
    num_classes : int   Number of output classes (default 5).
    pretrained  : bool  Whether to use ImageNet pretrained weights (default True).

    Returns
    -------
    nn.Module   Instantiated model with the correct output head.

    Raises
    ------
    ValueError   If model_name is not in the registry.

    Example
    -------
    >>> model = build_model('efficientnet_b0', num_classes=5, pretrained=True)
    """
    key = model_name.lower().strip()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[key](num_classes, pretrained)


# ════════════════════════════════════════════════════════════════
# PARAM COUNT
# ════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> tuple:
    """
    Count total and trainable parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    (total_params, trainable_params) : tuple of int
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(num_classes: int = 5) -> None:
    """
    Print a formatted parameter count table for all three model architectures.
    Uses pretrained=False to avoid downloading weights during summary.

    Parameters
    ----------
    num_classes : int   Number of output classes (default 5).
    """
    rows = [
        ('Baseline CNN',    'baseline_cnn',    False),
        ('EfficientNet-B0', 'efficientnet_b0', False),
        ('ResNet-50',       'resnet50',        False),
    ]

    print('┌─────────────────────────┬─────────────────┬─────────────────┐')
    print('│ Model                   │ Total Params    │ Trainable Params│')
    print('├─────────────────────────┼─────────────────┼─────────────────┤')
    for display_name, key, pt in rows:
        m = build_model(key, num_classes=num_classes, pretrained=pt)
        t, tr = count_params(m)
        print(f'│ {display_name:<23s} │ {t:>14,}  │ {tr:>14,}  │')
        del m
    print('└─────────────────────────┴─────────────────┴─────────────────┘')
    print('  Note: pretrained=False used here — no download. '
          'Pretrained weights load at training start.')
