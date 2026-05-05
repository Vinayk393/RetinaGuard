"""
utils.py
========
RetinaGuard — Shared Utilities: Seed Control, Logging, Output Management.
"""

import os
import json
import random
import logging
import datetime
from pathlib import Path

import numpy as np
import torch


# ════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """
    Set global random seed across Python, NumPy, PyTorch, and CUDA.

    Must be called before any model initialisation, data loading,
    or split generation to guarantee reproducible results.

    Parameters
    ----------
    seed : int   Global seed value (default 42 — matches paper).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"🌱 Seed set to {seed} — fully reproducible run.")


# ════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY MANAGEMENT
# ════════════════════════════════════════════════════════════════

def make_output_dirs(base_dir: str, experiment_name: str) -> dict:
    """
    Create a structured output directory tree for one experiment run.

    Directory structure
    -------------------
    {base_dir}/{experiment_name}/
        checkpoints/    ← best model weights (.pth)
        figures/        ← training curves, confusion matrices, ROC plots
        gradcam/        ← Grad-CAM++ activation maps
        logs/           ← training log file

    Parameters
    ----------
    base_dir        : str   Root output directory (e.g. 'outputs').
    experiment_name : str   Slug identifying this experiment (e.g. 'efficientnet_b0_focal').

    Returns
    -------
    dirs : dict
        Keys: 'root', 'checkpoints', 'figures', 'gradcam', 'logs'.
        Values: absolute path strings.
    """
    slug = experiment_name.lower().replace(' ', '_').replace('/', '_') \
                          .replace('(', '').replace(')', '').replace('-', '_')

    root = Path(base_dir) / slug
    dirs = {
        'root'        : str(root),
        'checkpoints' : str(root / 'checkpoints'),
        'figures'     : str(root / 'figures'),
        'gradcam'     : str(root / 'gradcam'),
        'logs'        : str(root / 'logs'),
    }
    for d in dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)

    return dirs


# ════════════════════════════════════════════════════════════════
# JSON SAVING
# ════════════════════════════════════════════════════════════════

def save_json(data: dict, path: str) -> None:
    """
    Serialise a dict to a JSON file, handling non-serialisable types.

    Converts numpy scalars and arrays to native Python types automatically.

    Parameters
    ----------
    data : dict   Dictionary to save.
    path : str    Full output file path (e.g. 'outputs/focal/metrics.json').
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_convert)

    print(f"  Saved: {path}")


def load_json(path: str) -> dict:
    """
    Load a JSON file from disk.

    Parameters
    ----------
    path : str   Full path to the JSON file.

    Returns
    -------
    dict
    """
    with open(path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════

def setup_logger(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    Set up a dual-output logger (console + file).

    Parameters
    ----------
    log_dir         : str   Directory to write the log file.
    experiment_name : str   Used for the log file name.

    Returns
    -------
    logging.Logger
        Configured logger instance. Use logger.info(), logger.warning(), etc.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    slug      = experiment_name.lower().replace(' ', '_').replace('/', '_') \
                               .replace('(', '').replace(')', '')
    log_path  = os.path.join(log_dir, f'{slug}_{timestamp}.log')

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-run

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger


# ════════════════════════════════════════════════════════════════
# DEVICE DETECTION
# ════════════════════════════════════════════════════════════════

def get_device() -> str:
    """
    Return the best available compute device.

    Returns
    -------
    str   'cuda' if GPU is available, else 'cpu'.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"  Device : {device} — {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  Device : {device} (no GPU found)")
    return device
