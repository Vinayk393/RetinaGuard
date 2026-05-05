"""
preprocess.py
=============
RetinaGuard — Preprocessing, Augmentation, Dataset, DataLoaders.

Extracted directly from the RetinaGuard Phase 4 Kaggle notebook.

Classes
-------
RetinopathyDataset
    PyTorch Dataset for DR fundus images.
    Applies CLAHE contrast enhancement + configurable torchvision transforms.

Functions
---------
apply_clahe(img_np)
    Apply CLAHE on the L-channel of LAB colourspace.

get_transforms()
    Return (train_transform, val_transform, tta_transform).

compute_class_weights(df_train, num_classes, device)
    Return inverse-frequency class weights as a torch.FloatTensor.

make_dataloaders(df_train, df_val, df_test, cfg)
    Build and return (train_loader, val_loader, test_loader,
                      train_ds, val_ds, test_ds).
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


# ── ImageNet statistics (for pretrained backbone normalisation)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════════
# CLAHE
# ════════════════════════════════════════════════════════════════

def apply_clahe(img_np: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE)
    on the Luminance (L) channel of the LAB colourspace.

    Parameters
    ----------
    img_np : np.ndarray
        RGB image as uint8 array, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        CLAHE-enhanced RGB image, same shape and dtype as input.

    Notes
    -----
    clip_limit=2.0 and tileGridSize=(8, 8) were selected based on
    empirical performance in DR fundus image preprocessing
    (Chakour et al., 2025; Aurangzeb et al., 2023).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab   = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ════════════════════════════════════════════════════════════════
# TRANSFORMS
# ════════════════════════════════════════════════════════════════

def get_transforms():
    """
    Return the three transform pipelines used in training.

    Returns
    -------
    train_transform : torchvision.transforms.Compose
        Stochastic augmentation pipeline for training images.
    val_transform : torchvision.transforms.Compose
        Deterministic normalisation only — for validation and test.
    tta_transform : torchvision.transforms.Compose
        Lightweight stochastic pipeline for Test-Time Augmentation.

    Augmentation pipeline (training)
    ---------------------------------
    - RandomHorizontalFlip  (p=0.5)
    - RandomVerticalFlip    (p=0.3)
    - RandomRotation        (±20°)
    - ColorJitter           (brightness ±0.2, contrast ±0.2, saturation ±0.1)
    - RandomAffine          (translate ±5%, scale 95–105%)
    - ToTensor + ImageNet Normalize
    """
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    tta_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    return train_transform, val_transform, tta_transform


# ════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════

class RetinopathyDataset(Dataset):
    """
    PyTorch Dataset for the DR Gaussian-Filtered fundus image dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain columns: 'filepath' (str) and 'label' (int 0–4).
    transform : torchvision.transforms.Compose, optional
        Transform pipeline applied after CLAHE.
    use_clahe : bool
        Whether to apply CLAHE contrast enhancement (default: True).
    img_size : int
        Target image size in pixels (default: 224).

    Returns (per __getitem__)
    -------------------------
    img   : torch.Tensor  shape (3, img_size, img_size)
    label : torch.Tensor  scalar int64
    """

    def __init__(self, dataframe, transform=None, use_clahe: bool = True, img_size: int = 224):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform
        self.use_clahe = use_clahe
        self.img_size  = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load → BGR→RGB → resize
        img = cv2.imread(row['filepath'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Optional CLAHE
        if self.use_clahe:
            img = apply_clahe(img)

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img, label


# ════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ════════════════════════════════════════════════════════════════

def compute_class_weights(df_train, num_classes: int, device: str) -> torch.FloatTensor:
    """
    Compute inverse-frequency class weights, normalised so weights sum to num_classes.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split DataFrame containing a 'label' column.
    num_classes : int
        Number of severity classes (5 for DR grading).
    device : str
        Target device, e.g. 'cuda' or 'cpu'.

    Returns
    -------
    torch.FloatTensor
        Shape (num_classes,), moved to the specified device.

    Example weights for DR dataset (n=2563 train images)
    ------------------------------------------------------
    No DR          | w=0.22
    Mild           | w=1.05
    Moderate       | w=0.39
    Severe         | w=2.02   ← highest weight (fewest samples)
    Proliferative  | w=1.32
    """
    counts = np.array([len(df_train[df_train['label'] == i]) for i in range(num_classes)],
                      dtype=np.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalise to sum = num_classes
    return torch.FloatTensor(weights).to(device)


# ════════════════════════════════════════════════════════════════
# DATALOADERS
# ════════════════════════════════════════════════════════════════

def make_dataloaders(df_train, df_val, df_test, cfg: dict):
    """
    Construct PyTorch DataLoaders for train, validation, and test splits.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Split DataFrames each containing 'filepath' and 'label' columns.
    cfg : dict
        Configuration dict. Required keys:
            img_size   (int)  : target image size (default 224)
            batch_size (int)  : batch size (default 32)
            use_clahe  (bool) : whether to apply CLAHE
            num_workers (int) : DataLoader worker processes (default 2)

    Returns
    -------
    train_loader, val_loader, test_loader : torch.utils.data.DataLoader
    train_ds, val_ds, test_ds             : RetinopathyDataset
    """
    train_tf, val_tf, _ = get_transforms()

    img_size    = cfg.get('img_size',    224)
    batch_size  = cfg.get('batch_size',   32)
    use_clahe   = cfg.get('use_clahe',  True)
    num_workers = cfg.get('num_workers',   2)

    train_ds = RetinopathyDataset(df_train, transform=train_tf, use_clahe=use_clahe, img_size=img_size)
    val_ds   = RetinopathyDataset(df_val,   transform=val_tf,   use_clahe=use_clahe, img_size=img_size)
    test_ds  = RetinopathyDataset(df_test,  transform=val_tf,   use_clahe=use_clahe, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
