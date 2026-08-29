"""
CSI Fall Detection Dataset Module.

Provides PyTorch Dataset and DataLoader utilities for loading CSI amplitude
data (post-PCA) with data augmentation for fall detection training.

Input shape: (batch, time_steps=100, features=20)
Labels: 0 = fall, 1 = non-fall (binary classification)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class CSIFallDataset(Dataset):
    """PyTorch Dataset for CSI-based fall detection.

    Loads pre-processed CSI amplitude matrices (X.npy) and labels (y.npy)
    from a data directory. Supports optional data augmentation for training.

    Args:
        X: CSI amplitude data of shape (num_samples, time_steps, features).
        y: Binary labels of shape (num_samples,). 0=fall, 1=non-fall.
        augment: Whether to apply data augmentation. Default False.
        augment_config: Dictionary of augmentation hyperparameters.
    """

    # Default augmentation hyperparameters
    DEFAULT_AUGMENT_CONFIG: Dict[str, Any] = {
        "time_shift_max": 10,
        "noise_std": 0.02,
        "amplitude_scale_range": (0.8, 1.2),
        "augment_prob": 0.5,
    }

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        augment_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        assert X.shape[0] == y.shape[0], (
            f"Sample count mismatch: X has {X.shape[0]}, y has {y.shape[0]}"
        )

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.augment_config = {
            **self.DEFAULT_AUGMENT_CONFIG,
            **(augment_config or {}),
        }

        logger.info(
            "Dataset created: %d samples, shape %s, augment=%s",
            len(self.y),
            tuple(self.X.shape),
            self.augment,
        )

        # Log class distribution
        unique, counts = torch.unique(self.y, return_counts=True)
        for cls, cnt in zip(unique.tolist(), counts.tolist()):
            logger.info("  Class %d: %d samples (%.1f%%)", cls, cnt, 100.0 * cnt / len(self.y))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single (sample, label) pair, optionally augmented.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (sample tensor [time_steps, features], label tensor []).
        """
        x = self.X[idx].clone()  # (time_steps, features)
        y = self.y[idx]

        if self.augment and torch.rand(1).item() < self.augment_config["augment_prob"]:
            x = self._apply_augmentation(x)

        return x, y

    # ------------------------------------------------------------------
    # Data augmentation methods
    # ------------------------------------------------------------------

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a random subset of augmentations to a single sample.

        Augmentations are applied sequentially, each with 50% probability:
          1. Time shift (circular roll along time axis)
          2. Gaussian noise injection
          3. Amplitude scaling

        Args:
            x: Input tensor of shape (time_steps, features).

        Returns:
            Augmented tensor of the same shape.
        """
        if torch.rand(1).item() < 0.5:
            x = self._time_shift(x)
        if torch.rand(1).item() < 0.5:
            x = self._noise_injection(x)
        if torch.rand(1).item() < 0.5:
            x = self._amplitude_scaling(x)
        return x

    def _time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Circular time shift along the temporal axis.

        Simulates small temporal misalignments in CSI capture windows.

        Args:
            x: Input tensor of shape (time_steps, features).

        Returns:
            Time-shifted tensor.
        """
        max_shift = self.augment_config["time_shift_max"]
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=int(shift), dims=0)

    def _noise_injection(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to simulate sensor noise.

        Args:
            x: Input tensor of shape (time_steps, features).

        Returns:
            Noisy tensor.
        """
        noise_std = self.augment_config["noise_std"]
        noise = torch.randn_like(x) * noise_std
        return x + noise

    def _amplitude_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Random amplitude scaling to simulate signal strength variation.

        Args:
            x: Input tensor of shape (time_steps, features).

        Returns:
            Scaled tensor.
        """
        low, high = self.augment_config["amplitude_scale_range"]
        scale = torch.empty(1).uniform_(low, high).item()
        return x * scale

    @classmethod
    def from_directory(
        cls,
        data_dir: str,
        augment: bool = False,
        augment_config: Optional[Dict[str, Any]] = None,
    ) -> "CSIFallDataset":
        """Create a dataset by loading X.npy and y.npy from a directory.

        Args:
            data_dir: Path to directory containing X.npy and y.npy.
            augment: Whether to enable data augmentation.
            augment_config: Optional augmentation parameters.

        Returns:
            CSIFallDataset instance.

        Raises:
            FileNotFoundError: If X.npy or y.npy is missing.
        """
        data_path = Path(data_dir)
        x_path = data_path / "X.npy"
        y_path = data_path / "y.npy"

        if not x_path.exists():
            raise FileNotFoundError(f"Data file not found: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Label file not found: {y_path}")

        logger.info("Loading data from %s", data_path)
        X = np.load(str(x_path))
        y = np.load(str(y_path))

        logger.info("Loaded X: %s, y: %s", X.shape, y.shape)
        return cls(X, y, augment=augment, augment_config=augment_config)


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets.

    Args:
        y: Label array of shape (num_samples,).

    Returns:
        Tensor of class weights, one per class.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = total / (len(classes) * counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info("Class weights: %s", dict(zip(classes.tolist(), weight_tensor.tolist())))
    return weight_tensor


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    augment: bool = True,
    augment_config: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Create stratified train/validation/test DataLoaders.

    Performs a two-stage stratified split:
      1. Split off test set.
      2. Split remaining into train and validation sets.

    Training DataLoader uses weighted random sampling for class balance.

    Args:
        data_dir: Path to directory containing X.npy and y.npy.
        batch_size: Batch size for all loaders.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        augment: Whether to augment training data.
        augment_config: Optional augmentation parameters.
        num_workers: DataLoader worker count.
        random_state: Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights).
    """
    data_path = Path(data_dir)
    X = np.load(str(data_path / "X.npy"))
    y = np.load(str(data_path / "y.npy"))

    logger.info("Total samples: %d, shape: %s", len(y), X.shape)

    # --- Stage 1: Split off test set ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )

    # --- Stage 2: Split remaining into train / val ---
    val_fraction = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=random_state,
    )

    logger.info("Split sizes — Train: %d, Val: %d, Test: %d", len(y_train), len(y_val), len(y_test))

    # --- Build datasets ---
    train_dataset = CSIFallDataset(X_train, y_train, augment=augment, augment_config=augment_config)
    val_dataset = CSIFallDataset(X_val, y_val, augment=False)
    test_dataset = CSIFallDataset(X_test, y_test, augment=False)

    # --- Weighted random sampler for training (handles class imbalance) ---
    class_weights = compute_class_weights(y_train)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights


# ------------------------------------------------------------------
# Standalone usage
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="CSI Fall Detection Dataset Utility")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory with X.npy / y.npy")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, weights = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # Verify a batch
    for batch_x, batch_y in train_loader:
        logger.info("Sample batch — X: %s, y: %s", batch_x.shape, batch_y.shape)
        logger.info("Label distribution in batch: %s", torch.bincount(batch_y).tolist())
        break
