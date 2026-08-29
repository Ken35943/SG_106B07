"""
CSI Fall Detection Training Script.

Trains the CSIFallDetector model on pre-processed CSI amplitude data.
Supports GPU acceleration, early stopping, learning rate scheduling,
checkpoint saving, and training history export.

Usage:
    python train.py --config ../config.yaml --epochs 100 --batch_size 32 --lr 0.001
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from dataset import compute_class_weights, create_dataloaders
from cnn_lstm import CSIFallDetector, get_model_summary

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------------

class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        verbose: Whether to log early stopping events.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter: int = 0
        self.best_loss: Optional[float] = None
        self.should_stop: bool = False

    def __call__(self, val_loss: float) -> bool:
        """Check whether training should stop.

        Args:
            val_loss: Current epoch's validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "EarlyStopping: %d / %d (best val_loss: %.6f)",
                    self.counter,
                    self.patience,
                    self.best_loss,
                )
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train the model for a single epoch.

    Args:
        model: The neural network model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Computation device.
        epoch: Current epoch number (for display).

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False, unit="batch")

    for batch_x, batch_y in progress:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.1f}%")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Args:
        model: The neural network model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Computation device.
        epoch: Current epoch number (for display).

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Val   Epoch {epoch}", leave=False, unit="batch")

    for batch_x, batch_y in progress:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        running_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning("Config file not found: %s — using defaults", config_path)
        return {}

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    logger.info("Loaded config from %s", config_path)
    return config


def train(
    config: Dict[str, Any],
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    data_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Full training pipeline.

    Args:
        config: Configuration dictionary (from config.yaml).
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate for Adam optimizer.
        data_dir: Path to data directory. Falls back to config or default.

    Returns:
        Training history dictionary.
    """
    # --- Resolve data directory ---
    if data_dir is None:
        data_dir = config.get("data", {}).get("processed_dir", "data/processed")
    logger.info("Data directory: %s", data_dir)

    # --- Resolve output directory ---
    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir / "saved"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # --- Model hyperparameters from config ---
    model_cfg = config.get("model", {})
    input_features = model_cfg.get("input_features", 20)
    num_classes = model_cfg.get("num_classes", 2)
    lstm_hidden = model_cfg.get("lstm_hidden", 128)
    lstm_layers = model_cfg.get("lstm_layers", 2)
    lstm_dropout = model_cfg.get("lstm_dropout", 0.3)
    attention_heads = model_cfg.get("attention_heads", 4)

    # --- Training hyperparameters from config (CLI args override) ---
    train_cfg = config.get("training", {})
    patience = train_cfg.get("early_stopping_patience", 15)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    scheduler_patience = train_cfg.get("scheduler_patience", 7)
    scheduler_factor = train_cfg.get("scheduler_factor", 0.5)

    # --- Data ---
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=True,
        num_workers=config.get("training", {}).get("num_workers", 0),
    )

    # --- Model ---
    model = CSIFallDetector(
        input_features=input_features,
        num_classes=num_classes,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        attention_heads=attention_heads,
    ).to(device)

    logger.info("\n%s", get_model_summary(model, input_features=input_features))

    # --- Loss, Optimizer, Scheduler ---
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
    )

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # --- Training history ---
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")

    logger.info("Starting training for up to %d epochs...", epochs)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("─" * 60)
        logger.info("Epoch %d/%d  (lr=%.2e)", epoch, epochs, current_lr)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # LR scheduling
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        logger.info(
            "  Train — loss: %.4f, acc: %.2f%%  |  Val — loss: %.4f, acc: %.2f%%",
            train_loss, train_acc, val_loss, val_acc,
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": config,
                "model_config": {
                    "input_features": input_features,
                    "num_classes": num_classes,
                    "lstm_hidden": lstm_hidden,
                    "lstm_layers": lstm_layers,
                    "lstm_dropout": lstm_dropout,
                    "attention_heads": attention_heads,
                },
            }
            checkpoint_path = save_dir / "best_model.pth"
            torch.save(checkpoint, str(checkpoint_path))
            logger.info("  ★ Best model saved (val_loss: %.6f)", val_loss)

        # Early stopping check
        if early_stopping(val_loss):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1f seconds (%d epochs)", elapsed, epoch)

    # --- Save training history ---
    history_path = save_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    return history


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CSI Fall Detection Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory (overrides config)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    args = parse_args()

    # Load YAML config
    config = load_config(args.config)

    # Override config with CLI args where provided
    logger.info("CLI args: epochs=%d, batch_size=%d, lr=%.2e", args.epochs, args.batch_size, args.lr)

    history = train(
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
    )

    logger.info("Final training loss:   %.4f", history["train_loss"][-1])
    logger.info("Final validation loss: %.4f", history["val_loss"][-1])
    logger.info("Best validation loss:  %.4f", min(history["val_loss"]))
