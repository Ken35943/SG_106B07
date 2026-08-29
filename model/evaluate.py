"""
CSI Fall Detection Model Evaluation Script.

Loads a trained model checkpoint and evaluates it on the test set, producing:
  - Classification report (precision, recall, F1)
  - Confusion matrix heatmap
  - ROC curve with AUC
  - Precision-Recall curve

All plots are saved to model/saved/evaluation/.

Usage:
    python evaluate.py --model_path saved/best_model.pth --data_dir ../data/processed
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from dataset import create_dataloaders
from cnn_lstm import CSIFallDetector

logger = logging.getLogger(__name__)

# Class label mapping
CLASS_NAMES = ["Fall (0)", "Non-Fall (1)"]


# ------------------------------------------------------------------
# Evaluation core
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on the test set and collect predictions.

    Args:
        model: Trained model in eval mode.
        loader: Test DataLoader.
        device: Computation device.

    Returns:
        Tuple of (true_labels, predicted_labels, predicted_probabilities).
        - true_labels: shape (N,)
        - predicted_labels: shape (N,)
        - predicted_probs: shape (N, num_classes)
    """
    model.eval()

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    progress = tqdm(loader, desc="Evaluating", unit="batch")

    for batch_x, batch_y in progress:
        batch_x = batch_x.to(device, non_blocking=True)
        logits = model(batch_x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_labels.append(batch_y.numpy())
        all_preds.append(preds)
        all_probs.append(probs)

    true_labels = np.concatenate(all_labels)
    predicted_labels = np.concatenate(all_preds)
    predicted_probs = np.concatenate(all_probs)

    return true_labels, predicted_labels, predicted_probs


# ------------------------------------------------------------------
# Plotting utilities
# ------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    class_names: List[str] = CLASS_NAMES,
) -> None:
    """Generate and save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_path: File path to save the plot.
        class_names: Display names for each class.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — CSI Fall Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Path,
) -> float:
    """Generate and save a ROC curve with AUC score.

    Uses the probability of the positive class (non-fall=1) or
    equivalently 1 minus the fall probability.

    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities of shape (N, num_classes).
        save_path: File path to save the plot.

    Returns:
        AUC score.
    """
    # For binary classification, use probability of class 1 (non-fall)
    y_score = y_probs[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr, tpr,
        color="#2563EB",
        lw=2,
        label=f"ROC Curve (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563EB")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — CSI Fall Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to %s (AUC: %.4f)", save_path, roc_auc)

    return roc_auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Path,
) -> float:
    """Generate and save a Precision-Recall curve.

    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities of shape (N, num_classes).
        save_path: File path to save the plot.

    Returns:
        Average precision (area under PR curve).
    """
    y_score = y_probs[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall, precision,
        color="#059669",
        lw=2,
        label=f"PR Curve (AUC = {pr_auc:.4f})",
    )
    ax.fill_between(recall, precision, alpha=0.1, color="#059669")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — CSI Fall Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved to %s (AUC: %.4f)", save_path, pr_auc)

    return pr_auc


# ------------------------------------------------------------------
# Main evaluation pipeline
# ------------------------------------------------------------------

def run_evaluation(
    model_path: str,
    data_dir: str,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Full evaluation pipeline: load model, predict, generate metrics and plots.

    Args:
        model_path: Path to the saved model checkpoint (.pth).
        data_dir: Path to the data directory containing X.npy and y.npy.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary of evaluation metrics.
    """
    # --- Resolve paths ---
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

    script_dir = Path(__file__).resolve().parent
    eval_dir = script_dir / "saved" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- Load checkpoint ---
    logger.info("Loading checkpoint from %s", model_file)
    checkpoint = torch.load(str(model_file), map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config", {})
    input_features = model_config.get("input_features", 20)
    num_classes = model_config.get("num_classes", 2)
    lstm_hidden = model_config.get("lstm_hidden", 128)
    lstm_layers = model_config.get("lstm_layers", 2)
    lstm_dropout = model_config.get("lstm_dropout", 0.3)
    attention_heads = model_config.get("attention_heads", 4)

    logger.info("Model config: %s", model_config)

    # --- Build and load model ---
    model = CSIFallDetector(
        input_features=input_features,
        num_classes=num_classes,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        attention_heads=attention_heads,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        "Model loaded from epoch %d (val_loss: %.6f, val_acc: %.2f%%)",
        checkpoint.get("epoch", -1),
        checkpoint.get("val_loss", float("nan")),
        checkpoint.get("val_acc", float("nan")),
    )

    # --- Data (use test split only) ---
    logger.info("Loading test data from %s", data_dir)
    _, _, test_loader, _ = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=False,
    )

    # --- Run inference ---
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    logger.info("Evaluated %d test samples", len(y_true))

    # --- Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    report_str = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    report_dict = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
    )

    logger.info("\n%s", "=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)
    logger.info("\n%s", report_str)
    logger.info("Overall Accuracy: %.4f", accuracy)
    logger.info("Weighted F1:      %.4f", f1)

    # --- Generate plots ---
    plot_confusion_matrix(y_true, y_pred, eval_dir / "confusion_matrix.png")
    roc_auc_val = plot_roc_curve(y_true, y_probs, eval_dir / "roc_curve.png")
    pr_auc_val = plot_precision_recall_curve(y_true, y_probs, eval_dir / "pr_curve.png")

    # --- Compile results ---
    results: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1),
        "roc_auc": float(roc_auc_val),
        "pr_auc": float(pr_auc_val),
        "classification_report": report_dict,
        "checkpoint_epoch": checkpoint.get("epoch", -1),
        "checkpoint_val_loss": checkpoint.get("val_loss", None),
        "checkpoint_val_acc": checkpoint.get("val_acc", None),
        "num_test_samples": int(len(y_true)),
    }

    # Save results JSON
    results_path = eval_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Evaluation results saved to %s", results_path)

    return results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate CSI Fall Detection Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved/best_model.pth",
        help="Path to the saved model checkpoint (.pth)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing X.npy and y.npy",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
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

    results = run_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    logger.info("─" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("  Accuracy:   %.4f", results["accuracy"])
    logger.info("  F1 (wt):    %.4f", results["f1_weighted"])
    logger.info("  ROC AUC:    %.4f", results["roc_auc"])
    logger.info("  PR AUC:     %.4f", results["pr_auc"])
    logger.info("─" * 60)
