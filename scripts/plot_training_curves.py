"""Plot Training and Validation Loss and Accuracy curves from training_history.json."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_curves(history_path: str, output_path: str):
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    epochs = range(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, train_loss, label="Training Loss", color="#2563EB", lw=2)
    ax1.plot(epochs, val_loss, label="Validation Loss", color="#DC2626", lw=2, linestyle="--")
    best_epoch = val_loss.index(min(val_loss)) + 1
    best_loss = min(val_loss)
    ax1.axvline(x=best_epoch, color="#059669", linestyle=":", label=f"Best Model (Epoch {best_epoch}: {best_loss:.4f})")
    ax1.set_title("Training and Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_acc, label="Training Accuracy", color="#2563EB", lw=2)
    ax2.plot(epochs, val_acc, label="Validation Accuracy", color="#DC2626", lw=2, linestyle="--")
    val_acc_at_best = val_acc[best_epoch - 1]
    ax2.axvline(x=best_epoch, color="#059669", linestyle=":", label=f"Checkpoint (Epoch {best_epoch}: {val_acc_at_best:.2f}%)")
    ax2.set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("CSI Fall Detection — Training Dynamics (Real Fall Dataset)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {output_path}")

if __name__ == "__main__":
    plot_curves(
        history_path="model/saved/training_history.json",
        output_path="model/saved/evaluation/training_curves.png"
    )
