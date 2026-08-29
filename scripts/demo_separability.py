#!/usr/bin/env python3
"""
demo_separability.py — Quick Visual Analysis: Can CSI Separate Sitting vs Walking?
===================================================================================

Loads the 6 collected CSV samples (3 sitting, 3 walking), applies light
preprocessing, and generates visual evidence showing whether the two
activities produce distinguishable CSI signatures.

Outputs 4 plots saved to data/analysis/:
  1. Raw amplitude time-series comparison (top-5 subcarriers)
  2. Mean amplitude spectrum per activity
  3. PCA 2D scatter plot (can the clouds be separated?)
  4. Simple classifier accuracy (k-NN leave-one-out)
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSI CSV and return (amplitudes, phases) as 2D arrays.

    Returns
    -------
    amplitudes : np.ndarray of shape (n_packets, n_subcarriers)
    phases     : np.ndarray of shape (n_packets, n_subcarriers)
    """
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader]

    amp_cols = [i for i, h in enumerate(header) if h.startswith("amplitude_")]
    phase_cols = [i for i, h in enumerate(header) if h.startswith("phase_")]

    data = np.array(rows, dtype=np.float64)
    amplitudes = data[:, amp_cols]
    phases = data[:, phase_cols]
    return amplitudes, phases


def butter_lowpass(data: np.ndarray, cutoff: float = 10.0,
                   fs: float = 100.0, order: int = 4) -> np.ndarray:
    """Apply a Butterworth low-pass filter along axis 0 (time)."""
    nyq = 0.5 * fs
    # Clamp cutoff to valid range
    norm_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, norm_cutoff, btype="low")
    return filtfilt(b, a, data, axis=0)


def extract_features(amplitudes: np.ndarray, window_size: int = 100,
                     overlap: float = 0.5) -> np.ndarray:
    """Segment into windows and extract statistical features per window.

    Features per window (per subcarrier):
        mean, std, max, min, range, energy
    Then flatten into a single feature vector per window.
    """
    step = int(window_size * (1 - overlap))
    n_samples, n_sub = amplitudes.shape
    windows = []

    for start in range(0, n_samples - window_size + 1, step):
        w = amplitudes[start:start + window_size]
        feats = np.concatenate([
            w.mean(axis=0),
            w.std(axis=0),
            w.max(axis=0) - w.min(axis=0),    # range
            np.sqrt((w ** 2).mean(axis=0)),    # RMS energy
        ])
        windows.append(feats)

    return np.array(windows) if windows else np.empty((0, n_sub * 4))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CSI Separability Demo: Sitting vs Walking")
    print("=" * 60)

    # Load all samples
    activities = {"sitting_down": [], "walking": []}
    for activity in activities:
        act_dir = RAW_DIR / activity
        if not act_dir.exists():
            print(f"  ❌ No data found for '{activity}' in {act_dir}")
            sys.exit(1)
        csv_files = sorted(act_dir.glob("*.csv"))
        print(f"  📂 {activity}: {len(csv_files)} samples")
        for f in csv_files:
            amp, phase = load_csv(f)
            activities[activity].append(amp)
            print(f"     └─ {f.name}: {amp.shape[0]} packets, {amp.shape[1]} subcarriers")

    n_sub = activities["sitting_down"][0].shape[1]
    print(f"\n  Subcarrier count: {n_sub}")

    # ── Apply low-pass filter to all samples
    print("\n  🔧 Applying Butterworth low-pass filter (cutoff=10Hz)...")
    for activity in activities:
        activities[activity] = [butter_lowpass(a) for a in activities[activity]]

    # ── Pick top-K most variable subcarriers (use all sitting+walking data)
    all_data = np.vstack([a for samples in activities.values() for a in samples])
    variance_per_sub = all_data.var(axis=0)
    top_k = 10
    top_k_idx = np.argsort(variance_per_sub)[-top_k:][::-1]
    print(f"  📊 Top-{top_k} most active subcarrier indices: {top_k_idx.tolist()}")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 1: Raw amplitude time-series comparison
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  📈 Generating Plot 1: Time-Series Comparison...")
    fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig1.suptitle("CSI Amplitude Time-Series: Sitting vs Walking",
                  fontsize=16, fontweight="bold")

    for ax, (activity, color, label) in zip(
        axes1,
        [("sitting_down", "#2196F3", "Sitting Down"),
         ("walking", "#FF5722", "Walking")]
    ):
        sample = activities[activity][0]  # First sample
        t = np.arange(sample.shape[0]) / 100.0  # Assuming 100 Hz
        for i, sub_idx in enumerate(top_k_idx[:5]):
            alpha = 1.0 - i * 0.15
            ax.plot(t, sample[:, sub_idx], alpha=alpha,
                    linewidth=0.8, label=f"SC {sub_idx}")
        ax.set_title(f"{label} — Top 5 Subcarriers", fontsize=12)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (seconds)")
        ax.legend(loc="upper right", fontsize=8, ncol=5)
        ax.grid(alpha=0.2)

    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "01_timeseries_comparison.png", dpi=150)
    print(f"     Saved → {OUT_DIR / '01_timeseries_comparison.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 2: Mean amplitude spectrum per activity
    # ═══════════════════════════════════════════════════════════════════════
    print("  📈 Generating Plot 2: Amplitude Spectrum...")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    fig2.suptitle("Mean Amplitude Spectrum per Activity", fontsize=16, fontweight="bold")

    for activity, color, label in [
        ("sitting_down", "#2196F3", "Sitting Down"),
        ("walking", "#FF5722", "Walking"),
    ]:
        all_amp = np.vstack(activities[activity])
        mean_amp = all_amp.mean(axis=0)
        std_amp = all_amp.std(axis=0)
        x = np.arange(len(mean_amp))
        ax2.plot(x, mean_amp, color=color, linewidth=1.5, label=f"{label} (mean)")
        ax2.fill_between(x, mean_amp - std_amp, mean_amp + std_amp,
                         color=color, alpha=0.15)

    ax2.set_xlabel("Subcarrier Index", fontsize=12)
    ax2.set_ylabel("Mean Amplitude", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "02_amplitude_spectrum.png", dpi=150)
    print(f"     Saved → {OUT_DIR / '02_amplitude_spectrum.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 3: PCA 2D scatter plot
    # ═══════════════════════════════════════════════════════════════════════
    print("  📈 Generating Plot 3: PCA Scatter Plot...")

    # Extract windowed features
    X_list, y_list = [], []
    for activity, label_id in [("sitting_down", 0), ("walking", 1)]:
        for sample_amp in activities[activity]:
            feats = extract_features(sample_amp, window_size=100, overlap=0.5)
            X_list.append(feats)
            y_list.extend([label_id] * feats.shape[0])

    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"     Total windows: {len(y)} (sitting={sum(y==0)}, walking={sum(y==1)})")

    # Standardize + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    fig3.suptitle("PCA Projection: Sitting vs Walking Windows",
                  fontsize=16, fontweight="bold")

    for label_id, color, name, marker in [
        (0, "#2196F3", "Sitting Down", "o"),
        (1, "#FF5722", "Walking", "^"),
    ]:
        mask = y == label_id
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, marker=marker, alpha=0.6, s=40,
                    edgecolors="white", linewidths=0.3, label=name)

    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax3.legend(fontsize=12, markerscale=1.5)
    ax3.grid(alpha=0.2)
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "03_pca_scatter.png", dpi=150)
    print(f"     Saved → {OUT_DIR / '03_pca_scatter.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 4: Classification accuracy (per-sample leave-one-out)
    # ═══════════════════════════════════════════════════════════════════════
    print("  📈 Generating Plot 4: Classification Test...")

    # Per-sample features (mean of all windows in each sample)
    X_sample, y_sample, sample_names = [], [], []
    for activity, label_id in [("sitting_down", 0), ("walking", 1)]:
        for i, sample_amp in enumerate(activities[activity]):
            feats = extract_features(sample_amp, window_size=100, overlap=0.5)
            if feats.shape[0] > 0:
                X_sample.append(feats.mean(axis=0))  # Aggregate per sample
                y_sample.append(label_id)
                sample_names.append(f"{activity}_{i+1}")

    X_sample = np.array(X_sample)
    y_sample = np.array(y_sample)

    # Leave-one-out cross-validation with k-NN
    loo = LeaveOneOut()
    knn = KNeighborsClassifier(n_neighbors=1)  # k=1 since we only have 6 samples
    y_pred = []
    for train_idx, test_idx in loo.split(X_sample):
        scaler_loo = StandardScaler()
        X_train = scaler_loo.fit_transform(X_sample[train_idx])
        X_test = scaler_loo.transform(X_sample[test_idx])
        knn.fit(X_train, y_sample[train_idx])
        y_pred.append(knn.predict(X_test)[0])

    y_pred = np.array(y_pred)
    accuracy = accuracy_score(y_sample, y_pred)

    # Window-level classification
    knn_window = KNeighborsClassifier(n_neighbors=3)
    scaler_w = StandardScaler()
    X_w_scaled = scaler_w.fit_transform(X)
    knn_window.fit(X_w_scaled, y)
    y_w_pred = knn_window.predict(X_w_scaled)
    window_acc = accuracy_score(y, y_w_pred)

    # Results plot
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
    fig4.suptitle("Classification Results", fontsize=16, fontweight="bold")

    # Confusion-style result
    result_labels = ["Correct ✅", "Wrong ❌"]
    correct = int(accuracy * len(y_sample))
    wrong = len(y_sample) - correct
    bars = ax4a.bar(result_labels, [correct, wrong],
                    color=["#4CAF50", "#F44336"], width=0.5, edgecolor="white")
    ax4a.set_title(f"Leave-One-Out (Sample-Level)\nAccuracy: {accuracy:.0%}", fontsize=13)
    ax4a.set_ylabel("Count")
    for bar, val in zip(bars, [correct, wrong]):
        ax4a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                  str(val), ha="center", fontsize=14, fontweight="bold")

    # Per-sample predictions table
    table_data = []
    for name, true, pred in zip(sample_names, y_sample, y_pred):
        true_label = "Sitting" if true == 0 else "Walking"
        pred_label = "Sitting" if pred == 0 else "Walking"
        result = "✅" if true == pred else "❌"
        table_data.append([name, true_label, pred_label, result])

    ax4b.axis("off")
    table = ax4b.table(
        cellText=table_data,
        colLabels=["Sample", "True", "Predicted", "Result"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4b.set_title(f"Window-Level Accuracy: {window_acc:.1%}\n({sum(y==0)} sitting + {sum(y==1)} walking windows)",
                   fontsize=12)

    fig4.tight_layout()
    fig4.savefig(OUT_DIR / "04_classification_results.png", dpi=150)
    print(f"     Saved → {OUT_DIR / '04_classification_results.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  📊 SEPARABILITY ANALYSIS RESULTS")
    print("=" * 60)
    print(f"  Total samples:         {len(y_sample)} (3 sitting + 3 walking)")
    print(f"  Total windows:         {len(y)} ({sum(y==0)} sitting + {sum(y==1)} walking)")
    print(f"  Subcarriers:           {n_sub}")
    print(f"  PCA variance (PC1+2):  {sum(pca.explained_variance_ratio_[:2]):.1%}")
    print(f"  Sample-Level Accuracy: {accuracy:.0%} (LOO cross-validation)")
    print(f"  Window-Level Accuracy: {window_acc:.1%} (k-NN, k=3)")
    print(f"  ")

    if accuracy >= 0.8:
        print(f"  🟢 VERDICT: The two activities are CLEARLY SEPARABLE!")
        print(f"     Even with only 3 samples each, the CSI signatures are")
        print(f"     distinct enough for reliable classification.")
        print(f"     → You can proceed to collect fall data with confidence!")
    elif accuracy >= 0.5:
        print(f"  🟡 VERDICT: Partially separable — more data recommended.")
        print(f"     → Try collecting 5-10 more samples of each activity.")
    else:
        print(f"  🔴 VERDICT: Not clearly separable yet.")
        print(f"     → Check ESP placement and collect more diverse samples.")

    print(f"\n  All plots saved to: {OUT_DIR}")
    print("=" * 60)

    # Show the plots
    plt.show()


if __name__ == "__main__":
    main()
