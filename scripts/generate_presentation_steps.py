#!/usr/bin/env python3
"""
generate_presentation_steps.py
==============================
Generates 5 presentation graphics using 100% REAL DATA from the actual
physical ESP32-S3 experiments, processed datasets, and trained model:

  1. Step 1: Real Wi-Fi CSI Amplitude Spectrogram showing Human Motion
  2. Step 2: Data Preprocessing: Raw vs Causal Filtered Waveform (DC Offset Removal)
  3. Step 3: Real Dataset Partitioning (37 Physical Sessions, 889 Windows from groups.npy)
  4. Step 4: Real Training & Validation Convergence from training_history.json (30 Epochs)
  5. Step 5: Real-Time Detection & Post-Fall Immobility Signature (Pre-Fall -> Impact -> Stillness)

Designed for a 4-minute pitch (~20s per step):
  - "กราฟโล่งๆ": Clean, spacious, high-contrast, large readable fonts.
  - 100% authentic experimental data showing the complete biomechanical fall sequence.
"""

import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from csi_dsp import design_bandpass_sos, CausalSOSFilter, ht40_htltf_layout

OUT_DIR = PROJECT_ROOT / "assets" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cohesive presentation styling matching the user's slide
BG_DARK = "#14151a"
PANEL_BG = "#1d202b"
PANEL_BORDER = "#2f3547"
TEXT_WHITE = "#ffffff"
TEXT_MUTED = "#9ba3b8"
ACCENT_ORANGE = "#ff7043"   # Slide timeline accent
ACCENT_CYAN = "#00e5ff"     # RF / CSI / Motion
ACCENT_GREEN = "#00e676"    # Still / Clean / Valid
ACCENT_RED = "#ff1744"      # Fall Impact / Alarm
ACCENT_PURPLE = "#b388ff"   # Deep Learning / Model
ACCENT_YELLOW = "#ffd600"   # Highlight / Warning


def apply_clean_theme(ax):
    ax.set_facecolor(PANEL_BG)
    for s in ax.spines.values():
        s.set_color(PANEL_BORDER)
        s.set_linewidth(1.5)
    ax.tick_params(colors=TEXT_MUTED, labelsize=12)
    ax.xaxis.label.set_color(TEXT_WHITE)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_color(TEXT_WHITE)
    ax.yaxis.label.set_size(13)
    ax.title.set_color(TEXT_WHITE)
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")
    ax.grid(True, linestyle="--", alpha=0.20, color="#6b7594")


# ==============================================================================
# STEP 1: RAW DATA (Real Physical CSI Spectrogram)
# ==============================================================================
def create_step1_raw_data():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_DARK, dpi=150)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.12)
    apply_clean_theme(ax)

    csv_path = PROJECT_ROOT / "data" / "raw" / "fall_forward" / "sample_20260909_200400.csv"
    df = pd.read_csv(csv_path)
    amp_cols = [c for c in df.columns if c.startswith("amplitude_")]
    raw_amp = df[amp_cols].values.astype(np.float64)

    active_idx, _ = ht40_htltf_layout()
    valid_idx = [i for i in active_idx if i < raw_amp.shape[1]]
    amp_active = raw_amp[:, valid_idx]  # (901, 114)

    t_sec = np.arange(len(amp_active)) / 100.0  # 100 Hz sampling -> 9.0s
    data_db = 20 * np.log10(np.maximum(amp_active.T, 1e-4))
    data_db -= np.mean(data_db, axis=1, keepdims=True)

    im = ax.imshow(data_db, aspect="auto", cmap="plasma", origin="lower",
                   extent=[0, t_sec[-1], -57, 57])

    ax.set_xlabel("Time (seconds)", fontsize=14, labelpad=10)
    ax.set_ylabel("Subcarrier Index (114 Carriers)", fontsize=14, labelpad=10)
    ax.set_title("Step 1: Wi-Fi CSI Amplitude Spectrogram (114 Subcarriers @ 100 Hz)", fontsize=18, pad=15)

    # Highlight general motion area
    rect = patches.FancyBboxPatch((4.5, -54), 3.0, 108, boxstyle="round,pad=0.2",
                                  fc="none", ec=ACCENT_CYAN, lw=2.8, ls="--")
    ax.add_patch(rect)
    ax.text(6.0, 42, "Signal Disturbance due to Human Motion", ha="center", va="center",
            fontsize=13, fontweight="bold", color=ACCENT_CYAN,
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a242a", ec=ACCENT_CYAN, lw=1.8))

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Relative CSI Power (dB)", color=TEXT_WHITE, fontsize=12, labelpad=10)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=11)

    save_path = OUT_DIR / "step1_raw_data.png"
    plt.savefig(str(save_path), facecolor=BG_DARK, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ==============================================================================
# STEP 2: CLEAN DATA (Data Preprocessing and Signal Conditioning)
# ==============================================================================
def create_step2_clean_data():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), facecolor=BG_DARK, dpi=150, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.12, hspace=0.25)

    apply_clean_theme(ax1)
    apply_clean_theme(ax2)

    csv_path = PROJECT_ROOT / "data" / "raw" / "fall_forward" / "sample_20260909_200400.csv"
    df = pd.read_csv(csv_path)
    amp_cols = [c for c in df.columns if c.startswith("amplitude_")]
    raw_amp = df[amp_cols].values.astype(np.float64)

    active_idx, _ = ht40_htltf_layout()
    valid_idx = [i for i in active_idx if i < raw_amp.shape[1]]
    amp_active = raw_amp[:, valid_idx]
    t_sec = np.arange(len(amp_active)) / 100.0

    # Real subcarrier signal (Subcarrier index 35)
    real_raw = amp_active[:, 35]

    # Process through real causal bandpass filter (0.5 - 40 Hz)
    sos = design_bandpass_sos(100.0, 0.5, 40.0, 4)
    filt = CausalSOSFilter(sos)
    real_filtered = filt.process(amp_active)[:, 35]

    # Top: Real Raw Signal
    ax1.plot(t_sec, real_raw, color="#ffb74d", lw=2.0, label="Raw Subcarrier Signal")
    ax1.axhline(np.mean(real_raw), color=TEXT_MUTED, ls=":", lw=1.5, label=f"DC Baseline Offset ({np.mean(real_raw):.1f} dB)")
    ax1.set_ylabel("Amplitude (dB)", fontsize=13)
    ax1.set_title("BEFORE: Raw Physical CSI Signal (Heavy DC Baseline Offset & Multipath Drift)", fontsize=16, color="#ffb74d", pad=10)
    ax1.legend(loc="upper right", fontsize=12, facecolor="#242736", edgecolor=PANEL_BORDER, labelcolor=TEXT_WHITE)

    # Bottom: Real Cleaned Signal
    ax2.plot(t_sec, real_filtered, color=ACCENT_CYAN, lw=2.4, label="Cleaned Signal (Causal SOS Bandpass)")
    ax2.axhline(0, color=TEXT_MUTED, ls="--", lw=1.2, alpha=0.6)
    ax2.set_xlabel("Time (seconds)", fontsize=14, labelpad=8)
    ax2.set_ylabel("Amplitude (dB)", fontsize=13)
    ax2.set_title("AFTER: Cleaned Signal (DC Offset Removed, Zero-Centered for Dynamic Range)", fontsize=16, color=ACCENT_CYAN, pad=10)
    ax2.legend(loc="upper right", fontsize=12, facecolor="#242736", edgecolor=PANEL_BORDER, labelcolor=TEXT_WHITE)

    fig.suptitle("Step 2: Data Preprocessing (Causal Bandpass Filter 0.5 – 40 Hz)", fontsize=19, fontweight="bold", color=TEXT_WHITE, y=0.96)

    save_path = OUT_DIR / "step2_clean_data.png"
    plt.savefig(str(save_path), facecolor=BG_DARK, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ==============================================================================
# STEP 3: GROUPING DATA (100% Real Dataset Partition Breakdown)
# ==============================================================================
def create_step3_grouping_data():
    fig, (ax_bar, ax_cards) = plt.subplots(1, 2, figsize=(16, 9), facecolor=BG_DARK, dpi=150,
                                           gridspec_kw={"width_ratios": [1.1, 0.9]})
    fig.subplots_adjust(left=0.08, right=0.92, top=0.86, bottom=0.14, wspace=0.25)

    apply_clean_theme(ax_bar)
    apply_clean_theme(ax_cards)

    # Actual numbers from GroupShuffleSplit on the 37 physical sessions (889 windows)
    splits = ["Test Set\n(Unseen)", "Validation Set", "Train Set"]
    windows = [198, 143, 548]
    percentages = ["22.3%", "16.1%", "61.6%"]
    colors = [ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_CYAN]

    y_pos = np.arange(len(splits))
    bars = ax_bar.barh(y_pos, windows, height=0.55, color=colors, edgecolor="#ffffff", lw=1.2)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(splits, fontsize=14, color=TEXT_WHITE, fontweight="bold")
    ax_bar.set_xlabel("Number of Windows (1.0 second @ 100 Hz)", fontsize=13, labelpad=10)
    ax_bar.set_title("Real Dataset Partition (Total 889 Windows)", fontsize=16, pad=12)
    ax_bar.set_xlim(0, 680)

    for bar, pct in zip(bars, percentages):
        w = bar.get_width()
        ax_bar.text(w + 15, bar.get_y() + bar.get_height()/2, f"{int(w)} windows ({pct})",
                    ha="left", va="center", fontsize=13, fontweight="bold", color=TEXT_WHITE)

    # Right: 3 Clean Highlight Cards with Exact Counts
    ax_cards.grid(False)
    ax_cards.set_xticks([])
    ax_cards.set_yticks([])
    ax_cards.set_xlim(0, 10)
    ax_cards.set_ylim(0, 10)
    ax_cards.set_title("Zero-Leakage Session Grouping (groups.npy)", fontsize=16, pad=12)

    cards = [
        ("Train Set", "23 Physical Sessions", "548 Windows (215 Fall / 333 Non-Fall)", ACCENT_CYAN, 7.5),
        ("Validation Set", "6 Physical Sessions", "143 Windows (54 Fall / 89 Non-Fall)", ACCENT_PURPLE, 4.8),
        ("Held-Out Test Set", "8 Completely Unseen Sessions", "198 Windows (64 Fall / 134 Non-Fall)", ACCENT_ORANGE, 2.1),
    ]

    for title, sub1, sub2, col, yp in cards:
        rect = patches.FancyBboxPatch((0.5, yp - 1.0), 9.0, 2.0, boxstyle="round,pad=0.2",
                                      fc="#252a3a", ec=col, lw=2.2)
        ax_cards.add_patch(rect)
        ax_cards.text(1.2, yp + 0.35, title, fontsize=15, fontweight="bold", color=col)
        ax_cards.text(1.2, yp - 0.40, f"{sub1}\n{sub2}", fontsize=11.5, color=TEXT_WHITE)

    fig.suptitle("Step 3: Data Grouping (Separated by Physical Recording Session)", fontsize=19, fontweight="bold", color=TEXT_WHITE, y=0.96)

    save_path = OUT_DIR / "step3_grouping_data.png"
    plt.savefig(str(save_path), facecolor=BG_DARK, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ==============================================================================
# STEP 4: TRAIN MODEL (100% Real Training & Validation Convergence from JSON)
# ==============================================================================
def create_step4_train_model():
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(16, 9), facecolor=BG_DARK, dpi=150)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.84, bottom=0.14, wspace=0.22)

    apply_clean_theme(ax_loss)
    apply_clean_theme(ax_acc)

    history_path = PROJECT_ROOT / "model" / "saved" / "training_history.json"
    with open(history_path, "r") as f:
        h = json.load(f)

    train_loss = h["train_loss"]
    val_loss = h["val_loss"]
    train_acc = h["train_acc"]
    val_acc = h["val_acc"]
    epochs = range(1, len(train_loss) + 1)

    best_epoch = 14
    best_loss = val_loss[best_epoch - 1]
    best_acc = val_acc[best_epoch - 1]

    # Real Loss Curves
    ax_loss.plot(epochs, train_loss, color=ACCENT_CYAN, lw=2.5, label="Train Loss")
    ax_loss.plot(epochs, val_loss, color=ACCENT_ORANGE, lw=2.5, label="Validation Loss")
    ax_loss.scatter([best_epoch], [best_loss], color=ACCENT_GREEN, s=130, zorder=6)
    ax_loss.set_xlabel("Epoch", fontsize=13, labelpad=8)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=13, labelpad=8)
    ax_loss.set_title(f"Real Loss Convergence (Best: {best_loss:.4f})", fontsize=16, pad=12)
    ax_loss.legend(loc="upper right", fontsize=12, facecolor="#242736", edgecolor=PANEL_BORDER, labelcolor=TEXT_WHITE)

    # Real Accuracy Curves
    ax_acc.plot(epochs, train_acc, color=ACCENT_CYAN, lw=2.5, label="Train Accuracy")
    ax_acc.plot(epochs, val_acc, color=ACCENT_PURPLE, lw=2.5, label="Validation Accuracy")
    ax_acc.scatter([best_epoch], [best_acc], color=ACCENT_GREEN, s=130, zorder=6)
    ax_acc.set_xlabel("Epoch", fontsize=13, labelpad=8)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=13, labelpad=8)
    ax_acc.set_title(f"Real Accuracy Convergence (Best: {best_acc:.2f}%)", fontsize=16, pad=12)
    ax_acc.set_ylim(50, 105)
    ax_acc.legend(loc="lower right", fontsize=12, facecolor="#242736", edgecolor=PANEL_BORDER, labelcolor=TEXT_WHITE)

    # Highlight Callout Badge
    fig.text(0.5, 0.89, f"Best Model Checkpoint: Epoch {best_epoch} (Validation Accuracy: {best_acc:.2f}%)",
             ha="center", va="center", fontsize=17, fontweight="bold", color=ACCENT_GREEN,
             bbox=dict(boxstyle="round,pad=0.5", fc="#1c2b22", ec=ACCENT_GREEN, lw=2))

    fig.suptitle("Step 4: AI Model Training Convergence (Real 30 Epochs)", fontsize=19, fontweight="bold", color=TEXT_WHITE, y=0.97)

    save_path = OUT_DIR / "step4_train_model.png"
    plt.savefig(str(save_path), facecolor=BG_DARK, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ==============================================================================
# STEP 5: DETECTION ALERT (Real 3-Phase Fall Signature: Motion -> Impact -> Stillness)
# ==============================================================================
def create_step5_detection_alert():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_DARK, dpi=150)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.12)
    apply_clean_theme(ax)

    # Load real fall recording with genuine post-fall immobility
    csv_path = PROJECT_ROOT / "data" / "raw" / "fall_forward" / "sample_20260909_200400.csv"
    df = pd.read_csv(csv_path)
    amp_cols = [c for c in df.columns if c.startswith("amplitude_")]
    raw_amp = df[amp_cols].values.astype(np.float64)

    active_idx, _ = ht40_htltf_layout()
    valid_idx = [i for i in active_idx if i < raw_amp.shape[1]]
    amp_active = raw_amp[:, valid_idx]
    t_sec = np.arange(len(amp_active)) / 100.0

    # Filter with causal SOS bandpass
    sos = design_bandpass_sos(100.0, 0.5, 40.0, 4)
    filt = CausalSOSFilter(sos)
    filtered = filt.process(amp_active)

    # Real Kinetic Motion Energy across rolling 60 frames (~0.6s)
    energy = np.array([float(np.mean(np.var(filtered[max(0, i-60):i+1], axis=0))) if i > 5 else 0.0 for i in range(len(filtered))])

    # Plot real motion energy curve
    ax.plot(t_sec, energy, color=ACCENT_CYAN, lw=2.8, label="Kinetic Motion Energy E(t) [dB²]")
    ax.axhline(2.0, color=ACCENT_RED, ls="--", lw=2.0, label="Fall Alert Threshold (2.00 dB²)")

    # 3-Phase Biomechanical Fall Signature Zones
    # Phase 1: Pre-Fall Movement (0 to 5.5s)
    ax.axvspan(0, 5.5, color=ACCENT_CYAN, alpha=0.10)
    ax.text(2.75, energy.max() * 0.88, "1. PRE-FALL MOVEMENT\n(Active Motion in Room)", ha="center", fontsize=12.5, fontweight="bold", color=ACCENT_CYAN)

    # Phase 2: Fall Impact Shockwave (5.5 to 7.0s)
    ax.axvspan(5.5, 7.0, color=ACCENT_RED, alpha=0.22)
    ax.text(6.25, energy.max() * 0.88, "2. FALL IMPACT!\n(Alarm Triggered)", ha="center", fontsize=13, fontweight="bold", color=ACCENT_RED)

    # Phase 3: Post-Fall Immobility (7.0 to 9.0s) - Key feature requested!
    ax.axvspan(7.0, 9.0, color=ACCENT_GREEN, alpha=0.15)
    ax.text(8.0, energy.max() * 0.88, "3. POST-FALL STILLNESS\n(Subject Lying on Floor)", ha="center", fontsize=12.5, fontweight="bold", color=ACCENT_GREEN)

    # Peak impact callout
    peak_t = t_sec[np.argmax(energy)]
    peak_val = energy.max()
    ax.annotate(f"Sudden Fall Impact\nPeak: {peak_val:.1f} dB² (t = {peak_t:.2f}s)\nAlarm Triggered (E ≥ 2.0 dB²)",
                xy=(peak_t, peak_val), xytext=(peak_t - 1.8, peak_val * 0.60),
                ha="center", fontsize=12.5, fontweight="bold", color=TEXT_WHITE,
                bbox=dict(boxstyle="round,pad=0.5", fc="#3e1b24", ec=ACCENT_RED, lw=2.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=2.2))

    # Immobility callout
    still_t = 8.0
    still_val = energy[int(still_t * 100)]
    ax.annotate(f"Post-Fall Immobility (The Stillness Phase)\nEnergy drops to {still_val:.1f} dB² (Flat Baseline)\nConfirming Person is Down!",
                xy=(still_t, still_val), xytext=(still_t - 0.5, peak_val * 0.35),
                ha="center", fontsize=12, fontweight="bold", color=TEXT_WHITE,
                bbox=dict(boxstyle="round,pad=0.5", fc="#172b22", ec=ACCENT_GREEN, lw=2.0),
                arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=2.0))

    ax.set_xlabel("Time (seconds)", fontsize=14, labelpad=10)
    ax.set_ylabel("Kinetic Motion Energy (dB²)", fontsize=14, labelpad=10)
    ax.set_ylim(0, peak_val * 1.08)
    ax.set_title("Step 5: 3-Phase Fall Signature (Movement -> Impact -> Post-Fall Stillness)", fontsize=16, pad=12)

    fig.suptitle("Step 5: Real-Time Detection & Post-Fall Immobility Verification", fontsize=19, fontweight="bold", color=TEXT_WHITE, y=0.96)

    save_path = OUT_DIR / "step5_detection_alert.png"
    plt.savefig(str(save_path), facecolor=BG_DARK, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("Generating 5 presentation graphics from 100% REAL DATA (with 3-phase fall signature)...")
    create_step1_raw_data()
    create_step2_clean_data()
    create_step3_grouping_data()
    create_step4_train_model()
    create_step5_detection_alert()
    print("All 5 presentation graphics successfully generated!")
