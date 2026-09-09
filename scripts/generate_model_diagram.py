#!/usr/bin/env python3
"""
generate_model_diagram.py
=========================
Generates an ultra-minimal, high-contrast, dark-themed 16:9 diagram
for the CNN-BiLSTM-Attention architecture tailored for a 20-second slide pitch.
Uses 100% clean English typography to avoid font rendering issues.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "assets" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG_DARK = "#14151a"
PANEL_BG = "#1d202b"
PANEL_BORDER = "#2f3547"
TEXT_WHITE = "#ffffff"
TEXT_MUTED = "#9ba3b8"

COLOR_INPUT = "#38bdf8"      # Light Cyan
COLOR_CNN = "#00e5ff"        # Vivid Cyan
COLOR_LSTM = "#b388ff"       # Vivid Purple
COLOR_ATTN = "#ffd600"       # Vivid Gold / Yellow
COLOR_OUT = "#00e676"        # Green / Decision


def create_model_slide_diagram():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_DARK, dpi=150)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.08)
    
    ax.set_facecolor(PANEL_BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

    # Title & Header
    fig.suptitle("AI Architecture: CNN + BiLSTM + Attention",
                 fontsize=24, fontweight="bold", color=TEXT_WHITE, y=0.95)
    ax.text(50, 46, "Real-Time 1.0-Second Sliding Window Processing (100 Hz Sampling)",
            ha="center", va="center", fontsize=13.5, color=TEXT_MUTED)

    # 4 Main Blocks
    blocks = [
        {
            "x": 4, "w": 18, "h": 31, "y": 10.5,
            "title": "1. Input Window",
            "subtitle": "CSI Matrix (100x20)",
            "color": COLOR_INPUT,
            "duty": "100 Time-steps (1.0s)\n20 Subcarriers (PCA)\nCausal Filtered (SOS)",
            "tag": "Preprocessed Input"
        },
        {
            "x": 27, "w": 20, "h": 31, "y": 10.5,
            "title": "2. 1D-CNN",
            "subtitle": "Local Waveform Features",
            "color": COLOR_CNN,
            "duty": "Extracts sudden spikes\nCaptures sharp edges\nFilters ambient multipath",
            "tag": "Spatial Patterns"
        },
        {
            "x": 52, "w": 20, "h": 31, "y": 10.5,
            "title": "3. Bi-LSTM",
            "subtitle": "Temporal Context",
            "color": COLOR_LSTM,
            "duty": "Learns time sequence\nPast & future context\n(Before -> During -> After)",
            "tag": "Sequence Memory"
        },
        {
            "x": 77, "w": 20, "h": 31, "y": 10.5,
            "title": "4. Self-Attention",
            "subtitle": "Focus on Critical Moment",
            "color": COLOR_ATTN,
            "duty": "Weights decisive frames\nFocuses on impact shock\nSuppresses background",
            "tag": "Impact Weighting"
        }
    ]

    for b in blocks:
        # Card outline
        rect = patches.FancyBboxPatch(
            (b["x"], b["y"]), b["w"], b["h"],
            boxstyle="round,pad=0.8",
            fc="#181b24", ec=b["color"], lw=2.5
        )
        ax.add_patch(rect)

        # Tag pill
        tag_w = len(b["tag"]) * 0.72 + 2.5
        rect_tag = patches.FancyBboxPatch(
            (b["x"] + b["w"]/2 - tag_w/2, b["y"] + b["h"] - 2.0), tag_w, 2.6,
            boxstyle="round,pad=0.3",
            fc=b["color"], ec=b["color"], lw=1
        )
        ax.add_patch(rect_tag)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"] - 0.7, b["tag"],
                ha="center", va="center", fontsize=11, fontweight="bold", color="#0b0f19")

        # Title & Subtitle
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"] - 6.5, b["title"],
                ha="center", va="center", fontsize=16.5, fontweight="bold", color=TEXT_WHITE)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"] - 10.0, b["subtitle"],
                ha="center", va="center", fontsize=12, color=b["color"], fontweight="bold")

        # Divider
        ax.plot([b["x"] + 2, b["x"] + b["w"] - 2], [b["y"] + b["h"] - 13.5, b["y"] + b["h"] - 13.5],
                color=PANEL_BORDER, lw=1.5, ls="--")

        # Duty / Core concept
        ax.text(b["x"] + b["w"]/2, b["y"] + 8.0, b["duty"],
                ha="center", va="center", fontsize=13, color="#d1d5db", linespacing=1.7)

    # Arrows between blocks
    arrow_pairs = [(22.8, 26.2), (47.8, 51.2), (72.8, 76.2)]
    for x1, x2 in arrow_pairs:
        ax.annotate("", xy=(x2, 26), xytext=(x1, 26),
                    arrowprops=dict(arrowstyle="->,head_width=0.6,head_length=0.8",
                                    color=TEXT_MUTED, lw=2.5))

    # Final Output Badge - Generous width and proper padding so it completely covers the text!
    box_w = 78
    box_x = 50 - box_w / 2  # 11 to 89
    rect_out = patches.FancyBboxPatch(
        (box_x, 1.5), box_w, 5.8,
        boxstyle="round,pad=0.6",
        fc="#102318", ec=COLOR_OUT, lw=2.5
    )
    ax.add_patch(rect_out)
    ax.text(50, 4.4, "DECISION OUTPUT :   FALL (Alarm Triggered)   vs   NON-FALL (Normal Activity)",
            ha="center", va="center", fontsize=13, fontweight="bold", color=COLOR_OUT)

    save_path = OUT_DIR / "step_model_architecture.png"
    plt.savefig(str(save_path), facecolor=BG_DARK, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    create_model_slide_diagram()
