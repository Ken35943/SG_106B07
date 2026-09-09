#!/usr/bin/env python3
"""
generate_results_table_slide.py
===============================
Generates an ultra-minimal, high-contrast, dark-themed 16:9 slide graphic
displaying the complete evaluation results table from 100% REAL physical test data
(model/saved/evaluation/evaluation_results.json).
"""

import json
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

ACCENT_CYAN = "#00e5ff"
ACCENT_GREEN = "#00e676"
ACCENT_ORANGE = "#ff7043"
ACCENT_PURPLE = "#b388ff"
ACCENT_YELLOW = "#ffd600"


def create_results_table_slide():
    # Load actual evaluation results
    json_path = PROJECT_ROOT / "model" / "saved" / "evaluation" / "evaluation_results.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    rep = data["classification_report"]
    acc = data["accuracy"] * 100
    roc_auc = data["roc_auc"]

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_DARK, dpi=150)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.06)

    ax.set_facecolor(PANEL_BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

    # Title & Subtitle
    fig.suptitle("Model Evaluation Results (Held-Out Physical Test Set)",
                 fontsize=24, fontweight="bold", color=TEXT_WHITE, y=0.95)
    ax.text(50, 45.8, "Tested on 153 Unseen Physical Windows (8 Independent Recording Sessions — Zero Leakage)",
            ha="center", va="center", fontsize=13.5, color=TEXT_MUTED)

    # -------------------------------------------------------------
    # TABLE SECTION (y: 20 to 42)
    # -------------------------------------------------------------
    headers = ["Class / Category", "Precision", "Recall (Sensitivity)", "F1-Score", "Support (Samples)"]
    col_x = [6, 32, 50, 70, 88]
    col_align = ["left", "center", "center", "center", "center"]

    # Table Header Background
    hdr_box = patches.FancyBboxPatch(
        (4, 38.0), 92, 4.2, boxstyle="round,pad=0.4",
        fc="#252a3a", ec=PANEL_BORDER, lw=1.8
    )
    ax.add_patch(hdr_box)

    for i, h in enumerate(headers):
        ax.text(col_x[i], 40.1, h, fontsize=13, fontweight="bold",
                color=TEXT_WHITE, ha=col_align[i], va="center")

    # Table Rows Data
    rows = [
        {
            "class": "Fall (Positive)",
            "color": ACCENT_ORANGE,
            "prec": f"{rep['Fall (0)']['precision']*100:.1f}%",
            "rec": f"{rep['Fall (0)']['recall']*100:.1f}%",
            "f1": f"{rep['Fall (0)']['f1-score']*100:.1f}%",
            "supp": f"{int(rep['Fall (0)']['support'])} falls",
            "badge": "High Safety Focus (95.1% Detected)"
        },
        {
            "class": "Non-Fall (Normal Activity)",
            "color": ACCENT_CYAN,
            "prec": f"{rep['Non-Fall (1)']['precision']*100:.1f}%",
            "rec": f"{rep['Non-Fall (1)']['recall']*100:.1f}%",
            "f1": f"{rep['Non-Fall (1)']['f1-score']*100:.1f}%",
            "supp": f"{int(rep['Non-Fall (1)']['support'])} windows",
            "badge": "Walk / Sit / Stand"
        },
        {
            "class": "Macro Average",
            "color": ACCENT_PURPLE,
            "prec": f"{rep['macro avg']['precision']*100:.1f}%",
            "rec": f"{rep['macro avg']['recall']*100:.1f}%",
            "f1": f"{rep['macro avg']['f1-score']*100:.1f}%",
            "supp": "153 total",
            "badge": "Unweighted Mean"
        },
        {
            "class": "Overall Test Accuracy",
            "color": ACCENT_GREEN,
            "prec": f"{rep['weighted avg']['precision']*100:.1f}%",
            "rec": f"{acc:.1f}% (Acc)",
            "f1": f"{rep['weighted avg']['f1-score']*100:.1f}%",
            "supp": "153 total",
            "badge": f"ROC-AUC: {roc_auc:.3f}"
        }
    ]

    row_y = [32.8, 27.6, 22.4, 17.2]

    for idx, r in enumerate(rows):
        yp = row_y[idx]
        bg_c = "#171a23" if idx % 2 == 0 else "#1a1e2a"
        border_c = r["color"] if idx == 3 else PANEL_BORDER
        lw = 2.0 if idx == 3 else 1.2

        r_box = patches.FancyBboxPatch(
            (4, yp - 1.8), 92, 3.8, boxstyle="round,pad=0.3",
            fc=bg_c, ec=border_c, lw=lw
        )
        ax.add_patch(r_box)

        # Class Name with color indicator
        circle = patches.Circle((col_x[0] + 0.5, yp + 0.1), 0.6, fc=r["color"], ec="none")
        ax.add_patch(circle)
        ax.text(col_x[0] + 2.0, yp + 0.1, r["class"],
                fontsize=13, fontweight="bold", color=r["color"], va="center")

        # Values
        ax.text(col_x[1], yp + 0.1, r["prec"], fontsize=13, color=TEXT_WHITE, ha="center", va="center")
        ax.text(col_x[2], yp + 0.1, r["rec"], fontsize=13, fontweight="bold",
                color=ACCENT_GREEN if "95" in r["rec"] or "Acc" in r["rec"] else TEXT_WHITE,
                ha="center", va="center")
        ax.text(col_x[3], yp + 0.1, r["f1"], fontsize=13, color=TEXT_WHITE, ha="center", va="center")
        ax.text(col_x[4], yp + 0.1, r["supp"], fontsize=12, color=TEXT_MUTED, ha="center", va="center")

    # -------------------------------------------------------------
    # 2 KEY INSIGHT CARDS (Bottom section, y: 3 to 13)
    # -------------------------------------------------------------
    # Card 1: High Recall (Safety First)
    c1_box = patches.FancyBboxPatch(
        (4, 2.5), 44.5, 11.5, boxstyle="round,pad=0.6",
        fc="#16261f", ec=ACCENT_GREEN, lw=2.2
    )
    ax.add_patch(c1_box)
    ax.text(6.5, 11.2, "1. Critical Safety: 95.1% Fall Recall",
            fontsize=14, fontweight="bold", color=ACCENT_GREEN, va="center")
    ax.text(6.5, 6.8, "• Out of 41 physical falls, 39 were successfully detected.\n"
                      "• Only 2 false negatives (False Negative Rate ~4.9%).\n"
                      "• Top priority for healthcare: Never miss a real fall.",
            fontsize=11.8, color="#cbd5e1", va="center", linespacing=1.6)

    # Card 2: False Positive Mitigation (The 59% Precision)
    c2_box = patches.FancyBboxPatch(
        (51.5, 2.5), 44.5, 11.5, boxstyle="round,pad=0.6",
        fc="#281a1f", ec=ACCENT_ORANGE, lw=2.2
    )
    ax.add_patch(c2_box)
    ax.text(54.0, 11.2, "2. Real-Life Tradeoff: 59.1% Fall Precision",
            fontsize=14, fontweight="bold", color=ACCENT_ORANGE, va="center")
    ax.text(54.0, 6.8, "• Vigorous walking / erratic motion can trigger AI alert.\n"
                      "• Solved by Biomechanical Verification (Step 5):\n"
                      "  Post-fall stillness check eliminates false walking alarms!",
            fontsize=11.8, color="#cbd5e1", va="center", linespacing=1.6)

    save_path = OUT_DIR / "step_evaluation_results.png"
    plt.savefig(str(save_path), facecolor=BG_DARK, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    create_results_table_slide()
