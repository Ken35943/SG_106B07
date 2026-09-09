#!/usr/bin/env python3
"""
generate_conclusion_slide.py
============================
Generates an ultra-minimal, modern, dark-themed 16:9 Conclusion & Summary slide
graphic for the presentation.
Highlights:
  1. Key Achievements (Proof-of-Concept validated)
  2. Honest Academic Limitations (Preliminary metrics, small dataset)
  3. Future Work & Roadmap (Scale dataset, multi-node mesh, on-chip edge AI)
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

COLOR_SUMMARY = "#00e5ff"    # Cyan
COLOR_LIMIT = "#ff7043"      # Orange / Honest Limitation
COLOR_FUTURE = "#00e676"     # Green / Roadmap


def create_conclusion_slide():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_DARK, dpi=150)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.08)
    
    ax.set_facecolor(PANEL_BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

    # Title & Subtitle
    fig.suptitle("Project Summary & Future Roadmap",
                 fontsize=24, fontweight="bold", color=TEXT_WHITE, y=0.95)
    ax.text(50, 45.8, "ESP32-S3 Wi-Fi CSI Contactless Fall Detection System",
            ha="center", va="center", fontsize=13.5, color=TEXT_MUTED)

    # 3 Strategic Column Cards
    cards = [
        {
            "x": 4, "w": 29, "h": 36, "y": 6,
            "title": "1. Proof of Concept",
            "tag": "Key Takeaways",
            "color": COLOR_SUMMARY,
            "points": [
                ("100% Privacy-Preserving", "No cameras, no wearables,\nuses standard Wi-Fi RF waves"),
                ("3-Phase Biomechanical Signal", "Captures Movement -> Impact\n-> Post-fall stillness clearly"),
                ("Full Real-Time Pipeline", "ESP32-S3 CSI streaming to\nAI detection + Hardware Alarm")
            ]
        },
        {
            "x": 35.5, "w": 29, "h": 36, "y": 6,
            "title": "2. Current Limitations",
            "tag": "Academic Honesty",
            "color": COLOR_LIMIT,
            "points": [
                ("Preliminary Accuracy", "Results are proof-of-concept;\nnot yet full clinical validation"),
                ("Limited Dataset Scale", "37 physical lab recordings;\nmore subject diversity needed"),
                ("False Positive Challenge", "Vigorous walking can trigger\nalert; tuned stillness required")
            ]
        },
        {
            "x": 67, "w": 29, "h": 36, "y": 6,
            "title": "3. Future Roadmap",
            "tag": "Next Steps",
            "color": COLOR_FUTURE,
            "points": [
                ("Dataset Expansion", "Multi-person, diverse rooms,\nand daily activity edge cases"),
                ("Multi-Node CSI Mesh", "Deploy multi-ESP32 mesh to\neliminate room blind spots"),
                ("Edge AI Quantization", "Quantize model (INT8/TFLite)\nto run entirely on ESP32-S3")
            ]
        }
    ]

    for c in cards:
        # Outer Card Box
        rect = patches.FancyBboxPatch(
            (c["x"], c["y"]), c["w"], c["h"],
            boxstyle="round,pad=0.8",
            fc="#171a23", ec=c["color"], lw=2.4
        )
        ax.add_patch(rect)

        # Tag Header Pill
        tag_w = len(c["tag"]) * 0.72 + 2.5
        rect_tag = patches.FancyBboxPatch(
            (c["x"] + c["w"]/2 - tag_w/2, c["y"] + c["h"] - 2.0), tag_w, 2.6,
            boxstyle="round,pad=0.3",
            fc=c["color"], ec=c["color"], lw=1
        )
        ax.add_patch(rect_tag)
        ax.text(c["x"] + c["w"]/2, c["y"] + c["h"] - 0.7, c["tag"],
                ha="center", va="center", fontsize=11, fontweight="bold", color="#0b0f19")

        # Title
        ax.text(c["x"] + c["w"]/2, c["y"] + c["h"] - 6.0, c["title"],
                ha="center", va="center", fontsize=17, fontweight="bold", color=TEXT_WHITE)

        # Divider
        ax.plot([c["x"] + 2, c["x"] + c["w"] - 2], [c["y"] + c["h"] - 8.5, c["y"] + c["h"] - 8.5],
                color=PANEL_BORDER, lw=1.5, ls="--")

        # 3 Bullet Items per Card
        y_cursor = c["y"] + c["h"] - 12.0
        for header, detail in c["points"]:
            # Bullet icon / badge
            circle = patches.Circle((c["x"] + 2.2, y_cursor - 0.2), 0.7, fc=c["color"], ec="none")
            ax.add_patch(circle)

            # Bullet header
            ax.text(c["x"] + 4.0, y_cursor, header,
                    fontsize=13, fontweight="bold", color=c["color"], va="center")

            # Bullet detail
            ax.text(c["x"] + 4.0, y_cursor - 3.8, detail,
                    fontsize=11.5, color="#cbd5e1", va="center", linespacing=1.4)

            y_cursor -= 8.0

    # Bottom Takeaway Message Bar
    rect_bot = patches.FancyBboxPatch(
        (8, 0.8), 84, 3.8,
        boxstyle="round,pad=0.3",
        fc="#1e2230", ec=PANEL_BORDER, lw=1.5
    )
    ax.add_patch(rect_bot)
    ax.text(50, 2.7, "Takeaway: Feasible contactless fall detection achieved on low-cost hardware with clear path to deployment.",
            ha="center", va="center", fontsize=12.5, color="#94a3b8", fontstyle="italic")

    save_path = OUT_DIR / "step_conclusion_roadmap.png"
    plt.savefig(str(save_path), facecolor=BG_DARK, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    create_conclusion_slide()
