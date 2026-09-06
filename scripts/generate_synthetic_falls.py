#!/usr/bin/env python3
"""
generate_synthetic_falls.py — Realistic Fall Sample Generator
============================================================

Generates synthetic CSI fall recordings matching the exact CSV structure and
subcarrier count (190 subcarriers in HT40) of existing data.

CSI Fall Dynamics Modeled:
1. Phase 1 (0.0s - 3.5s): Pre-fall normal movement (moderate variance)
2. Phase 2 (3.5s - 5.0s): Sudden descent / impact transient (high amplitude swing & frequency shift)
3. Phase 3 (5.0s - 10.0s): Post-fall lying still on the floor (very low variance)

Usage:
    python scripts/generate_synthetic_falls.py --count 5
"""

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FALL_GENERATOR")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw"


def find_reference_csv() -> Path:
    """Find an existing CSV to clone header and subcarrier format."""
    csvs = list(RAW_DIR.glob("**/*.csv"))
    if not csvs:
        raise FileNotFoundError("No reference CSV found in data/raw to copy schema from")
    return sorted(csvs)[0]


def synthesize_fall_sample(
    header: list[str],
    n_packets: int = 750,
    fall_type: str = "fall_forward",
) -> list[dict]:
    """Generate a single 10-second fall recording with 750-1000 packets."""
    amp_cols = [c for c in header if c.startswith("amplitude_")]
    phase_cols = [c for c in header if c.startswith("phase_")]
    n_sub = len(amp_cols)

    # Base profile across subcarriers
    center_sub = n_sub // 2
    carrier_curve = 30.0 - 10.0 * (np.abs(np.arange(n_sub) - center_sub) / center_sub) ** 2
    carrier_curve = np.clip(carrier_curve, 10.0, 40.0)

    rows = []
    t_start = 1721045000.0 + np.random.uniform(0, 100000)

    # Dynamics timing (fall happens between packet 250 and 370)
    fall_start = int(n_packets * 0.35)
    impact = int(n_packets * 0.45)
    still_end = n_packets

    for i in range(n_packets):
        t = t_start + i * 0.0133  # ~75 Hz
        rssi = -45 + int(np.random.normal(0, 1.5))
        channel = 6

        if i < fall_start:
            # Phase 1: Pre-fall movement
            motion = np.sin(i * 0.2) * 4.0 + np.random.normal(0, 2.0, n_sub)
            amp = carrier_curve + motion
            rssi = -46 + int(np.random.normal(0, 1))

        elif fall_start <= i < impact:
            # Phase 2: Active fall descent and impact (huge amplitude spike & fluctuations)
            descent_progress = (i - fall_start) / (impact - fall_start)
            burst = np.sin(i * 0.8) * (20.0 + 10.0 * descent_progress)
            high_freq_jitter = np.random.normal(0, 8.0, n_sub)
            amp = carrier_curve + burst + high_freq_jitter
            rssi = -52 + int(np.random.normal(0, 3))  # RSSI drops during fall

        else:
            # Phase 3: Lying still on the floor (flat line, low variance, ground multipath shift)
            flat_shift = -5.0 if "backward" in fall_type else -3.0
            quiet_jitter = np.random.normal(0, 0.4, n_sub)
            amp = (carrier_curve + flat_shift) + quiet_jitter
            rssi = -49 + int(np.random.normal(0, 0.5))

        amp = np.clip(amp, 0.0, 80.0)
        # Synthetic phase (uniform circular)
        phases = np.random.uniform(-np.pi, np.pi, n_sub)

        row_dict = {
            "timestamp": f"{t:.6f}",
            "rssi": str(rssi),
            "channel": str(channel),
        }
        for idx, col in enumerate(amp_cols):
            row_dict[col] = f"{amp[idx]:.4f}"
        for idx, col in enumerate(phase_cols):
            row_dict[col] = f"{phases[idx]:.4f}"

        rows.append(row_dict)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CSI fall recordings")
    parser.add_argument("--count", type=int, default=4, help="Number of files to generate per fall category")
    args = parser.parse_args()

    ref_csv = find_reference_csv()
    logger.info("Using reference CSV for schema: %s", ref_csv)

    with open(ref_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    fall_activities = ["fall_forward", "fall_backward", "fall_sideways"]

    for act in fall_activities:
        act_dir = RAW_DIR / act
        act_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.count):
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i:02d}"
            filename = act_dir / f"sample_synth_{ts_str}.csv"
            logger.info("Generating %s -> %s", act, filename.name)

            rows = synthesize_fall_sample(header, n_packets=750, fall_type=act)
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)

    logger.info("Generated %d synthetic fall samples across %s ✓", args.count * len(fall_activities), fall_activities)


if __name__ == "__main__":
    main()
