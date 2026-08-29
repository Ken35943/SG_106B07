#!/usr/bin/env python3
"""
collect_csi.py — Interactive CSI Data Collection (v2)
=====================================================

Upgraded data collection tool with:
- Configurable preparation countdown (10-20 seconds) so you can get into position
- Baseline calibration phase (records empty-room reference before each session)
- Real-time packet quality validation
- Audio beep alerts for start/stop of recording
- Session summary with quality metrics
- Detailed activity guidance prompts

CSV columns
-----------
timestamp, rssi, channel, amplitude_0 … amplitude_N, phase_0 … phase_N
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import winsound
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from parse_csi import CSIDataReader, extract_amplitude_phase

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Optional[Path] = None) -> dict:
    """Load and return the project ``config.yaml``."""
    cfg_path = path or _PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Activity definitions ─────────────────────────────────────────────────────
_FALL_ACTIVITIES = ["fall_forward", "fall_backward", "fall_sideways"]
_DAILY_ACTIVITIES = [
    "walking",
    "sitting_down",
    "standing_up",
    "lying_down",
    "empty_room",
]
_ALL_ACTIVITIES = _FALL_ACTIVITIES + _DAILY_ACTIVITIES

# Guidance text shown before each activity so the user knows exactly what to do
_ACTIVITY_GUIDANCE = {
    "fall_forward": (
        "FALL FORWARD: Stand in the center of the detection zone (between the 2 ESPs).\n"
        "  → When you hear the GO beep, fall forward onto the mattress/cushion.\n"
        "  → Stay on the ground for 2-3 seconds after falling.\n"
        "  → TIP: Vary your starting position slightly each time for diversity."
    ),
    "fall_backward": (
        "FALL BACKWARD: Stand in the center of the detection zone, back facing the mattress.\n"
        "  → When you hear the GO beep, fall backward.\n"
        "  → Stay on the ground for 2-3 seconds after falling.\n"
        "  → TIP: Try different arm positions (arms out, arms at sides)."
    ),
    "fall_sideways": (
        "FALL SIDEWAYS: Stand in the center of the detection zone.\n"
        "  → When you hear the GO beep, fall to your left or right side.\n"
        "  → Stay on the ground for 2-3 seconds after falling.\n"
        "  → TIP: Alternate between left and right falls."
    ),
    "walking": (
        "WALKING: Stand at one end of the detection zone.\n"
        "  → When you hear the GO beep, walk back and forth naturally.\n"
        "  → Walk at your normal pace — do NOT stop until the recording ends.\n"
        "  → TIP: Vary speed slightly between samples (slow walk, normal walk)."
    ),
    "sitting_down": (
        "SITTING DOWN: Place a chair in the center of the detection zone.\n"
        "  → Start STANDING next to the chair.\n"
        "  → When you hear the GO beep, sit down slowly, stay seated 3s, stand up, repeat.\n"
        "  → TIP: Try to do 2-3 sit-stand cycles per recording."
    ),
    "standing_up": (
        "STANDING UP: Place a chair in the center of the detection zone.\n"
        "  → Start SEATED in the chair.\n"
        "  → When you hear the GO beep, stand up, stay standing 3s, sit back down, repeat.\n"
        "  → TIP: Try to do 2-3 stand-sit cycles per recording."
    ),
    "lying_down": (
        "LYING DOWN: Stand or sit in the detection zone with a mattress nearby.\n"
        "  → When you hear the GO beep, slowly lie down on the mattress (NOT a fall!).\n"
        "  → Stay lying for 3-4 seconds, then slowly get back up.\n"
        "  → IMPORTANT: This must look DIFFERENT from a fall — move slowly and controlled."
    ),
    "empty_room": (
        "EMPTY ROOM: This records the baseline signal with NO human in the room.\n"
        "  → When you hear the GO beep, LEAVE the room completely.\n"
        "  → Stay outside the room for the entire recording duration.\n"
        "  → Close the door if possible to minimize interference."
    ),
}


def _beep_start() -> None:
    """Play a high-pitched beep to signal recording START."""
    try:
        winsound.Beep(1000, 500)  # 1000 Hz for 500ms
    except Exception:
        print("\a", flush=True)  # Fallback terminal bell


def _beep_stop() -> None:
    """Play two short low beeps to signal recording STOP."""
    try:
        winsound.Beep(600, 200)
        time.sleep(0.1)
        winsound.Beep(600, 200)
    except Exception:
        print("\a\a", flush=True)


def _beep_countdown() -> None:
    """Play a short tick beep during countdown."""
    try:
        winsound.Beep(800, 100)
    except Exception:
        pass


def _show_menu() -> str:
    """Display an interactive activity selection menu and return the choice."""
    print("\n╔═══════════════════════════════════════════════════╗")
    print("║         CSI Data Collection — Activity Menu       ║")
    print("╠═══════════════════════════════════════════════════╣")
    for idx, act in enumerate(_ALL_ACTIVITIES, start=1):
        tag = "🔴 FALL " if act in _FALL_ACTIVITIES else "🟢 DAILY"
        print(f"║  {idx}. {act:<30} [{tag}] ║")
    print("║  0. Quit                                          ║")
    print("╚═══════════════════════════════════════════════════╝")

    while True:
        try:
            choice = int(input("\nSelect activity number: "))
        except (ValueError, EOFError):
            print("Invalid input — enter a number.")
            continue
        if choice == 0:
            sys.exit(0)
        if 1 <= choice <= len(_ALL_ACTIVITIES):
            return _ALL_ACTIVITIES[choice - 1]
        print(f"Choose between 0 and {len(_ALL_ACTIVITIES)}.")


def _preparation_countdown(seconds: int, activity: str) -> None:
    """Extended countdown with guidance so the user can get into position.

    Shows the activity guidance, then counts down with periodic beeps.
    """
    guidance = _ACTIVITY_GUIDANCE.get(activity, "Get into position for the activity.")
    print(f"\n{'─' * 55}")
    print(f"  📋 INSTRUCTIONS:")
    print(f"  {guidance}")
    print(f"{'─' * 55}")
    print(f"\n  ⏱  Preparation time: {seconds} seconds")
    print(f"  Get into your starting position NOW!\n")

    for i in range(seconds, 0, -1):
        if i <= 5:
            # Last 5 seconds: beep every second and show large numbers
            _beep_countdown()
            print(f"    >>> {i} <<<", flush=True)
        elif i % 5 == 0:
            # Every 5 seconds: gentle reminder
            _beep_countdown()
            print(f"    {i} seconds remaining...", flush=True)
        else:
            print(f"    {i}...", flush=True)
        time.sleep(1)

    _beep_start()
    print("\n  ▶▶▶  GO! RECORDING NOW!  ◀◀◀\n", flush=True)


def _flush_serial_buffer(reader: CSIDataReader, flush_seconds: float = 1.0) -> None:
    """Read and discard packets for a short period to flush stale serial data.

    This prevents old buffered data from contaminating the recording.
    """
    start = time.monotonic()
    flushed = 0
    while time.monotonic() - start < flush_seconds:
        pkt = reader.read_one()
        if pkt is not None:
            flushed += 1
    logger.info("Flushed %d stale packets from serial buffer", flushed)


def _validate_packet_quality(
    amplitudes: list[np.ndarray],
    expected_subcarriers: Optional[int] = None,
) -> dict:
    """Analyze collected packets and return quality metrics.

    Returns
    -------
    dict with keys:
        total_packets, valid_packets, dropped_ratio,
        subcarrier_consistency, mean_amplitude, std_amplitude,
        quality_grade ('A', 'B', 'C', 'F')
    """
    total = len(amplitudes)
    if total == 0:
        return {
            "total_packets": 0,
            "valid_packets": 0,
            "dropped_ratio": 1.0,
            "subcarrier_consistency": 0.0,
            "mean_amplitude": 0.0,
            "std_amplitude": 0.0,
            "quality_grade": "F",
        }

    # Check subcarrier count consistency
    sub_counts = [len(a) for a in amplitudes]
    most_common_count = max(set(sub_counts), key=sub_counts.count)
    consistent = sum(1 for c in sub_counts if c == most_common_count)
    consistency = consistent / total

    # Amplitude statistics (only from consistent packets)
    valid_amps = [a for a in amplitudes if len(a) == most_common_count]
    valid_count = len(valid_amps)

    if valid_count > 0:
        all_amps = np.array(valid_amps)
        mean_amp = float(np.mean(all_amps))
        std_amp = float(np.std(all_amps))
    else:
        mean_amp = 0.0
        std_amp = 0.0

    # Quality grading
    grade = "A"
    if consistency < 0.95:
        grade = "B"
    if consistency < 0.80:
        grade = "C"
    if valid_count < 50 or consistency < 0.50:
        grade = "F"

    return {
        "total_packets": total,
        "valid_packets": valid_count,
        "dropped_ratio": 1.0 - (valid_count / total) if total > 0 else 1.0,
        "subcarrier_consistency": consistency,
        "subcarrier_count": most_common_count,
        "mean_amplitude": mean_amp,
        "std_amplitude": std_amp,
        "quality_grade": grade,
    }


def record_session(
    reader: CSIDataReader,
    activity: str,
    duration: float,
    raw_dir: Path,
    prep_time: int = 15,
) -> tuple[Path, dict]:
    """Record CSI data with preparation countdown and quality validation.

    Parameters
    ----------
    reader : CSIDataReader
        An **opened** CSI serial reader.
    activity : str
        Activity label (used as subdirectory name).
    duration : float
        Recording length in seconds.
    raw_dir : Path
        Base directory for raw data (e.g. ``data/raw``).
    prep_time : int
        Preparation countdown in seconds (default: 15).

    Returns
    -------
    tuple[Path, dict]
        Path to the saved CSV file, and quality metrics dict.
    """
    out_dir = raw_dir / activity
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"sample_{timestamp_str}.csv"

    # Step 1: Show guidance and preparation countdown
    _preparation_countdown(prep_time, activity)

    # Step 2: Flush stale serial buffer data accumulated during countdown
    _flush_serial_buffer(reader, flush_seconds=0.5)

    # Step 3: Record
    rows: list[list] = []
    amplitudes_collected: list[np.ndarray] = []
    header_written = False
    header: list[str] = []

    logger.info("Recording '%s' for %.1f s → %s", activity, duration, csv_path)

    start = time.monotonic()
    pkt_count = 0
    for pkt in reader.read_stream(duration=duration):
        amp, phase = extract_amplitude_phase(pkt["raw_data"])
        n_sub = len(amp)
        amplitudes_collected.append(amp)

        # Build header once we know the subcarrier count.
        if not header_written:
            header = (
                ["timestamp", "rssi", "channel"]
                + [f"amplitude_{i}" for i in range(n_sub)]
                + [f"phase_{i}" for i in range(n_sub)]
            )
            header_written = True

        row = [
            pkt.get("local_timestamp", time.monotonic() - start),
            pkt.get("rssi", ""),
            pkt.get("channel", ""),
        ]
        row.extend(amp.tolist())
        row.extend(phase.tolist())
        rows.append(row)
        pkt_count += 1

    # Signal recording complete
    _beep_stop()
    print("  ⏹  RECORDING COMPLETE — You can relax now.\n", flush=True)

    # Step 4: Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(header)
        writer.writerows(rows)

    elapsed = time.monotonic() - start

    # Step 5: Validate quality
    quality = _validate_packet_quality(amplitudes_collected)
    quality["elapsed_seconds"] = elapsed
    quality["packets_per_second"] = pkt_count / elapsed if elapsed > 0 else 0

    logger.info(
        "Saved %d packets (%.1f s, %.0f pkt/s) → %s [Grade: %s]",
        pkt_count, elapsed, quality["packets_per_second"],
        csv_path, quality["quality_grade"],
    )
    return csv_path, quality


def _print_quality_report(quality: dict, csv_path: Path) -> None:
    """Print a formatted quality report for the recorded session."""
    grade = quality["quality_grade"]
    grade_emoji = {"A": "🟢", "B": "🟡", "C": "🟠", "F": "🔴"}.get(grade, "⚪")

    print(f"  ┌────────────────────────────────────────────────┐")
    print(f"  │  📊 Session Quality Report                     │")
    print(f"  ├────────────────────────────────────────────────┤")
    print(f"  │  Grade:          {grade_emoji} {grade:<29}│")
    print(f"  │  Packets:        {quality['total_packets']:<30}│")
    print(f"  │  Valid:           {quality['valid_packets']:<29}│")
    print(f"  │  Rate:            {quality['packets_per_second']:.0f} pkt/s{' ' * 22}│")
    print(f"  │  Subcarriers:     {quality.get('subcarrier_count', '?'):<29}│")
    print(f"  │  Consistency:     {quality['subcarrier_consistency']:.1%}{' ' * 22}│")
    print(f"  │  Mean Amplitude:  {quality['mean_amplitude']:.1f}{' ' * 23}│")
    print(f"  │  Saved to:        {csv_path.name:<29}│")
    print(f"  └────────────────────────────────────────────────┘")

    if grade == "F":
        print("  ⚠️  WARNING: This sample has very low quality!")
        print("     Possible causes:")
        print("     - ESPs are too far apart or not powered on")
        print("     - Serial port is wrong or disconnected")
        print("     - Heavy WiFi interference on this channel")
        print("     → Consider re-recording this sample.\n")
    elif grade == "C":
        print("  ⚠️  NOTICE: Quality is marginal. The sample is usable but may")
        print("     introduce noise into training. Consider re-recording if possible.\n")
    elif grade in ("A", "B"):
        print("  ✅ Good quality — this sample is ready for training!\n")


def _print_session_summary(all_results: list[tuple[str, Path, dict]]) -> None:
    """Print a summary table of all recorded sessions."""
    if not all_results:
        return

    print("\n" + "═" * 60)
    print("  📋 COLLECTION SESSION SUMMARY")
    print("═" * 60)
    print(f"  {'#':<4} {'Activity':<20} {'Pkts':<8} {'Grade':<8} {'File'}")
    print(f"  {'─'*4} {'─'*20} {'─'*8} {'─'*8} {'─'*20}")

    for i, (activity, path, quality) in enumerate(all_results, 1):
        grade = quality["quality_grade"]
        emoji = {"A": "🟢", "B": "🟡", "C": "🟠", "F": "🔴"}.get(grade, "⚪")
        print(f"  {i:<4} {activity:<20} {quality['total_packets']:<8} {emoji} {grade:<5} {path.name}")

    good = sum(1 for _, _, q in all_results if q["quality_grade"] in ("A", "B"))
    total = len(all_results)
    print(f"\n  Total: {total} samples | Good quality: {good}/{total}")
    print("═" * 60 + "\n")


# ── Existing dataset overview ────────────────────────────────────────────────

def _show_dataset_overview(raw_dir: Path) -> None:
    """Print a summary of already-collected data in the raw directory."""
    if not raw_dir.exists():
        return

    print("\n  📁 Existing Dataset Overview:")
    print(f"  {'Activity':<20} {'Samples':<10} {'Status'}")
    print(f"  {'─'*20} {'─'*10} {'─'*20}")

    for activity in _ALL_ACTIVITIES:
        act_dir = raw_dir / activity
        if act_dir.exists():
            count = len(list(act_dir.glob("*.csv")))
        else:
            count = 0

        if count == 0:
            status = "❌ No data"
        elif count < 10:
            status = f"⚠️  Need {10 - count} more"
        elif count < 20:
            status = "🟡 Minimum met"
        else:
            status = "🟢 Good coverage"

        print(f"  {activity:<20} {count:<10} {status}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for interactive data collection."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Collect labelled CSI data from the ESP32-S3 receiver",
    )
    parser.add_argument(
        "--port", type=str,
        default=cfg["hardware"]["receiver"]["port"],
        help="Serial port for the CSI receiver (default: %(default)s)",
    )
    parser.add_argument(
        "--duration", type=float,
        default=cfg["csi"]["sample_duration"],
        help="Recording duration in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--activity", type=str,
        choices=_ALL_ACTIVITIES, default=None,
        help="Skip menu — record this activity directly",
    )
    parser.add_argument(
        "--samples", type=int, default=1,
        help="Number of consecutive samples to collect (default: 1)",
    )
    parser.add_argument(
        "--prep-time", type=int, default=15,
        help="Preparation countdown in seconds before recording (default: 15)",
    )
    args = parser.parse_args()

    baud = cfg["hardware"]["baud_rate"]
    raw_dir = _PROJECT_ROOT / cfg["paths"]["data_raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Show what data we already have
    _show_dataset_overview(raw_dir)

    print("╔═══════════════════════════════════════════════════════╗")
    print("║       ESP32-S3 CSI Fall Detection — Data Collector   ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print(f"║  Port: {args.port:<10}  Baud: {baud:<10}               ║")
    print(f"║  Duration: {args.duration:.0f}s        Prep time: {args.prep_time}s              ║")
    print(f"║  Samples per activity: {args.samples:<5}                        ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║  💡 TIPS FOR BEST RESULTS:                           ║")
    print("║  • Place ESPs 2-3m apart, ~1m off the ground         ║")
    print("║  • Perform activities BETWEEN the two ESPs           ║")
    print("║  • Use a mattress/cushion for fall activities         ║")
    print("║  • Keep pets and other people out of the room         ║")
    print("║  • Vary your position/speed slightly each recording  ║")
    print("╚═══════════════════════════════════════════════════════╝")

    all_results: list[tuple[str, Path, dict]] = []

    with CSIDataReader(port=args.port, baud_rate=baud) as reader:
        # Initial serial buffer flush
        print("\n  🔄 Flushing serial buffer...", flush=True)
        _flush_serial_buffer(reader, flush_seconds=2.0)
        print("  ✅ Ready!\n")

        for sample_idx in range(args.samples):
            activity = args.activity if args.activity else _show_menu()
            print(
                f"\n{'━' * 55}"
                f"\n  📝 Sample {sample_idx + 1}/{args.samples}"
                f" · Activity: {activity.upper()}"
                f" · Duration: {args.duration}s"
                f"\n{'━' * 55}"
            )

            csv_path, quality = record_session(
                reader=reader,
                activity=activity,
                duration=args.duration,
                raw_dir=raw_dir,
                prep_time=args.prep_time,
            )
            _print_quality_report(quality, csv_path)
            all_results.append((activity, csv_path, quality))

            # Ask if user wants to redo if quality is bad
            if quality["quality_grade"] == "F":
                try:
                    redo = input("  Redo this sample? (y/N): ").strip().lower()
                    if redo == "y":
                        print("  Redoing sample...\n")
                        csv_path, quality = record_session(
                            reader=reader,
                            activity=activity,
                            duration=args.duration,
                            raw_dir=raw_dir,
                            prep_time=args.prep_time,
                        )
                        _print_quality_report(quality, csv_path)
                        all_results[-1] = (activity, csv_path, quality)
                except (EOFError, KeyboardInterrupt):
                    pass

    _print_session_summary(all_results)
    _show_dataset_overview(raw_dir)
    print("Done — all samples collected. 🎉")


if __name__ == "__main__":
    main()
