#!/usr/bin/env python3
"""
collect_dual_csi.py — Synchronized Dual-Receiver CSI Data Collection
====================================================================
Records CSI data from two ESP32-S3 receivers simultaneously (e.g. COM3 & COM11)
while a standalone Tx broadcasts from the corner of the room.

Outputs per session:
data/raw_dual/<activity>/<session_id>/
├── rx1.csv       (Receiver 1 data with Tx packet IDs)
├── rx2.csv       (Receiver 2 data with Tx packet IDs)
└── meta.json     (Session metadata & sync statistics)
"""

from __future__ import annotations
import sys
import os
import time
import json
import argparse
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_csi import CSIDataReader, extract_amplitude_phase

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ALL_ACTIVITIES = [
    "fall_forward", "fall_backward", "fall_sideways",
    "walking", "sitting_down", "standing_up", "lying_down", "empty_room"
]


def _beep_countdown():
    try:
        import winsound
        winsound.Beep(1000, 100)
    except Exception:
        pass


def _beep_start():
    try:
        import winsound
        winsound.Beep(1800, 400)
    except Exception:
        pass


def _beep_stop():
    try:
        import winsound
        winsound.Beep(1200, 200)
        time.sleep(0.05)
        winsound.Beep(800, 300)
    except Exception:
        pass


class SingleLinkRecorder(threading.Thread):
    """Records one receiver port into a list of row dicts."""

    def __init__(self, name: str, port: str, baud: int = 921600):
        super().__init__(daemon=True, name=f"Rec-{name}")
        self.name = name
        self.port = port
        self.baud = baud
        self.rows = []
        self.running = True
        self.error = None
        self.packet_ids = set()

    def run(self):
        logger.info("[%s] Connecting to %s...", self.name, self.port)
        try:
            with CSIDataReader(port=self.port, baud_rate=self.baud, timeout=1.0) as reader:
                if reader._serial:
                    reader._serial.reset_input_buffer()

                while self.running:
                    pkt = reader.read_one()
                    if pkt is None:
                        continue

                    amp, phase = extract_amplitude_phase(pkt.get("raw_data", []))
                    pkt_id = int(pkt.get("id", -1))
                    ts = pkt.get("local_timestamp", 0)
                    rssi = pkt.get("rssi", 0)
                    channel = pkt.get("channel", 6)

                    self.packet_ids.add(pkt_id)
                    row = [pkt_id, ts, rssi, channel] + amp.tolist() + phase.tolist()
                    self.rows.append(row)

        except Exception as e:
            self.error = str(e)
            logger.error("[%s] Error on %s: %s", self.name, self.port, e)

    def stop(self):
        self.running = False


def record_dual_session(
    ports: tuple[str, str],
    activity: str,
    duration: float = 10.0,
    prep_time: int = 5,
    baud: int = 921600,
    out_dir: Optional[Path] = None,
):
    print("\n" + "═" * 60)
    print(f"  📡 DUAL-LINK CSI RECORDER: {activity.upper()}")
    print(f"  Ports: Rx1 = {ports[0]}  |  Rx2 = {ports[1]}")
    print(f"  Duration: {duration:.1f}s  |  Preparation Countdown: {prep_time}s")
    print("═" * 60)

    # Countdown
    if prep_time > 0:
        print(f"\n👉 Get into position! Recording starts in {prep_time} seconds...")
        for sec in range(prep_time, 0, -1):
            print(f"   >>> {sec} <<<")
            _beep_countdown()
            time.sleep(1.0)

    print("\n🔴 [RECORDING IN PROGRESS... PERFORM ACTIVITY NOW]")
    _beep_start()

    # Start recorders
    rec1 = SingleLinkRecorder("Rx1", ports[0], baud)
    rec2 = SingleLinkRecorder("Rx2", ports[1], baud)
    rec1.start()
    rec2.start()

    # Progress bar
    start_t = time.monotonic()
    while time.monotonic() - start_t < duration:
        elapsed = time.monotonic() - start_t
        pct = min(1.0, elapsed / duration)
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        sys.stdout.write(f"\r   [{bar}] {elapsed:.1f}s / {duration:.1f}s | Rx1: {len(rec1.rows)} pkts | Rx2: {len(rec2.rows)} pkts")
        sys.stdout.flush()
        time.sleep(0.1)

    print()
    rec1.stop()
    rec2.stop()
    rec1.join(timeout=1.5)
    rec2.join(timeout=1.5)
    _beep_stop()

    # Session directory
    base_out = out_dir or (_PROJECT_ROOT / "data" / "raw_dual")
    session_id = f"{activity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = base_out / activity / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Common IDs & Quality calculation
    common_ids = rec1.packet_ids.intersection(rec2.packet_ids)
    sync_rate = (len(common_ids) / max(len(rec1.packet_ids), len(rec2.packet_ids), 1)) * 100

    print("\n" + "─" * 60)
    print(f"✅ Recording Complete!")
    print(f"   Rx1 ({ports[0]}): {len(rec1.rows)} packets")
    print(f"   Rx2 ({ports[1]}): {len(rec2.rows)} packets")
    print(f"   Synchronized Frames (Common Tx IDs): {len(common_ids)} ({sync_rate:.1f}%)")
    print(f"   Saved to: {session_dir}")
    print("─" * 60)

    # Save rx1.csv
    if rec1.rows:
        n_sub = (len(rec1.rows[0]) - 4) // 2
        header = ["id", "timestamp", "rssi", "channel"] + [f"amplitude_{i}" for i in range(n_sub)] + [f"phase_{i}" for i in range(n_sub)]
        df1 = pd.DataFrame(rec1.rows, columns=header)
        df1.to_csv(session_dir / "rx1.csv", index=False)

    # Save rx2.csv
    if rec2.rows:
        n_sub = (len(rec2.rows[0]) - 4) // 2
        header = ["id", "timestamp", "rssi", "channel"] + [f"amplitude_{i}" for i in range(n_sub)] + [f"phase_{i}" for i in range(n_sub)]
        df2 = pd.DataFrame(rec2.rows, columns=header)
        df2.to_csv(session_dir / "rx2.csv", index=False)

    # Save meta.json
    meta = {
        "activity": activity,
        "session_id": session_id,
        "ports": list(ports),
        "baud_rate": baud,
        "duration_s": duration,
        "rx1_packets": len(rec1.rows),
        "rx2_packets": len(rec2.rows),
        "common_ids": len(common_ids),
        "sync_rate_pct": float(sync_rate),
        "recorded_at": datetime.now().isoformat(),
    }
    with open(session_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Dual-Receiver CSI Data Collection")
    parser.add_argument("--ports", type=str, default="COM3,COM11", help="Comma-separated COM ports (e.g. COM3,COM11)")
    parser.add_argument("--activity", type=str, default="walking", choices=_ALL_ACTIVITIES, help="Activity to collect")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to record")
    parser.add_argument("--prep", type=int, default=5, help="Preparation countdown in seconds")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate")
    args = parser.parse_args()

    port_list = [p.strip() for p in args.ports.split(",")]
    if len(port_list) != 2:
        print("❌ Error: Must specify exactly two ports, e.g. --ports COM3,COM11")
        return

    record_dual_session(
        ports=(port_list[0], port_list[1]),
        activity=args.activity,
        duration=args.duration,
        prep_time=args.prep,
        baud=args.baud,
    )


if __name__ == "__main__":
    main()
