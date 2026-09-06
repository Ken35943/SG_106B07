#!/usr/bin/env python3
"""
realtime_activity_demo.py — Live Human Activity Monitor (Walking vs Sitting)
=============================================================================
Demonstrates real-time WiFi CSI classification between Walking and Sitting Still.
Features:
- Big glowing activity status banner (Walking / Sitting)
- Live Motion Intensity Gauge (dB²)
- 10-Subcarrier real-time filtered waveform (0.5–40 Hz Causal SOS Filter)
- Supports live serial (COM3), CSV replay, and mock mode
"""

from __future__ import annotations
import sys
import os
import time
import argparse
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
        QWidget, QLabel, QProgressBar, QFrame
    )
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QFont
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
        QWidget, QLabel, QProgressBar, QFrame
    )
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QFont

import pyqtgraph as pg

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_csi import CSIDataReader, extract_amplitude_phase
from csi_dsp import (
    select_subcarriers, hampel_filter, design_bandpass_sos,
    CausalSOSFilter, ht40_htltf_layout, HT40_TOTAL_ENTRIES
)

logger = logging.getLogger(__name__)

_DEFAULT_PORT = "COM3"
_DEFAULT_BAUD = 921600
_HISTORY_LEN = 200
_NUM_SUBCARRIERS_TO_PLOT = 10

_PALETTE_10 = [
    "#00E5FF", "#E040FB", "#7C4DFF", "#00E676", "#FFEA00",
    "#FF6D00", "#FF1744", "#1DE9B6", "#2979FF", "#F50057",
]


class ActivityWorker(threading.Thread):
    """Processes streaming CSI packets, applies causal DSP, and classifies activity."""

    def __init__(self, port: str = _DEFAULT_PORT, baud_rate: int = _DEFAULT_BAUD,
                 replay: Optional[str] = None, mock: bool = False):
        super().__init__(daemon=True)
        self.port = port
        self.baud_rate = baud_rate
        self.replay_path = replay
        self.mock = mock
        self.running = True
        self.lock = threading.Lock()

        # Shared state
        self.latest_raw_amp = None
        self.history_filtered = np.zeros((_HISTORY_LEN, _NUM_SUBCARRIERS_TO_PLOT), dtype=np.float32)
        self.motion_energy = 0.0
        self.is_walking = False
        self.walking_prob = 0.0
        self.packet_rate = 0.0
        self.has_new_data = False

        # DSP Setup
        self.sos = design_bandpass_sos(100.0, 0.5, 40.0, 4)
        self.filter = CausalSOSFilter(self.sos)
        self.sub_cols = None
        self.plot_indices = None

        self._pkt_count = 0
        self._last_rate_time = time.time()

    def run(self):
        if self.replay_path:
            self._run_replay()
        elif self.mock:
            self._run_mock()
        else:
            self._run_serial()

    def _process_packet(self, amp: np.ndarray):
        num_sub = len(amp)
        if self.sub_cols is None:
            self.sub_cols, _ = select_subcarriers(amp[None, :])
            if num_sub == HT40_TOTAL_ENTRIES:
                cols, _ = ht40_htltf_layout()
                step = len(cols) // _NUM_SUBCARRIERS_TO_PLOT
                self.plot_indices = np.arange(0, len(cols), step)[:_NUM_SUBCARRIERS_TO_PLOT]
            else:
                self.plot_indices = np.linspace(0, len(self.sub_cols) - 1, _NUM_SUBCARRIERS_TO_PLOT, dtype=int)

        # Occupied carriers + Log transform
        x_occ = amp[self.sub_cols]
        x_log = 20.0 * np.log10(x_occ + 1.0)

        # Filter step (causal SOS)
        filtered = self.filter.process(x_log[None, :])[0]

        # Rate counter
        self._pkt_count += 1
        now = time.time()
        if now - self._last_rate_time >= 1.0:
            self.packet_rate = self._pkt_count / (now - self._last_rate_time)
            self._pkt_count = 0
            self._last_rate_time = now

        # Update buffer
        with self.lock:
            self.history_filtered = np.roll(self.history_filtered, -1, axis=0)
            self.history_filtered[-1, :] = filtered[self.plot_indices]

            # Motion energy: variance across the recent 60 frames (~0.6s)
            recent = self.history_filtered[-60:]
            var_sub = np.var(recent, axis=0)
            self.motion_energy = float(np.mean(var_sub))

            # Dynamic classification: Walking vs Sitting
            # Baseline quiet sitting in room is ~0.15–0.40, walking is >0.52
            energy_score = np.clip((self.motion_energy - 0.25) / 0.52, 0.0, 1.0)
            self.walking_prob = float(energy_score)
            self.is_walking = self.walking_prob > 0.50
            self.has_new_data = True

    def _run_replay(self):
        logger.info(f"Replaying: {self.replay_path}")
        df = pd.read_csv(self.replay_path)
        amp_cols = [c for c in df.columns if c.startswith("amplitude_")]
        data = df[amp_cols].values.astype(np.float64)
        n = len(data)
        idx = 0
        while self.running:
            self._process_packet(data[idx])
            idx = (idx + 1) % n
            time.sleep(0.01)

    def _run_mock(self):
        logger.info("Running mock generator...")
        t = 0
        while self.running:
            is_walk_phase = (t // 500) % 2 == 1
            n_sub = 190
            base = 25.0
            if is_walk_phase:
                noise = np.random.normal(0, 4.0, n_sub) * np.sin(t * 0.1)
            else:
                noise = np.random.normal(0, 0.4, n_sub)
            amp = np.clip(base + noise, 1.0, 150.0).astype(np.float64)
            self._process_packet(amp)
            t += 1
            time.sleep(0.01)

    def _run_serial(self):
        logger.info(f"Connecting to receiver on {self.port} @ {self.baud_rate}...")
        try:
            with CSIDataReader(port=self.port, baud_rate=self.baud_rate) as reader:
                if reader._serial:
                    reader._serial.reset_input_buffer()
                target_mac = None
                while self.running:
                    if reader._serial and reader._serial.in_waiting > 4096:
                        reader._serial.reset_input_buffer()
                    pkt = reader.read_one()
                    if pkt is not None:
                        mac = pkt.get("mac", "unknown")
                        if target_mac is None:
                            target_mac = mac
                            logger.info(f"Locked MAC: {target_mac}")
                        if mac == target_mac:
                            amp, _ = extract_amplitude_phase(pkt["raw_data"])
                            self._process_packet(amp)
        except Exception as e:
            logger.error(f"Serial worker error: {e}")

    def stop(self):
        self.running = False


class ActivityMonitorWindow(QMainWindow):
    def __init__(self, port: str = _DEFAULT_PORT, baud: int = _DEFAULT_BAUD,
                 replay: Optional[str] = None, mock: bool = False):
        super().__init__()
        mode_title = f"Replay ({Path(replay).name})" if replay else ("Mock" if mock else f"Live ({port})")
        self.setWindowTitle(f"ESP32-S3 WiFi CSI Real-Time Activity Classifier — [{mode_title}]")
        self.resize(1050, 750)
        self.setStyleSheet("background-color: #0d1117; color: #c9d1d9;")

        pg.setConfigOptions(useOpenGL=False, antialias=False)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # ── 1. Top Status Banner ─────────────────────────────────────────────
        self.banner = QFrame()
        self.banner.setFrameShape(QFrame.StyledPanel)
        banner_layout = QVBoxLayout(self.banner)
        banner_layout.setContentsMargins(16, 16, 16, 16)

        self.label_status = QLabel("INITIALIZING...")
        self.label_status.setAlignment(Qt.AlignCenter)
        font = QFont("Consolas", 26, QFont.Bold)
        self.label_status.setFont(font)
        banner_layout.addWidget(self.label_status)

        self.label_sub = QLabel("Detecting WiFi multi-path perturbations...")
        self.label_sub.setAlignment(Qt.AlignCenter)
        self.label_sub.setStyleSheet("font-size: 13px; color: #8b949e;")
        banner_layout.addWidget(self.label_sub)
        layout.addWidget(self.banner)

        # ── 2. Gauge & Stats Row ────────────────────────────────────────────
        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(12)

        lbl_gauge = QLabel("Motion Energy:")
        lbl_gauge.setStyleSheet("font-weight: bold; font-size: 13px;")
        gauge_row.addWidget(lbl_gauge)

        self.progress_energy = QProgressBar()
        self.progress_energy.setRange(0, 100)
        self.progress_energy.setValue(0)
        self.progress_energy.setTextVisible(True)
        self.progress_energy.setFixedHeight(24)
        self.progress_energy.setStyleSheet("""
            QProgressBar {
                border: 1px solid #30363d;
                border-radius: 4px;
                background-color: #161b22;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00E676, stop:0.6 #00E5FF, stop:1 #FF1744);
                border-radius: 3px;
            }
        """)
        gauge_row.addWidget(self.progress_energy, stretch=3)

        self.label_stats = QLabel("Rate: 0 Hz | Energy: 0.00 dB²")
        self.label_stats.setStyleSheet("font-family: Consolas; font-size: 13px; color: #58a6ff;")
        gauge_row.addWidget(self.label_stats, stretch=1)
        layout.addLayout(gauge_row)

        # ── 3. Waveform Plot ────────────────────────────────────────────────
        self.plot_widget = pg.PlotWidget(title="Real-Time Filtered CSI Subcarrier Waveforms (0.5–40 Hz Causal)")
        self.plot_widget.setLabel("bottom", "Time (frames @ 100 Hz)")
        self.plot_widget.setLabel("left", "Amplitude Perturbation (dB)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setXRange(0, _HISTORY_LEN, padding=0)
        self.plot_widget.setYRange(-6, 6, padding=0.1)
        self.plot_widget.setMouseEnabled(x=False, y=True)
        layout.addWidget(self.plot_widget, stretch=3)

        self.curves = []
        for i in range(_NUM_SUBCARRIERS_TO_PLOT):
            pen = pg.mkPen(color=_PALETTE_10[i % len(_PALETTE_10)], width=1.6)
            curve = self.plot_widget.plot(pen=pen)
            self.curves.append(curve)

        # Start worker thread
        self.worker = ActivityWorker(port=port, baud_rate=baud, replay=replay, mock=mock)
        self.worker.start()

        # Update Timer (60 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(16)

    def update_ui(self):
        if not self.worker.has_new_data:
            return

        with self.worker.lock:
            data = self.worker.history_filtered.copy()
            energy = self.worker.motion_energy
            is_walking = self.worker.is_walking
            prob = self.worker.walking_prob
            rate = self.worker.packet_rate
            self.worker.has_new_data = False

        # Update Banner
        if is_walking:
            self.banner.setStyleSheet("""
                QFrame {
                    background-color: #052636;
                    border: 2px solid #00E5FF;
                    border-radius: 8px;
                }
            """)
            self.label_status.setText("🚶  WALKING / ACTIVE MOTION")
            self.label_status.setStyleSheet("color: #00E5FF;")
            self.label_sub.setText(f"Active movement detected — Motion Probability: {prob*100:.0f}%")
        else:
            self.banner.setStyleSheet("""
                QFrame {
                    background-color: #092615;
                    border: 2px solid #00E676;
                    border-radius: 8px;
                }
            """)
            self.label_status.setText("🪑  SITTING STILL / QUIET")
            self.label_status.setStyleSheet("color: #00E676;")
            self.label_sub.setText(f"Person stationary — Motion Probability: {prob*100:.0f}%")

        # Update Gauge
        gauge_val = int(np.clip(energy / 1.0 * 100, 0, 100))
        self.progress_energy.setValue(gauge_val)
        self.progress_energy.setFormat(f"Energy: {energy:.2f} dB² ({gauge_val}%)")
        self.label_stats.setText(f"Rate: {rate:.0f} Hz | Energy: {energy:.2f} dB²")

        # Update Waveforms
        for i in range(_NUM_SUBCARRIERS_TO_PLOT):
            self.curves[i].setData(data[:, i])

    def closeEvent(self, event):
        self.timer.stop()
        self.worker.stop()
        event.accept()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Real-Time CSI Activity Monitor")
    parser.add_argument("--port", type=str, default=_DEFAULT_PORT, help="Serial port")
    parser.add_argument("--baud", type=int, default=_DEFAULT_BAUD, help="Baud rate")
    parser.add_argument("--replay", type=str, default=None, help="Replay CSV path")
    parser.add_argument("--mock", action="store_true", help="Run in mock simulation mode")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = ActivityMonitorWindow(port=args.port, baud=args.baud, replay=args.replay, mock=args.mock)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
