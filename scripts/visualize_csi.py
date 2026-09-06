#!/usr/bin/env python3
"""
visualize_csi.py — Ultra-Smooth Zero-Latency CSI Visualisation
================================================================
Connects to the ESP32-S3 CSI receiver via serial.
Visualizes the raw amplitude of 10 active subcarriers over time,
and an instant snapshot of all subcarriers using an ultra-fast stem plot.

Architecture:
- Background serial reader thread: continuous non-blocking ingest with backlog purge.
- Shared ring buffer: thread-safe preallocated numpy array with minimal copying.
- GUI thread: locked 60 FPS QTimer with native Qt QPainter rendering (150+ FPS capable).
"""

from __future__ import annotations
import sys
import argparse
import logging
import threading
import numpy as np

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from PySide6.QtCore import QTimer, Qt
except ImportError:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from PyQt5.QtCore import QTimer, Qt

import pyqtgraph as pg
from parse_csi import CSIDataReader, extract_amplitude_phase

logger = logging.getLogger(__name__)

_DEFAULT_PORT = "COM3"
_DEFAULT_BAUD = 921600
_HISTORY_LEN = 200
_NUM_SUBCARRIERS_TO_PLOT = 10

# 10 High-contrast neon colors matching oscilloscope theme
_PALETTE_10 = [
    "#00E5FF",  # Neon Cyan
    "#E040FB",  # Neon Magenta
    "#7C4DFF",  # Vivid Purple
    "#00E676",  # Lime Green
    "#FFEA00",  # Bright Yellow
    "#FF6D00",  # Neon Orange
    "#FF1744",  # Neon Red
    "#1DE9B6",  # Bright Teal
    "#2979FF",  # Electric Blue
    "#F50057",  # Deep Pink
]

# Active subcarrier indices for 64-subcarrier HT20 (avoiding null/guard and DC subcarriers)
# 6..31 (lower band), 33..58 (upper band)
_DEFAULT_ACTIVE_INDICES = np.array([7, 12, 17, 22, 27, 34, 39, 44, 49, 54], dtype=int)


import time
import pandas as pd
from pathlib import Path

# Subcarrier selection
from csi_dsp import ht40_htltf_layout, HT40_TOTAL_ENTRIES


class CSIReaderThread(threading.Thread):
    """Dedicated high-speed thread for streaming CSI data (live serial, replay, or mock)."""

    def __init__(self, port: str = _DEFAULT_PORT, baud_rate: int = _DEFAULT_BAUD,
                 replay: Optional[str] = None, mock: bool = False,
                 history_len: int = _HISTORY_LEN):
        super().__init__(daemon=True)
        self.port = port
        self.baud_rate = baud_rate
        self.replay_path = replay
        self.mock = mock
        self.history_len = history_len
        self.running = True
        self.target_mac = None

        # Thread-safe shared buffers
        self.lock = threading.Lock()
        self.history_buffer = None  # shape: (history_len, num_subcarriers)
        self.latest_amp = None
        self.has_new_data = False
        self.packet_count = 0

    def _push_amp(self, amp: np.ndarray):
        num_sub = len(amp)
        with self.lock:
            if self.history_buffer is None or self.history_buffer.shape[1] != num_sub:
                self.history_buffer = np.zeros((self.history_len, num_sub), dtype=np.float32)

            self.history_buffer = np.roll(self.history_buffer, -1, axis=0)
            self.history_buffer[-1, :] = amp
            self.latest_amp = amp
            self.has_new_data = True
            self.packet_count += 1

    def run(self):
        if self.replay_path:
            self._run_replay()
        elif self.mock:
            self._run_mock()
        else:
            self._run_serial()

    def _run_replay(self):
        logger.info(f"Replaying CSI data from {self.replay_path}...")
        df = pd.read_csv(self.replay_path)
        amp_cols = [c for c in df.columns if c.startswith("amplitude_")]
        if not amp_cols:
            logger.error("No amplitude columns found in replay CSV!")
            return
        data = df[amp_cols].values.astype(np.float32)
        n_rows = len(data)
        idx = 0
        while self.running:
            self._push_amp(data[idx])
            idx = (idx + 1) % n_rows
            time.sleep(0.01)  # ~100 Hz replay rate

    def _run_mock(self):
        logger.info("Running mock CSI generator at 100 Hz...")
        t = 0
        n_sub = 190
        while self.running:
            base = 25.0 + 10.0 * np.sin(np.linspace(0, 4 * np.pi, n_sub) + t * 0.05)
            noise = np.random.normal(0, 1.5, n_sub)
            amp = np.clip(base + noise, 0, 150).astype(np.float32)
            self._push_amp(amp)
            t += 1
            time.sleep(0.01)

    def _run_serial(self):
        logger.info(f"Connecting to CSI receiver on {self.port} @ {self.baud_rate}...")
        try:
            with CSIDataReader(port=self.port, baud_rate=self.baud_rate) as reader:
                if reader._serial:
                    reader._serial.reset_input_buffer()

                while self.running:
                    if reader._serial and reader._serial.in_waiting > 4096:
                        reader._serial.reset_input_buffer()

                    pkt = reader.read_one()
                    if pkt is not None:
                        mac = pkt.get("mac", "unknown")
                        if self.target_mac is None:
                            self.target_mac = mac
                            logger.info(f"Locked onto Sender MAC: {self.target_mac}")

                        if mac == self.target_mac:
                            amp, _ = extract_amplitude_phase(pkt["raw_data"])
                            self._push_amp(amp)
        except Exception as e:
            logger.error(f"Serial reader exception: {e}")

    def stop(self):
        self.running = False


class CSIVisualiser(QMainWindow):
    def __init__(self, port: str = _DEFAULT_PORT, baud: int = _DEFAULT_BAUD,
                 replay: Optional[str] = None, mock: bool = False):
        super().__init__()
        mode_str = f"Replay: {Path(replay).name}" if replay else ("Mock Mode" if mock else f"Live: {port}")
        self.setWindowTitle(f"CSI Raw Subcarrier Amplitudes (Real-Time 60 FPS) — [{mode_str}]")
        self.resize(1100, 850)

        # Native C++ QPainter (150+ FPS capable)
        pg.setConfigOptions(useOpenGL=False, antialias=False)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 1. Line Plot (Top-10 Subcarriers over time)
        self.plot_line = pg.PlotWidget(title="Top-10 Subcarrier Amplitudes")
        self.plot_line.setLabel("bottom", "Time (frames)")
        self.plot_line.setLabel("left", "Amplitude")
        self.plot_line.showGrid(x=True, y=True, alpha=0.25)
        self.plot_line.setXRange(0, _HISTORY_LEN, padding=0)
        self.plot_line.setYRange(0, 50, padding=0.05)
        self.plot_line.setMouseEnabled(x=False, y=True)
        layout.addWidget(self.plot_line, stretch=3)

        # 2. Bar Plot (Current Amplitude Snapshot - Stem Bar Style)
        self.plot_bar = pg.PlotWidget(title="Current Amplitude Snapshot (All Subcarriers)")
        self.plot_bar.setLabel("bottom", "Subcarrier Index")
        self.plot_bar.setLabel("left", "Amplitude")
        self.plot_bar.showGrid(x=False, y=True, alpha=0.25)
        self.plot_bar.setYRange(0, 160, padding=0.02)
        self.plot_bar.setMouseEnabled(x=False, y=True)
        layout.addWidget(self.plot_bar, stretch=2)

        # Pre-create curves for the 10 subcarriers
        self.curves = []
        for i in range(_NUM_SUBCARRIERS_TO_PLOT):
            pen = pg.mkPen(color=_PALETTE_10[i % len(_PALETTE_10)], width=1.8)
            curve = self.plot_line.plot(pen=pen)
            self.curves.append(curve)

        # Snapshot bar plot using connect='pairs' for lightning fast single-draw rendering
        self.bar_curve = self.plot_bar.plot(
            pen=pg.mkPen(color='#00E5FF', width=2.5),
            connect='pairs'
        )

        # Pre-allocated arrays for snapshot plot
        self.x_bars = None
        self.y_bars = None
        self.subcarrier_indices = None

        # Start data acquisition thread
        self.reader_thread = CSIReaderThread(port, baud, replay=replay, mock=mock, history_len=_HISTORY_LEN)
        self.reader_thread.start()

        # GUI Update Timer locked to 60 FPS (16 ms)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

    def update_frame(self):
        if not self.reader_thread.has_new_data:
            return

        with self.reader_thread.lock:
            if self.reader_thread.history_buffer is None or self.reader_thread.latest_amp is None:
                return
            curr_history = self.reader_thread.history_buffer.copy()
            curr_amp = self.reader_thread.latest_amp.copy()
            self.reader_thread.has_new_data = False

        num_sub = curr_history.shape[1]

        # Initialize subcarrier indices once
        if self.subcarrier_indices is None:
            if num_sub == 64:
                self.subcarrier_indices = _DEFAULT_ACTIVE_INDICES
            elif num_sub == HT40_TOTAL_ENTRIES:
                cols, _ = ht40_htltf_layout()
                step = len(cols) // _NUM_SUBCARRIERS_TO_PLOT
                self.subcarrier_indices = cols[::step][:_NUM_SUBCARRIERS_TO_PLOT]
            else:
                self.subcarrier_indices = np.linspace(0, num_sub - 1, _NUM_SUBCARRIERS_TO_PLOT, dtype=int)

        # Update the 10 line plots
        for i, idx in enumerate(self.subcarrier_indices):
            self.curves[i].setData(curr_history[:, idx])

        # Update snapshot bar plot
        if self.x_bars is None or len(self.x_bars) != num_sub * 2:
            self.x_bars = np.repeat(np.arange(num_sub), 2)
            self.y_bars = np.zeros(num_sub * 2, dtype=np.float32)
            self.plot_bar.setXRange(0, num_sub, padding=0.01)

        self.y_bars[1::2] = curr_amp
        self.bar_curve.setData(self.x_bars, self.y_bars)

    def closeEvent(self, event):
        self.timer.stop()
        self.reader_thread.stop()
        event.accept()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Real-Time CSI Visualisation")
    parser.add_argument("--port", type=str, default=_DEFAULT_PORT, help="Serial port of receiver")
    parser.add_argument("--baud", type=int, default=_DEFAULT_BAUD, help="Baud rate (default: 921600)")
    parser.add_argument("--replay", type=str, default=None, help="Path to CSV file to replay")
    parser.add_argument("--mock", action="store_true", help="Run with simulated CSI frames")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = CSIVisualiser(args.port, args.baud, replay=args.replay, mock=args.mock)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
