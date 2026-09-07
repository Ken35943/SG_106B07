#!/usr/bin/env python3
"""
visualize_dual.py — Real-Time Dual-Receiver CSI Waveform Monitor
=================================================================
Visualizes real-time CSI subcarrier waveforms simultaneously from two receivers
(e.g. COM3 & COM11), with cross-link motion energy and sync statistics.
"""

from __future__ import annotations
import sys
import time
import argparse
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel
    from PySide6.QtCore import QTimer, Qt
except ImportError:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel
    from PyQt5.QtCore import QTimer, Qt

import pyqtgraph as pg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dual_link import DualCSIReader, LinkWorker
from csi_dsp import ht40_htltf_layout, HT40_TOTAL_ENTRIES

logger = logging.getLogger(__name__)

_HISTORY_LEN = 200
_NUM_SUBCARRIERS_TO_PLOT = 8

_PALETTE_8 = [
    "#00E5FF", "#E040FB", "#7C4DFF", "#00E676",
    "#FFEA00", "#FF6D00", "#FF1744", "#2979FF"
]


class DualVisualizerWorker(threading.Thread):
    def __init__(self, ports: tuple[str, str], baud: int = 921600):
        super().__init__(daemon=True)
        self.ports = ports
        self.baud = baud
        self.running = True
        self.lock = threading.Lock()

        self.history_rx1 = np.zeros((_HISTORY_LEN, _NUM_SUBCARRIERS_TO_PLOT), dtype=np.float32)
        self.history_rx2 = np.zeros((_HISTORY_LEN, _NUM_SUBCARRIERS_TO_PLOT), dtype=np.float32)
        self.rate_rx1 = 0.0
        self.rate_rx2 = 0.0
        self.sync_fps = 0.0
        self.has_new_data = False
        self.plot_indices = None

    def run(self):
        try:
            with DualCSIReader(ports=self.ports, baud=self.baud) as reader:
                cols, _ = ht40_htltf_layout()
                step = len(cols) // _NUM_SUBCARRIERS_TO_PLOT
                self.plot_indices = cols[::step][:_NUM_SUBCARRIERS_TO_PLOT]

                count = 0
                t0 = time.time()

                while self.running:
                    res = reader.read_fused(timeout=0.03)
                    if res is not None:
                        t_sec, fused_amp = res  # shape: (2, 190)
                        amp1 = fused_amp[0]
                        amp2 = fused_amp[1]

                        with self.lock:
                            self.history_rx1 = np.roll(self.history_rx1, -1, axis=0)
                            self.history_rx1[-1, :] = amp1[self.plot_indices]

                            self.history_rx2 = np.roll(self.history_rx2, -1, axis=0)
                            self.history_rx2[-1, :] = amp2[self.plot_indices]

                            count += 1
                            now = time.time()
                            if now - t0 >= 1.0:
                                self.sync_fps = count / (now - t0)
                                count = 0
                                t0 = now
                                self.rate_rx1 = reader.workers[0].total_packets
                                self.rate_rx2 = reader.workers[1].total_packets

                            self.has_new_data = True
        except Exception as e:
            logger.error("Dual visualizer reader error: %s", e)

    def stop(self):
        self.running = False


class DualVisualizerWindow(QMainWindow):
    def __init__(self, ports: tuple[str, str], baud: int = 921600):
        super().__init__()
        self.setWindowTitle(f"Dual-Receiver CSI Monitor — Rx1: {ports[0]} | Rx2: {ports[1]}")
        self.resize(1100, 800)
        self.setStyleSheet("background-color: #0d1117; color: #c9d1d9;")

        pg.setConfigOptions(useOpenGL=False, antialias=False)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header info
        hdr = QHBoxLayout()
        self.lbl_info = QLabel(f"📡 Dual-Link Ingestion: Rx1 ({ports[0]}) & Rx2 ({ports[1]}) | Sync Rate: 0 Hz")
        self.lbl_info.setStyleSheet("font-family: Consolas; font-size: 14px; font-weight: bold; color: #58a6ff;")
        hdr.addWidget(self.lbl_info)
        layout.addLayout(hdr)

        # 1. Plot Rx1
        self.plot1 = pg.PlotWidget(title=f"Receiver 1 ({ports[0]}) — Subcarrier Amplitudes")
        self.plot1.setLabel("left", "Amplitude")
        self.plot1.setXRange(0, _HISTORY_LEN, padding=0)
        self.plot1.setYRange(0, 100, padding=0.05)
        self.plot1.showGrid(x=True, y=True, alpha=0.25)
        self.plot1.setMouseEnabled(x=False, y=True)
        layout.addWidget(self.plot1)

        self.curves1 = []
        for i in range(_NUM_SUBCARRIERS_TO_PLOT):
            pen = pg.mkPen(color=_PALETTE_8[i % len(_PALETTE_8)], width=1.8)
            self.curves1.append(self.plot1.plot(pen=pen))

        # 2. Plot Rx2
        self.plot2 = pg.PlotWidget(title=f"Receiver 2 ({ports[1]}) — Subcarrier Amplitudes")
        self.plot2.setLabel("bottom", "Time (frames)")
        self.plot2.setLabel("left", "Amplitude")
        self.plot2.setXRange(0, _HISTORY_LEN, padding=0)
        self.plot2.setYRange(0, 100, padding=0.05)
        self.plot2.showGrid(x=True, y=True, alpha=0.25)
        self.plot2.setMouseEnabled(x=False, y=True)
        layout.addWidget(self.plot2)

        self.curves2 = []
        for i in range(_NUM_SUBCARRIERS_TO_PLOT):
            pen = pg.mkPen(color=_PALETTE_8[i % len(_PALETTE_8)], width=1.8)
            self.curves2.append(self.plot2.plot(pen=pen))

        self.worker = DualVisualizerWorker(ports=ports, baud=baud)
        self.worker.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(16)

    def update_ui(self):
        if not self.worker.has_new_data:
            return

        with self.worker.lock:
            data1 = self.worker.history_rx1.copy()
            data2 = self.worker.history_rx2.copy()
            fps = self.worker.sync_fps
            r1 = self.worker.rate_rx1
            r2 = self.worker.rate_rx2
            self.worker.has_new_data = False

        self.lbl_info.setText(
            f"📡 Dual-Link Ingestion | Rx1 ({self.worker.ports[0]}): {r1} pkts | "
            f"Rx2 ({self.worker.ports[1]}): {r2} pkts | Synchronized: {fps:.0f} Hz"
        )

        for i in range(_NUM_SUBCARRIERS_TO_PLOT):
            self.curves1[i].setData(data1[:, i])
            self.curves2[i].setData(data2[:, i])

    def closeEvent(self, event):
        self.timer.stop()
        self.worker.stop()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Dual-Receiver CSI Real-Time Visualizer")
    parser.add_argument("--ports", type=str, default="COM3,COM11", help="Comma-separated COM ports")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate")
    args = parser.parse_args()

    port_list = [p.strip() for p in args.ports.split(",")]
    app = QApplication(sys.argv)
    win = DualVisualizerWindow(ports=(port_list[0], port_list[1]), baud=args.baud)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
