#!/usr/bin/env python3
"""
visualize_csi.py — High-Performance Multi-Node CSI Visualisation
================================================================
Connects to the ESP32-S3 CSI receiver via serial.
Tracks multiple ESP8266 TX Beacons by MAC address and computes a 
"Movement Score" (variance of CSI amplitude) for each node.
"""

from __future__ import annotations
import sys
import argparse
import logging
import collections
import numpy as np

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from PyQt5.QtCore import pyqtSignal, QThread
except ImportError:
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from PySide6.QtCore import Signal as pyqtSignal, QThread

import pyqtgraph as pg
from parse_csi import CSIDataReader, extract_amplitude_phase

logger = logging.getLogger(__name__)

_DEFAULT_PORT = "COM3"
_DEFAULT_BAUD = 921600
_HISTORY_LEN = 100
_WINDOW_SIZE = 20  # Frames to compute variance over

class CSIReaderThread(QThread):
    data_ready = pyqtSignal(str, np.ndarray)

    def __init__(self, port: str, baud_rate: int):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.running = True

    def run(self):
        try:
            with CSIDataReader(port=self.port, baud_rate=self.baud_rate) as reader:
                while self.running:
                    pkt = reader.read_one()
                    if pkt is not None:
                        mac = pkt.get("mac", "unknown")
                        amp, _ = extract_amplitude_phase(pkt["raw_data"])
                        self.data_ready.emit(mac, amp)
        except Exception as e:
            logger.error(f"Serial read error: {e}")

    def stop(self):
        self.running = False
        self.wait()


class CSIVisualiser(QMainWindow):
    def __init__(self, port: str, baud: int):
        super().__init__()
        self.setWindowTitle("Multi-Node CSI Movement Coverage (Real-Time)")
        self.resize(1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        pg.setConfigOptions(antialias=True)
        
        # 1. Line Plot (Movement Score over time)
        self.plot_line = pg.PlotWidget(title="Movement Score (Variance) per Node")
        self.plot_line.setLabel("bottom", "Time (frames)")
        self.plot_line.setLabel("left", "Movement Score")
        self.plot_line.addLegend()
        self.plot_line.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_line)

        # State for multiple MACs
        self.mac_history = {}  # mac -> deque of amplitudes
        self.mac_score_history = {} # mac -> deque of movement scores
        self.curves = {}
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (0, 255, 255), (255, 0, 255)
        ]

        self.reader_thread = CSIReaderThread(port, baud)
        self.reader_thread.data_ready.connect(self.update_plots)
        self.reader_thread.start()

    def update_plots(self, mac: str, amp: np.ndarray):
        # Ignore non-target MACs if needed, or dynamically add them
        if mac not in self.mac_history:
            self.mac_history[mac] = collections.deque(maxlen=_WINDOW_SIZE)
            self.mac_score_history[mac] = collections.deque(
                [0]*_HISTORY_LEN, maxlen=_HISTORY_LEN
            )
            color = self.colors[len(self.curves) % len(self.colors)]
            curve_name = f"Node {mac[-5:]}"
            self.curves[mac] = self.plot_line.plot(pen=color, name=curve_name)
        
        self.mac_history[mac].append(amp)
        
        if len(self.mac_history[mac]) == _WINDOW_SIZE:
            # 1. Compute standard deviation (std) across the window
            mat = np.array(self.mac_history[mac])
            std_dev = np.std(mat, axis=0)
            
            # 2. Compute mean amplitude to normalize the score (Coefficient of Variation)
            mean_amp = np.mean(mat, axis=0)
            mean_amp[mean_amp < 1] = 1  # Avoid division by zero
            
            # 3. Normalized score: (Std / Mean) * 100
            normalized_var = (std_dev / mean_amp) * 100
            current_score = np.mean(normalized_var)
            
            # 4. Smooth the score using Exponential Moving Average (EMA)
            if len(self.mac_score_history[mac]) == 0 or np.all(np.array(self.mac_score_history[mac]) == 0):
                smoothed_score = current_score
            else:
                last_score = self.mac_score_history[mac][-1]
                smoothed_score = (current_score * 0.3) + (last_score * 0.7)
                
            self.mac_score_history[mac].append(smoothed_score)
            
            # Update plot
            self.curves[mac].setData(np.array(self.mac_score_history[mac]))

    def closeEvent(self, event):
        self.reader_thread.stop()
        event.accept()

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=_DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=_DEFAULT_BAUD)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = CSIVisualiser(args.port, args.baud)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
