#!/usr/bin/env python3
"""
visualize_heatmap.py — Real-Time 2D Spatial Heatmap for Multi-Node CSI
========================================================================
Reads from both COM3 and COM5 simultaneously.
Maps the Movement Score of each ESP8266 onto a virtual 2D room layout.
Generates a dynamic glowing heatmap to visualize physical movement in real-time.
"""

from __future__ import annotations
import sys
import argparse
import logging
import collections
import numpy as np

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from PyQt5.QtCore import pyqtSignal, QThread, QTimer
except ImportError:
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from PySide6.QtCore import Signal as pyqtSignal, QThread, QTimer

import pyqtgraph as pg
from parse_csi import CSIDataReader, extract_amplitude_phase

logger = logging.getLogger(__name__)

# Config
WINDOW_SIZE = 20
RESOLUTION = 100  # 100x100 grid for the heatmap

# Virtual Room Layout (X, Y) - Normalized between 0 and 100
# Estimated from CSI Amplitude/RSSI
RX_POSITIONS = {
    "COM5": (25, 50), # RX 2: Left
    "COM3": (75, 50), # RX 1: Right
}

NODE_POSITIONS = {
    "1a:00:00:00:00:01": (85, 50),  # TX 1: Very close to Right (COM3)
    "1a:00:00:00:00:02": (15, 90),  # TX 2: Far away, closer to Left (COM5)
    "1a:00:00:00:00:03": (65, 75),  # TX 3: Mid distance, closer to Right (COM3)
    "1a:00:00:00:00:04": (35, 20),  # TX 4: Mid distance, closer to Left (COM5)
    "1a:00:00:00:00:05": (85, 85),  # TX 5: Top-Right corner (between 1 and 3)
    "1a:00:00:00:00:06": (50, 90),  # TX 6: (Reserved placeholder)
}

class CSIReaderThread(QThread):
    data_ready = pyqtSignal(str, str, np.ndarray) # port, mac, amplitude

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
                        self.data_ready.emit(self.port, mac, amp)
        except Exception as e:
            logger.error(f"Error on {self.port}: {e}")

    def stop(self):
        self.running = False
        self.wait()

class HeatmapVisualiser(QMainWindow):
    def __init__(self, port1: str, port2: str, baud: int):
        super().__init__()
        self.setWindowTitle("CSI 2D Spatial Coverage Heatmap")
        self.resize(800, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Pyqtgraph plot
        self.plot = pg.PlotWidget(title="Live Room Movement Heatmap")
        self.plot.setAspectLocked(True)
        self.plot.setXRange(0, 100)
        self.plot.setYRange(0, 100)
        layout.addWidget(self.plot)

        # Heatmap Image Item
        self.img = pg.ImageItem()
        # Colormap: Black -> Blue -> Red -> Yellow -> White (Heat)
        pos = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        color = np.array([
            [0, 0, 0, 255],
            [0, 0, 100, 255],
            [255, 0, 0, 255],
            [255, 255, 0, 255],
            [255, 255, 255, 255]
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.img.setLookupTable(cmap.getLookupTable())
        self.plot.addItem(self.img)

        # Draw Node Positions
        scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 200))
        spots = []
        # Add TX
        for mac, (x, y) in NODE_POSITIONS.items():
            spots.append({'pos': (x, y), 'data': 1})
            text = pg.TextItem(f"TX {mac[-2:]}", anchor=(0.5, -0.5), color=(255,255,255))
            text.setPos(x, y)
            self.plot.addItem(text)
        # Add RX
        for rx, (x, y) in RX_POSITIONS.items():
            spots.append({'pos': (x, y), 'data': 1, 'brush': pg.mkBrush(0, 255, 0, 200)})
            text = pg.TextItem(f"RX ({rx})", anchor=(0.5, -0.5), color=(0,255,0))
            text.setPos(x, y)
            self.plot.addItem(text)
        scatter.addPoints(spots)
        self.plot.addItem(scatter)

        # State mapping
        self.link_history = {} # (port, mac) -> deque
        self.link_score = {}   # (port, mac) -> current score

        # X/Y grid for heatmap calculation
        self.x_grid, self.y_grid = np.meshgrid(np.arange(RESOLUTION), np.arange(RESOLUTION))

        # Start Threads
        self.threads = []
        for port in [port1, port2]:
            t = CSIReaderThread(port, baud)
            t.data_ready.connect(self.process_data)
            self.threads.append(t)
            t.start()

        # Render Timer (30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_heatmap)
        self.timer.start(33)

    def process_data(self, port: str, mac: str, amp: np.ndarray):
        link = (port, mac)
        if link not in self.link_history:
            self.link_history[link] = collections.deque(maxlen=WINDOW_SIZE)
            self.link_score[link] = 0.0

        self.link_history[link].append(amp)
        
        if len(self.link_history[link]) == WINDOW_SIZE:
            mat = np.array(self.link_history[link])
            std_dev = np.std(mat, axis=0)
            mean_amp = np.mean(mat, axis=0)
            mean_amp[mean_amp < 1] = 1
            
            score = np.mean((std_dev / mean_amp) * 100)
            
            # EMA Smoothing
            last_score = self.link_score[link]
            self.link_score[link] = (score * 0.4) + (last_score * 0.6)

    def update_heatmap(self):
        heatmap = np.zeros((RESOLUTION, RESOLUTION))
        
        # Calculate heat blobs for each active link
        for (port, mac), score in self.link_score.items():
            if score < 1.0: # Noise floor
                continue
                
            tx_pos = NODE_POSITIONS.get(mac)
            rx_pos = RX_POSITIONS.get(port)
            
            if tx_pos and rx_pos:
                # The activity is mostly detected in the path between TX and RX
                mid_x = (tx_pos[0] + rx_pos[0]) / 2
                mid_y = (tx_pos[1] + rx_pos[1]) / 2
                
                # Add a Gaussian blob
                # Spread (sigma) defines how wide the heat area is
                sigma = 15.0
                intensity = min(max(score * 3, 0), 255) # Scale score to visual intensity
                
                blob = intensity * np.exp(
                    -(((self.x_grid - mid_y)**2 + (self.y_grid - mid_x)**2) / (2.0 * sigma**2))
                )
                heatmap += blob

        # Cap max value
        heatmap = np.clip(heatmap, 0, 255)
        
        # Transpose needed for pyqtgraph image orientation
        self.img.setImage(heatmap.T, autoLevels=False, levels=(0, 255))

    def closeEvent(self, event):
        for t in self.threads:
            t.stop()
        event.accept()

def main():
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    window = HeatmapVisualiser("COM3", "COM5", 921600)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
