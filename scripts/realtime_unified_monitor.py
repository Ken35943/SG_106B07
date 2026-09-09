#!/usr/bin/env python3
"""
realtime_unified_monitor.py — Unified CSI Real-Time Activity & Fall Detection Dashboard
========================================================================================
All-in-one 60 FPS interactive dashboard for ESP32-S3 WiFi CSI sensing.
Combines:
  1. Activity Classification: SITTING STILL (🧘) vs. WALKING / ACTIVE MOTION (🚶)
  2. Deep Learning Fall Detection: CNN-BiLSTM-Attention inference on sliding windows (🚨)
  3. Real-Time Telemetry: Motion Energy Gauge (dB²) and AI Fall Risk Gauge (%)
  4. Real-Time Waveform Oscilloscope: 10 filtered subcarrier streams (0.5–40 Hz Causal SOS)
  5. Live Event History Log: Timestamped transitions and emergency alert alarms

Usage:
    python scripts/realtime_unified_monitor.py --port COM3
    python scripts/realtime_unified_monitor.py --replay data/raw/walking/sample_20260715_192540.csv
    python scripts/realtime_unified_monitor.py --replay data/raw/fall_forward/sample_20260909_200400.csv
    python scripts/realtime_unified_monitor.py --mock
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import pickle
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# GUI Framework: PySide6 preferred, fallback to PyQt5
try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QProgressBar,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QColor, QFont
    from PyQt5.QtWidgets import (
        QApplication,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QProgressBar,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

import pyqtgraph as pg

# Local imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(_PROJECT_ROOT / "model"))

from cnn_lstm import CSIFallDetector
from csi_dsp import (
    CausalSOSFilter,
    HT40_TOTAL_ENTRIES,
    StreamingResampler,
    design_bandpass_sos,
    ht40_htltf_layout,
    select_subcarriers,
)
from dual_link import AGCFaultGuard
from parse_csi import CSIDataReader, extract_amplitude_phase

logger = logging.getLogger("UNIFIED_MONITOR")

_DEFAULT_PORT = "COM3"
_DEFAULT_BAUD = 921600
_HISTORY_LEN = 200
_NUM_SUBCARRIERS_TO_PLOT = 10

_PALETTE_10 = [
    "#00E5FF", "#E040FB", "#7C4DFF", "#00E676", "#FFEA00",
    "#FF6D00", "#FF1744", "#1DE9B6", "#2979FF", "#F50057",
]


# ═════════════════════════════════════════════════════════════════════════════
# Causal Streaming Hampel Filter
# ═════════════════════════════════════════════════════════════════════════════

class StreamingHampel:
    """Causal streaming Hampel outlier suppressor with bounded delay."""

    def __init__(self, half_window: int = 3, threshold: float = 2.5) -> None:
        self.k = int(half_window)
        self.threshold = float(threshold)
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=2 * self.k + 1)

    def process(self, x: np.ndarray) -> Optional[np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        if not self._buf:
            self._buf.extend([x] * (self.k + 1))
            return None
        self._buf.append(x)
        if len(self._buf) < 2 * self.k + 1:
            return None

        mat = np.asarray(self._buf, dtype=np.float64)
        med = np.median(mat, axis=0)
        mad = 1.4826 * np.median(np.abs(mat - med), axis=0)
        out = mat[self.k].copy()
        mask = (mad > 0) & (np.abs(mat[self.k] - med) > self.threshold * mad)
        out[mask] = med[mask]
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Multi-Threaded CSI Processing & Inference Worker
# ═════════════════════════════════════════════════════════════════════════════

class UnifiedWorker(threading.Thread):
    """Processes streaming CSI packets, applies causal DSP, and runs AI inference."""

    def __init__(
        self,
        port: str = _DEFAULT_PORT,
        baud_rate: int = _DEFAULT_BAUD,
        replay_path: Optional[str] = None,
        mock: bool = False,
        fall_threshold: float = 0.80,
        consecutive_required: int = 2,
        stride: int = 25,
        cooldown_seconds: float = 5.0,
        beep_enabled: bool = True,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baud_rate = baud_rate
        self.replay_path = replay_path
        self.mock = mock
        self.fall_threshold = fall_threshold
        self.consecutive_required = consecutive_required
        self.stride = stride
        self.cooldown_seconds = cooldown_seconds
        self.beep_enabled = beep_enabled and HAS_WINSOUND
        self.running = True
        self.lock = threading.Lock()

        # Shared telemetry state for GUI
        self.history_filtered = np.zeros((_HISTORY_LEN, _NUM_SUBCARRIERS_TO_PLOT), dtype=np.float32)
        self.motion_energy = 0.0
        self.packet_rate = 0.0
        self.is_walking = False
        self.fall_probability = 0.0
        self.consecutive_fall_count = 0
        self.is_fall_alarm = False
        self.last_alarm_time = 0.0
        self.alarm_cooldown_remaining = 0.0
        self.has_new_data = False
        self.new_event_log: Optional[str] = None

        # ── Load Preprocessing Pipeline State ───────────────────────────────
        pipe_path = _PROJECT_ROOT / "data" / "processed" / "pipeline_state.pkl"
        if not pipe_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline state not found: {pipe_path}\n"
                f"Please run 'python scripts/preprocess.py' first."
            )
        with open(pipe_path, "rb") as f:
            st = pickle.load(f)

        self.pca_model = st["pca_model"]
        self.scaler = st["scaler"]
        self.subcarrier_cols = st.get("subcarrier_cols")
        self.n_raw_subcarriers = st.get("n_raw_subcarriers", 190)
        self.window_size = int(st.get("window_size", 100))

        # ── Load Trained CSIFallDetector Model ──────────────────────────────
        model_path = _PROJECT_ROOT / "model" / "saved" / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                f"Please train model using 'python model/train.py' first."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        m_cfg = checkpoint.get("model_config", {})
        input_features = m_cfg.get("input_features", self.pca_model.n_components_)

        self.model = CSIFallDetector(
            input_features=input_features,
            num_classes=m_cfg.get("num_classes", 2),
            lstm_hidden=m_cfg.get("lstm_hidden", 128),
            lstm_layers=m_cfg.get("lstm_layers", 2),
            lstm_dropout=m_cfg.get("lstm_dropout", 0.3),
            attention_heads=m_cfg.get("attention_heads", 4),
            attention_dropout=m_cfg.get("attention_dropout", 0.1),
            cnn_dropout=m_cfg.get("cnn_dropout", 0.2),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Loaded trained model from %s (val_acc: %.2f%%)", model_path.name, checkpoint.get("val_acc", 0.0))

        # ── Causal DSP Chain ────────────────────────────────────────────────
        self.sos = design_bandpass_sos(100.0, 0.5, 40.0, 4)
        self.filter = CausalSOSFilter(self.sos)
        self.resampler = StreamingResampler(100.0)
        self.guard = AGCFaultGuard(log_mag_thresh=0.69, uniformity_tol=0.35, hold_limit=3)
        self.hampel = StreamingHampel(half_window=3, threshold=2.5)

        self._last_good_sel = None
        self.plot_indices = None
        self.filt_buf: collections.deque[np.ndarray] = collections.deque(maxlen=self.window_size)
        self.filt_count = 0

        self._pkt_count = 0
        self._last_rate_time = time.time()
        self._prev_status = "INITIALIZING"

    def run(self) -> None:
        if self.replay_path:
            self._run_replay()
        elif self.mock:
            self._run_mock()
        else:
            self._run_serial()

    def stop(self) -> None:
        self.running = False

    def _play_buzzer_async(self) -> None:
        if self.beep_enabled:
            def _beep():
                try:
                    winsound.Beep(1800, 600)
                except Exception:
                    pass
            threading.Thread(target=_beep, daemon=True).start()

    def _process_packet(self, raw_amp: np.ndarray, t: Optional[float] = None) -> None:
        if t is None:
            t = time.time()

        if len(raw_amp) != self.n_raw_subcarriers:
            return

        if self.subcarrier_cols is None:
            self.subcarrier_cols, _ = select_subcarriers(raw_amp[None, :])
        
        sel = raw_amp[self.subcarrier_cols]

        if self.plot_indices is None:
            step = len(sel) // _NUM_SUBCARRIERS_TO_PLOT
            self.plot_indices = np.arange(0, len(sel), step)[:_NUM_SUBCARRIERS_TO_PLOT]

        # 1. Hardware AGC Fault Guard
        is_ok, _ = self.guard.check(sel)
        if not is_ok and self._last_good_sel is not None:
            sel = self._last_good_sel.copy()
        else:
            self._last_good_sel = sel.copy()

        # 2. Resample onto strict 100 Hz grid
        for row in self.resampler.push(t, sel):
            # 3. Log-amplitude transformation
            row_log = 20.0 * np.log10(row + 1.0)

            # 4. Streaming Hampel filter
            h = self.hampel.process(row_log)
            if h is None:
                continue

            # 5. Causal SOS Bandpass Filter (0.5–40 Hz)
            filtered = self.filter.process(h[None, :])[0]
            self.filt_buf.append(filtered)
            self.filt_count += 1

            # 6. Rate calculation
            self._pkt_count += 1
            now = time.time()
            if now - self._last_rate_time >= 1.0:
                self.packet_rate = self._pkt_count / (now - self._last_rate_time)
                self._pkt_count = 0
                self._last_rate_time = now

            # 7. Update waveform history buffer
            with self.lock:
                self.history_filtered = np.roll(self.history_filtered, -1, axis=0)
                self.history_filtered[-1, :] = filtered[self.plot_indices]

                # Motion energy: variance across recent 60 frames (~0.6s)
                recent = self.history_filtered[-60:]
                var_sub = np.var(recent, axis=0)
                self.motion_energy = float(np.mean(var_sub))
                self.is_walking = self.motion_energy >= 0.40

                # 8. Neural Network Fall Detection every `stride` frames
                if len(self.filt_buf) >= self.window_size and self.filt_count % self.stride == 0:
                    self._infer_sliding_window()

                # Update cooldown timer
                time_since_alarm = now - self.last_alarm_time
                if time_since_alarm < self.cooldown_seconds:
                    self.is_fall_alarm = True
                    self.alarm_cooldown_remaining = max(0.0, self.cooldown_seconds - time_since_alarm)
                else:
                    self.is_fall_alarm = False
                    self.alarm_cooldown_remaining = 0.0

                self.has_new_data = True

    def _infer_sliding_window(self) -> None:
        """Run CNN-BiLSTM-Attention inference on the latest 1.0-second window."""
        window = np.asarray(self.filt_buf, dtype=np.float64)  # (100, 114)
        reduced = self.pca_model.transform(window)             # (100, 20)
        normed = self.scaler.transform(reduced)                # (100, 20)
        tensor_in = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        p_fall = float(probs[0])      # Class 0: Fall
        self.fall_probability = p_fall
        now = time.time()

        # Fall alarm condition requires BOTH:
        # 1. High neural network fall confidence across consecutive sliding windows
        # 2. Dynamic kinetic energy impact burst (>= 2.0 dB^2, distinct from quiet sitting down)
        if p_fall >= self.fall_threshold:
            self.consecutive_fall_count += 1
            if self.consecutive_fall_count >= self.consecutive_required and self.motion_energy >= 2.0:
                if now - self.last_alarm_time > self.cooldown_seconds:
                    self.last_alarm_time = now
                    self.is_fall_alarm = True
                    self.alarm_cooldown_remaining = self.cooldown_seconds
                    self._play_buzzer_async()
                    ts = datetime.now().strftime("%H:%M:%S")
                    self.new_event_log = f"[{ts}] 🚨 *** FALL DETECTED! *** Confidence: {p_fall*100:.1f}%, Energy: {self.motion_energy:.2f} dB²"
                    logger.warning(self.new_event_log)
        else:
            self.consecutive_fall_count = 0

    def _run_serial(self) -> None:
        logger.info("Connecting to receiver on %s @ %d...", self.port, self.baud_rate)
        try:
            with CSIDataReader(port=self.port, baud_rate=self.baud_rate) as reader:
                if reader._serial:
                    reader._serial.reset_input_buffer()
                target_mac = None
                while self.running:
                    if reader._serial and reader._serial.in_waiting > 65536:
                        reader._serial.reset_input_buffer()
                    pkt = reader.read_one()
                    if pkt is not None:
                        mac = pkt.get("mac", "unknown")
                        if target_mac is None:
                            target_mac = mac
                            logger.info("Locked to MAC: %s", target_mac)
                        if mac == target_mac:
                            t_pkt = time.time()
                            amp, _ = extract_amplitude_phase(pkt["raw_data"])
                            self._process_packet(amp, t_pkt)
        except Exception as e:
            logger.error("Serial worker error: %s", e)

    def _run_replay(self) -> None:
        logger.info("Replaying dataset: %s", self.replay_path)
        df = pd.read_csv(self.replay_path)
        amp_cols = [c for c in df.columns if c.startswith("amplitude_")]
        data = df[amp_cols].values.astype(np.float64)
        n = len(data)
        idx = 0
        while self.running:
            self._process_packet(data[idx], time.time())
            idx += 1
            if idx >= n:
                # End of file reached: pause briefly and reset filters before next replay cycle
                time.sleep(1.0)
                idx = 0
                with self.lock:
                    self.filt_buf.clear()
                    self.filt_count = 0
                    self._last_good_sel = None
                self.sos = design_bandpass_sos(100.0, 0.5, 40.0, 4)
                self.filter = CausalSOSFilter(self.sos)
                self.resampler = StreamingResampler(100.0)
                self.guard = AGCFaultGuard(log_mag_thresh=0.69, uniformity_tol=0.35, hold_limit=3)
                self.hampel = StreamingHampel(half_window=3, threshold=2.5)
            time.sleep(0.01)

    def _run_mock(self) -> None:
        logger.info("Running synthetic mock CSI generator...")
        t = 0
        while self.running:
            is_fall_moment = (t % 600) > 280 and (t % 600) < 320
            is_walk_moment = (t % 600) >= 320 and (t % 600) < 550
            n_sub = 190
            base = 25.0
            if is_fall_moment:
                noise = np.random.normal(0, 12.0, n_sub) * np.sin(t * 0.4)
            elif is_walk_moment:
                noise = np.random.normal(0, 4.0, n_sub) * np.sin(t * 0.1)
            else:
                noise = np.random.normal(0, 0.4, n_sub)
            amp = np.clip(base + noise, 1.0, 150.0).astype(np.float64)
            self._process_packet(amp, time.time())
            t += 1
            time.sleep(0.01)


# ═════════════════════════════════════════════════════════════════════════════
# Flagship Unified Dashboard Window
# ═════════════════════════════════════════════════════════════════════════════

class UnifiedMonitorWindow(QMainWindow):
    """60 FPS Real-time Activity and Fall Detection GUI Dashboard."""

    def __init__(
        self,
        port: str = _DEFAULT_PORT,
        baud: int = _DEFAULT_BAUD,
        replay: Optional[str] = None,
        mock: bool = False,
        fall_threshold: float = 0.80,
        consecutive_required: int = 2,
        stride: int = 25,
        cooldown_seconds: float = 5.0,
        beep_enabled: bool = True,
    ) -> None:
        super().__init__()
        mode_title = f"Replay ({Path(replay).name})" if replay else ("Mock Stream" if mock else f"Live Hardware ({port})")
        self.setWindowTitle(f"ESP32-S3 WiFi CSI Unified Activity & Fall Detection Dashboard — [{mode_title}]")
        self.resize(1180, 820)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d1117;
            }
            QWidget {
                color: #c9d1d9;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QFrame.card {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
            QTextEdit {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #8b949e;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)

        pg.setConfigOptions(useOpenGL=False, antialias=False)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # ── 1. Top HUD Status Banner ─────────────────────────────────────────
        self.banner = QFrame()
        self.banner.setFrameShape(QFrame.StyledPanel)
        banner_layout = QVBoxLayout(self.banner)
        banner_layout.setContentsMargins(14, 14, 14, 14)
        banner_layout.setSpacing(4)

        self.label_status = QLabel("INITIALIZING SYSTEM...")
        self.label_status.setAlignment(Qt.AlignCenter)
        self.label_status.setFont(QFont("Consolas", 26, QFont.Bold))
        banner_layout.addWidget(self.label_status)

        self.label_sub = QLabel("Synchronizing 114 subcarriers across dual ESP32-S3 wireless link...")
        self.label_sub.setAlignment(Qt.AlignCenter)
        self.label_sub.setStyleSheet("font-size: 13px; color: #8b949e;")
        banner_layout.addWidget(self.label_sub)
        main_layout.addWidget(self.banner)

        # ── 2. Dual Real-Time Telemetry Gauges ────────────────────────────────
        telemetry_card = QFrame()
        telemetry_card.setProperty("class", "card")
        telemetry_card.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 8px;")
        t_layout = QGridLayout(telemetry_card)
        t_layout.setContentsMargins(14, 12, 14, 12)
        t_layout.setHorizontalSpacing(16)
        t_layout.setVerticalSpacing(6)

        # Gauge 1: Motion Energy
        lbl_e = QLabel("Kinetic Motion Energy:")
        lbl_e.setStyleSheet("font-weight: bold; font-size: 12px; color: #e6edf3;")
        t_layout.addWidget(lbl_e, 0, 0)

        self.bar_energy = QProgressBar()
        self.bar_energy.setRange(0, 100)
        self.bar_energy.setValue(0)
        self.bar_energy.setFixedHeight(22)
        self.bar_energy.setStyleSheet("""
            QProgressBar {
                border: 1px solid #30363d;
                border-radius: 4px;
                background-color: #0d1117;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00E676, stop:0.4 #00E5FF, stop:0.8 #FFD600, stop:1.0 #FF1744);
                border-radius: 3px;
            }
        """)
        t_layout.addWidget(self.bar_energy, 0, 1)

        self.lbl_energy_val = QLabel("0.00 dB² (Quiet)")
        self.lbl_energy_val.setStyleSheet("font-family: Consolas; font-size: 12px; color: #58a6ff; font-weight: bold;")
        t_layout.addWidget(self.lbl_energy_val, 0, 2)

        # Gauge 2: AI Fall Risk
        lbl_f = QLabel("AI Fall Risk (CNN-LSTM):")
        lbl_f.setStyleSheet("font-weight: bold; font-size: 12px; color: #e6edf3;")
        t_layout.addWidget(lbl_f, 1, 0)

        self.bar_fall = QProgressBar()
        self.bar_fall.setRange(0, 100)
        self.bar_fall.setValue(0)
        self.bar_fall.setFixedHeight(22)
        self.bar_fall.setStyleSheet("""
            QProgressBar {
                border: 1px solid #30363d;
                border-radius: 4px;
                background-color: #0d1117;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00E676, stop:0.6 #FF9100, stop:0.8 #FF5252, stop:1.0 #FF1744);
                border-radius: 3px;
            }
        """)
        t_layout.addWidget(self.bar_fall, 1, 1)

        self.lbl_fall_val = QLabel("0.0% (Safe)")
        self.lbl_fall_val.setStyleSheet("font-family: Consolas; font-size: 12px; color: #7ee787; font-weight: bold;")
        t_layout.addWidget(self.lbl_fall_val, 1, 2)

        # Hardware info row
        self.lbl_hw_info = QLabel("PHY Rate: 0 Hz  |  Bandwidth: HT40 (114 Subcarriers)  |  Inference Stride: 250 ms")
        self.lbl_hw_info.setStyleSheet("font-family: Consolas; font-size: 11px; color: #8b949e;")
        t_layout.addWidget(self.lbl_hw_info, 2, 0, 1, 3)

        main_layout.addWidget(telemetry_card)

        # ── 3. Middle Section: Waveforms + Live Event History ────────────────
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(4)

        # Waveform Plot
        self.plot_widget = pg.PlotWidget(title="Real-Time Causal CSI Subcarrier Perturbations (0.5–40 Hz SOS Bandpass)")
        self.plot_widget.setLabel("bottom", "Sliding Time Frames (@ 100 Hz Uniform Grid)")
        self.plot_widget.setLabel("left", "Amplitude Perturbation (dB)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.20)
        self.plot_widget.setXRange(0, _HISTORY_LEN, padding=0)
        self.plot_widget.setYRange(-7, 7, padding=0.1)
        self.plot_widget.setMouseEnabled(x=False, y=True)
        splitter.addWidget(self.plot_widget)

        self.curves = []
        for i in range(_NUM_SUBCARRIERS_TO_PLOT):
            pen = pg.mkPen(color=_PALETTE_10[i % len(_PALETTE_10)], width=1.6)
            curve = self.plot_widget.plot(pen=pen)
            self.curves.append(curve)

        # Event History Drawer
        log_frame = QFrame()
        log_frame.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px;")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(10, 8, 10, 8)
        log_layout.setSpacing(4)

        lbl_log = QLabel("Real-Time Telemetry & Event Audit Log:")
        lbl_log.setStyleSheet("font-weight: bold; font-size: 11px; color: #8b949e;")
        log_layout.addWidget(lbl_log)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(110)
        log_layout.addWidget(self.log_text)
        splitter.addWidget(log_frame)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, stretch=4)

        # ── Start Background Worker ──────────────────────────────────────────
        self.worker = UnifiedWorker(
            port=port,
            baud_rate=baud,
            replay_path=replay,
            mock=mock,
            fall_threshold=fall_threshold,
            consecutive_required=consecutive_required,
            stride=stride,
            cooldown_seconds=cooldown_seconds,
            beep_enabled=beep_enabled,
        )
        self.worker.start()

        # ── 60 FPS UI Refresh Timer ──────────────────────────────────────────
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(16)

        self._prev_state_name = "INIT"
        self._pulse_state = False

    def update_ui(self) -> None:
        if not self.worker.has_new_data:
            return

        with self.worker.lock:
            data = self.worker.history_filtered.copy()
            energy = self.worker.motion_energy
            rate = self.worker.packet_rate
            is_walking = self.worker.is_walking
            p_fall = self.worker.fall_probability
            is_fall_alarm = self.worker.is_fall_alarm
            cooldown = self.worker.alarm_cooldown_remaining
            new_log = self.worker.new_event_log
            self.worker.new_event_log = None
            self.worker.has_new_data = False

        # Add event to log
        if new_log:
            self.log_text.append(new_log)

        # ── Update HUD Banner & State ────────────────────────────────────────
        if is_fall_alarm:
            self._pulse_state = not self._pulse_state
            border_color = "#FF1744" if self._pulse_state else "#FF5252"
            self.banner.setStyleSheet(f"""
                QFrame {{
                    background-color: #3d080e;
                    border: 3px solid {border_color};
                    border-radius: 8px;
                }}
            """)
            self.label_status.setText("🚨  *** EMERGENCY: FALL DETECTED! ***")
            self.label_status.setStyleSheet("color: #FF1744; font-weight: bold;")
            self.label_sub.setText(
                f"AI Confidence: {p_fall*100:.1f}%  |  Audible Buzzer Triggered  |  Alarm Cooldown: {cooldown:.1f}s"
            )
            current_state = "FALL"
        elif is_walking:
            self.banner.setStyleSheet("""
                QFrame {{
                    background-color: #052636;
                    border: 2px solid #00E5FF;
                    border-radius: 8px;
                }}
            """)
            self.label_status.setText("🚶  WALKING / ACTIVE MOTION")
            self.label_status.setStyleSheet("color: #00E5FF; font-weight: bold;")
            self.label_sub.setText(f"Active movement detected  |  Motion Energy: {energy:.2f} dB²  |  Fall Risk: Low ({p_fall*100:.1f}%)")
            current_state = "WALK"
        else:
            self.banner.setStyleSheet("""
                QFrame {{
                    background-color: #092615;
                    border: 2px solid #00E676;
                    border-radius: 8px;
                }}
            """)
            self.label_status.setText("🧘  SITTING STILL / RESTING")
            self.label_status.setStyleSheet("color: #00E676; font-weight: bold;")
            self.label_sub.setText(f"Stationary baseline  |  Motion Energy: {energy:.2f} dB²  |  Fall Risk: None ({p_fall*100:.1f}%)")
            current_state = "STILL"

        # Log state transition
        if current_state != self._prev_state_name:
            ts = datetime.now().strftime("%H:%M:%S")
            if current_state == "WALK":
                self.log_text.append(f"[{ts}] 🚶 Movement Started — Energy: {energy:.2f} dB²")
            elif current_state == "STILL":
                self.log_text.append(f"[{ts}] 🧘 Stationary Stillness — Energy: {energy:.2f} dB²")
            self._prev_state_name = current_state

        # ── Update Gauges ────────────────────────────────────────────────────
        energy_pct = int(np.clip(energy / 20.0 * 100, 0, 100))
        self.bar_energy.setValue(energy_pct)
        self.lbl_energy_val.setText(f"{energy:.2f} dB² ({energy_pct}%)")

        fall_pct = int(np.clip(p_fall * 100, 0, 100))
        self.bar_fall.setValue(fall_pct)
        if fall_pct >= 80:
            self.lbl_fall_val.setText(f"{p_fall*100:.1f}% (CRITICAL)")
            self.lbl_fall_val.setStyleSheet("font-family: Consolas; font-size: 12px; color: #FF1744; font-weight: bold;")
        elif fall_pct >= 40:
            self.lbl_fall_val.setText(f"{p_fall*100:.1f}% (Elevated)")
            self.lbl_fall_val.setStyleSheet("font-family: Consolas; font-size: 12px; color: #FFD600; font-weight: bold;")
        else:
            self.lbl_fall_val.setText(f"{p_fall*100:.1f}% (Safe)")
            self.lbl_fall_val.setStyleSheet("font-family: Consolas; font-size: 12px; color: #00E676; font-weight: bold;")

        # ── Update Telemetry Info ────────────────────────────────────────────
        self.lbl_hw_info.setText(
            f"PHY Rate: {rate:.0f} Hz  |  Bandwidth: HT40 (114 Carriers)  |  "
            f"Inference: Stride 250ms  |  Model: CNN-BiLSTM-Attention (968k params)"
        )

        # ── Update Waveforms ─────────────────────────────────────────────────
        for i, curve in enumerate(self.curves):
            curve.setData(data[:, i])

    def closeEvent(self, event) -> None:
        self.worker.stop()
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ESP32-S3 WiFi CSI Unified Activity & Fall Detection Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=str, default=_DEFAULT_PORT, help="Serial COM port for receiver")
    parser.add_argument("--baud", type=int, default=_DEFAULT_BAUD, help="Serial baud rate")
    parser.add_argument("--replay", type=str, default=None, help="Replay CSV recording file")
    parser.add_argument("--mock", action="store_true", help="Run autonomous mock generator")
    parser.add_argument("--threshold", type=float, default=0.80, help="Fall probability threshold (0.0–1.0)")
    parser.add_argument("--consecutive", type=int, default=2, help="Consecutive windows required for alarm")
    parser.add_argument("--stride", type=int, default=25, help="Inference stride in samples (25 pkts = 250ms)")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Alarm cooldown in seconds")
    parser.add_argument("--no-beep", action="store_true", help="Disable audible buzzer")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    win = UnifiedMonitorWindow(
        port=args.port,
        baud=args.baud,
        replay=args.replay,
        mock=args.mock,
        fall_threshold=args.threshold,
        consecutive_required=args.consecutive,
        stride=args.stride,
        cooldown_seconds=args.cooldown,
        beep_enabled=not args.no_beep,
    )
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
