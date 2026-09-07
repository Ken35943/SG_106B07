#!/usr/bin/env python3
"""
visualize_dual.py — Real-Time Dual-Receiver CSI Waveform Monitor
=================================================================
Visualizes real-time CSI subcarrier waveforms simultaneously from two receivers
(e.g. COM3 & COM11), with per-link packet rates, Tx sequence IDs, and sync stats.

ZERO-LAG ARCHITECTURE:
- Each receiver streams into its OWN preallocated RingBuffer at full native
  speed. Rx1 NEVER freezes because Rx2 dropped a packet in the air.
- No Tx-ID intersection gating on the display path (fusion lives in
  dual_link.DualCSIReader and is used only where alignment is required).
- GUI refreshes on a precision 60 FPS QTimer; Y-axis auto-expands (0–120)
  so peaks are never clipped, while user zoom still works.
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
from dual_link import LinkWorker, RingBuffer
from csi_dsp import ht40_htltf_layout, HT40_TOTAL_ENTRIES

logger = logging.getLogger(__name__)

_HISTORY_LEN = 200
_NUM_SUBCARRIERS_TO_PLOT = 8
_Y_MAX_HARD = 5000.0   # allow near-field Tx high amplitude up to 5000
_Y_MIN_SPAN = 50.0     # never auto-scale below this

_PALETTE_8 = [
    "#00E5FF", "#E040FB", "#7C4DFF", "#00E676",
    "#FFEA00", "#FF6D00", "#FF1744", "#2979FF"
]


def _select_plot_indices(num_sub: int) -> np.ndarray:
    if num_sub == HT40_TOTAL_ENTRIES:
        cols, _ = ht40_htltf_layout()
        step = max(1, len(cols) // _NUM_SUBCARRIERS_TO_PLOT)
        return np.asarray(cols[::step][:_NUM_SUBCARRIERS_TO_PLOT], dtype=int)
    return np.linspace(0, max(0, num_sub - 1), _NUM_SUBCARRIERS_TO_PLOT, dtype=int)


class DualVizPump(threading.Thread):
    """Drains both links and feeds two independent ring buffers at native speed."""

    def __init__(
        self,
        ports: tuple[str, str],
        baud: int = 921600,
        mock: bool = False,
        replay: Optional[tuple[str, str]] = None,
        history_len: int = _HISTORY_LEN,
        n_plot: int = _NUM_SUBCARRIERS_TO_PLOT,
    ):
        super().__init__(daemon=True, name="DualVizPump")
        self.ports = ports
        self.baud = baud
        self.mock = mock
        self.replay = replay
        self.history_len = history_len
        self.n_plot = n_plot
        self.running = True

        self.workers: list[LinkWorker] = []
        self.rings = [RingBuffer(history_len, n_plot), RingBuffer(history_len, n_plot)]
        self.plot_idx: list[Optional[np.ndarray]] = [None, None]
        self.last_id = [-1, -1]
        self.rate_hz = [0.0, 0.0]
        self.has_new = [False, False]
        self._last_row: list[Optional[np.ndarray]] = [None, None]
        self.glitches = [0, 0]
        self._lock = threading.Lock()   # guards stats only; rings lock themselves
        self._cnt = [0, 0]
        self._t0 = time.monotonic()
        self.pump_hz = 0.0
        self._pump_n = 0
        self._pump_t0 = time.monotonic()

    # ── sources ────────────────────────────────────────────────────────
    def _open_live(self) -> None:
        self.workers = [LinkWorker(f"rx{i+1}", p, self.baud) for i, p in enumerate(self.ports)]
        for w in self.workers:
            w.start()

    def _close_live(self) -> None:
        for w in self.workers:
            w.stop()
        self.workers = []

    def _mock_packets(self, link: int):
        """Synthetic 190-ch frames with independent 15% air drops (benchmark/offline)."""
        rng = np.random.default_rng(1000 + link)
        pid = 0
        t = 0.0
        while self.running and self.mock:
            t0 = time.monotonic()
            if rng.random() > 0.15:  # 15% independent drop per link
                base = 25.0 + 10.0 * np.sin(np.linspace(0, 4 * np.pi, 190) + pid * 0.05 + link)
                amp = np.clip(base + rng.normal(0, 1.5, 190), 0, 150)
                self._feed(link, pid, amp)
            pid += 1
            # pace ~100 Hz generation
            dt = 0.01 - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)

    def _replay_packets(self, link: int, csv_path: str):
        import pandas as pd
        df = pd.read_csv(csv_path)
        amp_cols = [c for c in df.columns if c.startswith("amplitude_")]
        data = df[amp_cols].values.astype(np.float64)
        ids = df["id"].values if "id" in df.columns else np.arange(len(data))
        n = len(data)
        idx = 0
        while self.running and self.replay:
            t0 = time.monotonic()
            self._feed(link, int(ids[idx]), data[idx])
            idx = (idx + 1) % n
            dt = 0.01 - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)

    def _feed(self, link: int, pid: int, amp: np.ndarray, gain_ok: bool = True) -> None:
        if self.plot_idx[link] is None or len(amp) != getattr(self, "_nsub", [None, None])[link]:
            self._nsub = getattr(self, "_nsub", [None, None])
            self._nsub[link] = len(amp)
            self.plot_idx[link] = _select_plot_indices(len(amp))
        if not gain_ok and self._last_row[link] is not None:
            # Firmware AGC gain-glitch frame: hold the previous row instead
            # of plotting a 200–400% common-mode spike. The spike never
            # enters the ring buffer, so waveforms, autoscale and any
            # downstream consumer stay clean.
            self.rings[link].push(self._last_row[link])
            with self._lock:
                self.glitches[link] += 1
                self.last_id[link] = pid
                self._cnt[link] += 1
                self.has_new[link] = True
            return
        try:
            row = np.asarray(amp[self.plot_idx[link]], dtype=np.float32)
            self.rings[link].push(row)
            self._last_row[link] = row
        except Exception:
            return  # length changed mid-stream; re-lock indices next packet
        with self._lock:
            self.last_id[link] = pid
            self._cnt[link] += 1
            self.has_new[link] = True

    # ── main loop ──────────────────────────────────────────────────────
    def run(self) -> None:
        self._nsub = [None, None]
        sources: list[threading.Thread] = []
        if self.mock:
            logger.info("Mock mode: two synthetic 190-ch streams @ ~100 Hz, 15%% drops")
            for link in range(2):
                t = threading.Thread(target=self._mock_packets, args=(link,), daemon=True)
                t.start()
                sources.append(t)
        elif self.replay:
            logger.info("Replay mode: %s", self.replay)
            for link, path in enumerate(self.replay):
                t = threading.Thread(target=self._replay_packets, args=(link, path), daemon=True)
                t.start()
                sources.append(t)
        else:
            self._open_live()

        try:
            while self.running:
                drained = 0
                if not self.mock and not self.replay:
                    for link, w in enumerate(self.workers):
                        pkts = w.drain()          # single lock per link per loop
                        for p in pkts:
                            self._feed(link, p.pkt_id, p.amp,
                                       gain_ok=getattr(p, "gain_ok", True))
                            drained += 1
                        st = w.stats()
                        with self._lock:
                            self.rate_hz[link] = st["rate_hz"]
                else:
                    with self._lock:
                        drained = int(self.has_new[0] or self.has_new[1])
                    # mock/replay threads already paced; just update 1 s rate windows
                    now = time.monotonic()
                    if now - self._t0 >= 1.0:
                        with self._lock:
                            for link in range(2):
                                self.rate_hz[link] = self._cnt[link] / (now - self._t0)
                                self._cnt[link] = 0
                            self._t0 = now
                # pump-rate accounting + 1 s stats tick for live mode
                self._pump_n += 1
                now = time.monotonic()
                if now - self._pump_t0 >= 1.0:
                    self.pump_hz = self._pump_n / (now - self._pump_t0)
                    self._pump_n = 0
                    self._pump_t0 = now
                # ALWAYS yield the GIL once per loop: caps the pump at ~1 kHz
                # (data arrives at <= ~150 Hz combined) and prevents a tight
                # lock/poll loop from starving the Qt event loop -> frozen GUI.
                time.sleep(0.001)
        finally:
            if not self.mock and not self.replay:
                self._close_live()

    def snapshot(self, link: int) -> tuple[np.ndarray, bool]:
        with self._lock:
            new = self.has_new[link]
            self.has_new[link] = False
        return self.rings[link].snapshot(), new

    def stats(self) -> dict:
        with self._lock:
            return {
                "rate": list(self.rate_hz),
                "last_id": list(self.last_id),
                "pump_hz": self.pump_hz,
                "glitches": list(self.glitches),
            }

    def stop(self) -> None:
        self.running = False
        self.join(timeout=3.0)


class DualVisualizerWindow(QMainWindow):
    def __init__(
        self,
        ports: tuple[str, str],
        baud: int = 921600,
        mock: bool = False,
        replay: Optional[tuple[str, str]] = None,
        use_gl: bool = False,
    ):
        super().__init__()
        mode = "MOCK" if mock else (f"REPLAY" if replay else f"{ports[0]}|{ports[1]}")
        self.setWindowTitle(f"Dual-Receiver CSI Monitor — [{mode}]")
        self.resize(1100, 800)
        self.setStyleSheet("background-color: #0d1117; color: #c9d1d9;")

        # GPU-accelerated curve rendering when requested (needs PyOpenGL +
        # a real display; offscreen/headless always uses the raster engine).
        pg.setConfigOptions(useOpenGL=use_gl, antialias=False)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        hdr = QHBoxLayout()
        self.lbl_info = QLabel("starting…")
        self.lbl_info.setStyleSheet("font-family: Consolas; font-size: 14px; font-weight: bold; color: #58a6ff;")
        hdr.addWidget(self.lbl_info)
        layout.addLayout(hdr)

        self.plots, self.curves = [], []
        for k, name in enumerate((f"Receiver 1 ({ports[0]})", f"Receiver 2 ({ports[1]})")):
            plot = pg.PlotWidget(title=f"{name} — Subcarrier Amplitudes")
            if k == 1:
                plot.setLabel("bottom", "Time (frames)")
            plot.setLabel("left", "Amplitude")
            plot.setXRange(0, _HISTORY_LEN, padding=0)
            plot.setYRange(0, _Y_MIN_SPAN, padding=0.02)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setMouseEnabled(x=False, y=True)
            layout.addWidget(plot)
            curves = []
            for i in range(_NUM_SUBCARRIERS_TO_PLOT):
                curves.append(plot.plot(pen=pg.mkPen(color=_PALETTE_8[i % len(_PALETTE_8)], width=1.8)))
            self.plots.append(plot)
            self.curves.append(curves)

        # Auto-scale state per plot: expand instantly on clipping, decay slowly.
        self._ymax = [_Y_MIN_SPAN, _Y_MIN_SPAN]

        self.pump = DualVizPump(ports=ports, baud=baud, mock=mock, replay=replay)
        self.pump.start()

        self._frames = 0
        self._fps_t0 = time.monotonic()
        self._gui_fps = 0.0
        self._last_text_t = 0.0   # stats label throttled to ~4 Hz (re-layout is costly)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(16)

    def _autoscale(self, k: int, data: np.ndarray) -> None:
        peak = float(data.max()) if data.size else 0.0
        cur = self._ymax[k]
        if peak > cur * 0.98:
            target = min(_Y_MAX_HARD, max(_Y_MIN_SPAN, peak * 1.12))
            if abs(target - cur) / max(cur, 1e-6) > 0.03:
                self._ymax[k] = target
                self.plots[k].setYRange(0, target, padding=0.02)
        elif peak < cur * 0.65 and cur > _Y_MIN_SPAN:
            target = min(_Y_MAX_HARD, max(_Y_MIN_SPAN, peak * 1.25))
            # slow decay: move 8% toward target per frame (~0.5 s to settle)
            new = cur + (target - cur) * 0.08
            if abs(new - cur) / max(cur, 1e-6) > 0.03:
                self._ymax[k] = new
                self.plots[k].setYRange(0, new, padding=0.02)

    def update_ui(self):
        self._frames += 1
        now = time.monotonic()
        if now - self._fps_t0 >= 1.0:
            self._gui_fps = self._frames / (now - self._fps_t0)
            self._frames = 0
            self._fps_t0 = now

        st = self.pump.stats()
        d0, n0 = self.pump.snapshot(0)
        d1, n1 = self.pump.snapshot(1)
        if not (n0 or n1):
            return  # nothing new on either link: skip redraw entirely

        self._autoscale(0, d0)
        self._autoscale(1, d1)
        for i in range(_NUM_SUBCARRIERS_TO_PLOT):
            # skipFiniteCheck: data is finite by construction; skips a full
            # NaN scan per curve per frame.
            self.curves[0][i].setData(d0[:, i], skipFiniteCheck=True)
            self.curves[1][i].setData(d1[:, i], skipFiniteCheck=True)

        if now - self._last_text_t >= 0.25:
            self._last_text_t = now
            skew = abs(st["last_id"][0] - st["last_id"][1]) if min(st["last_id"]) >= 0 else -1
            gl = st.get("glitches", [0, 0])
            self.lbl_info.setText(
                f"Rx1: {st['rate'][0]:.0f} Hz (id {st['last_id'][0]}, glitch {gl[0]})  |  "
                f"Rx2: {st['rate'][1]:.0f} Hz (id {st['last_id'][1]}, glitch {gl[1]})  |  "
                f"id-skew: {skew}  |  pump: {st['pump_hz']:.0f} Hz  |  GUI: {self._gui_fps:.0f} FPS"
            )

    def closeEvent(self, event):
        self.timer.stop()
        self.pump.stop()
        event.accept()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Dual-Receiver CSI Real-Time Visualizer")
    parser.add_argument("--ports", type=str, default="COM3,COM11", help="Comma-separated COM ports")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate")
    parser.add_argument("--mock", action="store_true", help="Synthetic dual streams (no hardware)")
    parser.add_argument("--gl", action="store_true",
                        help="GPU-accelerated curve rendering (needs PyOpenGL + real display)")
    parser.add_argument("--replay", type=str, default=None,
                        help="One CSV (both links) or two CSVs 'a.csv,b.csv' to replay")
    args = parser.parse_args()

    port_list = [p.strip() for p in args.ports.split(",")]
    replay = None
    if args.replay:
        parts = [p.strip() for p in args.replay.split(",")]
        replay = (parts[0], parts[1] if len(parts) > 1 else parts[0])

    app = QApplication(sys.argv)
    win = DualVisualizerWindow(ports=(port_list[0], port_list[1]), baud=args.baud,
                               mock=args.mock, replay=replay, use_gl=args.gl)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
