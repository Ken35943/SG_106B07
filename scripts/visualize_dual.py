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
- FIXED-TIME-BASE RENDERING (anti-judder): packets arrive in USB bursts
  (0/2/1/3 per 16.6 ms frame), so plotting one step per packet produces
  variable stride = visible micro-stutter. Instead each stored sample
  carries its wall-clock arrival time and every GUI tick renders a UNIFORM
  time grid (default last 3.0 s) via linear interpolation. Scroll speed is
  therefore constant in pixels/second regardless of arrival bursts; gaps
  render as truthful flatlines (edge-hold) instead of timeline leaps.
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
_Y_MAX_HARD = 120.0   # never auto-scale above this
_Y_MIN_SPAN = 50.0    # never auto-scale below this

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
        # Ring holds n_plot data channels + 1 trailing WALL-CLOCK arrival-time
        # channel (seconds, monotonic). float64: float32 would quantize large
        # monotonic timestamps to ~ms steps and break interpolation order.
        # One snapshot() therefore always returns time-aligned (data, time).
        self.rings = [RingBuffer(history_len, n_plot + 1, dtype=np.float64),
                      RingBuffer(history_len, n_plot + 1, dtype=np.float64)]
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

    def _feed(self, link: int, pid: int, amp: np.ndarray, gain_ok: bool = True,
              t_wall: Optional[float] = None) -> None:
        if self.plot_idx[link] is None or len(amp) != getattr(self, "_nsub", [None, None])[link]:
            self._nsub = getattr(self, "_nsub", [None, None])
            self._nsub[link] = len(amp)
            self.plot_idx[link] = _select_plot_indices(len(amp))
        now = t_wall if t_wall is not None else time.monotonic()
        if not gain_ok and self._last_row[link] is not None:
            # Firmware AGC gain-glitch frame: hold the previous row instead
            # of plotting a 200–400% common-mode spike. The spike never
            # enters the ring buffer, so waveforms, autoscale and any
            # downstream consumer stay clean. The held row is stamped with
            # the CURRENT time so the fixed time base keeps scrolling.
            row = np.empty(self.n_plot + 1, dtype=np.float64)
            row[:-1] = self._last_row[link]
            row[-1] = now
            self.rings[link].push(row)
            with self._lock:
                self.glitches[link] += 1
                self.last_id[link] = pid
                self._cnt[link] += 1
                self.has_new[link] = True
            return
        try:
            row = np.empty(self.n_plot + 1, dtype=np.float64)
            row[:-1] = amp[self.plot_idx[link]]
            row[-1] = now
            self.rings[link].push(row)
            self._last_row[link] = row[:-1].copy()
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
        span_s: float = 3.0,
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

        # Fixed display time base (seconds). The x-axis is wall-clock time
        # ("seconds ago"), NOT packet index — this is what makes scroll
        # speed constant regardless of USB burst arrivals.
        self._span_s = float(span_s)
        self._xgrid = -self._span_s + (np.arange(_HISTORY_LEN) + 0.5) / _HISTORY_LEN * self._span_s

        self.plots, self.curves = [], []
        for k, name in enumerate((f"Receiver 1 ({ports[0]})", f"Receiver 2 ({ports[1]})")):
            plot = pg.PlotWidget(title=f"{name} — Subcarrier Amplitudes")
            plot.setLabel("bottom", "Time (s ago)")
            plot.setLabel("left", "Amplitude")
            plot.setXRange(-self._span_s, 0, padding=0)
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

    def _interp_frame(self, snap: np.ndarray, now: float) -> Optional[np.ndarray]:
        """Resample one ring snapshot onto the uniform display time grid.

        ``snap`` is (H, n_plot+1) with wall-clock arrival times in the last
        column (0 = empty slot). Returns (H, n_plot) float32 sampled at
        ``now - span ... now``, or None if fewer than 2 valid samples exist.
        np.interp edge-holds outside the data range, so starvation renders
        as a truthful flatline instead of a timeline leap.
        """
        data = snap[:, :-1]
        t = snap[:, -1]
        valid = t > 0
        if int(valid.sum()) < 2:
            return None
        grid = now - self._span_s + (np.arange(_HISTORY_LEN) + 0.5) / _HISTORY_LEN * self._span_s
        tv = t[valid]
        out = np.empty((_HISTORY_LEN, data.shape[1]), dtype=np.float32)
        dv = data[valid]
        for i in range(dv.shape[1]):
            out[:, i] = np.interp(grid, tv, dv[:, i])
        return out

    def update_ui(self):
        self._frames += 1
        now = time.monotonic()
        if now - self._fps_t0 >= 1.0:
            self._gui_fps = self._frames / (now - self._fps_t0)
            self._frames = 0
            self._fps_t0 = now

        st = self.pump.stats()
        # NOTE: deliberately re-render EVERY tick (no has_new early-return).
        # The time base is anchored to wall-clock `now`, so the trace scrolls
        # at constant speed even through arrival gaps (flatline), instead of
        # freezing then leaping. Per-frame cost is ~8×np.interp(200) ≈ 0.2 ms.
        for k in (0, 1):
            snap, _new = self.pump.snapshot(k)
            frame = self._interp_frame(snap, now)
            if frame is None:
                continue  # buffer still empty (startup): leave previous paint
            self._autoscale(k, frame)
            for i in range(_NUM_SUBCARRIERS_TO_PLOT):
                # skipFiniteCheck: data is finite by construction; skips a full
                # NaN scan per curve per frame.
                self.curves[k][i].setData(self._xgrid, frame[:, i], skipFiniteCheck=True)

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
    parser.add_argument("--span", type=float, default=3.0,
                        help="Display time-base span in seconds (constant scroll speed)")
    args = parser.parse_args()

    port_list = [p.strip() for p in args.ports.split(",")]
    replay = None
    if args.replay:
        parts = [p.strip() for p in args.replay.split(",")]
        replay = (parts[0], parts[1] if len(parts) > 1 else parts[0])

    app = QApplication(sys.argv)
    win = DualVisualizerWindow(ports=(port_list[0], port_list[1]), baud=args.baud,
                               mock=args.mock, replay=replay, use_gl=args.gl,
                               span_s=args.span)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
