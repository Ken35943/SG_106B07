#!/usr/bin/env python3
"""
dual_link.py — Dual-Port CSI Ingestion with Tx-ID Aligned Synchronization
==========================================================================
Connects concurrently to two ESP32-S3 receiver nodes (e.g. COM3 & COM11).
Synchronizes packets using the Tx-generated sequence ID (`pkt_id`) across links.
Supports offline fusion, gap interpolation, and independent clock reconstruction.
"""

from __future__ import annotations
import collections
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from parse_csi import CSIDataReader, extract_amplitude_phase

logger = logging.getLogger(__name__)

_SERIAL_TIMEOUT_S = 1.0


@dataclass
class LinkPacket:
    pkt_id: int          # Tx frame sequence ID (frame-global across all receivers)
    t_us: float          # THIS link's local microsecond clock (wrap-corrected)
    amp: np.ndarray      # (190,) raw amplitude vector


class LinkWorker(threading.Thread):
    """Worker thread that owns a single serial port and pushes packets into a bounded deque."""

    def __init__(self, name: str, port: str, baud: int = 921600, max_q: int = 2048):
        super().__init__(daemon=True, name=f"LinkWorker-{name}")
        self.name = name
        self.port = port
        self.baud = baud
        self.max_q = max_q
        self.q: collections.deque[LinkPacket] = collections.deque(maxlen=max_q)
        self.lock = threading.Lock()
        self.running = True
        self.dropped = 0
        self.t_us = 0.0
        self._prev_raw = None
        self.total_packets = 0

    def run(self) -> None:
        logger.info("[%s] Opening serial port %s @ %d baud...", self.name, self.port, self.baud)
        try:
            with CSIDataReader(port=self.port, baud_rate=self.baud) as reader:
                if reader._serial:
                    reader._serial.reset_input_buffer()

                while self.running:
                    pkt = reader.read_one()
                    if pkt is None:
                        continue

                    # Local timestamp wrap handling (32-bit uint)
                    raw_ts = pkt.get("local_timestamp")
                    if isinstance(raw_ts, int):
                        if self._prev_raw is None:
                            self._prev_raw = raw_ts
                        else:
                            d = raw_ts - self._prev_raw
                            if d < 0:
                                d += 2 ** 32
                            if 0 < d < 2 ** 31:
                                self.t_us += d
                                self._prev_raw = raw_ts

                    amp, _ = extract_amplitude_phase(pkt.get("raw_data", []))
                    pkt_id = int(pkt.get("id", -1))

                    with self.lock:
                        if len(self.q) == self.q.maxlen:
                            self.dropped += 1
                            self.q.popleft()
                        self.q.append(LinkPacket(pkt_id=pkt_id, t_us=self.t_us, amp=amp))
                        self.total_packets += 1

        except Exception as e:
            logger.error("[%s] Serial reader error on %s: %s", self.name, self.port, e)
        finally:
            self.running = False
            logger.info("[%s] Stopped. Total packets read: %d (dropped: %d)", self.name, self.total_packets, self.dropped)

    def drain(self) -> list[LinkPacket]:
        with self.lock:
            out = list(self.q)
            self.q.clear()
        return out

    def stop(self) -> None:
        self.running = False
        self.join(timeout=2.0)


class DualCSIReader:
    """Fuses two links into synchronized (t_seconds, [amp_A, amp_B]) frames aligned by Tx ID."""

    def __init__(
        self,
        ports: tuple[str, str],
        baud: int = 921600,
        max_lag_ids: int = 40,
        max_gap: int = 20,
    ):
        self.ports = ports
        self.baud = baud
        self.max_lag_ids = max_lag_ids
        self.max_gap = max_gap
        self.workers = [LinkWorker(f"rx{i+1}", p, baud) for i, p in enumerate(ports)]
        self._buf: dict[int, dict] = {}
        self._next_emit_id: Optional[int] = None
        self._last_t_us: Optional[float] = None

    def open(self) -> None:
        for w in self.workers:
            w.start()

    def close(self) -> None:
        for w in self.workers:
            w.stop()

    def __enter__(self) -> DualCSIReader:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _interp(self, link: str, k: int, k1: int, k2: int) -> tuple[np.ndarray, bool]:
        x1, x2 = self._buf[k1][link][1], self._buf[k2][link][1]
        w1, w2 = (k2 - k) / (k2 - k1), (k - k1) / (k2 - k1)
        return w1 * x1 + w2 * x2, True

    def read_fused(self, timeout: float = 0.05) -> Optional[tuple[float, np.ndarray]]:
        """Return (t_seconds_linkA, fused_amp (2, 190)) aligned by Tx frame ID."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for w in self.workers:
                link_key = "A" if w.name == "rx1" else "B"
                for p in w.drain():
                    self._buf.setdefault(p.pkt_id, {})[link_key] = (p.t_us, p.amp.astype(np.float64))

            if not self._buf:
                time.sleep(0.002)
                continue

            # Find common IDs present in both links
            common_ids = [i for i, entry in self._buf.items() if "A" in entry and "B" in entry]

            if common_ids:
                eid = min(common_ids)
                entry = self._buf.pop(eid)

                # Purge stale single-link entries older than the emitted ID
                stale_ids = [i for i in self._buf if i < eid]
                for si in stale_ids:
                    del self._buf[si]

                t_sec = entry["A"][0] / 1e6
                fused = np.stack([entry["A"][1], entry["B"][1]])
                return t_sec, fused

            # Prevent buffer buildup if one link is lagging or dropped
            if len(self._buf) > self.max_lag_ids:
                oldest_id = min(self._buf.keys())
                del self._buf[oldest_id]

            time.sleep(0.002)

        return None
