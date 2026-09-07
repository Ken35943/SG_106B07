#!/usr/bin/env python3
"""
dual_link.py — Dual-Port CSI Ingestion with Tx-ID Aligned Synchronization
==========================================================================
Connects concurrently to two ESP32-S3 receiver nodes (e.g. COM3 & COM11).
Synchronizes packets using the Tx-generated sequence ID (`pkt_id`) across links.
Supports offline fusion, gap interpolation, and independent clock reconstruction.

PERFORMANCE ARCHITECTURE (v2 — zero-lag rewrite):
- Each link ingests on its own thread into a bounded deque (never blocks).
- Serial backlog is purged aggressively so latency can never accumulate.
- Visualization reads each link INDEPENDENTLY via lock-free-ish latest
  snapshots + preallocated ring buffers: Link 1 NEVER freezes because
  Link 2 dropped a packet in the air.
- Tx-ID fusion (`DualCSIReader.read_fused` / `read_available`) is used ONLY
  where frame alignment is actually required (offline fusion, future
  dual-link inference) — never on the visualization hot path.
"""

from __future__ import annotations
import collections
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_csi import CSIDataReader, extract_amplitude_phase

logger = logging.getLogger(__name__)

_SERIAL_TIMEOUT_S = 1.0
# Purge the OS serial buffer once it holds more than this many bytes.
# One full HT40 CSI line is ~1.7 kB, so 16 kB ≈ 9 stale frames. Purging
# keeps displayed data fresh instead of replaying ancient backlog.
_DEFAULT_PURGE_THRESH_BYTES = 16384


@dataclass
class LinkPacket:
    pkt_id: int          # Tx frame sequence ID (frame-global across all receivers)
    t_us: float          # THIS link's local microsecond clock (wrap-corrected)
    amp: np.ndarray      # (190,) raw amplitude vector


class LinkWorker(threading.Thread):
    """Worker thread that owns a single serial port and pushes packets into a bounded deque.

    Never blocks the consumer: the deque is keep-latest (oldest dropped first,
    counted in ``dropped``), and serial backlog is purged so a slow consumer
    sees fresh data instead of an ever-growing delay.
    """

    def __init__(
        self,
        name: str,
        port: str,
        baud: int = 921600,
        max_q: int = 2048,
        purge_backlog: bool = True,
        purge_thresh_bytes: int = _DEFAULT_PURGE_THRESH_BYTES,
    ):
        super().__init__(daemon=True, name=f"LinkWorker-{name}")
        self.name = name
        self.port = port
        self.baud = baud
        self.max_q = max_q
        self.purge_backlog = purge_backlog
        self.purge_thresh_bytes = purge_thresh_bytes
        self.q: collections.deque[LinkPacket] = collections.deque(maxlen=max_q)
        self.lock = threading.Lock()
        self.running = True
        self.dropped = 0
        self.purges = 0
        self.t_us = 0.0
        self._prev_raw = None
        self.total_packets = 0
        # Live stats (updated under lock, cheap 1 s window accounting)
        self.rate_hz = 0.0
        self.last_id = -1
        self._latest_amp: Optional[np.ndarray] = None
        self._win_count = 0
        self._win_t0 = time.monotonic()

    def run(self) -> None:
        logger.info("[%s] Opening serial port %s @ %d baud...", self.name, self.port, self.baud)
        try:
            with CSIDataReader(port=self.port, baud_rate=self.baud) as reader:
                if reader._serial:
                    reader._serial.reset_input_buffer()

                while self.running:
                    # ── Aggressive backlog purge: stale bytes are latency. ──
                    ser = reader._serial
                    if self.purge_backlog and ser is not None:
                        try:
                            if ser.in_waiting > self.purge_thresh_bytes:
                                ser.reset_input_buffer()
                                self.purges += 1
                        except Exception:
                            pass  # port disappearing mid-read is handled below

                    pkt = reader.read_one()  # blocks ≤ serial timeout (1 s)
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
                        self.last_id = pkt_id
                        self._latest_amp = amp
                        self._win_count += 1
                        now = time.monotonic()
                        if now - self._win_t0 >= 1.0:
                            self.rate_hz = self._win_count / (now - self._win_t0)
                            self._win_count = 0
                            self._win_t0 = now

        except Exception as e:
            logger.error("[%s] Serial reader error on %s: %s", self.name, self.port, e)
        finally:
            self.running = False
            logger.info("[%s] Stopped. Total packets read: %d (dropped: %d, purges: %d)",
                        self.name, self.total_packets, self.dropped, self.purges)

    def drain(self) -> list[LinkPacket]:
        """Take all queued packets (single lock acquisition)."""
        with self.lock:
            if not self.q:
                return []
            out = list(self.q)
            self.q.clear()
        return out

    def read_latest(self) -> Optional[tuple[int, float, np.ndarray]]:
        """Newest packet snapshot without fusion gating: (id, t_us, amp copy)."""
        with self.lock:
            if self._latest_amp is None:
                return None
            return self.last_id, self.t_us, self._latest_amp.copy()

    def stats(self) -> dict:
        with self.lock:
            return {
                "name": self.name,
                "port": self.port,
                "rate_hz": self.rate_hz,
                "total": self.total_packets,
                "dropped": self.dropped,
                "purges": self.purges,
                "last_id": self.last_id,
                "alive": self.running,
            }

    def stop(self) -> None:
        self.running = False
        # readline() can block up to the serial timeout; allow margin.
        self.join(timeout=_SERIAL_TIMEOUT_S + 1.5)


class RingBuffer:
    """Preallocated single-producer/single-consumer ring buffer (NumPy).

    ``push`` writes one row in place — zero allocations on the hot path.
    ``snapshot`` returns a chronological copy (one allocation per GUI frame,
    ~6 kB for 200×8 float32 — negligible).
    """

    def __init__(self, capacity: int, n_channels: int, dtype=np.float32):
        self.capacity = int(capacity)
        self.n_channels = int(n_channels)
        self.buf = np.zeros((self.capacity, self.n_channels), dtype=dtype)
        self.lock = threading.Lock()
        self.head = 0          # next write position
        self.count = 0         # rows ever written (saturates at capacity for fullness check)

    def push(self, row: np.ndarray) -> None:
        with self.lock:
            self.buf[self.head] = row  # in-place, casts to dtype
            self.head = (self.head + 1) % self.capacity
            self.count += 1

    def snapshot(self) -> np.ndarray:
        """Chronological (oldest → newest) copy of the full window."""
        with self.lock:
            if self.count < self.capacity:
                out = self.buf.copy()
            else:
                out = np.empty_like(self.buf)
                h = self.head
                out[: self.capacity - h] = self.buf[h:]
                out[self.capacity - h :] = self.buf[:h]
        return out

    @property
    def filled(self) -> int:
        with self.lock:
            return min(self.count, self.capacity)


class DualCSIReader:
    """Fuses two links into synchronized (t_seconds, [amp_A, amp_B]) frames aligned by Tx ID.

    Fusion is STRICTLY opt-in: use ``read_fused``/``read_available`` only where
    frame alignment is required (offline fusion, dual-link inference). For
    visualization, read each link independently via ``read_latest`` so one
    link's air drops can never stall the other.
    """

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

    # ── Decoupled per-link access (visualization hot path) ──────────────
    def read_latest(self, link: int) -> Optional[tuple[int, float, np.ndarray]]:
        """Newest packet of link 0/1 with NO fusion gating."""
        return self.workers[link].read_latest()

    def link_stats(self) -> list[dict]:
        return [w.stats() for w in self.workers]

    # ── Tx-ID fusion (offline / inference path) ──────────────────────────
    def _interp(self, link: str, k: int, k1: int, k2: int) -> tuple[np.ndarray, bool]:
        x1, x2 = self._buf[k1][link][1], self._buf[k2][link][1]
        w1, w2 = (k2 - k) / (k2 - k1), (k - k1) / (k2 - k1)
        return w1 * x1 + w2 * x2, True

    def _ingest(self) -> None:
        """Drain both workers into the fusion buffer (single pass, bounded)."""
        for w in self.workers:
            link_key = "A" if w.name == "rx1" else "B"
            for p in w.drain():
                self._buf.setdefault(p.pkt_id, {})[link_key] = (p.t_us, p.amp.astype(np.float64))
        # Hard bound: a dead link must never grow this dict without limit.
        over = len(self._buf) - self.max_lag_ids * 4
        if over > 0:
            for old in sorted(self._buf)[:over]:
                del self._buf[old]
            if self._next_emit_id is not None and self._buf:
                self._next_emit_id = max(self._next_emit_id, min(self._buf))

    def _emit_one(self) -> Optional[tuple[float, np.ndarray]]:
        """Pop the smallest common ID and purge everything older. No sleeping."""
        if not self._buf:
            return None
        # Smallest ID present on BOTH links.
        eid = None
        for i in sorted(self._buf):
            e = self._buf[i]
            if "A" in e and "B" in e:
                eid = i
                break
        if eid is None:
            return None
        entry = self._buf.pop(eid)
        # Purge stale single-link entries older than the emitted ID. IDs from
        # one Tx are monotonic per link, so these can never match in future.
        for si in [i for i in self._buf if i < eid]:
            del self._buf[si]
        if self._last_t_us is not None and entry["A"][0] < self._last_t_us:
            return None  # out-of-order stamp guard (strictly less: equal stamps
        # legitimately occur with coarse/broken clocks and must still emit —
        # id-keyed dict entries are unique per emission by construction)
        self._last_t_us = entry["A"][0]
        return entry["A"][0] / 1e6, np.stack([entry["A"][1], entry["B"][1]])

    def read_available(self, max_frames: int = 256) -> list[tuple[float, np.ndarray]]:
        """Emit ALL currently-fused frames (no waiting, no sleeping)."""
        self._ingest()
        out: list[tuple[float, np.ndarray]] = []
        while len(out) < max_frames:
            f = self._emit_one()
            if f is None:
                break
            out.append(f)
        return out

    def read_fused(self, timeout: float = 0.0) -> Optional[tuple[float, np.ndarray]]:
        """Return ONE fused frame (t_seconds_linkA, fused_amp (2, 190)).

        Non-blocking by default (``timeout=0``): ingests once and returns
        immediately. Only waits up to ``timeout`` seconds when the caller
        explicitly needs a blocking read (e.g. offline drains).
        """
        self._ingest()
        f = self._emit_one()
        if f is not None or timeout <= 0:
            return f
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.002)
            self._ingest()
            f = self._emit_one()
            if f is not None:
                return f
        return None
