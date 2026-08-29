#!/usr/bin/env python3
"""
parse_csi.py — ESP32-S3 CSI Data Parser
========================================

Parses raw CSI serial output from the ESP32-S3 receiver, extracts
amplitude / phase arrays, and provides a streaming serial reader.

Serial line format
------------------
CSI_DATA,<id>,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<bandwidth>,
<smoothing>,<not_sounding>,<aggregation>,<stbc>,<fec_coding>,<sgi>,
<noise_floor>,<ampdu_cnt>,<channel>,<secondary_channel>,
<local_timestamp>,<ant>,<sig_len>,<rx_state>,<len>,<first_word>,<data>

where ``<data>`` is a bracketed comma-separated list of signed integers:
``"[Q0,I0,Q1,I1,...]"``

The first 4 values (2 IQ pairs → ``first_word_invalid``) are skipped.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np
import serial

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
_CSI_PREFIX = "CSI_DATA"
_FIRST_WORD_SKIP = 4  # Skip first 4 raw values (2 IQ pairs)
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")

# Field names in the CSV-style serial line (after splitting on commas that
# are *outside* the bracket-delimited data payload).
_FIELD_NAMES = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
    "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word",
    "data",
]


# ── Public helpers ───────────────────────────────────────────────────────────

def parse_csi_line(line: str) -> dict:
    """Parse a single CSI serial output line into a structured dict.

    Parameters
    ----------
    line : str
        Raw line read from the serial port.

    Returns
    -------
    dict
        Parsed fields including ``raw_data`` (list[int]) extracted from the
        bracketed payload.

    Raises
    ------
    ValueError
        If the line does not start with ``CSI_DATA`` or the payload cannot
        be parsed.
    """
    line = line.strip()
    if not line.startswith(_CSI_PREFIX):
        raise ValueError(f"Line does not start with {_CSI_PREFIX!r}")

    # Extract bracketed data first so commas inside it don't split.
    bracket_match = _BRACKET_RE.search(line)
    if bracket_match is None:
        raise ValueError("No bracketed data payload found in line")

    raw_ints_str = bracket_match.group(1)
    raw_data = [int(v) for v in raw_ints_str.split(",")]

    # Remove the bracket section and split the remaining header fields.
    header = line[: bracket_match.start()].rstrip(",")
    parts = [p.strip() for p in header.split(",")]

    result: dict = {}
    for idx, name in enumerate(_FIELD_NAMES[:-1]):  # all except "data"
        if idx < len(parts):
            result[name] = parts[idx]
    result["raw_data"] = raw_data

    # Cast numeric fields where useful.
    for int_field in (
        "id", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing",
        "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
        "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
        "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word",
    ):
        if int_field in result:
            try:
                result[int_field] = int(result[int_field])
            except (ValueError, TypeError):
                pass  # keep as string

    return result


def extract_amplitude_phase(
    raw_data: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert interleaved IQ values to amplitude and phase arrays.

    The raw payload is ``[Q0, I0, Q1, I1, ...]``.  The first
    ``_FIRST_WORD_SKIP`` (4) values are discarded (first_word_invalid).

    Parameters
    ----------
    raw_data : list[int]
        Raw signed-integer IQ samples from the CSI payload.

    Returns
    -------
    amplitude : np.ndarray
        ``sqrt(I² + Q²)`` for each valid subcarrier.
    phase : np.ndarray
        ``atan2(Q, I)`` for each valid subcarrier (radians).
    """
    trimmed = raw_data[_FIRST_WORD_SKIP:]
    if len(trimmed) % 2 != 0:
        logger.debug(
            "Odd number of IQ values after trimming (%d); dropping last value",
            len(trimmed),
        )
        trimmed = trimmed[:-1]

    iq = np.array(trimmed, dtype=np.float64).reshape(-1, 2)
    q_vals = iq[:, 0]
    i_vals = iq[:, 1]

    amplitude = np.sqrt(i_vals ** 2 + q_vals ** 2)
    phase = np.arctan2(q_vals, i_vals)
    return amplitude, phase


# ── CSIDataReader ────────────────────────────────────────────────────────────

@dataclass
class CSIDataReader:
    """Blocking serial reader for ESP32-S3 CSI data.

    Usage
    -----
    >>> with CSIDataReader(port="COM3", baud_rate=921600) as reader:
    ...     for packet in reader.read_stream(duration=10.0):
    ...         print(packet["rssi"])
    """

    port: str
    baud_rate: int = 921_600
    timeout: float = 1.0
    _serial: Optional[serial.Serial] = field(default=None, init=False, repr=False)

    # ── Connection lifecycle ─────────────────────────────────────────────
    def open(self) -> None:
        """Open the serial connection."""
        if self._serial is not None and self._serial.is_open:
            logger.debug("Serial port %s already open", self.port)
            return
        logger.info("Opening serial port %s @ %d baud", self.port, self.baud_rate)
        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baud_rate,
            timeout=self.timeout,
        )
        # Flush any stale bytes sitting in the buffer.
        self._serial.reset_input_buffer()

    def close(self) -> None:
        """Close the serial connection."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            logger.info("Closed serial port %s", self.port)
        self._serial = None

    def __enter__(self) -> "CSIDataReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.close()

    # ── Reading ──────────────────────────────────────────────────────────
    def read_one(self) -> Optional[dict]:
        """Read and parse a single CSI packet (blocking).

        Returns ``None`` if a non-CSI line is encountered or the line
        cannot be parsed.
        """
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("Serial port is not open — call open() first")

        raw_line = self._serial.readline()
        if not raw_line:
            return None

        try:
            line = raw_line.decode("utf-8", errors="replace")
        except Exception:
            logger.debug("Failed to decode serial bytes")
            return None

        if not line.startswith(_CSI_PREFIX):
            return None

        try:
            return parse_csi_line(line)
        except ValueError as exc:
            logger.debug("Skipping unparseable CSI line: %s", exc)
            return None

    def read_stream(
        self,
        duration: float = 10.0,
        max_packets: Optional[int] = None,
    ) -> Generator[dict, None, None]:
        """Yield parsed CSI packets for *duration* seconds or *max_packets*.

        Parameters
        ----------
        duration : float
            Maximum wall-clock seconds to collect data.
        max_packets : int | None
            Stop after this many valid packets (``None`` = no limit).

        Yields
        ------
        dict
            Parsed CSI packet (see :func:`parse_csi_line`).
        """
        start = time.monotonic()
        count = 0
        while time.monotonic() - start < duration:
            pkt = self.read_one()
            if pkt is not None:
                yield pkt
                count += 1
                if max_packets is not None and count >= max_packets:
                    logger.info("Reached max_packets=%d — stopping", max_packets)
                    return


# ── CLI entry point ──────────────────────────────────────────────────────────

def _main() -> None:
    """Quick smoke test: read 20 packets from the receiver and print them."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Parse ESP32-S3 CSI serial data")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port")
    parser.add_argument("--baud", type=int, default=921_600, help="Baud rate")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to read")
    parser.add_argument("--max-packets", type=int, default=20, help="Max packets")
    args = parser.parse_args()

    with CSIDataReader(port=args.port, baud_rate=args.baud) as reader:
        for pkt in reader.read_stream(
            duration=args.duration,
            max_packets=args.max_packets,
        ):
            amp, phase = extract_amplitude_phase(pkt["raw_data"])
            print(
                f"RSSI={pkt['rssi']:>4}  ch={pkt['channel']:>2}  "
                f"subcarriers={len(amp)}  amp_mean={amp.mean():.1f}"
            )


if __name__ == "__main__":
    _main()
