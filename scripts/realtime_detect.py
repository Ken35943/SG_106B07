#!/usr/bin/env python3
"""
realtime_detect.py — Real-Time CSI Fall Detection Runtime Engine
=================================================================

Streams live WiFi CSI data from the ESP32-S3 receiver and reproduces the
EXACT training-time signal chain, sample-for-sample, in a causal streaming
fashion:

    190-entry amplitude vector
        → occupied-carrier selection (114 HT-LTF carriers, from pipeline state)
        → StreamingResampler   (irregular 60-65 Hz arrivals → uniform 100 Hz grid)
        → log-amplitude        (20·log10(a+1))
        → StreamingHampel      (same window/threshold/MAD constant as offline)
        → CausalSOSFilter      (stateful 0.5-40 Hz Butterworth band-pass)
        → 100-sample window    → PCA (fitted) → z-score scaler (fitted)
        → CSIFallDetector inference every `stride` grid samples

Every DSP stage is CAUSAL and STATEFUL: no look-ahead, no block-boundary
artefacts, bit-identical to ``preprocess.py`` on the same data.

Modes:
    python scripts/realtime_detect.py --port COM3            # live ESP32-S3
    python scripts/realtime_detect.py --replay <file.csv>    # offline replay
    python scripts/realtime_detect.py --mock                 # synthetic stream
"""

from __future__ import annotations

import argparse
import collections
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(_PROJECT_ROOT / "model"))

from parse_csi import CSIDataReader, extract_amplitude_phase
from csi_dsp import CausalSOSFilter, StreamingResampler, design_bandpass_sos, timestamps_to_seconds
from cnn_lstm import CSIFallDetector

logger = logging.getLogger("CSI_FALL_DETECTOR")

# Serial I/O: 921600 baud ≈ 92 KB/s; one HT40 text line ≈ 1.8 KB → ~50 lines/s
# sustainable.  Below this the resampler cannot cover the nominal grid.
_LOW_RATE_WARN_HZ = 30.0
_MAD_TO_SIGMA = 1.4826


def load_config(config_path: Path) -> dict:
    """Load config.yaml with fallback defaults."""
    if not config_path.exists():
        logger.warning("Config file not found at %s; using internal defaults", config_path)
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ═════════════════════════════════════════════════════════════════════════════
# Streaming Hampel — matches csi_dsp.hampel_filter exactly, causally
# ═════════════════════════════════════════════════════════════════════════════

class StreamingHampel:
    """Causal streaming version of :func:`csi_dsp.hampel_filter`.

    Holds a rolling window of the last ``2k+1`` raw rows.  Sample ``i`` is
    emitted when raw sample ``i+k`` arrives (the median filter's inherent
    group delay), using the same edge-padded window construction, the same
    MAD consistency constant (1.4826) and the same replacement rule as the
    offline filter — so offline and online outputs are bit-identical."""

    def __init__(self, half_window: int = 5, threshold: float = 3.0) -> None:
        self.k = int(half_window)
        self.threshold = float(threshold)
        self._buf: collections.deque[np.ndarray] = collections.deque(
            maxlen=2 * self.k + 1
        )

    def process(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Feed one row ``(F,)``; return the filtered row for sample ``i-k``
        (or ``None`` while the look-ahead window is still filling)."""
        x = np.asarray(x, dtype=np.float64)
        if not self._buf:
            # First sample defines the edge-padding (offline 'edge' mode pads
            # the start of the recording with k copies of x[0]).
            self._buf.extend([x] * (self.k + 1))
            return None
        self._buf.append(x)
        if len(self._buf) < 2 * self.k + 1:
            return None

        mat = np.asarray(self._buf, dtype=np.float64)          # (2k+1, F)
        med = np.median(mat, axis=0)
        mad = _MAD_TO_SIGMA * np.median(np.abs(mat - med), axis=0)
        out = mat[self.k].copy()
        mask = (mad > 0) & (np.abs(mat[self.k] - med) > self.threshold * mad)
        out[mask] = med[mask]
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Detector
# ═════════════════════════════════════════════════════════════════════════════

class RealtimeFallDetector:
    """End-to-end streaming fall detector: online DSP + trained model.

    The DSP parameters (subcarrier columns, band-pass edges, Hampel window,
    log-amplitude flag, PCA, scaler) come from ``pipeline_state.pkl`` written
    by ``scripts/preprocess.py``, guaranteeing train/inference identity."""

    def __init__(
        self,
        model_path: Path,
        pipeline_state_path: Path,
        config: dict,
        threshold: float = 0.80,
        consecutive_required: int = 2,
        stride: int = 25,
        cooldown_seconds: float = 5.0,
        beep_enabled: bool = True,
    ) -> None:
        self.config = config
        self.threshold = threshold
        self.consecutive_required = consecutive_required
        self.stride = stride
        self.cooldown_seconds = cooldown_seconds
        self.beep_enabled = beep_enabled and HAS_WINSOUND

        # ── 1. Load fitted preprocessing state ──────────────────────────────
        if not pipeline_state_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline state not found: {pipeline_state_path}\n"
                f"Please run 'python scripts/preprocess.py' first."
            )
        with open(pipeline_state_path, "rb") as f:
            st = pickle.load(f)

        self.pca_model = st["pca_model"]
        self.scaler = st["scaler"]
        self.subcarrier_cols = st.get("subcarrier_cols")
        self.n_raw_subcarriers = st.get("n_raw_subcarriers")
        self.log_amplitude = bool(st.get("log_amplitude", True))
        self.hampel_window = int(st.get("hampel_window", 5))
        self.hampel_threshold = float(st.get("hampel_threshold", 3.0))
        bp = st.get("bandpass", (0.5, 40.0, 4))
        self.bandpass_low, self.bandpass_high, self.bandpass_order = (
            float(bp[0]), float(bp[1]), int(bp[2]),
        )
        self.fs = float(st.get("sample_rate", 100.0))
        self.window_size = int(st.get("window_size", 100))

        # Migration guard: the fitted state carries its own sample rate, so a
        # stale pipeline_state.pkl (e.g. 100 Hz artifacts after the 50 Hz
        # transition) would silently run the wrong filter/window timing.
        cfg_fs = float(config.get("csi", {}).get("sample_rate", self.fs))
        if abs(cfg_fs - self.fs) / max(self.fs, 1e-9) > 0.01:
            logger.warning(
                "pipeline_state sample_rate (%.1f Hz) != config csi.sample_rate "
                "(%.1f Hz) — re-run scripts/preprocess.py before trusting output!",
                self.fs, cfg_fs,
            )

        if self.subcarrier_cols is None:
            logger.warning(
                "pipeline_state.pkl has no subcarrier selection — using ALL raw "
                "columns. Re-run preprocess.py for exact train/inference parity."
            )

        # ── 2. Build the streaming DSP chain ────────────────────────────────
        self.sos_filter = CausalSOSFilter(
            design_bandpass_sos(
                self.fs, self.bandpass_low, self.bandpass_high, self.bandpass_order
            )
        )
        self.resampler = StreamingResampler(self.fs)
        self.hampel = StreamingHampel(self.hampel_window, self.hampel_threshold)
        self.filt_buf: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.window_size
        )
        self.filt_count = 0

        # ── 3. Load the trained model (valid constructor kwargs only) ──────
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                f"Please train a model using 'python model/train.py' first."
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
        logger.info(
            "Loaded model checkpoint %s (trained val_acc %.2f%%)",
            model_path.name, checkpoint.get("val_acc", float("nan")),
        )

        if input_features != self.pca_model.n_components_:
            raise ValueError(
                f"Model expects {input_features} features but the PCA produces "
                f"{self.pca_model.n_components_} — retrain or re-run preprocess.py."
            )
        if self.pca_model.n_features_in_ != (
            len(self.subcarrier_cols) if self.subcarrier_cols is not None else self.n_raw_subcarriers
        ):
            logger.warning(
                "PCA expects %d features; the selected-carrier count is %d. "
                "Live packets will be truncated/padded and results may be wrong.",
                self.pca_model.n_features_in_,
                len(self.subcarrier_cols) if self.subcarrier_cols is not None else 0,
            )

        # ── 4. Alarm state ──────────────────────────────────────────────────
        self.packet_counter = 0
        self.consecutive_fall_count = 0
        self.last_alert_time = 0.0

    # ── Alarm ───────────────────────────────────────────────────────────────

    def trigger_alarm(self, p_fall: float) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print("\n" + "!" * 78)
        print(f"  [ALARM - {timestamp}] *** FALL DETECTED! *** Confidence: {p_fall * 100:.1f}%")
        print("!" * 78 + "\n")
        if self.beep_enabled:
            try:
                winsound.Beep(1800, 600)
            except Exception:
                pass

    # ── Streaming ingestion ─────────────────────────────────────────────────

    def process_packet(self, t: float, raw_amp: np.ndarray) -> Optional[tuple[float, float, str]]:
        """Feed one packet ``(time_seconds, amplitude_190)`` into the online
        pipeline.  Returns ``(p_fall, p_daily, status)`` when a windowed
        inference occurs, else ``None``."""
        self.packet_counter += 1

        if raw_amp.size != self.n_raw_subcarriers:
            logger.debug(
                "Packet subcarrier count %d != %d — dropped", raw_amp.size, self.n_raw_subcarriers
            )
            return None

        sel = raw_amp[self.subcarrier_cols] if self.subcarrier_cols is not None else raw_amp

        for row in self.resampler.push(t, sel):
            # Identical chain to PreprocessingPipeline.filter_recording
            if self.log_amplitude:
                row = 20.0 * np.log10(row + 1.0)
            h = self.hampel.process(row)
            if h is None:
                continue
            filt = self.sos_filter.process(h[None, :])[0]
            self.filt_buf.append(filt)
            self.filt_count += 1

            if (
                len(self.filt_buf) >= self.window_size
                and self.filt_count % self.stride == 0
            ):
                result = self._infer_window()
                if result is not None:
                    return result
        return None

    def _infer_window(self) -> Optional[tuple[float, float, str]]:
        window = np.asarray(self.filt_buf, dtype=np.float64)          # (100, F)
        reduced = self.pca_model.transform(window)                    # (100, 20)
        normed = self.scaler.transform(reduced)                       # (100, 20)
        tensor_in = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        p_fall = float(probs[0])      # class 0 = fall
        p_daily = float(probs[1])
        now = time.monotonic()
        status = "NORMAL"

        if p_fall >= self.threshold:
            self.consecutive_fall_count += 1
            if self.consecutive_fall_count >= self.consecutive_required:
                status = "FALL_DETECTED"
                if now - self.last_alert_time > self.cooldown_seconds:
                    self.trigger_alarm(p_fall)
                    self.last_alert_time = now
            else:
                status = "VERIFYING"
        else:
            self.consecutive_fall_count = 0

        return p_fall, p_daily, status


# ═════════════════════════════════════════════════════════════════════════════
# Stream sources — all yield (t_seconds, amplitude_190)
# ═════════════════════════════════════════════════════════════════════════════

def stream_from_serial(
    port: str,
    baud_rate: int,
    raw_subcarriers: int,
    monitor_seconds: float = 10.0,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Yield ``(t, amp)`` from the physical ESP32-S3 receiver.

    ``t`` is reconstructed from the firmware's microsecond ``local_timestamp``
    (32-bit counter, wraps every ~71.6 min) so the resampler sees the TRUE
    packet timing instead of assuming 100 Hz."""
    logger.info("Connecting to ESP32-S3 on %s @ %d baud...", port, baud_rate)
    with CSIDataReader(port=port, baud_rate=baud_rate) as reader:
        logger.info("Connected! Streaming live CSI packets...")
        t = 0.0
        prev_raw = None
        n_pkts = 0
        t_mon = time.monotonic()

        while True:
            try:
                pkt = reader.read_one()
            except Exception:
                logger.exception("Serial read failed — stopping cleanly")
                return
            if pkt is None:
                continue
            raw = pkt.get("local_timestamp")
            if isinstance(raw, int):
                if prev_raw is None:
                    prev_raw = raw
                else:
                    d = raw - prev_raw
                    if d < 0:                       # 32-bit microsecond wrap
                        d += 2 ** 32
                    if 0 < d < 2 ** 31:             # sane inter-packet delta
                        t += d / 1e6
                        prev_raw = raw
                    # else: duplicate/garbage stamp — keep t unchanged
            else:
                t += 1.0 / 100.0                    # fallback: nominal grid

            amp, _ = extract_amplitude_phase(pkt.get("raw_data", []))
            if amp.size != raw_subcarriers:
                continue
            yield t, amp

            n_pkts += 1
            if time.monotonic() - t_mon >= monitor_seconds:
                rate = n_pkts / (time.monotonic() - t_mon)
                if rate < _LOW_RATE_WARN_HZ:
                    logger.warning(
                        "Low packet rate: %.1f pkt/s (nominal 100). The resampler "
                        "bridges gaps but fidelity above %.1f Hz is lost.",
                        rate, rate / 2,
                    )
                n_pkts = 0
                t_mon = time.monotonic()


def stream_from_csv(csv_path: Path) -> Generator[tuple[float, np.ndarray], None, None]:
    """Replay a recorded CSV (uses its real timestamps)."""
    import pandas as pd

    logger.info("Replaying CSI recording from %s...", csv_path)
    df = pd.read_csv(csv_path)
    amp_cols = [c for c in df.columns if c.startswith("amplitude_")]
    if not amp_cols:
        raise ValueError(f"No amplitude columns found in {csv_path}")
    amps = df[amp_cols].values.astype(np.float64)
    if "timestamp" in df.columns:
        t_sec = timestamps_to_seconds(df["timestamp"].values.astype(np.float64))
    else:
        t_sec = np.arange(len(amps)) / 100.0

    while True:  # loop for continuous demonstration
        for t, row in zip(t_sec, amps):
            yield float(t), row


def stream_mock(
    n_subcarriers: int,
    target_hz: float = 100.0,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Synthetic CSI stream (amplitudes in the same range as real hardware)
    with a periodic simulated fall every ~10 s."""
    logger.info("Starting synthetic mock CSI stream (%d subcarriers, %.0f Hz)...", n_subcarriers, target_hz)
    delay = 1.0 / target_hz
    rng = np.random.default_rng(7)
    t = 0
    while True:
        base = 28.0 + 4.0 * np.sin(0.06 * t) + rng.normal(0, 1.2, n_subcarriers)
        cycle = t % 1000
        if 400 <= cycle <= 430:
            base += 25.0 + rng.normal(0, 8.0, n_subcarriers)     # fall transient
        elif 431 <= cycle <= 600:
            base = 24.0 + rng.normal(0, 0.4, n_subcarriers)      # stillness on floor
        yield t / target_hz, base.astype(np.float64)
        t += 1
        time.sleep(delay)


# ═════════════════════════════════════════════════════════════════════════════
# Visual terminal meter
# ═════════════════════════════════════════════════════════════════════════════

def render_meter(p_fall: float, bar_width: int = 20) -> str:
    filled = int(round(p_fall * bar_width))
    bar = "=" * filled + "-" * (bar_width - filled)
    return f"[{bar}] {p_fall * 100:5.1f}%"


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Real-time CSI Fall Detection Runtime Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=_PROJECT_ROOT / "config.yaml", help="Path to config.yaml")
    parser.add_argument("--port", type=str, default=None, help="Receiver COM port (overrides config)")
    parser.add_argument("--baud", type=int, default=None, help="Baud rate (overrides config)")
    parser.add_argument("--model", type=Path, default=_PROJECT_ROOT / "model" / "saved" / "best_model.pth", help="Model checkpoint path")
    parser.add_argument("--pipeline", type=Path, default=_PROJECT_ROOT / "data" / "processed" / "pipeline_state.pkl", help="Fitted pipeline state path")
    parser.add_argument("--replay", type=Path, default=None, help="Replay an existing CSV file instead of live serial")
    parser.add_argument("--mock", action="store_true", help="Run with simulated synthetic CSI packets")
    parser.add_argument("--threshold", type=float, default=None, help="Fall confidence threshold (e.g. 0.80)")
    parser.add_argument("--stride", type=int, default=None, help="Inference stride in grid samples (e.g. 12 ≈ 240 ms at 50 Hz)")
    parser.add_argument("--no-beep", action="store_true", help="Disable audio alarm")
    args = parser.parse_args()

    cfg = load_config(args.config)
    port = args.port or cfg.get("hardware", {}).get("receiver", {}).get("port", "COM3")
    baud = args.baud or cfg.get("hardware", {}).get("baud_rate", 921600)
    rt_cfg = cfg.get("realtime", {})
    threshold = args.threshold if args.threshold is not None else rt_cfg.get("fall_threshold", 0.80)
    stride = args.stride if args.stride is not None else rt_cfg.get("stride", 25)
    consecutive = rt_cfg.get("consecutive_windows", 2)
    cooldown = rt_cfg.get("cooldown_seconds", 5.0)
    beep_enabled = not args.no_beep and rt_cfg.get("beep_enabled", True)

    print("=" * 76)
    print("  ESP32-S3 CSI Real-Time Fall Detection System")
    print("=" * 76)
    print(f"  Model checkpoint : {args.model}")
    print(f"  Pipeline state   : {args.pipeline}")
    print(f"  Fall Threshold   : {threshold * 100:.0f}%")
    print(f"  Alarm Cooldown   : {cooldown}s")
    print("=" * 76)

    detector = RealtimeFallDetector(
        model_path=args.model,
        pipeline_state_path=args.pipeline,
        config=cfg,
        threshold=threshold,
        consecutive_required=consecutive,
        stride=stride,
        cooldown_seconds=cooldown,
        beep_enabled=beep_enabled,
    )
    # fs-aware stride readout (was hardcoded as stride*10 ms = 100 Hz assumption)
    print(f"  Inference Stride : every {stride} grid samples (~{stride * 1000 / detector.fs:.0f} ms @ {detector.fs:.0f} Hz)")

    if args.replay:
        stream = stream_from_csv(args.replay)
    elif args.mock:
        stream = stream_mock(detector.n_raw_subcarriers or 190)
    else:
        try:
            stream = stream_from_serial(
                port=port, baud_rate=baud, raw_subcarriers=detector.n_raw_subcarriers or 190
            )
        except Exception as e:
            logger.error("Failed to connect to %s: %s", port, e)
            logger.info("Tip: run with '--replay <file.csv>' or '--mock' to test without hardware!")
            sys.exit(1)

    print("\n[READY] Monitoring live CSI activity... (Press Ctrl+C to stop)\n")

    pkt_count = 0
    start_time = time.monotonic()

    try:
        for t, amp in stream:
            pkt_count += 1
            result = detector.process_packet(t, amp)
            if result is not None:
                p_fall, p_daily, status = result
                now_str = datetime.now().strftime("%H:%M:%S")
                rate = pkt_count / max(0.001, time.monotonic() - start_time)

                if status == "FALL_DETECTED":
                    tag = "FALL!   "
                elif status == "VERIFYING":
                    tag = "CHECKING"
                else:
                    tag = "NORMAL  "

                meter = render_meter(p_fall)
                sys.stdout.write(
                    f"\r[{now_str}] Status: {tag} | Fall: {meter} | Daily: {p_daily * 100:5.1f}% | {rate:4.0f} pkt/s"
                )
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\n[EXIT] Real-time detection stopped by user.")


if __name__ == "__main__":
    main()
