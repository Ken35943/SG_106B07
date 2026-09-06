#!/usr/bin/env python3
"""
csi_dsp.py — Shared CSI signal-processing primitives (offline + real-time)
==========================================================================

Every function here is used by BOTH ``preprocess.py`` (training) and
``realtime_detect.py`` (inference) so that the two pipelines cannot drift
apart.  All time-domain filtering is CAUSAL and STATEFUL so that a live
stream and an offline recording produce bit-identical results.

Contents
--------
* Subcarrier layout of the ESP32-S3 HT40 CSI buffer (LLTF + HT-LTF) and
  null-carrier removal.
* Vectorised Hampel outlier filter (≈100× faster than the Python loop).
* Phase sanitisation (least-squares removal of the linear STO/CFO term).
* ``CausalSOSFilter`` — stateful second-order-section IIR band-pass.
* ``StreamingResampler`` — irregular packet arrivals → uniform time grid.
* STFT Doppler spectrogram + compact per-frame Doppler descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import ShortTimeFFT, butter, sosfilt, sosfilt_zi
from scipy.signal.windows import hann

# 2.4 GHz channel 6 centre frequency → wavelength used for Doppler → velocity.
SPEED_OF_LIGHT = 299_792_458.0
CHANNEL_6_HZ = 2.437e9
WAVELENGTH_CH6_M = SPEED_OF_LIGHT / CHANNEL_6_HZ  # ≈ 0.123 m


# ═════════════════════════════════════════════════════════════════════════════
# 1. Subcarrier layout
# ═════════════════════════════════════════════════════════════════════════════
#
# The stock esp-csi ``csi_recv`` firmware on ESP32-S3 in HT40 mode with
# ``lltf_en = htltf_en = true`` emits 384 int8 values = 192 complex numbers.
# ``parse_csi.extract_amplitude_phase`` discards the first 2 complex values
# (``first_word_invalid``) leaving 190 entries laid out as:
#
#   entry   0 ..  61 : LLTF  (legacy 20 MHz preamble, 64 pts, first 2 dropped)
#                      physical index k = entry + 2 - 32   (-30 … +31)
#   entry  62 .. 189 : HT-LTF (40 MHz, 128 pts)
#                      j = entry - 62 ;  k = j if j < 64 else j - 128
#
# Empirically verified on data/raw/walking/*.csv: the zero-amplitude
# (null) entries are exactly {0-3, 30, 57-63, 121-131, 189}, i.e.
# LLTF guards/DC, HT-LTF k∈{0,1,59..63,-64..-59,-1}.  The 802.11n HT40
# occupied set k ∈ [-58,-2] ∪ [2,58] (114 carriers) survives.

HT40_TOTAL_ENTRIES = 190
_HT40_LLTF_LEN = 62          # 64 minus the 2 invalid first-word entries
_HT40_HTLTF_LEN = 128


def ht40_htltf_layout() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(columns, k)`` for the 114 occupied HT-LTF carriers in the
    190-entry ESP32-S3 HT40 amplitude vector.

    ``columns`` indexes into the 190-vector; ``k`` is the physical
    subcarrier index (needed for phase sanitisation)."""
    j = np.arange(_HT40_HTLTF_LEN)
    k = np.where(j < 64, j, j - 128)
    occupied = (np.abs(k) >= 2) & (np.abs(k) <= 58)
    cols = _HT40_LLTF_LEN + j[occupied]
    return cols, k[occupied]


def detect_valid_subcarriers(amp: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Data-driven fallback: columns whose mean amplitude exceeds *eps*.

    Use this when the firmware layout is unknown (e.g. after switching to
    HT20 / HT-LTF-only binary frames)."""
    return np.flatnonzero(amp.mean(axis=0) > eps)


def select_subcarriers(amp: np.ndarray, eps: float = 1.0) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Pick the occupied subcarrier columns for an amplitude matrix (T, F).

    Returns ``(columns, k_or_None)``.  Uses the verified HT40 map when the
    vector has 190 entries, otherwise the data-driven mask."""
    if amp.shape[1] == HT40_TOTAL_ENTRIES:
        return ht40_htltf_layout()
    return detect_valid_subcarriers(amp, eps), None


# ═════════════════════════════════════════════════════════════════════════════
# 2. Hampel outlier filter (vectorised)
# ═════════════════════════════════════════════════════════════════════════════

_MAD_TO_SIGMA = 1.4826  # consistency constant for Gaussian data


def hampel_filter(x: np.ndarray, half_window: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """Vectorised Hampel filter along axis 0 of ``x`` (T, F).

    A sample is replaced by the local median when it deviates from it by more
    than ``n_sigmas × 1.4826 × MAD``.  Edges use edge-padding so the window is
    always full (the loop version silently shrank the window there).

    Complexity: one ``np.median`` over a (T, F, 2k+1) view — ~5 ms for
    (100, 114) versus ~500 ms for the per-element Python loop."""
    if x.ndim != 2:
        raise ValueError("hampel_filter expects a 2-D array (T, F)")
    k = int(half_window)
    if k <= 0 or x.shape[0] < 2:
        return x.copy()
    xp = np.pad(x, ((k, k), (0, 0)), mode="edge")
    win = sliding_window_view(xp, 2 * k + 1, axis=0)          # (T, F, 2k+1)
    med = np.median(win, axis=-1)                              # (T, F)
    mad = _MAD_TO_SIGMA * np.median(np.abs(win - med[..., None]), axis=-1)
    out = x.copy()
    mask = (mad > 0) & (np.abs(x - med) > n_sigmas * mad)
    out[mask] = med[mask]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3. Phase sanitisation
# ═════════════════════════════════════════════════════════════════════════════

def sanitize_phase(phase: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Remove the per-packet linear phase term (STO / SFO) and constant offset
    (CFO / PLL) from raw CSI phase.

    Model per packet ``t``:  φ̂_t(k) = φ_t(k) + 2π·k·τ_t/N + β_t + noise
    We unwrap along the subcarrier axis and subtract the least-squares line
    ``a_t·k + b_t``:

        a_t = Σ_k (k − k̄)(φ_t(k) − φ̄_t) / Σ_k (k − k̄)²
        b_t = φ̄_t − a_t·k̄

    The LS fit is the minimum-variance estimator of the slope (the classic
    end-point formula (φ_N − φ_1)/(k_N − k_1) is a noisier special case).
    It also handles the non-uniform gap at DC because the true ``k`` are used.

    Parameters
    ----------
    phase : (T, F) raw ``atan2(Q, I)`` in radians, occupied carriers only.
    k     : (F,) physical subcarrier indices matching the columns of *phase*.

    Notes
    -----
    A single-antenna ESP32-S3 receiving from an unsynchronised transmitter
    still leaves a random residual per packet after this step.  Treat
    sanitised phase as a *secondary* feature; validate its benefit on real
    data before relying on it."""
    if phase.shape[1] != k.shape[0]:
        raise ValueError("phase columns must match k")
    # The ESP32 buffer stores carriers in FFT order (0..63, -64..-1); unwrapping
    # must follow increasing physical k or the 58 → -58 seam produces a bogus
    # 2π jump.  Sort, unwrap, fit, then restore the caller's column order.
    order = np.argsort(k, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    kf = k[order].astype(np.float64)
    unwrapped = np.unwrap(phase[:, order], axis=1)
    k_c = kf - kf.mean()
    denom = float((k_c ** 2).sum())
    phi_mean = unwrapped.mean(axis=1, keepdims=True)
    a = ((unwrapped - phi_mean) * k_c[None, :]).sum(axis=1, keepdims=True) / denom
    b = phi_mean - a * kf.mean()
    clean = unwrapped - (a * kf[None, :] + b)
    return clean[:, inv]


# ═════════════════════════════════════════════════════════════════════════════
# 4. Causal stateful IIR filtering
# ═════════════════════════════════════════════════════════════════════════════

def design_bandpass_sos(
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Butterworth band-pass as second-order sections.

    ``order`` is the order of *each* edge (SciPy doubles it for band-pass), so
    ``order=4`` → 8th-order filter → 4 biquads.  The upper edge is clamped to
    0.45·fs so the design stays valid when the true packet rate is lower than
    the nominal one (e.g. 64 Hz instead of 100 Hz)."""
    nyq = fs / 2.0
    high = min(float(high_hz), 0.9 * nyq)
    low = float(low_hz)
    if not (0.0 < low < high < nyq):
        raise ValueError(f"Invalid band [{low}, {high}] for fs={fs}")
    return butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


class CausalSOSFilter:
    """Stateful, causal SOS filter for multichannel data (T, F).

    Call :meth:`process` repeatedly with consecutive blocks — the internal
    delay-line state carries over, so processing a recording in one call or
    in 1-sample blocks yields identical output (no block-boundary artefacts).

    The state is initialised with ``sosfilt_zi × x[0]`` (steady-state
    response to a step of the first sample), which suppresses the start-up
    transient of the high-pass section."""

    def __init__(self, sos: np.ndarray) -> None:
        self.sos = np.asarray(sos, dtype=np.float64)
        self._zi: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] == 0:
            return x
        if self._zi is None:
            zi0 = sosfilt_zi(self.sos)                    # (n_sections, 2)
            self._zi = zi0[:, :, None] * x[0][None, None, :]  # (n_sections, 2, F)
        y, self._zi = sosfilt(self.sos, x, axis=0, zi=self._zi)
        return y


def causal_bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """One-shot convenience wrapper (offline use) — identical to streaming."""
    return CausalSOSFilter(design_bandpass_sos(fs, low_hz, high_hz, order)).process(x)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Timing: sample-rate estimation and uniform resampling
# ═════════════════════════════════════════════════════════════════════════════

def timestamps_to_seconds(ts: np.ndarray) -> np.ndarray:
    """Normalise a CSV/serial timestamp column to seconds, starting at 0.

    The ESP32 ``rx_ctrl->timestamp`` is a 32-bit microsecond counter (wraps
    every ~71.6 min); synthetic files store float seconds.  Heuristic: a median
    inter-packet interval > 100 means microseconds."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size < 2:
        return np.zeros_like(ts)
    d = np.diff(ts)
    if np.median(np.abs(d)) > 100.0:          # microseconds
        wrap = d < 0
        if wrap.any():
            ts = ts.copy()
            ts[1:] += np.cumsum(wrap) * 2.0 ** 32
        ts = ts / 1e6
    return ts - ts[0]


def estimate_sample_rate(t_sec: np.ndarray) -> float:
    span = float(t_sec[-1] - t_sec[0])
    return (len(t_sec) - 1) / span if span > 0 else float("nan")


def resample_uniform(t_sec: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate irregularly-timed rows ``x`` (T, F) onto a uniform
    grid at ``fs`` Hz.  Returns ``(t_grid, x_grid)``.

    Linear interpolation is exact for the information actually present; it
    cannot recover content above half the *true* packet rate."""
    t_sec = np.asarray(t_sec, dtype=np.float64)
    order = np.argsort(t_sec, kind="stable")
    t_sec, x = t_sec[order], x[order]
    keep = np.concatenate(([True], np.diff(t_sec) > 0))     # drop duplicate stamps
    t_sec, x = t_sec[keep], x[keep]
    n = int(np.floor((t_sec[-1] - t_sec[0]) * fs)) + 1
    t_grid = t_sec[0] + np.arange(n) / fs
    out = np.empty((n, x.shape[1]), dtype=np.float64)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(t_grid, t_sec, x[:, c])
    return t_grid, out


class StreamingResampler:
    """Online version of :func:`resample_uniform`.

    Feed ``(t, row)`` pairs as packets arrive; receive the list of grid rows
    that became computable (all grid instants in ``(t_prev, t_now]``).  A gap
    of many grid steps (dropped packets) is bridged linearly — the caller
    should monitor :attr:`max_gap` and treat > ~5 steps as degraded input."""

    def __init__(self, fs: float) -> None:
        self.dt = 1.0 / fs
        self._t_prev: Optional[float] = None
        self._x_prev: Optional[np.ndarray] = None
        self._next_grid_t: Optional[float] = None
        self.max_gap = 0

    def reset(self) -> None:
        self._t_prev = self._x_prev = self._next_grid_t = None
        self.max_gap = 0

    def push(self, t: float, x: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        if self._t_prev is None:
            # First packet defines grid instant 0.
            self._t_prev, self._x_prev = t, x
            self._next_grid_t = t + self.dt
            return [x.copy()]
        if t <= self._t_prev:                  # duplicate / out-of-order stamp
            return []
        out: list[np.ndarray] = []
        while self._next_grid_t <= t:
            w = (self._next_grid_t - self._t_prev) / (t - self._t_prev)
            out.append(self._x_prev + w * (x - self._x_prev))
            self._next_grid_t += self.dt
        self.max_gap = max(self.max_gap, len(out))
        self._t_prev, self._x_prev = t, x
        return out


# ═════════════════════════════════════════════════════════════════════════════
# 6. Doppler spectrogram & descriptors
# ═════════════════════════════════════════════════════════════════════════════

def motion_principal_components(x_bp: np.ndarray, n_components: int = 1) -> np.ndarray:
    """First ``n`` principal components (over subcarriers) of a band-passed
    amplitude block (T, F) → (T, n).  Because the input is already zero-mean
    per subcarrier (high-passed), no centring is applied and the result is a
    pure motion signal."""
    u, s, _ = np.linalg.svd(x_bp, full_matrices=False)
    return u[:, :n_components] * s[:n_components]


@dataclass
class DopplerSpectrogram:
    f: np.ndarray        # (n_freq,) Hz  (unsigned for real input, signed for complex)
    t: np.ndarray        # (n_frames,) seconds
    S: np.ndarray        # (n_freq, n_frames) power


def doppler_spectrogram(
    x: np.ndarray,
    fs: float,
    win_sec: float = 0.5,
    hop: int = 5,
    nfft: int = 128,
) -> DopplerSpectrogram:
    """STFT power spectrogram of a 1-D motion signal.

    Defaults at fs = 100 Hz: 50-sample Hann window → 2 Hz resolution (before
    zero-padding to 128 bins ⇒ 0.78 Hz/bin display), hop 5 → 50 ms frames.
    A 0.5 s window is the shortest that still resolves a 0.4–0.7 s fall
    transient as a distinct chirp rather than a single smeared frame.

    Real input → one-sided (unsigned Doppler: |v| only, which is all a single
    antenna amplitude stream can provide).  Complex input (e.g. CSI ratio)
    → two-sided centred spectrum with signed Doppler."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[:, 0]
    nperseg = max(8, int(round(win_sec * fs)))
    nfft = max(nfft, nperseg)
    win = hann(nperseg, sym=False)
    is_complex = np.iscomplexobj(x)
    sft = ShortTimeFFT(
        win, hop=hop, fs=fs, mfft=nfft,
        fft_mode="centered" if is_complex else "onesided",
        scale_to="psd",
    )
    S = sft.spectrogram(x)                      # |STFT|², (n_freq, n_frames)
    t = sft.t(len(x))
    return DopplerSpectrogram(f=sft.f, t=t, S=S)


def doppler_descriptors(
    spec: DopplerSpectrogram,
    wavelength_m: float = WAVELENGTH_CH6_M,
    noise_floor: Optional[float] = None,
    gate_factor: float = 4.0,
) -> dict[str, np.ndarray]:
    """Per-frame Doppler descriptors that discriminate fall dynamics.

    For a reflector moving with radial speed ``v`` on a bistatic link the
    Doppler shift is ``f_D = (v/λ)(cos θ_tx + cos θ_rx) ≤ 2v/λ``.  We map
    frequency → velocity with the worst-case factor 2 (a lower bound on the
    true speed).

    Energy gate: a frame whose total power is below ``gate_factor × noise_floor``
    (default floor = 10th percentile of frame power, i.e. ≥ 10 % of the clip is
    assumed still) is "inactive" and its frequency/velocity descriptors are
    forced to 0.  Without this, a noise-only frame has a flat spectrum whose
    95th percentile sits at 0.95·f_Nyquist ⇒ a fictitious ~2.9 m/s.

    Returns arrays of length n_frames:
      energy      : log total motion power
      active      : bool gate
      centroid_hz : power-weighted mean |Doppler|
      f50_hz      : median frequency (torso-speed proxy)
      f95_hz      : 95th-percentile frequency (limb / peak-speed proxy)
      v50_mps     : f50 · λ / 2
      v95_mps     : f95 · λ / 2
      bandwidth_hz: power-weighted std around the centroid"""
    S = spec.S
    f = np.abs(spec.f)
    eps = 1e-12
    total = S.sum(axis=0) + eps
    if noise_floor is None:
        noise_floor = float(np.percentile(total, 10))
    active = total > gate_factor * noise_floor
    P = S / total
    centroid = (f[:, None] * P).sum(axis=0)
    bw = np.sqrt((((f[:, None] - centroid[None, :]) ** 2) * P).sum(axis=0))
    cdf = np.cumsum(P, axis=0)
    f50 = f[np.argmax(cdf >= 0.5, axis=0)] * active
    f95 = f[np.argmax(cdf >= 0.95, axis=0)] * active
    return {
        "energy": np.log(total),
        "active": active,
        "centroid_hz": centroid * active,
        "f50_hz": f50,
        "f95_hz": f95,
        "v50_mps": f50 * wavelength_m / 2.0,
        "v95_mps": f95 * wavelength_m / 2.0,
        "bandwidth_hz": bw * active,
    }


def fall_doppler_summary(desc: dict[str, np.ndarray], frame_dt: float) -> dict[str, float]:
    """Clip-level scalars from per-frame descriptors — a physics-based
    baseline detector and a sanity check for any learned model:

      peak_v95        : peak limb speed (m/s).  Falls: ≳ 1.5–2.5, sit-down: ≲ 1.
      burst_duration_s: time v95 stays above 50 % of its peak (falls: 0.3–0.8 s).
      post_quiet_ratio: mean energy in the 1.5 s after the peak relative to the
                        1.5 s before it — falls end in stillness (ratio ≪ 1),
                        walking does not."""
    v = desc["v95_mps"]
    e = desc["energy"]
    if v.size == 0:
        return {"peak_v95": 0.0, "burst_duration_s": 0.0, "post_quiet_ratio": 1.0}
    i_pk = int(np.argmax(v))
    above = v >= 0.5 * v[i_pk]
    # contiguous run around the peak
    lo = i_pk
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = i_pk
    while hi < len(v) - 1 and above[hi + 1]:
        hi += 1
    n15 = max(1, int(round(1.5 / frame_dt)))
    pre = e[max(0, i_pk - n15):i_pk]
    post = e[i_pk + 1:i_pk + 1 + n15]
    ratio = float(np.exp(post.mean() - pre.mean())) if pre.size and post.size else 1.0
    return {
        "peak_v95": float(v[i_pk]),
        "burst_duration_s": float((hi - lo + 1) * frame_dt),
        "post_quiet_ratio": ratio,
    }
