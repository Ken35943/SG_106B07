#!/usr/bin/env python3
"""
preprocess.py — CSI Preprocessing Pipeline
============================================

Reads raw CSI CSV files produced by ``collect_csi.py``, applies a chain of
signal-processing and dimensionality-reduction steps, segments the result
into fixed-length sliding windows, and writes ``X.npy`` / ``y.npy`` arrays
ready for model training.

Pipeline stages
---------------
0. **Subcarrier selection** — drop the 24 null/guard carriers and the
   redundant legacy LLTF block (HT40: 190 → 114 occupied HT-LTF carriers).
1. **Uniform resampling** — packets arrive irregularly (measured 60–65 Hz with
   10–77 ms jitter); interpolate onto the nominal grid so every window has the
   same physical duration.
2. **Log-amplitude** — turns AGC / distance gain differences between sessions
   into an additive offset that the high-pass removes.
3. **Hampel filter** — vectorised outlier removal per subcarrier.
4. **Causal Butterworth band-pass** — removes the static path (< 0.5 Hz) and
   anti-aliases; *identical* code path to the real-time detector.
5. **PCA** — dimensionality reduction across subcarriers.
6. **Sliding-window segmentation** — fixed-length windows with overlap;
   every window carries its recording id (``groups.npy``) so the train/test
   split can be done per recording and not per overlapping window.
7. **Normalisation** — z-score or min-max per feature.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csi_dsp import (  # noqa: E402
    CausalSOSFilter,
    design_bandpass_sos,
    estimate_sample_rate,
    hampel_filter as _hampel_vectorised,
    resample_uniform,
    select_subcarriers,
    timestamps_to_seconds,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# Individual processing functions
# ═════════════════════════════════════════════════════════════════════════════

def hampel_filter(
    data: np.ndarray,
    window_size: int = 5,
    threshold: float = 3.0,
) -> np.ndarray:
    """Hampel outlier filter along axis-0 of *data* ``(n_samples, n_features)``.

    Thin wrapper over the vectorised implementation in :mod:`csi_dsp`
    (kept for API compatibility; ~100× faster than the former Python loop).
    """
    return _hampel_vectorised(data, half_window=window_size, n_sigmas=threshold)


def causal_bandpass(
    data: np.ndarray,
    low_hz: float,
    high_hz: float,
    sample_rate: float,
    order: int = 4,
) -> np.ndarray:
    """Causal, stateful Butterworth band-pass (see :class:`csi_dsp.CausalSOSFilter`).

    Using the *same* causal filter offline and online guarantees that the
    training distribution matches what the detector sees at run time; the
    previous ``filtfilt`` (zero-phase, non-causal) could never be reproduced
    on a live stream.
    """
    sos = design_bandpass_sos(sample_rate, low_hz, high_hz, order)
    return CausalSOSFilter(sos).process(data)


def apply_pca(
    data: np.ndarray,
    n_components: int = 20,
    pca_model: Optional[PCA] = None,
) -> tuple[np.ndarray, PCA]:
    """Reduce feature dimensionality with PCA.

    Parameters
    ----------
    data : np.ndarray
        ``(n_samples, n_features)``
    n_components : int
        Target number of principal components.
    pca_model : PCA | None
        A **fitted** PCA to reuse (inference mode).  If ``None`` a new
        PCA is fitted.

    Returns
    -------
    transformed : np.ndarray
        ``(n_samples, n_components)``
    pca : PCA
        The fitted PCA object.
    """
    if pca_model is not None:
        return pca_model.transform(data), pca_model

    n_components = min(n_components, data.shape[1], data.shape[0])
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "PCA: %d → %d components (%.1f%% variance explained)",
        data.shape[1],
        n_components,
        explained * 100,
    )
    return transformed, pca


def segment_sliding_window(
    data: np.ndarray,
    window_size: int = 100,
    overlap: float = 0.5,
) -> np.ndarray:
    """Split a 2-D array into overlapping windows.

    Parameters
    ----------
    data : np.ndarray
        ``(n_samples, n_features)``
    window_size : int
        Samples per window.
    overlap : float
        Fractional overlap in ``[0, 1)``.

    Returns
    -------
    np.ndarray
        ``(n_windows, window_size, n_features)``
    """
    step = max(1, int(window_size * (1 - overlap)))
    n_samples, n_features = data.shape
    windows: list[np.ndarray] = []

    for start in range(0, n_samples - window_size + 1, step):
        windows.append(data[start : start + window_size])

    if not windows:
        logger.warning(
            "Data too short (%d) for window_size=%d; returning empty array",
            n_samples,
            window_size,
        )
        return np.empty((0, window_size, n_features))

    return np.stack(windows)


def normalize(
    data: np.ndarray,
    method: Literal["zscore", "minmax"] = "zscore",
    scaler: Optional[StandardScaler | MinMaxScaler] = None,
) -> tuple[np.ndarray, StandardScaler | MinMaxScaler]:
    """Normalise features (last axis) across the dataset.

    Parameters
    ----------
    data : np.ndarray
        2-D ``(n_samples, n_features)`` or 3-D ``(n_windows, win_size, n_features)``.
    method : str
        ``"zscore"`` or ``"minmax"``.
    scaler : sklearn scaler | None
        Pre-fitted scaler for inference.

    Returns
    -------
    normalised : np.ndarray
        Same shape as input.
    scaler : StandardScaler | MinMaxScaler
        Fitted scaler object.
    """
    original_shape = data.shape
    if data.ndim == 3:
        n_win, win_size, n_feat = data.shape
        data_2d = data.reshape(-1, n_feat)
    else:
        data_2d = data

    if scaler is None:
        if method == "zscore":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalisation method: {method!r}")
        scaler.fit(data_2d)

    normalised = scaler.transform(data_2d)

    if len(original_shape) == 3:
        normalised = normalised.reshape(original_shape)

    return normalised, scaler


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═════════════════════════════════════════════════════════════════════════════

class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for CSI amplitude data.

    The pipeline is configured via the ``preprocessing`` section of
    ``config.yaml`` and chains:  Hampel → Butterworth → PCA → Window → Norm.
    """

    def __init__(self, config: dict) -> None:
        pp = config["preprocessing"]
        self.hampel_window: int = pp["hampel_window"]
        self.hampel_threshold: float = pp["hampel_threshold"]
        self.bandpass_low: float = float(pp.get("bandpass_low_hz", 0.5))
        self.bandpass_high: float = float(pp.get("bandpass_high_hz", 40.0))
        self.bandpass_order: int = int(pp.get("bandpass_order", 4))
        self.log_amplitude: bool = bool(pp.get("log_amplitude", True))
        self.resample: bool = bool(pp.get("resample_to_uniform", True))
        self.min_rate_ratio: float = float(pp.get("min_rate_ratio", 0.5))
        self.sample_rate: float = float(config["csi"]["sample_rate"])
        self.pca_components: int = pp["pca_components"]
        self.window_size: int = pp["window_size"]
        self.window_overlap: float = pp["window_overlap"]
        self.norm_method: str = pp["normalization"]

        # Fitted transformers (populated after fit_transform)
        self.pca_model: Optional[PCA] = None
        self.scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.subcarrier_cols: Optional[np.ndarray] = None   # occupied carriers
        self.subcarrier_k: Optional[np.ndarray] = None      # physical indices
        self.n_raw_subcarriers: Optional[int] = None

        # Label mapping from config
        self._label_map = self._build_label_map(config)

    # ── Per-recording signal chain (shared with realtime via csi_dsp) ────

    def filter_recording(self, amp: np.ndarray) -> np.ndarray:
        """Log-amp → Hampel → causal band-pass on an occupied-carrier matrix
        (T, F) that is already on a uniform time grid."""
        if self.log_amplitude:
            amp = 20.0 * np.log10(amp + 1.0)
        amp = hampel_filter(amp, window_size=self.hampel_window, threshold=self.hampel_threshold)
        return causal_bandpass(
            amp,
            low_hz=self.bandpass_low,
            high_hz=self.bandpass_high,
            sample_rate=self.sample_rate,
            order=self.bandpass_order,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_label_map(config: dict) -> dict[str, int]:
        """Return ``{activity_name: int_label}``."""
        lm: dict[str, int] = {}
        for group in config["activity_labels"].values():
            label = int(group["label"])
            for act in group["activities"]:
                lm[act] = label
        return lm

    def _load_csv(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load a raw CSV → ``(t_seconds, amplitude)`` with amplitude (T, F_raw)."""
        df = pd.read_csv(path)
        amp_cols = [c for c in df.columns if c.startswith("amplitude_")]
        if not amp_cols:
            raise ValueError(f"No amplitude columns found in {path}")
        amp = df[amp_cols].values.astype(np.float64)
        if "timestamp" in df.columns:
            t = timestamps_to_seconds(df["timestamp"].values.astype(np.float64))
        else:
            t = np.arange(len(df)) / self.sample_rate
        return t, amp

    def _select_subcarriers(self, amp: np.ndarray) -> np.ndarray:
        """Lock the occupied-carrier column set on first use, then apply it."""
        if self.subcarrier_cols is None:
            cols, k = select_subcarriers(amp)
            self.subcarrier_cols, self.subcarrier_k = cols, k
            self.n_raw_subcarriers = amp.shape[1]
            logger.info(
                "Subcarrier selection: %d raw entries → %d occupied carriers%s",
                amp.shape[1], len(cols),
                " (verified ESP32-S3 HT40 HT-LTF map)" if k is not None else " (data-driven mask)",
            )
        if amp.shape[1] != self.n_raw_subcarriers:
            raise ValueError(
                f"Subcarrier count {amp.shape[1]} differs from pipeline's {self.n_raw_subcarriers}"
            )
        return amp[:, self.subcarrier_cols]

    def prepare_recording(self, t_sec: np.ndarray, amp_raw: np.ndarray, name: str = "") -> Optional[np.ndarray]:
        """Raw CSV arrays → filtered occupied-carrier matrix on a uniform grid.

        Returns ``None`` (and logs why) when the recording is unusable."""
        amp = self._select_subcarriers(amp_raw)
        fs_meas = estimate_sample_rate(t_sec)
        if not np.isfinite(fs_meas) or fs_meas < self.min_rate_ratio * self.sample_rate:
            logger.warning(
                "%s: measured packet rate %.1f Hz < %.0f%% of nominal %.0f Hz — skipping",
                name, fs_meas, 100 * self.min_rate_ratio, self.sample_rate,
            )
            return None
        if fs_meas < 0.9 * self.sample_rate:
            logger.warning(
                "%s: measured %.1f Hz (nominal %.0f Hz) — content above %.0f Hz is NOT recoverable",
                name, fs_meas, self.sample_rate, fs_meas / 2,
            )
        if self.resample:
            _, amp = resample_uniform(t_sec, amp, self.sample_rate)
        return self.filter_recording(amp)

    # ── Core API ─────────────────────────────────────────────────────────

    def fit_transform(self, raw_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Read all raw CSVs, run the full pipeline, and return
        ``(X, y, groups, group_names)``.

        Collects and filters data across all activity files, fits a global PCA
        across the entire pooled dataset (eliminating file-ordering bias),
        transforms each recording, and segments into sliding windows.

        Parameters
        ----------
        raw_dir : Path
            Directory containing per-activity subdirectories of CSV files.

        Returns
        -------
        X : np.ndarray
            ``(n_windows, window_size, n_pca_components)``
        y : np.ndarray
            ``(n_windows,)`` integer labels.
        groups : np.ndarray
            ``(n_windows,)`` integer recording id of every window — REQUIRED by
            ``model/dataset.py`` to split by recording and avoid leakage
            between overlapping windows of the same clip.
        group_names : list[str]
            ``group_names[g]`` is the source file of recording ``g``.
        """
        # Phase 1: Load and filter all files
        file_entries: list[tuple[np.ndarray, int, str]] = []

        for activity_dir in sorted(raw_dir.iterdir()):
            if not activity_dir.is_dir():
                continue
            activity = activity_dir.name
            if activity not in self._label_map:
                logger.warning("Unknown activity dir '%s' — skipping", activity)
                continue
            label = self._label_map[activity]

            csv_files = sorted(activity_dir.glob("*.csv"))
            logger.info(
                "Processing %d files for '%s' (label=%d)",
                len(csv_files),
                activity,
                label,
            )

            for csv_path in csv_files:
                try:
                    t_sec, amp_raw = self._load_csv(csv_path)
                    amp = self.prepare_recording(t_sec, amp_raw, name=csv_path.name)
                except Exception as exc:
                    logger.error("Failed to load %s: %s", csv_path, exc)
                    continue

                if amp is None:
                    continue

                if amp.shape[0] < self.window_size:
                    logger.warning(
                        "File %s too short (%d rows after resampling) — skipping",
                        csv_path.name,
                        amp.shape[0],
                    )
                    continue

                file_entries.append((amp, label, f"{activity}/{csv_path.name}"))

        if not file_entries:
            raise RuntimeError(f"No valid data produced — check {raw_dir}")

        # Phase 2: Fit global PCA across all pooled samples if not pre-loaded
        if self.pca_model is None:
            all_filtered = np.concatenate([amp for amp, _, _ in file_entries], axis=0)
            n_comp = min(self.pca_components, all_filtered.shape[1], all_filtered.shape[0])
            self.pca_model = PCA(n_components=n_comp)
            self.pca_model.fit(all_filtered)
            explained = self.pca_model.explained_variance_ratio_.sum()
            logger.info(
                "Global PCA fitted across %d samples: %d → %d components (%.1f%% variance explained)",
                all_filtered.shape[0],
                all_filtered.shape[1],
                n_comp,
                explained * 100,
            )

        # Phase 3: Transform with PCA, segment into sliding windows
        all_windows: list[np.ndarray] = []
        all_labels: list[int] = []
        all_groups: list[int] = []
        group_names: list[str] = []

        for amp, label, fname in file_entries:
            # 3. PCA transform
            amp_pca = self.pca_model.transform(amp)

            # 4. Sliding window segmentation
            windows = segment_sliding_window(
                amp_pca,
                window_size=self.window_size,
                overlap=self.window_overlap,
            )

            if windows.shape[0] == 0:
                continue

            group_id = len(group_names)
            group_names.append(fname)
            all_windows.append(windows)
            all_labels.extend([label] * windows.shape[0])
            all_groups.extend([group_id] * windows.shape[0])

        if not all_windows:
            raise RuntimeError("No valid windows generated after segmentation.")

        X = np.concatenate(all_windows, axis=0)

        # 5. Normalisation
        X, self.scaler = normalize(X, method=self.norm_method)

        y = np.array(all_labels, dtype=np.int64)
        groups = np.array(all_groups, dtype=np.int64)
        logger.info(
            "Pipeline complete — X.shape=%s  y.shape=%s  recordings=%d",
            X.shape, y.shape, len(group_names),
        )
        return X, y, groups, group_names

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist everything the real-time detector needs to reproduce the
        exact training transform."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "pca_model": self.pca_model,
            "scaler": self.scaler,
            "subcarrier_cols": self.subcarrier_cols,
            "subcarrier_k": self.subcarrier_k,
            "n_raw_subcarriers": self.n_raw_subcarriers,
            "sample_rate": self.sample_rate,
            "log_amplitude": self.log_amplitude,
            "hampel_window": self.hampel_window,
            "hampel_threshold": self.hampel_threshold,
            "bandpass": (self.bandpass_low, self.bandpass_high, self.bandpass_order),
            "window_size": self.window_size,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        logger.info("Pipeline state saved → %s", path)

    def load(self, path: Path) -> None:
        """Restore fitted state from *path*."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)  # noqa: S301
        self.pca_model = state["pca_model"]
        self.scaler = state["scaler"]
        self.subcarrier_cols = state.get("subcarrier_cols")
        self.subcarrier_k = state.get("subcarrier_k")
        self.n_raw_subcarriers = state.get("n_raw_subcarriers")
        logger.info("Pipeline state loaded ← %s", path)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the preprocessing pipeline from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess raw CSI data for the fall detection model",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "config.yaml",
        help="Path to config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override raw data directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override processed output directory",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    raw_dir = args.raw_dir or _PROJECT_ROOT / config["paths"]["data_raw"]
    out_dir = args.out_dir or _PROJECT_ROOT / config["paths"]["data_processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PreprocessingPipeline(config)
    X, y, groups, group_names = pipeline.fit_transform(raw_dir)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "groups.npy", groups)
    with open(out_dir / "groups.json", "w", encoding="utf-8") as fh:
        json.dump({str(i): name for i, name in enumerate(group_names)}, fh, indent=2)
    logger.info(
        "Saved X.npy %s, y.npy %s, groups.npy (%d recordings) → %s",
        X.shape, y.shape, len(group_names), out_dir,
    )

    pipeline.save(out_dir / "pipeline_state.pkl")
    logger.info("All done ✓")


if __name__ == "__main__":
    main()
