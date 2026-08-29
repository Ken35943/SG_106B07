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
1. **Hampel filter** — outlier removal per subcarrier.
2. **Butterworth low-pass filter** — noise suppression.
3. **PCA** — dimensionality reduction across subcarriers.
4. **Sliding-window segmentation** — fixed-length windows with overlap.
5. **Normalisation** — z-score or min-max per feature.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    """Apply a Hampel filter along axis-0 of *data* (samples × features).

    Replaces outliers (points that deviate from the local median by more
    than *threshold* × MAD) with the local median.

    Parameters
    ----------
    data : np.ndarray
        2-D array of shape ``(n_samples, n_features)``.
    window_size : int
        Half-window size for the rolling median / MAD.
    threshold : float
        Number of MADs beyond which a sample is flagged.

    Returns
    -------
    np.ndarray
        Filtered copy of *data* with same shape.
    """
    filtered = data.copy()
    n_samples, n_features = data.shape
    k = 1.4826  # consistency constant for Gaussian distribution

    for col in range(n_features):
        series = data[:, col]
        for i in range(n_samples):
            lo = max(0, i - window_size)
            hi = min(n_samples, i + window_size + 1)
            window = series[lo:hi]
            median = np.median(window)
            mad = k * np.median(np.abs(window - median))
            if mad == 0:
                continue
            if np.abs(series[i] - median) > threshold * mad:
                filtered[i, col] = median

    return filtered


def butterworth_lowpass(
    data: np.ndarray,
    cutoff: float = 30.0,
    sample_rate: float = 100.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter column-wise.

    Parameters
    ----------
    data : np.ndarray
        ``(n_samples, n_features)``
    cutoff : float
        Cut-off frequency in Hz.
    sample_rate : float
        Sampling rate in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered data, same shape.
    """
    nyquist = sample_rate / 2.0
    normalised_cutoff = cutoff / nyquist
    if normalised_cutoff >= 1.0:
        logger.warning(
            "Cutoff (%.1f Hz) ≥ Nyquist (%.1f Hz); skipping Butterworth filter",
            cutoff,
            nyquist,
        )
        return data

    b, a = butter(order, normalised_cutoff, btype="low")

    # filtfilt needs ≥ 3×padlen samples; fall back to raw if too short.
    padlen = 3 * max(len(a), len(b))
    if data.shape[0] <= padlen:
        logger.warning(
            "Signal too short (%d) for filter padlen (%d); returning raw data",
            data.shape[0],
            padlen,
        )
        return data

    return filtfilt(b, a, data, axis=0).astype(data.dtype)


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
        self.butterworth_cutoff: float = pp["butterworth_cutoff"]
        self.butterworth_order: int = pp["butterworth_order"]
        self.sample_rate: float = config["csi"]["sample_rate"]
        self.pca_components: int = pp["pca_components"]
        self.window_size: int = pp["window_size"]
        self.window_overlap: float = pp["window_overlap"]
        self.norm_method: str = pp["normalization"]

        # Fitted transformers (populated after fit_transform)
        self.pca_model: Optional[PCA] = None
        self.scaler: Optional[StandardScaler | MinMaxScaler] = None

        # Label mapping from config
        self._label_map = self._build_label_map(config)

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

    def _load_csv(self, path: Path) -> np.ndarray:
        """Load a raw CSV and return the amplitude columns as a 2-D array."""
        df = pd.read_csv(path)
        amp_cols = [c for c in df.columns if c.startswith("amplitude_")]
        if not amp_cols:
            raise ValueError(f"No amplitude columns found in {path}")
        return df[amp_cols].values.astype(np.float64)

    # ── Core API ─────────────────────────────────────────────────────────

    def fit_transform(self, raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read all raw CSVs, run the full pipeline, and return ``(X, y)``.

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
        """
        all_windows: list[np.ndarray] = []
        all_labels: list[int] = []

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
                    amp = self._load_csv(csv_path)
                except Exception as exc:
                    logger.error("Failed to load %s: %s", csv_path, exc)
                    continue

                if amp.shape[0] < self.window_size:
                    logger.warning(
                        "File %s too short (%d rows) — skipping",
                        csv_path.name,
                        amp.shape[0],
                    )
                    continue

                # 1. Hampel
                amp = hampel_filter(
                    amp,
                    window_size=self.hampel_window,
                    threshold=self.hampel_threshold,
                )

                # 2. Butterworth
                amp = butterworth_lowpass(
                    amp,
                    cutoff=self.butterworth_cutoff,
                    sample_rate=self.sample_rate,
                    order=self.butterworth_order,
                )

                # 3. PCA — fit on first file, transform on rest
                amp, self.pca_model = apply_pca(
                    amp,
                    n_components=self.pca_components,
                    pca_model=self.pca_model,
                )

                # 4. Sliding window
                windows = segment_sliding_window(
                    amp,
                    window_size=self.window_size,
                    overlap=self.window_overlap,
                )

                if windows.shape[0] == 0:
                    continue

                all_windows.append(windows)
                all_labels.extend([label] * windows.shape[0])

        if not all_windows:
            raise RuntimeError("No valid data produced — check data/raw/")

        X = np.concatenate(all_windows, axis=0)

        # 5. Normalisation
        X, self.scaler = normalize(X, method=self.norm_method)

        y = np.array(all_labels, dtype=np.int64)
        logger.info("Pipeline complete — X.shape=%s  y.shape=%s", X.shape, y.shape)
        return X, y

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist the fitted PCA + scaler to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "pca_model": self.pca_model,
            "scaler": self.scaler,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        logger.info("Pipeline state saved → %s", path)

    def load(self, path: Path) -> None:
        """Restore fitted PCA + scaler from *path*."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)  # noqa: S301
        self.pca_model = state["pca_model"]
        self.scaler = state["scaler"]
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
    X, y = pipeline.fit_transform(raw_dir)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    logger.info("Saved X.npy (%s) and y.npy (%s) → %s", X.shape, y.shape, out_dir)

    pipeline.save(out_dir / "pipeline_state.pkl")
    logger.info("All done ✓")


if __name__ == "__main__":
    main()
