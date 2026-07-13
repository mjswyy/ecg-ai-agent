"""Feature Bank — Aggregated ECG feature extraction.

Combines R-peak detection, HRV analysis, and QT measurement into a
unified interface. Produces both structured dict and flat vector outputs.

Usage:
    bank = FeatureBank()
    features = bank.extract_all(ecg_12lead, fs=500)
    vector = bank.to_vector(features)  # → np.ndarray (N_features,)
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np

from .r_peak_detector import RPeakDetector
from .hrv_analyzer import HRVAnalyzer
from .qt_analyzer import QTAnalyzer

logger = logging.getLogger(__name__)


class FeatureBank:
    """Aggregated ECG feature extraction.

    Extracts R-peaks, HRV metrics, and QT intervals in one call.
    """

    def __init__(
        self,
        r_peak_method: str = "pan_tompkins",
        lead_for_analysis: int = 1,  # Lead II
    ):
        self.r_peak_detector = RPeakDetector(method=r_peak_method)
        self.hrv_analyzer = HRVAnalyzer()
        self.qt_analyzer = QTAnalyzer()
        self.lead_for_analysis = lead_for_analysis

    def extract_all(
        self,
        ecg: np.ndarray,
        fs: float,
        sex: str = "Unknown",
        include_raw: bool = False,
    ) -> Dict:
        """Extract all ECG features.

        Args:
            ecg: 12-lead ECG signal (12, L) or single-lead (L,).
            fs: Sampling frequency in Hz.
            sex: Patient sex for QT thresholds.
            include_raw: If True, include raw R-peak positions and RR intervals.

        Returns:
            Comprehensive feature dict.
        """
        # Extract lead for analysis
        if ecg.ndim == 2:
            lead_idx = min(self.lead_for_analysis, ecg.shape[0] - 1)
            lead_signal = ecg[lead_idx]
        else:
            lead_signal = ecg

        # R-peak detection
        r_result = self.r_peak_detector.detect(lead_signal, fs)

        features = {
            "heart_rate": r_result["heart_rate"],
            "hr_std": r_result["hr_std"],
            "num_beats": r_result["num_beats"],
        }

        # HRV analysis
        rr = r_result.get("rr_intervals", np.array([]))
        rr_ts = r_result.get("r_peaks_ms", None)
        # Ensure numpy arrays
        if isinstance(rr, list):
            rr = np.array(rr)
        if rr_ts is not None and isinstance(rr_ts, list):
            rr_ts = np.array(rr_ts)

        if len(rr) >= 3:
            hrv_result = self.hrv_analyzer.analyze(rr, rr_timestamps_ms=rr_ts)
            features.update(hrv_result)
        else:
            features.update(self.hrv_analyzer._empty_result())

        # QT analysis
        r_peaks = r_result["r_peaks"]
        if isinstance(r_peaks, np.ndarray):
            r_peaks = r_peaks.tolist()
        if r_result["num_beats"] >= 2:
            qt_result = self.qt_analyzer.analyze(
                lead_signal, r_peaks, fs, sex
            )
            features.update(qt_result)
        else:
            features.update(QTAnalyzer._empty_result())

        # Raw data (optional)
        if include_raw:
            features["r_peaks_raw"] = r_result.get("r_peaks", [])
            features["rr_intervals_raw"] = (
                r_result["rr_intervals"].tolist()
                if isinstance(r_result.get("rr_intervals"), np.ndarray)
                else r_result.get("rr_intervals", [])
            )

        return features

    def to_vector(
        self,
        features: Dict,
        keys: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Convert feature dict to flat numpy vector.

        Args:
            features: Output from extract_all().
            keys: Specific keys to include (default: all numeric features).

        Returns:
            1D numpy array of float32 values.
        """
        if keys is None:
            keys = self._default_feature_keys()

        values = []
        for key in keys:
            if key not in features:
                logger.warning(f"Feature '{key}' missing from dict, using 0.0")
            val = features.get(key, 0.0)
            if isinstance(val, (int, float)):
                values.append(float(val))
            elif isinstance(val, bool):
                values.append(1.0 if val else 0.0)
            else:
                values.append(0.0)

        return np.array(values, dtype=np.float32)

    @staticmethod
    def _default_feature_keys() -> List[str]:
        """Default ordered list of feature keys for vectorization."""
        return [
            # Basic
            "heart_rate", "hr_std", "num_beats",
            # HRV time-domain
            "mean_hr", "sdnn", "rmssd", "pnn50", "cvrr",
            # HRV frequency-domain
            "lf_power", "hf_power", "lf_hf_ratio", "total_power",
            # HRV nonlinear
            "sd1", "sd2", "sd_ratio", "sample_entropy",
            # QT
            "qt_ms", "qtc_bazett", "qtc_fridericia", "qtc_framingham",
            "qrs_duration_ms", "pr_interval_ms",
        ]

    @property
    def feature_dim(self) -> int:
        """Number of features in the output vector."""
        return len(self._default_feature_keys())
