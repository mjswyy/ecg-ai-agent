"""
HRV 分析器 — 心率变异性分析 / Heart Rate Variability Analysis.

计算时域、频域和非线性 HRV 指标。基于 ESC/ASPE 1996 标准。

Computes time-domain, frequency-domain, and nonlinear HRV metrics.
Reference: Task Force of the ESC/ASPE (1996).

使用示例 / Usage:
    analyzer = HRVAnalyzer()
    metrics = analyzer.analyze(rr_intervals_ms)
    # → {sdnn: 45.2, rmssd: 32.1, lf_hf_ratio: 1.8, ...}
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HRVAnalyzer:
    """心率变异性分析器 / Heart Rate Variability analyzer.

    参数 / Args:
        interpolation_fs: 频域分析的插值重采样率 (Hz) / Resampling rate for frequency-domain analysis.
    """

    # 频段定义 (Hz) / Frequency band definitions
    LF_BAND = (0.04, 0.15)    # 低频 / Low frequency
    HF_BAND = (0.15, 0.4)     # 高频 / High frequency

    def __init__(self, interpolation_fs: float = 4.0):
        self.interpolation_fs = interpolation_fs

    def analyze(self, rr_intervals: np.ndarray, rr_timestamps_ms: Optional[np.ndarray] = None) -> Dict:
        """计算全面的 HRV 指标 / Compute comprehensive HRV metrics.

        参数 / Args:
            rr_intervals: RR 间期序列 (ms) / RR intervals in milliseconds.
            rr_timestamps_ms: R峰时间戳 (ms), 可选 / R-peak timestamps, optional.

        返回 / Returns:
            字典, 包含 时域/频域/非线性 三类指标 / Dict with time/frequency/nonlinear metrics.
        """
        if len(rr_intervals) < 3:
            logger.warning("RR 间期不足 (< 3), 无法进行 HRV 分析")
            return self._empty_result()

        result = {}
        result.update(self._time_domain(rr_intervals))  # 时域
        result.update(self._frequency_domain(rr_intervals, rr_timestamps_ms))  # 频域
        result.update(self._nonlinear(rr_intervals))  # 非线性
        return result

    def _time_domain(self, rr: np.ndarray) -> Dict:
        """时域 HRV 指标 / Time-domain HRV metrics: SDNN, RMSSD, pNN50, CV."""
        # 过滤极端间期 / Filter extreme intervals
        rr_clean = rr[(rr > 300) & (rr < 2000)]
        if len(rr_clean) < 2:
            return {"mean_hr": 0, "sdnn": 0, "rmssd": 0, "pnn50": 0, "cvrr": 0}

        heart_rates = 60000.0 / rr_clean
        sdnn = float(np.std(rr_clean))
        diff = np.diff(rr_clean)
        rmssd = float(np.sqrt(np.mean(diff ** 2)))
        nn50 = np.sum(np.abs(diff) > 50)
        pnn50 = float(nn50 / len(diff) * 100) if len(diff) > 0 else 0.0
        cvrr = float(sdnn / np.mean(rr_clean) * 100) if np.mean(rr_clean) > 0 else 0.0

        return {
            "mean_hr": round(float(np.mean(heart_rates)), 1),
            "sdnn": round(sdnn, 1), "rmssd": round(rmssd, 1),
            "pnn50": round(pnn50, 1), "cvrr": round(cvrr, 1),
        }

    def _frequency_domain(self, rr: np.ndarray, timestamps_ms: Optional[np.ndarray] = None) -> Dict:
        """频域 HRV 指标 (Welch 方法) / Frequency-domain HRV via Welch's method.

        先对 RR 间期进行线性插值到均匀网格，再用 Welch 周期图法计算 PSD。
        注意: 频域分析前会过滤极端间期 (同时间域) / RR intervals are filtered before frequency analysis.
        """
        # 过滤极端间期 / Filter extreme intervals (same as time-domain)
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) < 10:
            return {"lf_power": 0, "hf_power": 0, "lf_hf_ratio": 0, "total_power": 0}

        # 构建时间戳（对齐 RR 间期）/ Build timestamp aligned to RR intervals
        if timestamps_ms is None:
            t = np.cumsum(np.concatenate([[0], rr[:-1]])) / 1000.0
        else:
            if len(timestamps_ms) == len(rr) + 1:
                t = timestamps_ms[:-1] / 1000.0  # R峰位置 → RR起点
            elif len(timestamps_ms) == len(rr):
                t = timestamps_ms / 1000.0
            else:
                t = np.cumsum(np.concatenate([[0], rr[:-1]])) / 1000.0

        # 线性插值到均匀网格 / Linear interpolation to uniform grid
        from scipy import interpolate
        t_uniform = np.arange(t[0], t[-1], 1.0 / self.interpolation_fs)
        rr_interp = interpolate.interp1d(
            t, rr, kind="linear", bounds_error=False, fill_value="extrapolate"
        )(t_uniform)

        # 去趋势 + Welch PSD / Detrend + Welch PSD
        rr_detrended = rr_interp - np.mean(rr_interp)
        from scipy import signal
        nperseg = min(256, len(rr_detrended) // 2)
        if nperseg < 16:
            return {"lf_power": 0, "hf_power": 0, "lf_hf_ratio": 0, "total_power": 0}

        freqs, psd = signal.welch(rr_detrended, fs=self.interpolation_fs, nperseg=nperseg, scaling="density")

        # 积分频段功率 / Integrate power in bands
        lf_power = float(np.trapz(psd[(freqs >= self.LF_BAND[0]) & (freqs < self.LF_BAND[1])],
                                  freqs[(freqs >= self.LF_BAND[0]) & (freqs < self.LF_BAND[1])]))
        hf_power = float(np.trapz(psd[(freqs >= self.HF_BAND[0]) & (freqs < self.HF_BAND[1])],
                                  freqs[(freqs >= self.HF_BAND[0]) & (freqs < self.HF_BAND[1])]))
        total_power = float(np.trapz(psd, freqs))
        lf_hf_ratio = lf_power / hf_power if hf_power > 1e-10 else 0.0

        return {"lf_power": round(lf_power, 2), "hf_power": round(hf_power, 2),
                "lf_hf_ratio": round(lf_hf_ratio, 2), "total_power": round(total_power, 2)}

    def _nonlinear(self, rr: np.ndarray) -> Dict:
        """非线性 HRV 指标 / Nonlinear HRV metrics: SD1/SD2 (Poincaré), 样本熵 / Sample Entropy."""
        rr_clean = rr[(rr > 300) & (rr < 2000)]
        if len(rr_clean) < 3:
            return {"sd1": 0, "sd2": 0, "sd_ratio": 0, "sample_entropy": 0}

        # Poincaré 图 / Poincaré plot
        rr_n, rr_n1 = rr_clean[:-1], rr_clean[1:]
        sd1 = float(np.std((rr_n1 - rr_n) / np.sqrt(2)))
        sd2 = float(np.std((rr_n1 + rr_n) / np.sqrt(2)))
        sd_ratio = sd1 / sd2 if sd2 > 0 else 0.0

        # 样本熵 / Sample entropy (m=2, r=0.2*std)
        sample_entropy = self._sample_entropy(rr_clean, m=2, r=0.2 * np.std(rr_clean))

        return {"sd1": round(sd1, 1), "sd2": round(sd2, 1),
                "sd_ratio": round(sd_ratio, 3), "sample_entropy": round(sample_entropy, 4)}

    @staticmethod
    def _sample_entropy(signal: np.ndarray, m: int = 2, r: float = None) -> float:
        """样本熵计算 / Compute sample entropy of a time series.

        负值会被钳制为0 / Negative values are clamped to 0.
        """
        N = len(signal)
        if r is None: r = 0.2 * np.std(signal)
        if N <= m + 1 or r <= 0: return 0.0

        def _count_matches(template_len):
            count = 0
            templates = np.array([signal[i:i+template_len] for i in range(N-template_len)])
            for i in range(len(templates)):
                dist = np.max(np.abs(templates - templates[i]), axis=1)
                count += np.sum(dist < r) - 1
            return count

        A, B = _count_matches(m + 1), _count_matches(m)
        if B == 0: return 0.0
        return max(0.0, -np.log(A / B)) if A > 0 else 0.0

    @staticmethod
    def _empty_result() -> Dict:
        """空结果（数据不足时返回）/ Empty result when insufficient data."""
        return {"mean_hr": 0, "sdnn": 0, "rmssd": 0, "pnn50": 0, "cvrr": 0,
                "lf_power": 0, "hf_power": 0, "lf_hf_ratio": 0, "total_power": 0,
                "sd1": 0, "sd2": 0, "sd_ratio": 0, "sample_entropy": 0}
