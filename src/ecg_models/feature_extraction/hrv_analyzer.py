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
            # 第三轮审查 3C-HRV-2：数据不足返回 None（旧版全 0 把测量失败
            # 伪装成有效值；与 QT 分析器的 None 口径一致）
            return {"mean_hr": None, "sdnn": None, "rmssd": None,
                    "pnn50": None, "cvrr": None, "insufficient": True}

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
        # 第三轮审查 3C-HRV-1：过滤时同步保留有效索引，时间戳与 RR 一一对应
        # （旧版过滤后长度不匹配 → 真实 R 峰时间戳被丢弃、改用压缩时间轴插值）
        # 4D-ORANGE-2 修复：长度比对必须用过滤前的原始间期数——旧版在
        # rr=rr[idx] 之后才比较，任何间期被过滤都会长度不匹配 → 时间戳恒置
        # None → LF/HF 在压缩时间轴上失真
        n_orig = len(rr)
        valid = (rr > 300) & (rr < 2000)
        idx = np.where(valid)[0]
        rr = rr[idx]
        if timestamps_ms is not None and len(timestamps_ms) == n_orig + 1:
            # 时间戳含起点（N+1 点 ↔ N 间期）：每个保留间期取其实点时间戳
            timestamps_ms = timestamps_ms[idx]
        elif timestamps_ms is not None and len(timestamps_ms) == n_orig:
            timestamps_ms = timestamps_ms[idx]
        else:
            timestamps_ms = None
        if len(rr) < 10:
            return {"lf_power": None, "hf_power": None, "lf_hf_ratio": None,
                    "total_power": None, "insufficient": True}

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
            return {"lf_power": None, "hf_power": None, "lf_hf_ratio": None,
                    "total_power": None, "insufficient": True}

        freqs, psd = signal.welch(rr_detrended, fs=self.interpolation_fs, nperseg=nperseg, scaling="density")

        # 积分频段功率 / Integrate power in bands
        lf_power = float(np.trapz(psd[(freqs >= self.LF_BAND[0]) & (freqs < self.LF_BAND[1])],
                                  freqs[(freqs >= self.LF_BAND[0]) & (freqs < self.LF_BAND[1])]))
        hf_power = float(np.trapz(psd[(freqs >= self.HF_BAND[0]) & (freqs < self.HF_BAND[1])],
                                  freqs[(freqs >= self.HF_BAND[0]) & (freqs < self.HF_BAND[1])]))
        total_power = float(np.trapz(psd, freqs))
        # 5D 审查（🟡-2）：恒定 RR（去趋势后功率≈0）时 hf/total 为 0，lf/hf=0/0
        # 未定义——旧版返回 0.0 冒充有效比值；现 hf≈0 时显式 None+insufficient
        if hf_power > 1e-10 and total_power > 1e-10:
            lf_hf_ratio = lf_power / hf_power
            insufficient = False
        else:
            lf_hf_ratio = None
            insufficient = True

        return {"lf_power": round(lf_power, 2), "hf_power": round(hf_power, 2),
                "lf_hf_ratio": (round(lf_hf_ratio, 2) if lf_hf_ratio is not None else None),
                "total_power": round(total_power, 2),
                "insufficient": insufficient}

    def _nonlinear(self, rr: np.ndarray) -> Dict:
        """非线性 HRV 指标 / Nonlinear HRV metrics: SD1/SD2 (Poincaré), 样本熵 / Sample Entropy."""
        rr_clean = rr[(rr > 300) & (rr < 2000)]
        if len(rr_clean) < 3:
            # 第三轮审查 3C-HRV-2：数据不足显式标记（旧版全 0）
            return {"sd1": None, "sd2": None, "sd_ratio": None,
                    "sample_entropy": None, "insufficient": True}

        # Poincaré 图 / Poincaré plot
        rr_n, rr_n1 = rr_clean[:-1], rr_clean[1:]
        sd1 = float(np.std((rr_n1 - rr_n) / np.sqrt(2)))
        sd2 = float(np.std((rr_n1 + rr_n) / np.sqrt(2)))
        sd_ratio = sd1 / sd2 if sd2 > 0 else 0.0

        # 样本熵 / Sample entropy (m=2, r=0.2*std)
        sample_entropy = self._sample_entropy(rr_clean, m=2, r=0.2 * np.std(rr_clean))

        return {"sd1": round(sd1, 1), "sd2": round(sd2, 1),
                "sd_ratio": round(sd_ratio, 3),
                "sample_entropy": (round(sample_entropy, 4)
                                   if sample_entropy is not None else None)}

    @staticmethod
    def _sample_entropy(signal: np.ndarray, m: int = 2, r: float = None):
        """样本熵计算 / Compute sample entropy of a time series.

        第三轮审查 3C-HRV-3：
        - 向量化（旧版 O(N²) 双层循环 + 模板广播）
        - B==0（无匹配）时返回 None 并置 insufficient（旧版返回 0.0
          把"过于不规则"误报为 0 熵）
        - 模板数修正为 N-m+1（旧版 range(N-m) 少取最后一个模板）
        """
        N = len(signal)
        if r is None:
            r = 0.2 * np.std(signal)
        if N <= m + 1 or r <= 0:
            return None

        def _count_matches(template_len):
            n_tpl = N - template_len + 1
            templates = np.stack([signal[i:i + template_len]
                                  for i in range(n_tpl)])  # (n_tpl, template_len)
            # 两两 Chebyshev 距离（向量化）
            diff = np.abs(templates[:, None, :] - templates[None, :, :])
            dist = diff.max(axis=2)  # (n_tpl, n_tpl)
            matches = (dist < r).sum() - n_tpl  # 减自身
            return int(matches)

        A, B = _count_matches(m + 1), _count_matches(m)
        if B == 0:
            return None  # 无匹配 → 数据不规则度超出模板粒度，非 0 熵
        return max(0.0, -np.log(A / B)) if A > 0 else 0.0

    @staticmethod
    def _empty_result() -> Dict:
        """空结果（数据不足时返回）/ Empty result when insufficient data.

        第三轮审查 3C-HRV-2：数值字段统一 None + insufficient 标志
        （旧版全 0 把测量失败伪装成有效测量值）。
        """
        return {"mean_hr": None, "sdnn": None, "rmssd": None, "pnn50": None,
                "cvrr": None, "lf_power": None, "hf_power": None,
                "lf_hf_ratio": None, "total_power": None,
                "sd1": None, "sd2": None, "sd_ratio": None, "sample_entropy": None,
                "insufficient": True}
