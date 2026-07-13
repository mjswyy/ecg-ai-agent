"""
R 峰检测器 — ECG 信号中的 QRS 波群检测。

封装 neurokit2 的 ECG 处理管道，支持跨导联一致性验证。
如果没有安装 neurokit2，自动回退到简单的阈值法。

可用算法:
    pan_tompkins : Pan-Tompkins 1985 (默认，最经典)
    hamilton     : Hamilton 2002 (OpenECG)
    elgendi      : Elgendi 2010 (快速，适合低资源)
    neurokit     : 让 neurokit2 自动选择最佳算法

使用示例:
    detector = RPeakDetector(method="pan_tompkins")
    result = detector.detect(ecg_lead_ii, fs=500)
    # → {r_peaks, rr_intervals, heart_rate: 72.5, rhythm: "normal rate"}
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RPeakDetector:
    """QRS 波群 / R 峰检测器。

    属性:
        method:      检测算法名称
        last_result: 最近一次检测结果（缓存用于调试/检查）
    """

    VALID_METHODS = {"pan_tompkins", "hamilton", "elgendi", "neurokit"}
    LEAD_II_INDEX = 1  # Lead II 在 12 导联中的索引

    def __init__(self, method: str = "pan_tompkins"):
        if method not in self.VALID_METHODS:
            raise ValueError(f"未知算法 '{method}'。可选: {self.VALID_METHODS}")
        self.method = method
        self.last_result: Optional[Dict] = None

    def detect(self, ecg: np.ndarray, fs: float, lead: Optional[int] = None) -> Dict:
        """检测 ECG 信号中的 R 峰。

        参数:
            ecg:  ECG 信号。可以是:
                 - 1D 数组 (单导联, 长度 L)
                 - 2D 数组 (12 导联, 形状 (12, L)) — 自动使用 Lead II
            fs:   采样频率 (Hz)
            lead: 指定导联索引（覆盖默认的 Lead II）

        返回:
            字典，包含 r_peaks, r_peaks_ms, rr_intervals, heart_rate, hr_std, num_beats
        """
        # 提取单导联信号
        if ecg.ndim == 2:
            lead_idx = lead if lead is not None else self.LEAD_II_INDEX
            if lead_idx >= ecg.shape[0]:
                lead_idx = 0
            signal = ecg[lead_idx]
        else:
            signal = ecg

        # 使用 neurokit2 进行鲁棒检测
        try:
            import neurokit2 as nk
            if self.method == "neurokit":
                _, info = nk.ecg_process(signal, sampling_rate=int(fs))
            else:
                _, info = nk.ecg_process(signal, sampling_rate=int(fs), method=self.method)
            r_peaks = info["ECG_R_Peaks"]
        except ImportError:
            logger.warning("neurokit2 未安装；使用简单阈值法检测")
            r_peaks = self._simple_peak_detect(signal, fs)

        # ---- 少于2个R峰时无法可靠计算心率和RR间期 ----
        if len(r_peaks) < 2:
            return {
                "r_peaks": r_peaks.tolist() if isinstance(r_peaks, np.ndarray) else r_peaks,
                "r_peaks_ms": [], "rr_intervals": np.array([]),
                "heart_rate": 0.0, "hr_std": 0.0,
                "num_beats": len(r_peaks), "method": self.method,
            }

        # ---- 计算 RR 间期 (ms) ----
        r_peaks_ms = (np.array(r_peaks) / fs) * 1000.0
        rr_intervals = np.diff(r_peaks_ms)

        # 过滤生理上合理的 RR 间期 (300-2000 ms)
        rr_valid = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]

        # 心率
        if len(rr_valid) > 0:
            heart_rate = 60000.0 / np.mean(rr_valid)
            hr_std = np.std(60000.0 / rr_valid)
            # 基础节律分类
            if heart_rate < 60:
                rhythm = "bradycardia"
            elif heart_rate > 100:
                rhythm = "tachycardia"
            else:
                rhythm = "normal rate"
            rhythm += " (irregular)" if hr_std > 15 else " (regular)"
        else:
            heart_rate, hr_std, rhythm = 0.0, 0.0, "unknown"

        result = {
            "r_peaks": r_peaks.tolist() if isinstance(r_peaks, np.ndarray) else r_peaks,
            "r_peaks_ms": r_peaks_ms.tolist(),
            "rr_intervals": rr_intervals,
            "heart_rate": round(float(heart_rate), 1),
            "hr_std": round(float(hr_std), 1),
            "rhythm": rhythm,
            "num_beats": len(r_peaks),
            "method": self.method,
        }
        self.last_result = result
        return result

    def detect_multi_lead(self, ecg: np.ndarray, fs: float, leads: Optional[List[int]] = None) -> Dict:
        """多导联 R 峰共识检测。

        在多个导联（默认 II, V1, V5）上独立检测 R 峰，
        使用 30ms 容差窗口进行共识投票（≥2 导联同意则该峰保留）。

        参数:
            ecg:  12 导联 ECG 形状 (12, L)
            fs:   采样频率
            leads: 要使用的导联索引列表（默认 [1, 6, 10]）

        返回:
            同 detect() 的结果字典，额外包含 "lead_agreement" 字段。
        """
        target_leads = leads or [1, 6, 10]  # II, V1, V5
        all_peaks = []

        for lead_idx in target_leads:
            if lead_idx < ecg.shape[0]:
                result = self.detect(ecg, fs, lead=lead_idx)
                if result["num_beats"] > 0:
                    all_peaks.append(set(result["r_peaks"]))

        if not all_peaks:
            return self.detect(ecg, fs)  # 回退到 Lead II

        # ---- 共识投票: 30ms 容差 ----
        tolerance = int(0.03 * fs)
        consensus = set()
        for peak in all_peaks[0]:
            votes = 0
            for peaks in all_peaks:
                if any(abs(peak - p) <= tolerance for p in peaks):
                    votes += 1
            if votes >= 2:
                consensus.add(peak)

        agreement = len(consensus) / max(len(p) for p in all_peaks) if all_peaks else 0.0

        # 使用共识峰重新计算
        consensus_list = sorted(consensus)
        if len(consensus_list) >= 2:
            r_peaks_ms = (np.array(consensus_list) / fs) * 1000.0
            rr = np.diff(r_peaks_ms)
            rr_valid = rr[(rr > 300) & (rr < 2000)]
            hr = 60000.0 / np.mean(rr_valid) if len(rr_valid) > 0 else 0.0
            hr_std = np.std(60000.0 / rr_valid) if len(rr_valid) > 0 else 0.0
        else:
            hr, hr_std = 0.0, 0.0

        result = {
            "r_peaks": consensus_list,
            "r_peaks_ms": (np.array(consensus_list) / fs * 1000.0).tolist(),
            "rr_intervals": np.diff(np.array(consensus_list) / fs * 1000.0),
            "heart_rate": round(float(hr), 1),
            "hr_std": round(float(hr_std), 1),
            "num_beats": len(consensus_list),
            "method": f"{self.method}_multi_lead",
            "lead_agreement": round(agreement, 3),
        }
        self.last_result = result
        return result

    @staticmethod
    def _simple_peak_detect(signal: np.ndarray, fs: float) -> np.ndarray:
        """简单的阈值法 R 峰检测（neurokit2 不可用时的回退方案）。

        使用自适应阈值（95 分位数）和 200ms 不应期。
        如果首次检测到的峰太少，降低阈值重试。
        """
        from scipy import signal as scipy_signal

        # QRS 频段带通滤波 (5-15 Hz)
        nyquist = fs / 2.0
        try:
            b, a = scipy_signal.butter(2, [5.0 / nyquist, 15.0 / nyquist], btype="band")
            filtered = scipy_signal.filtfilt(b, a, signal)
        except Exception:
            filtered = signal

        abs_signal = np.abs(filtered)
        threshold = 0.5 * np.percentile(abs_signal, 95)

        refractory = int(0.2 * fs)  # 200ms 不应期
        peaks, _ = scipy_signal.find_peaks(filtered, height=threshold, distance=refractory)

        # 峰太少？降低阈值重试
        if len(peaks) < 2:
            threshold = 0.3 * np.max(abs_signal)
            peaks, _ = scipy_signal.find_peaks(filtered, height=threshold, distance=refractory)

        return peaks
