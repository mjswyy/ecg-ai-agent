"""
QT 间期分析器 — QT/QTc 测量与临床解读 / QT/QTc Measurement and Clinical Interpretation.

测量 QT 间期、QRS 时限、PR 间期，并用多种临床公式计算校正 QT (QTc)。
基于 AHA/ACCF/HRS 2009 ECG 标准化建议。

Measures QT interval, QRS duration, and PR interval.
Computes corrected QT (QTc) using Bazett, Fridericia, and Framingham formulas.
Reference: AHA/ACCF/HRS Recommendations for Standardization of ECG (2009).

使用示例 / Usage:
    analyzer = QTAnalyzer()
    result = analyzer.analyze(ecg_lead_ii, r_peaks, fs=500, sex="Male")
    # → {qt_ms: 380, qtc_bazett: 410, qrs_duration_ms: 95, interpretation: "normal"}
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class QTAnalyzer:
    """QT 间期分析器 / QT interval analyzer.

    测量波形边界并计算 QTc。优先使用 neurokit2 的波形描述算法，
    如不可用则回退到简单的固定窗口法。
    """

    # 临床阈值 (ms) / Clinical thresholds
    QT_NORMAL_UPPER_MALE = 450
    QT_NORMAL_UPPER_FEMALE = 460
    QTC_NORMAL_UPPER = 440
    QTC_PROLONGED_LOWER = 500
    QRS_NORMAL_UPPER = 120
    PR_NORMAL_RANGE = (120, 200)

    def analyze(self, ecg: np.ndarray, r_peaks: List[int], fs: float, sex: str = "Unknown") -> Dict:
        """测量 QT 间期并计算 QTc / Measure QT and compute QTc.

        参数 / Args:
            ecg:     单导联 ECG 信号 (L,) / Single-lead ECG signal.
            r_peaks: R 峰索引列表 / R-peak indices.
            fs:      采样频率 (Hz) / Sampling frequency.
            sex:     性别 (用于阈值) / Sex for threshold.

        返回 / Returns:
            字典, 含 qt_ms, qtc_bazett, qtc_fridericia, qtc_framingham,
            qrs_duration_ms, pr_interval_ms, 解读文本等。
        """
        if len(r_peaks) < 2:
            return self._empty_result()

        # ---- 使用 neurokit2 进行波形描述 ----
        try:
            import neurokit2 as nk
            _, waves = nk.ecg_delineate(ecg, r_peaks, sampling_rate=int(fs), method="peak")
            qt_intervals, qrs_durations, pr_intervals = [], [], []

            for i in range(len(r_peaks)):
                # Q 起点 (临床标准) / Q onset (clinical standard) — 优先，回退到 Q peak
                q_onset = self._safe_wave_index(waves, "ECG_Q_Onsets", i)
                if q_onset is None:
                    q_onset = self._safe_wave_index(waves, "ECG_Q_Peaks", i)

                # QT: Q onset → T offset / QT interval: Q onset to T offset
                t_offset = self._safe_wave_index(waves, "ECG_T_Offsets", i)
                if q_onset is not None and t_offset is not None:
                    qt_intervals.append((t_offset - q_onset) / fs * 1000.0)

                # QRS: Q onset → S offset (优先), 回退到 S peak
                s_offset = self._safe_wave_index(waves, "ECG_S_Offsets", i)
                if s_offset is None:
                    s_offset = self._safe_wave_index(waves, "ECG_S_Peaks", i)
                if q_onset is not None and s_offset is not None:
                    qrs_durations.append((s_offset - q_onset) / fs * 1000.0)

                # PR: P onset → Q onset / PR interval: P onset to Q onset
                p_onset = self._safe_wave_index(waves, "ECG_P_Onsets", i)
                if p_onset is not None and q_onset is not None:
                    pr_intervals.append((q_onset - p_onset) / fs * 1000.0)

        except ImportError:
            logger.warning("neurokit2 未安装；使用简单阈值法测量")
            qt_intervals, qrs_durations, pr_intervals = self._simple_delineate(ecg, r_peaks, fs)

        return self._compute_results(qt_intervals, qrs_durations, pr_intervals, sex, r_peaks, fs)

    def _compute_results(self, qt_intervals, qrs_durations, pr_intervals, sex, r_peaks, fs) -> Dict:
        """汇总间期测量并计算 QTc / Aggregate interval measurements and compute QTc."""
        # 使用中位数（比均值更稳健）/ Use median (more robust to outliers)
        qt_ms = float(np.median(qt_intervals)) if qt_intervals else 0.0
        qrs_ms = float(np.median(qrs_durations)) if qrs_durations else 0.0
        pr_ms = float(np.median(pr_intervals)) if pr_intervals else 0.0

        # 心率 / Heart rate
        if len(r_peaks) >= 2:
            rr_ms = np.median(np.diff(np.array(r_peaks))) / fs * 1000.0
            hr = 60000.0 / rr_ms if rr_ms > 0 else 60.0
        else:
            rr_ms, hr = 1000.0, 60.0

        # QTc 三种公式 / Three QTc correction formulas
        qtc_bazett = self._qtc_bazett(qt_ms, rr_ms)        # Bazett: QTc = QT / sqrt(RR)
        qtc_fridericia = self._qtc_fridericia(qt_ms, rr_ms) # Fridericia: QTc = QT / cbrt(RR)
        qtc_framingham = self._qtc_framingham(qt_ms, rr_ms) # Framingham: QTc = QT + 154*(1-RR)

        # 临床解读 / Clinical interpretation
        qt_upper = self.QT_NORMAL_UPPER_MALE if sex == "Male" else self.QT_NORMAL_UPPER_FEMALE
        qt_interp = self._interpret_qt(qt_ms, qtc_bazett, qt_upper)
        qrs_interp = "wide" if qrs_ms > self.QRS_NORMAL_UPPER else "normal"

        return {
            "qt_ms": round(qt_ms, 1), "qtc_bazett": round(qtc_bazett, 1),
            "qtc_fridericia": round(qtc_fridericia, 1), "qtc_framingham": round(qtc_framingham, 1),
            "qrs_duration_ms": round(qrs_ms, 1), "pr_interval_ms": round(pr_ms, 1),
            "heart_rate": round(float(hr), 1),
            "qt_interpretation": qt_interp, "qrs_interpretation": qrs_interp,
            "num_beats_analyzed": len(qt_intervals),
        }

    # ---- QTc 校正公式 / Correction Formulas ----
    @staticmethod
    def _qtc_bazett(qt_ms, rr_ms):       return qt_ms / np.sqrt(rr_ms / 1000.0) if rr_ms > 0 else 0.0
    @staticmethod
    def _qtc_fridericia(qt_ms, rr_ms):   return qt_ms / np.cbrt(rr_ms / 1000.0) if rr_ms > 0 else 0.0
    @staticmethod
    def _qtc_framingham(qt_ms, rr_ms):   return qt_ms + 154.0 * (1.0 - rr_ms / 1000.0)

    @staticmethod
    def _interpret_qt(qt_ms, qtc_ms, upper):
        """临床 QT 解读 / Clinical QT interpretation."""
        if qtc_ms > QTAnalyzer.QTC_PROLONGED_LOWER:    return "prolonged"
        elif qtc_ms > QTAnalyzer.QTC_NORMAL_UPPER:       return "borderline prolonged"
        elif qt_ms < 300:                                return "shortened"
        return "normal"

    @staticmethod
    def _safe_wave_index(waves, key, idx):
        """安全提取 neurokit2 波形索引 / Safely extract wave index."""
        try:
            wave_list = waves.get(key)
            if wave_list is None: return None
            val = wave_list[idx] if isinstance(wave_list, (np.ndarray, list)) and idx < len(wave_list) else None
            return int(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None
        except (IndexError, TypeError, KeyError):
            return None

    @staticmethod
    def _simple_delineate(ecg, r_peaks, fs):
        """简单固定窗口法（回退方案）/ Simple fixed-window fallback."""
        qrs_w, qt_w = int(0.10 * fs), int(0.44 * fs)  # ~100ms QRS, ~440ms QT
        qt_intervals, qrs_durations, pr_intervals = [], [], []
        for r in r_peaks:
            q_onset = max(0, r - qrs_w // 2)
            s_offset = min(len(ecg) - 1, r + qrs_w // 2)
            t_offset = min(len(ecg) - 1, r + qt_w)
            qrs_durations.append((s_offset - q_onset) / fs * 1000.0)
            qt_intervals.append((t_offset - q_onset) / fs * 1000.0)
            pr_intervals.append((r - max(0, r - int(0.20 * fs))) / fs * 1000.0)
        return qt_intervals, qrs_durations, pr_intervals

    @staticmethod
    def _empty_result() -> Dict:
        return {"qt_ms": 0, "qtc_bazett": 0, "qtc_fridericia": 0, "qtc_framingham": 0,
                "qrs_duration_ms": 0, "pr_interval_ms": 0, "heart_rate": 0,
                "qt_interpretation": "insufficient_data", "qrs_interpretation": "insufficient_data",
                "num_beats_analyzed": 0}
