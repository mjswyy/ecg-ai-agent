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

        except (ImportError, ValueError, TypeError, KeyError, IndexError,
                RuntimeError, ZeroDivisionError, OverflowError) as e:
            # 4D-ORANGE-3/5 修复：除 ImportError 外的异常（NaN/导联脱落输入触发
            # neurokit ValueError 等）旧版会击穿整条管线；且旧版 ImportError 分支
            # 回退到固定窗口常量（QRS=100/PR=200/QT=490ms 与信号无关）——违反
            # "严禁伪造测量值"。现统一显式失败，绝不产出与信号无关的常量。
            logger.warning(f"QT 波形描绘失败（{type(e).__name__}: {e}）；"
                           f"返回 insufficient_data，不伪造固定窗口测量值")
            return self._empty_result()

        return self._compute_results(qt_intervals, qrs_durations, pr_intervals, sex, r_peaks, fs)

    def _compute_results(self, qt_intervals, qrs_durations, pr_intervals, sex, r_peaks, fs) -> Dict:
        """汇总间期测量并计算 QTc / Aggregate interval measurements and compute QTc."""
        # R3 修复（2D-R3）：QT 间期一个都没测出来 → 整体失败显式化，
        # 不再返回 "qt_ms=0, QTc=0" 被下游当真实测量值。
        if not qt_intervals:
            return self._empty_result()

        # 使用中位数（比均值更稳健）/ Use median (more robust to outliers)
        qt_ms = float(np.median(qt_intervals)) if qt_intervals else None
        qrs_ms = float(np.median(qrs_durations)) if qrs_durations else None
        pr_ms = float(np.median(pr_intervals)) if pr_intervals else None

        # 心率 / Heart rate
        # 5D 审查（🟡-4）：删除伪造的 60 bpm 兜底与死分支——旧版 rr_ms==0
        # 时 hr=60 是编造值（且 else 分支不可达：analyze() 在 len(r_peaks)<2
        # 时已提前返回）；心率以 R 峰检测为单一权威（feature_bank 覆盖），
        # 此处仅作 QTc 公式输入，无有效 RR 时 hr 显式 None
        rr_ms = np.median(np.diff(np.array(r_peaks))) / fs * 1000.0
        # 复检B（🟡-2）：rr_ms<=0（重复/非升序 R 峰）时整体失败显式化——
        # 旧版继续算 QTc 产出伪值（bazett/fridericia=0.0、framingham=534.0），
        # 下游 rule_reflection 会把 0.0 当"超出生理范围"伪告警
        if rr_ms <= 0:
            return self._empty_result()
        hr = 60000.0 / rr_ms

        # QTc 三种公式 / Three QTc correction formulas
        qtc_bazett = self._qtc_bazett(qt_ms, rr_ms)        # Bazett: QTc = QT / sqrt(RR)
        qtc_fridericia = self._qtc_fridericia(qt_ms, rr_ms) # Fridericia: QTc = QT / cbrt(RR)
        qtc_framingham = self._qtc_framingham(qt_ms, rr_ms) # Framingham: QTc = QT + 154*(1-RR)

        # 临床解读 / Clinical interpretation
        # 第三轮审查 AGENT-LLM-O6：sex 特异性 QT 上界真正用于解读
        # （旧版 upper 计算后从未被 _interpret_qt 使用——死参数）
        qt_upper = self.QT_NORMAL_UPPER_MALE if sex == "Male" else self.QT_NORMAL_UPPER_FEMALE
        qt_interp = self._interpret_qt(qt_ms, qtc_bazett, qt_upper)
        # R3 修复（2D-R3）：QRS 未测出时为失败标记而非 "normal"
        qrs_interp = ("wide" if qrs_ms > self.QRS_NORMAL_UPPER
                      else "normal") if qrs_ms is not None else "insufficient_data"

        def _r(v):
            return None if v is None else round(float(v), 1)

        return {
            "qt_ms": _r(qt_ms), "qtc_bazett": _r(qtc_bazett),
            "qtc_fridericia": _r(qtc_fridericia), "qtc_framingham": _r(qtc_framingham),
            "qrs_duration_ms": _r(qrs_ms), "pr_interval_ms": _r(pr_ms),
            "heart_rate": _r(hr),
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
        """临床 QT 解读 / Clinical QT interpretation.

        第三轮审查 AGENT-LLM-O6：使用传入的性别特异性 QTc 上界
        （QT_NORMAL_UPPER_MALE=450 / FEMALE=460，即常见 QTc 正常上限；
        旧版忽略 upper 只按 440 判定——性别相关判定整体失效）。
        第三轮审查 3C-QT-1：shortened 优先于 borderline 判定
        （旧版 QTc 440-500 时先判 borderline，掩盖 QT<300ms 的缩短发现）。
        """
        if qtc_ms > QTAnalyzer.QTC_PROLONGED_LOWER:    return "prolonged"
        elif qt_ms < 300:                               return "shortened"
        elif qtc_ms > upper:                            return "borderline prolonged"
        return "normal"

    @staticmethod
    def _safe_wave_index(waves, key, idx):
        """安全提取 neurokit2 波形索引 / Safely extract wave index.

        检查报告 1.9 修复：np.float64 不是 float 的实例，旧版 NaN 守卫对
        numpy 标量失效（int(nan) 崩溃且 ValueError 不在捕获列表内）。
        """
        try:
            wave_list = waves.get(key)
            if wave_list is None:
                return None
            val = (wave_list[idx]
                   if isinstance(wave_list, (np.ndarray, list))
                      and idx < len(wave_list)
                   else None)
            if val is None:
                return None
            if isinstance(val, (int, float, np.integer, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    return None
                return int(val)
            return None
        except (IndexError, TypeError, KeyError, ValueError, OverflowError):
            return None

    @staticmethod
    def _simple_delineate(ecg, r_peaks, fs):
        """【已停用】简单固定窗口法——4D-ORANGE-5 审查判定其输出与信号无关的
        常量（QRS=100/PR=200/QT=490ms）属伪造测量值，analyze() 已不再调用。
        保留仅为历史参考，禁止恢复调用。"""
        raise RuntimeError("_simple_delineate 已停用（伪造测量值），禁止调用")

    @staticmethod
    def _empty_result() -> Dict:
        """R3 修复（第二次审查 2D-R3）：失败显式化——数据不足时数值字段为 None
        （旧版全 0 会被下游当作真实测量值 QTc=0 ms 呈现），并带 error 标记；
        消费方必须呈现"测量失败"而不是 0。"""
        return {"qt_ms": None, "qtc_bazett": None, "qtc_fridericia": None,
                "qtc_framingham": None, "qrs_duration_ms": None, "pr_interval_ms": None,
                "heart_rate": None,
                "qt_interpretation": "insufficient_data", "qrs_interpretation": "insufficient_data",
                "num_beats_analyzed": 0, "error": "insufficient_data"}
