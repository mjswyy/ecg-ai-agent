"""
ECG 分析工具 — 提供给 Agent 调用的 R峰/HRV/QT 特征提取工具。

3 个工具:
    - extract_r_peaks:     R峰检测 → 心率、节律判断
    - compute_hrv:         心率变异性分析 → SDNN、RMSSD、LF/HF
    - measure_qt_interval: QT 间期测量 → QTc (Bazett/Fridericia/Framingham)

使用示例:
    registry = ToolRegistry()
    register_ecg_tools(registry)
    result = registry.call("extract_r_peaks", ecg_signal=..., fs=500)
"""

import numpy as np


def extract_r_peaks(ecg_signal=None, fs: float = 500.0, lead: int = 1, **kwargs) -> dict:
    """检测 ECG 信号中的 R 峰位置。

    参数:
        ecg_signal: 12导联 ECG，形状 (12, L) 或 (L,)
        fs:         采样频率 (Hz)
        lead:       使用的导联索引 (默认1 = Lead II)

    返回:
        {r_peaks, rr_intervals, heart_rate, hr_std, rhythm, num_beats}
    """
    if ecg_signal is None:
        return {"error": "未提供 ECG 信号"}

    from src.ecg_models.feature_extraction.r_peak_detector import RPeakDetector

    detector = RPeakDetector(method="pan_tompkins")
    result = detector.detect(ecg_signal, fs, lead=lead)

    # 基础节律分类
    hr = result.get("heart_rate", 0)
    hr_std = result.get("hr_std", 0)
    if hr < 60:
        rhythm = "bradycardia"
    elif hr > 100:
        rhythm = "tachycardia"
    else:
        rhythm = "normal rate"
    rhythm += " (irregular)" if hr_std > 15 else " (regular)"
    result["rhythm"] = rhythm
    return result


def compute_hrv(rr_intervals=None, r_peaks=None, ecg_signal=None, fs: float = 500.0, **kwargs) -> dict:
    """计算心率变异性 (HRV) 指标。

    提供 RR 间期 / R 峰 / ECG 信号三者之一即可，
    缺失的数据会自动从前一步推导。

    返回:
        {sdnn, rmssd, pnn50, cvrr, mean_hr, lf_power, hf_power, lf_hf_ratio, sd1, sd2, sample_entropy}
    """
    # 自动推导 RR 间期
    if rr_intervals is None:
        if r_peaks is not None:
            rr_intervals = np.diff(np.array(r_peaks)) / fs * 1000.0
        elif ecg_signal is not None:
            rr = extract_r_peaks(ecg_signal=ecg_signal, fs=fs)
            rr_intervals = rr.get("rr_intervals")

    if rr_intervals is None or len(rr_intervals) < 3:
        return {"error": "RR 间期不足，无法进行 HRV 分析"}

    from src.ecg_models.feature_extraction.hrv_analyzer import HRVAnalyzer
    return HRVAnalyzer().analyze(np.asarray(rr_intervals))


def measure_qt_interval(ecg_signal=None, r_peaks=None, fs: float = 500.0, sex: str = "Unknown", **kwargs) -> dict:
    """测量 QT 间期并计算校正 QT (QTc)。

    使用 Bazett、Fridericia、Framingham 三种校正公式。

    参数:
        ecg_signal: ECG 信号 (单导联或12导联, 自动取 Lead II)
        r_peaks:    R 峰索引列表 (缺失时自动检测)
        fs:         采样频率
        sex:        性别 (用于QT阈值判断)

    返回:
        {qt_ms, qtc_bazett, qtc_fridericia, qtc_framingham, qrs_duration_ms, pr_interval_ms}
    """
    if ecg_signal is None:
        return {"error": "未提供 ECG 信号"}

    # 取 Lead II
    if ecg_signal.ndim == 2:
        lead_signal = ecg_signal[min(1, ecg_signal.shape[0] - 1)]
    else:
        lead_signal = ecg_signal

    # 自动检测 R 峰
    if r_peaks is None:
        r_peaks = extract_r_peaks(ecg_signal=ecg_signal, fs=fs).get("r_peaks", [])

    from src.ecg_models.feature_extraction.qt_analyzer import QTAnalyzer
    return QTAnalyzer().analyze(lead_signal, r_peaks, fs, sex)


def register_ecg_tools(registry: "ToolRegistry"):
    """向工具注册表中注册所有 ECG 分析工具。"""
    registry.register(
        "extract_r_peaks", extract_r_peaks,
        description="检测 ECG 信号中的 R 峰位置。返回心率、RR间期和节律分类。",
        schema={"ecg_signal": {"type": "array"}, "fs": {"type": "number", "default": 500}},
    )
    registry.register(
        "compute_hrv", compute_hrv,
        description="计算心率变异性指标 (SDNN, RMSSD, LF/HF 等)。",
        schema={"rr_intervals": {"type": "array"}, "fs": {"type": "number", "default": 500}},
        dependencies=["extract_r_peaks"],  # 依赖R峰检测
    )
    registry.register(
        "measure_qt_interval", measure_qt_interval,
        description="测量 QT 间期并计算 QTc (Bazett/Fridericia/Framingham)。",
        schema={"ecg_signal": {"type": "array"}, "r_peaks": {"type": "array"}, "fs": {"type": "number", "default": 500}},
        dependencies=["extract_r_peaks"],  # 依赖R峰检测
    )
