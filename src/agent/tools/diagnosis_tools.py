"""
诊断工具 — 分类器和异常检测工具 / Diagnosis Tools — Classifier and anomaly detection.

提供给 Agent 调用的高级诊断工具:
    - classify_arrhythmia: 27 类心律失常分类器推理
    - detect_anomaly:      VAE 无监督异常检测
    - generate_report:     结构化诊断报告生成

使用示例 / Usage:
    register_diagnosis_tools(registry, model=model, detector=detector, label_extractor=le)
"""

import numpy as np
import torch


def classify_arrhythmia(ecg_signal=None, model=None, label_extractor=None,
                        top_k: int = 5, threshold: float = 0.3, **kwargs) -> dict:
    """运行心律失常分类器 / Run arrhythmia classifier.

    参数 / Args:
        ecg_signal: 预处理后的 ECG (12, 4096) / Preprocessed ECG.
        model:      训练好的 ArrhythmiaClassifier / Trained model.
        label_extractor: 标签解码器 / For decoding predictions.
        top_k:      返回前 K 个预测 / Top-K predictions to return.
        threshold:  概率阈值 / Probability threshold for positive.

    返回 / Returns:
        {diagnoses: [{name, snomed_code, probability, evidence}, ...], ...}
    """
    if ecg_signal is None: return {"error": "未提供 ECG 信号"}
    if model is None:       return {"error": "未加载分类模型。请先训练模型。"}
    if label_extractor is None:
        from src.data_pipeline.label_extractor import LabelExtractor
        label_extractor = LabelExtractor()

    x = torch.from_numpy(ecg_signal).float().unsqueeze(0) if isinstance(ecg_signal, np.ndarray) else ecg_signal
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x.to(device))).squeeze(0).cpu().numpy()

    decoded = label_extractor.decode(probs, threshold=threshold)
    diagnoses = [{"name": name, "snomed_code": snomed, "probability": round(float(prob), 4),
                  "evidence": f"分类器预测, 置信度 {prob:.1%}"}
                 for snomed, name, prob in decoded[:top_k]]

    return {"diagnoses": diagnoses,
            "top_prediction": diagnoses[0]["name"] if diagnoses else "无发现",
            "all_probabilities": {n: round(float(p), 4) for n, p in zip(label_extractor.class_names, probs)}}


def detect_anomaly(ecg_signal=None, detector=None, **kwargs) -> dict:
    """运行异常检测 / Run anomaly detection.

    参数 / Args:
        ecg_signal: ECG 信号 (12, L) / ECG signal.
        detector:   训练好的 ECGAnomalyDetector / Trained detector.

    返回 / Returns:
        {anomaly_score, is_anomalous, interpretation}
    """
    if ecg_signal is None: return {"error": "未提供 ECG 信号"}
    if detector is None:   return {"error": "未加载异常检测器。请先训练。"}

    x = torch.from_numpy(ecg_signal).float().unsqueeze(0) if isinstance(ecg_signal, np.ndarray) else ecg_signal
    device = next(detector.parameters()).device
    detector.eval()
    with torch.no_grad():
        score = detector.anomaly_score(x.to(device)).item()

    is_anomalous = detector._fitted and score > detector.threshold.item()
    return {"anomaly_score": round(score, 6), "is_anomalous": is_anomalous,
            "interpretation": "检测到异常 ECG 模式" if is_anomalous else "ECG 在正常范围内"}


def generate_report(features=None, diagnoses=None, patient_info=None, **kwargs) -> dict:
    """生成结构化诊断报告 / Generate structured diagnostic report.

    从收集到的所有发现中汇总生成。 / Synthesize from collected findings.
    """
    report = {"patient": patient_info or {}, "ecg_findings": features or {},
              "diagnoses": diagnoses or [], "summary": ""}
    parts = []
    hr = (features or {}).get("heart_rate", "N/A")
    parts.append(f"心率: {hr} bpm")
    rhythm = (features or {}).get("rhythm", "")
    if rhythm: parts.append(f"节律: {rhythm}")
    if diagnoses:
        top = diagnoses[0]
        parts.append(f"主要发现: {top.get('name', 'Unknown')} (置信度: {top.get('probability', 0):.1%})")
    report["summary"] = "。".join(parts) + "。"
    return report


def register_diagnosis_tools(registry, model=None, detector=None, label_extractor=None):
    """注册所有诊断工具 / Register all diagnosis tools."""
    import functools
    registry.register(
        "classify_arrhythmia",
        functools.partial(classify_arrhythmia, model=model, label_extractor=label_extractor),
        description="运行 27 类心律失常分类器。返回 top-K 诊断及置信度。",
        schema={"ecg_signal": {"type": "array"}, "top_k": {"type": "integer", "default": 5}},
        dependencies=["extract_r_peaks"],
    )
    registry.register(
        "detect_anomaly",
        functools.partial(detect_anomaly, detector=detector),
        description="检测异常/不寻常的 ECG 模式（无监督异常检测）。",
        schema={"ecg_signal": {"type": "array"}},
    )
    registry.register(
        "generate_report", generate_report,
        description="从收集到的发现中生成结构化诊断报告。",
        schema={"features": {"type": "object"}, "diagnoses": {"type": "array"}},
        dependencies=["classify_arrhythmia"],
    )
