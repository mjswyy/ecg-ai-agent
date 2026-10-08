#!/usr/bin/env python3
"""R3 回归验证：真实 ECG 记录上工具链与 agent 正常路径不受影响。"""
# 4A/4J 审查修复：💡-12 本脚本使用 "outputs/...""data/..." 相对路径，
# 需在项目根目录（ecg-ai-agent/）运行，否则 FileNotFoundError。
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ecg_models.feature_extraction.r_peak_detector import RPeakDetector
from src.ecg_models.feature_extraction.qt_analyzer import QTAnalyzer

# 取一条真实记录
import json
m = json.load(open(r"outputs/agent_evalset/evalset.json", encoding="utf-8"))["items"][0]
sig = np.load(Path(r"data/physionet2020/processed_5k") / m["signal_file"])
print(f"记录: {m['record_id']} 形状 {sig.shape} 标签 {m['dx_names']}")

rpk = RPeakDetector(method="pan_tompkins")
res = rpk.detect(sig[1], fs=500)
print(f"R峰: n={res['num_beats']} hr={res['heart_rate']} rhythm={res['rhythm']!r}")
assert res["num_beats"] >= 2 and res["heart_rate"] > 0
assert res["rhythm"] != "insufficient_data"

q = QTAnalyzer().analyze(sig[1], res["r_peaks"], fs=500)
print(f"QT: qt_ms={q['qt_ms']} qtc={q['qtc_bazett']}")
assert q["qt_ms"] is not None

from src.agent.core.agent import ECGAIAgent
from src.agent.llm.llm_interface import LLMInterface
from src.agent.tools.registry import ToolRegistry
from src.agent.tools.ecg_tools import register_ecg_tools

llm = LLMInterface(backend="deepseek", model="deepseek-chat", mock=True)
reg = ToolRegistry()
register_ecg_tools(reg)
# 复检D（🟡-2）：注册 ECGFounderClassifier——旧版 registry 无分类器，
# result.diagnosis 恒空、断言加强仍测不出分类回归；注册后真实分类器运行，
# diagnosis 可断言非空（mock 计划含 classify_arrhythmia）
from src.agent.tools.ecgfounder_classifier import ECGFounderClassifier
_clf = ECGFounderClassifier()
reg.register("classify_arrhythmia", _clf.predict,
             description="27 类心律失常多标签分类（ECGFounder+MLP，逐类 Youden 阈值）",
             schema={"ecg_signal": {"type": "array"}, "top_k": {"type": "integer", "default": 5}})
agent = ECGAIAgent(llm, reg)
result = agent.diagnose(sig, {"age": m.get("age"), "sex": m.get("sex"), "symptoms": "胸痛"})
print(f"Agent: hr={result.heart_rate} rhythm={result.rhythm!r} "
      f"n_dx={len(result.diagnosis)} recs={result.recommendations[:1]}")
# 5J 审查（🟡-4）+ 复检D（🟡-2）：断言加强——分类回归真实覆盖：
# 分类器已注册且输入为真实记录，diagnosis 必须非空（若分类器坏掉/契约断，
# 此处必然失败）；心率/节律有效；建议层非空
assert result.heart_rate is not None and result.heart_rate > 0, "正常路径心率不应缺失"
assert result.rhythm not in (None, "insufficient_data"), "正常路径节律不应失败"
assert result.diagnosis, "真实记录上分类器应产出诊断（分类回归断言）"
assert result.recommendations, "建议层不应为空"
print("\n正常路径回归通过 [PASS]")
