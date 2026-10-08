#!/usr/bin/env python3
"""R3 修复端到端验证（第二次审查 2D-R1/R2/R3 三条红线）。

场景:
  A. 无 QRS 合成信号（0 R 峰）→ rhythm 必须 insufficient_data，不得为 bradycardia
  B. QT 数据不足（0 R 峰）→ qtc_bazett 必须 None + error 标记，不得为 0 ms
  C. agent.diagnose 全工具失败 → recommendations 必须显式"分析未能完成"，
     不得输出"未检测到高置信度异常。建议常规随访。"

4A/4J 审查修复：💡-12 本脚本使用 "outputs/...""data/..." 相对路径，
需在项目根目录（ecg-ai-agent/）运行，否则 FileNotFoundError。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ecg_models.feature_extraction.r_peak_detector import RPeakDetector
from src.ecg_models.feature_extraction.qt_analyzer import QTAnalyzer

# A: 无 QRS 的合成信号（完全零信号——模拟导联脱落；噪声会被 Pan-Tompkins 误检）
flat = np.zeros(5000)

rpk = RPeakDetector(method="pan_tompkins")
res = rpk.detect(flat, fs=500)
print(f"[A] R峰检测: n_beats={res['num_beats']} hr={res['heart_rate']} "
      f"rhythm={res['rhythm']!r} insufficient={res.get('insufficient')}")
assert res["num_beats"] < 2, "预期 0 峰"
assert res["rhythm"] == "insufficient_data", f"rhythm 应为 insufficient_data: {res['rhythm']}"
print("    [ok] 不再标记为 bradycardia")

# D: 导联脱落 + 工频/肌电噪声（3F-YELLOW-3 修复：旧版只用全零信号——对全零
#    信号 num_beats<2 天然成立，未覆盖噪声场景。实测低幅噪声曾被误检为
#    tachycardia 154-188 bpm，现由伪峰幅度守卫判 insufficient）
_rng = np.random.RandomState(7)
_t = np.arange(5000) / 500.0
noisy_lead = (0.05 * np.sin(2 * np.pi * 50 * _t)      # 50Hz 工频残留
              + 0.01 * _rng.randn(5000)               # 肌电噪声
              + 0.02 * np.sin(2 * np.pi * 0.5 * _t))  # 呼吸基线漂移
res_d = rpk.detect(noisy_lead, fs=500)
print(f"[D] 噪声导联脱落: n_beats={res_d['num_beats']} hr={res_d['heart_rate']} "
      f"rhythm={res_d['rhythm']!r} insufficient={res_d.get('insufficient')}")
assert res_d["rhythm"] == "insufficient_data", \
    f"噪声导联脱落不得被误报为规律节律: {res_d['rhythm']}"
assert res_d.get("insufficient") is True
print("    [ok] 噪声伪峰不再被误报为心动过速")

# B: QT 数据不足
q = QTAnalyzer().analyze(flat, [], fs=500)
print(f"[B] QT: qt_ms={q['qt_ms']} qtc_bazett={q['qtc_bazett']} error={q.get('error')}")
assert q["qtc_bazett"] is None, "qtc_bazett 应为 None"
assert q.get("error") == "insufficient_data"
print("    [ok] 不再呈现 QTc=0 ms")

# C: agent.diagnose 全工具失败场景（mock LLM 出计划，工具全对坏信号失败）
from src.agent.core.agent import ECGAIAgent
from src.agent.llm.llm_interface import LLMInterface
from src.agent.tools.registry import ToolRegistry
from src.agent.tools.ecg_tools import register_ecg_tools

llm = LLMInterface(backend="deepseek", model="deepseek-chat", mock=True)
reg = ToolRegistry()
register_ecg_tools(reg)
agent = ECGAIAgent(llm, reg, verbose=False)

# 坏信号 12 导联：全部为平线噪声
bad = np.stack([flat] * 12)
result = agent.diagnose(bad, {"age": 60, "sex": "Male", "symptoms": "胸痛"})
print(f"[C] diagnose: heart_rate={result.heart_rate} rhythm={result.rhythm!r}")
print(f"    recommendations={result.recommendations}")
assert result.recommendations and "分析未能完成" in result.recommendations[0], \
    "应显式标注分析未完成"
assert "常规随访" not in result.recommendations[0]
print("    [ok] 全失败不再输出'常规随访'")

print("\nR3 三条红线全部通过 [PASS]")
