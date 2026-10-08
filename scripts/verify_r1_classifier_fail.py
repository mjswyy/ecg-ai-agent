#!/usr/bin/env python3
"""第三轮审查 AGENT-LLM-R1 验证：测量成功但分类器失败 → 显式标注。"""
# 4A/4J 审查修复：💡-12 本脚本使用 "outputs/...""data/..." 相对路径，
# 需在项目根目录（ecg-ai-agent/）运行，否则 FileNotFoundError。
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.core.agent import ECGAIAgent
from src.agent.llm.llm_interface import LLMInterface
from src.agent.tools.registry import ToolRegistry
from src.agent.tools.ecg_tools import register_ecg_tools
from src.agent.tools.diagnosis_tools import classify_arrhythmia  # noqa: F401

# 正常信号（真实记录第一条）
import json
m = json.load(open(r"outputs/agent_evalset/evalset.json", encoding="utf-8"))["items"][0]
sig = np.load(Path(r"data/physionet2020/processed_5k") / m["signal_file"])

# mock LLM：计划只含 extract_r_peaks + classify_arrhythmia；classify 工具不注册模型 → 返回 error
llm = LLMInterface(backend="zhipu", model="glm-4.5-air", mock=True)
reg = ToolRegistry()
register_ecg_tools(reg)
reg.register("classify_arrhythmia", classify_arrhythmia,
             description="27 类心律失常分类",
             schema={"ecg_signal": {"type": "array"}})
agent = ECGAIAgent(llm, reg)

result = agent.diagnose(sig, {"age": 60, "sex": "Male", "symptoms": "胸痛"})
print(f"heart_rate={result.heart_rate} rhythm={result.rhythm!r}")
print(f"diagnosis={result.diagnosis}")
print(f"recommendations={result.recommendations}")
assert result.heart_rate is not None, "测量应成功"
assert result.recommendations and "分类步骤未完成" in result.recommendations[0], \
    "应显式标注诊断分类步骤未完成"
assert "常规随访" not in result.recommendations[0]
print("\nAGENT-LLM-R1 验证通过 [PASS]（测量成功 + 分类失败 → 显式标注，不再'常规随访'）")
