"""
规划器 — 基于 LLM 的 ECG 诊断任务分解。

使用 Few-shot 提示和 Chain-of-Thought 推理，
将高层诊断查询分解为结构化的工具调用步骤序列。

使用示例:
    planner = Planner(llm)
    plan = planner.generate_plan(context)
    # → '{"plan": [{"action": "extract_r_peaks", "reason": "..."}, ...]}'
"""

import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ============================================================================
# Few-shot 示例: 3个典型病例的诊断计划
# ============================================================================

FEW_SHOT_EXAMPLES = """
示例诊断计划:

病例1: "分析65岁男性胸痛患者的心电图"
{
  "plan": [
    {"action": "extract_r_peaks", "reason": "评估心率和基础节律"},
    {"action": "compute_hrv", "reason": "胸痛需评估自主神经张力"},
    {"action": "classify_arrhythmia", "reason": "检测房颤或其他心律失常"},
    {"action": "measure_qt_interval", "reason": "排除QT间期延长"},
    {"action": "detect_anomaly", "reason": "检查是否有异常ECG模式"},
    {"action": "generate_report", "reason": "汇总所有发现为结构化诊断"}
  ]
}

病例2: "无症状30岁女性常规体检ECG"
{
  "plan": [
    {"action": "extract_r_peaks", "reason": "基础心率测量"},
    {"action": "classify_arrhythmia", "reason": "筛查常见心律失常"},
    {"action": "generate_report", "reason": "输出正常/简单发现"}
  ]
}

病例3: "已知房颤患者服药后随访ECG"
{
  "plan": [
    {"action": "extract_r_peaks", "reason": "检查心室率控制情况"},
    {"action": "compute_hrv", "reason": "评估节律变异性"},
    {"action": "classify_arrhythmia", "reason": "确认房颤状态并检测新发心律失常"},
    {"action": "measure_qt_interval", "reason": "监测药物引起的QT变化"},
    {"action": "query_medical_knowledge", "reason": "查阅房颤管理指南"},
    {"action": "generate_report", "reason": "与既往发现对比"}
  ]
}
"""


class Planner:
    """基于 LLM 的诊断计划生成器。

    参数:
        llm: LLMInterface 实例。
    """

    def __init__(self, llm: "LLMInterface"):
        self.llm = llm

    def generate_plan(self, context: Dict) -> str:
        """根据上下文生成诊断计划。

        参数:
            context: {
                query:           诊断问题文本
                ecg_shape:       ECG 信号维度
                patient_info:    患者元数据字典
                available_tools: 可用工具列表 [{name, description}, ...]
            }

        返回:
            JSON 字符串形式的计划。
        """
        tools_str = "\n".join(
            f"  - {t['name']}: {t['description']}"
            for t in context.get("available_tools", [])
        )

        prompt = f"""你是一个制定 ECG 诊断步骤的医学 AI。

可用工具:
{tools_str}

患者信息: {json.dumps(context.get('patient_info', {}))}
ECG 信号: {context.get('ecg_shape', '未知形状')}
问题: {context['query']}

{FEW_SHOT_EXAMPLES}

请为此病例生成诊断计划。只包含必要的工具。输出只有 JSON:

{{"plan": [{{"action": "...", "reason": "..."}}, ...]}}"""

        response = self.llm.chat([{"role": "user", "content": prompt}])
        return self._extract_json(response)

    def revise_plan(self, context: Dict) -> str:
        """根据反思反馈修正计划。

        参数:
            context: {
                current_plan: 当前计划步骤列表
                feedback:     反思反馈文本
            }
        """
        prompt = f"""你是一个修正 ECG 诊断计划的医学 AI。

当前计划: {json.dumps(context['current_plan'])}
反馈: {context['feedback']}

生成修正后的诊断计划。添加缺失步骤，移除不必要的步骤。
输出只有 JSON: {{"plan": [...]}}"""

        response = self.llm.chat([{"role": "user", "content": prompt}])
        return self._extract_json(response)

    @staticmethod
    def _extract_json(text: str) -> str:
        """从 LLM 回复中提取 JSON（可能包含 markdown 代码块包裹）。"""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
        return text.strip()
