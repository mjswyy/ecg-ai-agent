"""
自反思模块 — 验证中间结果和诊断计划的合理性。

双层验证机制:
    1. 快速规则检查: 基于生理范围的硬规则（不调用LLM，零延迟）
       - 心率: 20-300 bpm
       - QT间期: 200-600 ms
       - SDNN: <300 ms
       - 跨工具心率一致性: ±20 bpm

    2. LLM 深度验证: 对关键步骤（分类器、异常检测、报告）调用LLM审查

使用示例:
    reflector = Reflector(llm)
    should_continue, feedback = reflector.check(step, history)
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class Reflector:
    """自反思和验证模块。

    参数:
        llm: LLMInterface 实例（仅关键步骤使用）。
    """

    def __init__(self, llm: "LLMInterface"):
        self.llm = llm

    def check(self, step: "AgentStep", history: Dict) -> Tuple[bool, str]:
        """检查一个已完成的步骤并决定下一步操作。

        返回:
            (should_continue, feedback) 元组。
            should_continue=False 表示诊断已完成，可以停止。
            feedback 包含修正建议（以 "revise:" 开头）。
        """
        # 步骤失败 → 继续执行其他步骤
        if step.error:
            return True, f"步骤 {step.action} 失败: {step.error}。继续执行剩余步骤。"

        if step.result is None:
            return True, ""

        # ---- 第1层: 快速规则检查（不调用LLM） ----
        issue = self._quick_check(step, history)
        if issue:
            return True, issue

        # ---- 第2层: 对关键步骤使用LLM验证 ----
        if self._is_critical(step.action):
            return self._llm_verify(step, history)

        return True, ""

    def _quick_check(self, step: "AgentStep", history: Dict) -> str:
        """基于生理范围的硬规则检查（零LLM调用）。

        返回空字符串表示通过，否则返回修正建议。
        """
        result = step.result if isinstance(step.result, dict) else {}

        # 检查1: 心率在生理范围内？
        hr = result.get("heart_rate")
        if hr is not None and (hr < 20 or hr > 300):
            return f"revise: 心率 {hr} bpm 超出生理范围。请检查R峰检测。"

        # 检查2: HRV SDNN 是否异常大？
        sdnn = result.get("sdnn")
        if sdnn is not None and sdnn > 300:
            return "revise: SDNN > 300ms 极为异常。请检查RR间期。"

        # 检查3: QT间期在生理范围内？
        qt = result.get("qt_ms")
        if qt is not None and (qt < 200 or qt > 600):
            return f"revise: QT 间期 {qt}ms 超出生理范围。"

        # 检查4: 跨工具心率一致性（HRV工具 vs R峰检测）
        if step.action == "compute_hrv" and hr is not None:
            prev_hr = None
            for obs in history.get("observations", []):
                r = obs.get("result", {})
                if isinstance(r, dict) and "heart_rate" in r:
                    prev_hr = r["heart_rate"]
                    break
            if prev_hr and abs(prev_hr - hr) > 20:
                return f"revise: 心率不一致 (前次={prev_hr} vs 本次={hr} bpm)"

        return ""

    @staticmethod
    def _is_critical(action: str) -> bool:
        """判断是否是需要LLM深度验证的关键步骤。"""
        return action in ("classify_arrhythmia", "detect_anomaly", "generate_report")

    def _llm_verify(self, step, history) -> Tuple[bool, str]:
        """使用LLM验证关键步骤的发现。

        返回 (是否继续, 反馈文本)。
        """
        prompt = f"""你是审查 ECG 分析的医学 AI。

刚完成的步骤: {step.action}
结果: {step.result}

最近的观测: {history.get('observations', [])[-3:]}

请检查:
1. 结果是否与之前的发现一致？
2. 置信度是否合理？
3. 是否需要额外的检查？

回复:
- "continue" 如果发现一致
- "revise: <原因>" 如果计划需要调整
- "complete" 如果诊断已经足够"""

        response = self.llm.chat([{"role": "user", "content": prompt}])
        if "revise" in response.lower():
            return True, response.strip()
        elif "complete" in response.lower():
            return False, response.strip()
        return True, ""
