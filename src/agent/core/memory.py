"""
Agent 记忆模块 — 多步骤推理的上下文存储与检索。

在整个诊断工作流期间维护:
    - 患者上下文 (ECG 信号, 元数据)
    - 工具执行历史 (观测序列)
    - 对话轮次
    - 工具结果缓存 (按工具名索引)

关键设计: get_context_for_tool() 自动向前一步提供依赖结果。
例如: 调用"compute_hrv"时，自动注入前一步"extract_r_peaks"的R峰结果。
"""

from typing import Any, Dict, List, Optional


class AgentMemory:
    """ECG AI Agent 的对话和上下文记忆。

    存储:
        - _context:       当前诊断上下文 (ECG, 患者信息)
        - _observations:  工具执行历史 (按时间排序)
        - _conversation:  对话轮次
        - _tool_results:  按工具名索引的结果缓存
    """

    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self._context: Dict[str, Any] = {}
        self._observations: List[Dict[str, Any]] = []
        self._conversation: List[Dict[str, str]] = []
        self._tool_results: Dict[str, Any] = {}  # 工具名 → 结果

    # ---- 上下文管理 ----
    def add_context(self, context: Dict[str, Any]):  self._context.update(context)
    def get_context(self) -> Dict[str, Any]:          return dict(self._context)
    def update_patient_info(self, field: str, value: str):
        if "patient_info" not in self._context:
            self._context["patient_info"] = {}
        self._context["patient_info"][field] = value

    # ---- 观测（工具结果）管理 ----
    def add_observation(self, obs: Dict[str, Any]):
        self._observations.append(obs)
        if len(self._observations) > self.max_history:
            self._observations.pop(0)
        # 缓存: 工具名 → 结果
        if "step" in obs:
            self._tool_results[obs["step"]] = obs.get("result")

    def get_observation(self, step_name: str) -> Optional[Dict]:
        for obs in reversed(self._observations):
            if obs.get("step") == step_name:
                return obs.get("result")
        return None

    def get_context_for_tool(self, tool_name: str) -> Dict:
        """获取工具调用所需的上下文（自动注入依赖）。

        原理: 不同工具需要不同的前置结果。
        - R峰检测需要 ECG 信号
        - HRV/QT 分析需要 R 峰结果
        - 分类器需要预处理后的 ECG
        """
        context = {"patient_info": self._context.get("patient_info", {})}

        # ECG 信号（大多数工具都需要）
        if tool_name in ("extract_r_peaks", "classify_arrhythmia",
                         "detect_anomaly", "measure_qt_interval", "plot_waveform"):
            context["ecg_signal"] = self._context.get("ecg_signal")

        # 依赖 R 峰结果的工具 → 自动注入前一步的R峰数据
        if tool_name in ("compute_hrv", "measure_qt_interval"):
            r_result = self._tool_results.get("extract_r_peaks")
            if r_result:
                context["r_peaks"] = r_result.get("r_peaks")
                context["rr_intervals"] = r_result.get("rr_intervals")

        return context

    # ---- 对话管理 ----
    def add_conversation(self, role: str, content: str):
        self._conversation.append({"role": role, "content": content})
        if len(self._conversation) > self.max_history:
            self._conversation.pop(0)

    def get_conversation(self) -> List[Dict[str, str]]:
        return list(self._conversation)

    # ---- 完整历史（供反思器使用） ----
    def get_history(self) -> Dict[str, Any]:
        return {
            "context": dict(self._context),
            "observations": list(self._observations),
            "conversation": list(self._conversation),
        }

    # ---- 重置 ----
    def clear(self):
        """清空所有记忆，为新诊断会话做准备。"""
        self._context.clear()
        self._observations.clear()
        self._conversation.clear()
        self._tool_results.clear()
