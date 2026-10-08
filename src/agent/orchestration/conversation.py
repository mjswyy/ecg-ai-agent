"""Conversation Manager — Multi-turn interaction state machine.

Manages the agent-user conversation through states:
    COLLECTING_INFO → PLANNING → EXECUTING → REFLECTING → FINALIZING

Usage:
    conv = ConversationManager(agent)
    conv.start(ecg_signal)
    while not conv.is_finished:
        user_input = input(conv.prompt)
        conv.handle(user_input)
"""

import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States of the diagnostic conversation."""
    INIT = auto()
    COLLECTING_INFO = auto()
    PLANNING = auto()
    EXECUTING = auto()
    REFLECTING = auto()
    FINALIZING = auto()
    DONE = auto()


class ConversationManager:
    """Multi-turn conversation manager for the ECG AI Agent.

    Args:
        agent: ECGAIAgent instance.
    """

    # 4A/4J 审查修复：🟡-7 REFLECTING 为占位状态——反思实际在 EXECUTING 分支
    # 内逐步完成（_advance_state 的 EXECUTING 分支对每个步骤调用
    # reflector.check，见下），REFLECTING 状态无独立校验逻辑，handle() 下次
    # 调用时被 _advance_state() 直接跳过到 FINALIZING。保留该状态仅为状态机
    # 语义完整，未接入独立反思/多轮交互输入。
    STATE_TRANSITIONS = {
        ConversationState.INIT: ConversationState.COLLECTING_INFO,
        ConversationState.COLLECTING_INFO: ConversationState.PLANNING,
        ConversationState.PLANNING: ConversationState.EXECUTING,
        ConversationState.EXECUTING: ConversationState.REFLECTING,
        ConversationState.REFLECTING: ConversationState.FINALIZING,
        ConversationState.FINALIZING: ConversationState.DONE,
    }

    def __init__(self, agent: "ECGAIAgent"):
        self.agent = agent
        self.state = ConversationState.INIT
        self._collected_info: Dict[str, str] = {}
        self._results: List[Dict] = []
        self._turn = 0

    @property
    def is_finished(self) -> bool:
        return self.state == ConversationState.DONE

    @property
    def prompt(self) -> str:
        """Get the current prompt for the user."""
        prompts = {
            ConversationState.COLLECTING_INFO: self._info_prompt(),
            ConversationState.PLANNING: "Planning diagnostic steps...",
            ConversationState.EXECUTING: "Running diagnostic tools...",
            ConversationState.REFLECTING: "Verifying findings...",
            ConversationState.FINALIZING: "Generating report...",
            ConversationState.DONE: "Diagnosis complete.",
        }
        return prompts.get(self.state, "")

    def _info_prompt(self) -> str:
        """Generate information collection prompt."""
        missing = []
        if "age" not in self._collected_info:
            missing.append("patient's age")
        if "sex" not in self._collected_info:
            missing.append("patient's sex")
        if "symptoms" not in self._collected_info:
            missing.append("symptoms (e.g., chest pain, palpitations, dizziness)")
        if "history" not in self._collected_info:
            missing.append("relevant medical history")

        if missing:
            return f"Please provide: {', '.join(missing)}."
        return ""

    def start(self, ecg_signal: "np.ndarray", query: str = None):
        """Start a new diagnostic conversation.

        检查报告 1.9 修复（🔴）：
          - 重置 _collected_info/_results/_turn，防止第二名患者继承
            第一名患者的年龄/性别/症状与工具结果（跨患者污染）；
          - 直接进入 COLLECTING_INFO——旧版停在 INIT 且 handle() 无
            INIT 转移分支，按 docstring 用法会无限循环打印空提示。
        """
        self._collected_info = {}
        self._results = []
        self._turn = 0
        self.state = ConversationState.COLLECTING_INFO
        self.agent.memory.clear()
        self.agent.memory.add_context({
            "ecg_signal": ecg_signal,
            "query": query or "Analyze this ECG",
        })

    def handle(self, user_input: str) -> Optional[str]:
        """Handle a user input and advance the conversation.

        Args:
            user_input: User's text response.

        Returns:
            Agent response or None.
        """
        self._turn += 1

        if self.state == ConversationState.INIT:
            # 防御分支：INIT 也应能推进（正常路径 start() 已直达 COLLECTING_INFO）
            self._advance_state()
            return self.prompt
        if self.state == ConversationState.COLLECTING_INFO:
            return self._handle_info_collection(user_input)
        elif self.state in (ConversationState.PLANNING, ConversationState.EXECUTING,
                            ConversationState.REFLECTING, ConversationState.FINALIZING):
            # 4A/4J 审查修复：🟡-7 中间状态（含占位的 REFLECTING）直接推进，
            # 不解析用户输入——多轮交互的反思实际在 EXECUTING 分支内完成。
            return self._advance_state()

        return None

    def _handle_info_collection(self, text: str) -> str:
        """Parse user-provided patient information."""
        text_lower = text.lower()

        # Simple keyword-based extraction
        if any(w in text_lower for w in ("year", "age", "岁")):
            import re
            # 第三轮审查 AGENT-LLM-Y1：年龄锚定 age/岁/year 附近数字并做范围校验
            # （旧版取文本第一个数字——'3 stents placed, age 65' 会解析成 3）
            # 5A 审查（🟡-2）："old" 需词边界（"he told 3 doctors"→age=3、
            # "please hold 2 tablets"→age=2 旧版误匹配）
            m = re.search(r"(?:age|aged|岁|\bold\b)[^\d]{0,8}(\d{1,3})"
                          r"|(\d{1,3})\s*(?:year|yr|岁)", text_lower)
            if m:
                age_val = int(m.group(1) or m.group(2))
                if 0 < age_val < 120:
                    self._collected_info["age"] = str(age_val)

        if any(w in text_lower for w in ("male", "female", "男", "女")):
            # 2D-O2 修复："female" 含子串 "male" → 女性被误记为 Male
            # （旧版 `"male" in "female"` 恒真）
            self._collected_info["sex"] = (
                "Male" if ("男" in text or ("male" in text_lower and "female" not in text_lower))
                else "Female")

        # 5A 审查（🟡-3）：补 "dizziness"（旧版只配 "dizzy"，"I have dizziness"
        # 漏配）。复检C（🟡-1）登记接受现状：symptoms 存原始输入整段文本
        # （非结构化、仅供展示与 LLM 上下文；不实现"只存命中子串"——输入
        # 文本本身就是最完整的情境描述，截断会丢信息）
        if any(w in text_lower for w in ("pain", "palpitation", "dizzy", "dizziness",
                                         "chest", "胸痛", "心悸", "头晕")):
            self._collected_info["symptoms"] = text

        # Update agent memory
        for k, v in self._collected_info.items():
            self.agent.memory.update_patient_info(k, v)

        # Check if we have enough info
        if len(self._collected_info) >= 2:
            self._advance_state()
            return "Thank you. I'll now analyze the ECG with this information."

        return self._info_prompt()

    def _advance_state(self) -> Optional[str]:
        """Move to the next conversation state."""
        next_state = self.STATE_TRANSITIONS.get(self.state)
        if next_state:
            self.state = next_state

        if self.state == ConversationState.PLANNING:
            context = self.agent.memory.get_context()
            plan = self.agent._plan(
                context.get("ecg_signal"),
                context.get("patient_info"),
                context.get("query", "Diagnose"),
            )
            self.agent.memory.add_context({"plan": plan})

            tools_str = ", ".join(s.action for s in plan)
            return f"I'll run the following tests: {tools_str}."

        elif self.state == ConversationState.EXECUTING:
            # 批量执行未完成步骤（单轮完成，文档化语义）；检查报告 1.9 修复：
            # 每步后仍走反射器校验，避免完全绕过反思/修订。
            # 2D-O5 修复：replan 重绑后从新计划头部重新扫描（旧版循环仍迭代
            # 旧列表，修订计划的新步骤永不执行）。
            plan = list(self.agent.memory.get_context().get("plan", []))
            executed = set()
            # 第三轮审查 AGENT-LLM-O4：循环加硬上限（旧版 while True，
            # LLM 反复 revise 时可无限续跑）
            max_rounds = getattr(self.agent, "max_steps", 10) * 3
            rounds = 0
            while rounds < max_rounds:
                rounds += 1
                pending = [s for s in plan
                           if not s.completed and id(s) not in executed]
                if not pending:
                    break
                step = pending[0]
                self.agent._execute_step(step)
                executed.add(id(step))
                self._results.append({
                    "step": step.action,
                    "result": step.result,
                    "error": step.error,
                })
                try:
                    should_continue, feedback = self.agent.reflector.check(
                        step, self.agent.memory.get_history())
                except Exception as e:
                    logger.warning(f"反射器异常，继续执行: {e}")
                    should_continue, feedback = True, ""
                if not should_continue:
                    logger.warning(f"反思终止于 {step.action}: {feedback}")
                    break
                if feedback and "revise" in feedback.lower():
                    plan = self.agent._replan(plan, feedback)
                    self.agent.memory.add_context({"plan": plan})
            return self._summarize_results()

        elif self.state == ConversationState.FINALIZING:
            plan = self.agent.memory.get_context().get("plan", [])
            result = self.agent._synthesize_diagnosis(plan)
            return self.agent.format_report(result)

        return None

    def _summarize_results(self) -> str:
        """Summarize tool execution results."""
        lines = ["Results:"]
        for r in self._results:
            result = r.get("result", {})
            if isinstance(result, dict):
                if "heart_rate" in result:
                    lines.append(f"- Heart rate: {result['heart_rate']} bpm")
                # 检查报告 1.9 修复：数值格式化前校验类型（None/字符串不崩溃）
                sdnn, rmssd = result.get("sdnn"), result.get("rmssd")
                if isinstance(sdnn, (int, float)) and isinstance(rmssd, (int, float)):
                    lines.append(f"- HRV: SDNN={sdnn:.1f}ms, RMSSD={rmssd:.1f}ms")
                qtc = result.get("qtc_bazett")
                if isinstance(qtc, (int, float)):
                    lines.append(f"- QTc: {qtc:.0f}ms")
        return "\n".join(lines)

    def get_full_transcript(self) -> List[Dict[str, Any]]:
        """Get the full conversation transcript."""
        return [
            {"turn": self._turn, "state": self.state.name,
             "collected_info": dict(self._collected_info),
             "results": list(self._results)}
        ]
