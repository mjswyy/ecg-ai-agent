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

        Args:
            ecg_signal: 12-lead ECG array.
            query: Initial diagnostic query.
        """
        self.state = ConversationState.INIT
        self._turn = 0
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

        if self.state == ConversationState.COLLECTING_INFO:
            return self._handle_info_collection(user_input)
        elif self.state in (ConversationState.PLANNING, ConversationState.EXECUTING,
                            ConversationState.REFLECTING, ConversationState.FINALIZING):
            return self._advance_state()

        return None

    def _handle_info_collection(self, text: str) -> str:
        """Parse user-provided patient information."""
        text_lower = text.lower()

        # Simple keyword-based extraction
        if any(w in text_lower for w in ("year", "age", "old", "岁")):
            import re
            ages = re.findall(r'\d+', text)
            if ages:
                self._collected_info["age"] = ages[0]

        if any(w in text_lower for w in ("male", "female", "男", "女")):
            self._collected_info["sex"] = "Male" if "male" in text_lower or "男" in text else "Female"

        if any(w in text_lower for w in ("pain", "palpitation", "dizzy", "chest", "胸痛", "心悸", "头晕")):
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
            plan = self.agent.memory.get_context().get("plan", [])
            for step in plan:
                if not step.completed:
                    self.agent._execute_step(step)
                    self._results.append({
                        "step": step.action,
                        "result": step.result,
                        "error": step.error,
                    })
            return self._summarize_results()

        elif self.state == ConversationState.FINALIZING:
            plan = self.agent.memory.get_context().get("plan", [])
            result = self.agent._synthesize_diagnosis(plan)
            return self.agent._format_report(result) if hasattr(self.agent, '_format_report') else str(result.diagnosis)

        return None

    def _summarize_results(self) -> str:
        """Summarize tool execution results."""
        lines = ["Results:"]
        for r in self._results:
            result = r.get("result", {})
            if isinstance(result, dict):
                if "heart_rate" in result:
                    lines.append(f"- Heart rate: {result['heart_rate']} bpm")
                if "sdnn" in result:
                    lines.append(f"- HRV: SDNN={result['sdnn']:.1f}ms, RMSSD={result['rmssd']:.1f}ms")
                if "qtc_bazett" in result:
                    lines.append(f"- QTc: {result['qtc_bazett']:.0f}ms")
        return "\n".join(lines)

    def get_full_transcript(self) -> List[Dict[str, Any]]:
        """Get the full conversation transcript."""
        return [
            {"turn": self._turn, "state": self.state.name,
             "collected_info": dict(self._collected_info),
             "results": list(self._results)}
        ]
