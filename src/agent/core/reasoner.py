"""Medical Reasoner — Chain-of-Thought reasoning for ECG interpretation.

Generates step-by-step clinical reasoning chains using the LLM.

Usage:
    reasoner = MedicalReasoner(llm)
    reasoning = reasoner.reason(context, observations)
"""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MedicalReasoner:
    """Chain-of-thought clinical reasoning for ECG diagnosis.

    Args:
        llm: LLM interface instance.
    """

    def __init__(self, llm: "LLMInterface"):
        self.llm = llm

    def reason(
        self,
        context: Dict,
        observations: List[Dict],
    ) -> str:
        """Generate step-by-step clinical reasoning.

        Args:
            context: Patient context and ECG info.
            observations: Tool execution results.

        Returns:
            Text reasoning chain.
        """
        prompt = self._build_reasoning_prompt(context, observations)
        response = self.llm.chat([
            {"role": "system", "content": "You are a cardiologist performing systematic ECG interpretation."},
            {"role": "user", "content": prompt},
        ])
        return response

    def _build_reasoning_prompt(
        self, context: Dict, observations: List[Dict]
    ) -> str:
        """Build the reasoning prompt from context and observations."""
        patient = context.get("patient_info", {})

        prompt = f"""Patient: {json.dumps(patient)}

ECG Findings:
"""
        for obs in observations:
            result = obs.get("result", {})
            if isinstance(result, dict):
                prompt += f"- {obs.get('step', 'Unknown')}: {json.dumps(result)}\n"

        prompt += """
Please provide a systematic ECG interpretation following this structure:

1. Rate and Rhythm: What is the heart rate? Is the rhythm regular or irregular?
2. Axis: Any axis deviation?
3. Intervals: PR, QRS, QT — any abnormalities?
4. Morphology: Any ST-T changes? Q waves? Bundle branch blocks?
5. Diagnosis: What are the primary findings? Differential diagnoses?
6. Recommendations: What clinical actions are suggested?

Be specific, cite the measured values, and explain your reasoning step by step."""

        return prompt

    def explain_finding(
        self, finding_name: str, finding_value: Any, context: Dict
    ) -> str:
        """Explain what a specific ECG finding means clinically.

        Args:
            finding_name: Name of the finding (e.g., "QTc prolongation").
            finding_value: The measured value.
            context: Patient context.

        Returns:
            Clinical explanation.
        """
        prompt = f"""Explain the clinical significance of this ECG finding:

Finding: {finding_name}
Value: {finding_value}
Patient context: {json.dumps(context.get('patient_info', {}))}

Explain in 2-3 sentences what this means for the patient, what could cause it,
and what should be done next."""

        return self.llm.chat([{"role": "user", "content": prompt}])
