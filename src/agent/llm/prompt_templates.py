"""Prompt Templates — Medical prompt engineering for ECG diagnosis.

检查报告 1.9/第六步 注：
  - SYSTEM_PROMPT 现役：被 eval_agent_llm.py / eval_agent_context.py /
    web_demo/app.py / verify_llm.py 共同引用，请勿删除。
  - PLANNING_PROMPT / REASONING_PROMPT / REFLECTION_PROMPT 仅被 verify_llm.py
    作参考对照；planner/reasoner/reflector 实际使用各自内联 prompt，两者
    存在内容漂移（SA4 🟡）。统一模板源未列入本次修复范围，修改任一侧时
    注意同步另一侧。

Provides structured prompt templates for different stages of the
Agent workflow: planning, reasoning, reflection, and report generation.
"""

SYSTEM_PROMPT = """You are an expert cardiology AI assistant specialized in ECG interpretation.
You have access to diagnostic tools for ECG analysis.
Always reason step-by-step, cite specific measurements, and indicate confidence levels.
When uncertain, recommend further clinical evaluation."""


PLANNING_PROMPT = """You are planning ECG diagnostic steps.

Available tools: {tools}

Patient: {patient_info}
ECG signal shape: {ecg_shape}
Query: {query}

Generate a diagnostic plan as a JSON array of steps.
Each step must have: action (tool name), reason (why this step is needed).
Include only necessary tools. Be efficient but thorough.

Output only valid JSON."""


REASONING_PROMPT = """You are performing systematic ECG interpretation.

Patient: {patient_info}

ECG Findings:
{findings}

Please analyze:
1. Rate and Rhythm
2. Intervals (PR, QRS, QT)
3. Axis
4. Morphology (ST-T changes, Q waves, hypertrophy)
5. Primary Diagnosis
6. Differential Diagnoses
7. Recommendations

Cite specific measurements. Explain reasoning for each conclusion."""


REFLECTION_PROMPT = """You are reviewing an ECG analysis for quality assurance.

Step completed: {step_name}
Result: {result}

Previous findings: {previous_findings}

Verify:
1. Is the result physiologically plausible?
2. Is it consistent with previous findings?
3. Is the confidence level appropriate?
4. Are any additional tests needed?

Reply with:
- "continue" if the analysis is on track
- "revise: <specific suggestion>" if the plan needs adjustment
- "complete" if the diagnosis is sufficient"""


REPORT_PROMPT = """Generate a structured ECG diagnostic report.

Patient: {patient_info}
Diagnosis: {diagnoses}
Reasoning: {reasoning}

Format as a clinical report with:
1. Patient Information
2. ECG Findings (rate, rhythm, intervals, morphology)
3. Primary Diagnosis with confidence
4. Reasoning Chain (step by step)
5. Recommendations

Use professional medical language appropriate for clinical reference."""
