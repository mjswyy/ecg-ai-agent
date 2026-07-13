"""
ECG AI Agent — 基于 ReAct 模式的心电图诊断智能体。

这是整个项目的核心创新模块。Agent 遵循 ReAct (Reasoning + Acting) 循环:
    1. 收集患者信息（年龄、性别、症状）
    2. 将诊断任务分解为工具调用计划
    3. 按依赖顺序执行工具链
    4. 每步执行后进行自我反思验证
    5. 综合所有发现生成最终诊断报告

使用示例:
    llm = LLMInterface(backend="deepseek")
    agent = ECGAIAgent(llm, tool_registry)

    result = agent.diagnose(ecg_signal, patient_info)
    print(result.diagnosis)       # [{name, snomed_code, confidence, evidence}, ...]
    print(result.reasoning_chain) # [AgentStep, ...]
    print(result.recommendations) # ["临床建议1", "临床建议2"]
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class AgentStep:
    """诊断计划中的单一步骤。

    属性:
        action:   工具名称（如 "extract_r_peaks"）
        params:   工具参数字典
        reason:   为什么需要这一步
        result:   工具执行结果（执行后填充）
        error:    错误信息（如果执行失败）
        completed: 是否已完成
    """
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    result: Any = None
    error: Optional[str] = None
    completed: bool = False


@dataclass
class DiagnosisResult:
    """最终诊断输出。

    属性:
        diagnosis:        诊断列表 [{name, snomed_code, confidence, evidence}, ...]
        reasoning_chain:  推理步骤链 [AgentStep, ...]
        heart_rate:       心率 (bpm)
        rhythm:           节律描述
        intervals:        {qt_ms, qtc_bazett, qrs_duration_ms, pr_interval_ms}
        confidence:       整体置信度 (0-1)
        recommendations:  临床建议列表
    """
    diagnosis: List[Dict[str, Any]]
    reasoning_chain: List[AgentStep]
    heart_rate: Optional[float] = None
    rhythm: Optional[str] = None
    intervals: Optional[Dict] = None
    confidence: float = 0.0
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# 主 Agent 类
# ============================================================================

class ECGAIAgent:
    """ReAct 风格的 ECG 诊断 AI Agent。

    Agent 不直接诊断，而是像一个"医生"一样：
    1. 制定检查计划
    2. 调用各种工具（R峰检测、分类器、异常检测等）
    3. 检查每步结果是否合理
    4. 综合所有证据给出最终诊断

    参数:
        llm:           LLM 接口（用于推理和规划）
        tool_registry: 工具注册表（可用诊断工具集）
        max_steps:     最大 ReAct 循环迭代次数
        verbose:       是否输出详细日志
    """

    def __init__(
        self,
        llm: "LLMInterface",
        tool_registry: "ToolRegistry",
        max_steps: int = 10,
        verbose: bool = False,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.max_steps = max_steps
        self.verbose = verbose

        # 延迟导入子模块（避免循环依赖）
        from .planner import Planner
        from .memory import AgentMemory
        from .reflector import Reflector
        from .reasoner import MedicalReasoner

        self.planner = Planner(llm)           # 任务规划器
        self.memory = AgentMemory()           # 上下文记忆
        self.reflector = Reflector(llm)       # 自反思器
        self.reasoner = MedicalReasoner(llm)  # 医学推理器

    # ================================================================
    # 主入口: 完整诊断流程
    # ================================================================

    def diagnose(
        self,
        ecg_signal: "np.ndarray",
        patient_info: Optional[Dict] = None,
        query: str = "分析这份心电图并给出诊断。",
    ) -> DiagnosisResult:
        """运行完整的诊断工作流。

        参数:
            ecg_signal:   12 导联 ECG 信号，形状 (12, L)，单位 mV。
            patient_info: 可选的病人信息字典 {age, sex, symptoms, history}。
            query:        自然语言诊断查询。

        返回:
            DiagnosisResult 包含诊断结论和完整推理链。
        """
        # ---- 初始化上下文 ----
        self.memory.add_context({
            "ecg_signal": ecg_signal,
            "patient_info": patient_info or {},
            "query": query,
        })

        # ---- 步骤1: 收集缺失的患者信息 ----
        # 只在交互模式下询问（有 tty 时），批量/测试模式跳过
        if (patient_info is None or not patient_info.get("age")) and sys.stdin.isatty():
            self._collect_patient_info()

        # ---- 步骤2: 制定诊断计划 ----
        plan = self._plan(ecg_signal, patient_info, query)
        if self.verbose:
            logger.info(f"诊断计划: {[s.action for s in plan]}")

        # ---- 步骤3: ReAct 循环执行计划 ----
        for iteration in range(self.max_steps):
            # 找出所有未完成的步骤
            pending = [s for s in plan if not s.completed]
            if not pending:
                break  # 全部完成

            # 执行下一步
            step = pending[0]
            self._execute_step(step)
            if self.verbose:
                status = "OK" if not step.error else f"ERR: {step.error}"
                logger.info(f"  [{status}] {step.action}")

            # 自我反思: 结果是否合理？需要修正计划吗？
            should_continue, feedback = self.reflector.check(
                step, self.memory.get_history()
            )
            if not should_continue:
                if self.verbose:
                    logger.info(f"反思终止: {feedback}")
                break

            # 如果反思建议修正，重新制定计划
            if feedback and "revise" in feedback.lower():
                plan = self._replan(plan, feedback)

        # ---- 步骤4: 综合所有发现生成最终诊断 ----
        return self._synthesize_diagnosis(plan)

    # ================================================================
    # 内部方法
    # ================================================================

    def _collect_patient_info(self):
        """以交互方式询问缺失的患者信息。"""
        questions = [
            "患者的年龄是多少？",
            "患者的性别是？",
            "患者有什么症状？",
            "有什么相关的病史？",
        ]
        for q in questions:
            response = input(f"[Agent] {q}\n> ")
            if response.strip():
                self.memory.update_patient_info(q, response.strip())

    def _plan(
        self, ecg_signal, patient_info, query
    ) -> List[AgentStep]:
        """使用 LLM 制定诊断计划。

        LLM 根据患者信息和可用工具，生成一个结构化的步骤列表。
        如果 LLM 不可用，回退到预设的默认计划。
        """
        context = {
            "query": query,
            "ecg_shape": list(ecg_signal.shape),
            "patient_info": patient_info or {},
            "available_tools": self.tools.list_tools(),
        }
        plan_json = self.planner.generate_plan(context)
        return self._parse_plan(plan_json)

    def _parse_plan(self, plan_json: str) -> List[AgentStep]:
        """将 LLM 生成的 JSON 计划解析为 AgentStep 列表。"""
        try:
            data = json.loads(plan_json)
            if isinstance(data, dict):
                # 支持 {"plan": [...]} 和 {"steps": [...]} 两种格式
                data = data.get("plan", data.get("steps", []))
            return [
                AgentStep(
                    action=s.get("action", s.get("tool", "unknown")),
                    params=s.get("params", s.get("parameters", {})),
                    reason=s.get("reason", s.get("rationale", "")),
                )
                for s in data
            ]
        except json.JSONDecodeError:
            # 回退: 使用默认的诊断计划
            logger.warning("LLM 计划解析失败，使用默认计划")
            return [
                AgentStep("extract_r_peaks", {}, "测量心率和基础节律"),
                AgentStep("compute_hrv", {}, "评估自主神经功能"),
                AgentStep("measure_qt_interval", {}, "检查 QT 间期"),
                AgentStep("classify_arrhythmia", {}, "诊断心律失常"),
                AgentStep("generate_report", {}, "综合生成报告"),
            ]

    def _execute_step(self, step: AgentStep):
        """执行单个工具调用步骤。

        从记忆模块获取工具所需的上下文（如前一步的 R 峰结果）。
        """
        tool_name = step.action

        if tool_name not in self.tools:
            step.error = f"工具未找到: {tool_name}"
            step.completed = True
            return

        try:
            # 合并步骤参数和记忆上下文（自动注入依赖结果）
            params = {**step.params}
            params.update(self.memory.get_context_for_tool(tool_name))

            result = self.tools.call(tool_name, **params)
            step.result = result
            self.memory.add_observation({"step": tool_name, "result": result})
        except Exception as e:
            step.error = str(e)
            logger.error(f"工具 {tool_name} 执行失败: {e}")

        step.completed = True

    def _replan(self, plan: List[AgentStep], feedback: str) -> List[AgentStep]:
        """根据反思反馈重新制定计划。

        保留已完成的步骤，只修改未完成部分。
        """
        context = {
            "current_plan": [
                {"action": s.action, "params": s.params, "completed": s.completed}
                for s in plan
            ],
            "feedback": feedback,
        }
        new_plan_json = self.planner.revise_plan(context)
        new_plan = self._parse_plan(new_plan_json)

        # 保留已完成的步骤结果
        for i, s in enumerate(plan):
            if s.completed and i < len(new_plan):
                new_plan[i].result = s.result
                new_plan[i].completed = True

        return new_plan

    def _synthesize_diagnosis(self, plan: List[AgentStep]) -> DiagnosisResult:
        """从所有已完成的步骤中综合生成最终诊断。

        遍历已完成的工具调用结果，提取心率、节律、间期、
        诊断结论和置信度。
        """
        hr = None
        rhythm = None
        intervals = None
        diagnoses = []
        confidence = 0.0
        num_results = 0

        for step in plan:
            if not step.completed or step.error:
                continue

            result = step.result
            if not isinstance(result, dict):
                continue

            # 提取基础测量值
            if "heart_rate" in result:
                hr = result["heart_rate"]
            if "rhythm" in result:
                rhythm = result["rhythm"]

            # 提取间期测量值
            if "qt_ms" in result:
                intervals = {
                    "qt_ms": result.get("qt_ms"),
                    "qtc_bazett": result.get("qtc_bazett"),
                    "qrs_duration_ms": result.get("qrs_duration_ms"),
                    "pr_interval_ms": result.get("pr_interval_ms"),
                }

            # 收集分类器诊断结果
            if "diagnoses" in result:
                for d in result["diagnoses"]:
                    diagnoses.append({
                        "name": d.get("name", "Unknown"),
                        "snomed_code": d.get("snomed_code", ""),
                        "confidence": d.get("probability", 0.0),
                        "evidence": d.get("evidence", ""),
                    })
                    confidence += d.get("probability", 0.0)
                    num_results += 1

        if num_results > 0:
            confidence /= num_results

        return DiagnosisResult(
            diagnosis=sorted(diagnoses, key=lambda x: -x["confidence"]),
            reasoning_chain=plan,
            heart_rate=hr,
            rhythm=rhythm,
            intervals=intervals,
            confidence=confidence,
            recommendations=self._generate_recommendations(diagnoses),
        )

    def _generate_recommendations(self, diagnoses: List[Dict]) -> List[str]:
        """根据诊断结果生成临床建议。"""
        recs = []
        for d in diagnoses:
            if d["confidence"] > 0.7:
                recs.append(
                    f"高置信度发现: {d['name']} — 建议临床对照确认"
                )
        if not recs:
            recs.append("未检测到高置信度异常。建议常规随访。")
        return recs
