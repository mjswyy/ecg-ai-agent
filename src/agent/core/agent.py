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
    incomplete: bool = False  # max_steps 耗尽时未完成全部计划步骤则为 True


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
        # 检查报告 1.9 修复（🔴 跨患者污染）：每次诊断前清空记忆，
        # 防止上一名患者的工具结果（R 峰/心率/缓存诊断）注入本次计划。
        self.memory.clear()
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
            # 检查报告 1.9 修复：反射器异常不炸穿主流程，降级为"继续执行"
            try:
                should_continue, feedback = self.reflector.check(
                    step, self.memory.get_history()
                )
            except Exception as e:
                logger.warning(f"反射器异常，按继续执行处理: {e}")
                should_continue, feedback = True, ""
            if not should_continue:
                if self.verbose:
                    logger.info(f"反思终止: {feedback}")
                break

            # 如果反思建议修正，重新制定计划
            if feedback and "revise" in feedback.lower():
                plan = self._replan(plan, feedback)

        # 检查报告 1.9 修复：max_steps 耗尽仍有未完成步骤时显式告警
        if any(not s.completed for s in plan):
            logger.warning(f"达到 max_steps={self.max_steps}，"
                           f"仍有 {sum(1 for s in plan if not s.completed)} 步未完成")

        # ---- 步骤4: 综合所有发现生成最终诊断 ----
        result = self._synthesize_diagnosis(plan)
        result.incomplete = any(not s.completed for s in plan)
        return result

    # ================================================================
    # 内部方法
    # ================================================================

    def _collect_patient_info(self):
        """以交互方式询问缺失的患者信息。

        2D-O4 修复：用规范字段名存储（旧版把完整问句当 key，
        "患者的年龄是多少？" 而非 "age"，合并时被调用方 dict 覆盖丢弃）。
        """
        questions = [
            ("age", "患者的年龄是多少？"),
            ("sex", "患者的性别是？"),
            ("symptoms", "患者有什么症状？"),
            ("history", "有什么相关的病史？"),
        ]
        for field, q in questions:
            response = input(f"[Agent] {q}\n> ")
            if response.strip():
                self.memory.update_patient_info(field, response.strip())

    def _plan(
        self, ecg_signal, patient_info, query
    ) -> List[AgentStep]:
        """使用 LLM 制定诊断计划。

        LLM 根据患者信息和可用工具，生成一个结构化的步骤列表。
        如果 LLM 不可用，回退到预设的默认计划。
        """
        # 检查报告 1.9 修复：交互收集的患者信息写入 memory，规划时合并使用
        # 2D-O4 修复：合并而非"全有或全无"（旧版调用方传部分 dict 时丢弃交互信息）
        memory_pi = self.memory.get_context().get("patient_info") or {}
        merged_pi = {**memory_pi, **(patient_info or {})}
        patient_info = merged_pi
        context = {
            "query": query,
            "ecg_shape": list(ecg_signal.shape),
            "patient_info": patient_info or {},
            "available_tools": self.tools.list_tools(),
        }
        # 2D-O3 修复：LLM 不可用/网络故障不再裸抛，回退默认计划并告警
        # （docstring 承诺的"回退默认计划"旧版未实现）
        try:
            plan_json = self.planner.generate_plan(context)
        except Exception as e:
            logger.warning(f"规划器调用失败（{e}），使用默认计划")
            plan_json = ""
        return self._parse_plan(plan_json)

    def _parse_plan(self, plan_json: str) -> List[AgentStep]:
        """将 LLM 生成的 JSON 计划解析为 AgentStep 列表。

        检查报告 1.9 修复：全面容错——JSON 结构缺失/None/数组内含非
        dict 元素/params 非 dict 均不崩溃，逐条校验并跳过非法步骤。
        """
        try:
            data = json.loads(plan_json)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            # 支持 {"plan": [...]} 和 {"steps": [...]} 两种格式
            data = data.get("plan", data.get("steps", []))
        if not isinstance(data, list):
            data = []

        steps = []
        for s in data:
            if not isinstance(s, dict):
                logger.warning(f"计划中含非 dict 元素，已跳过: {s!r}")
                continue
            action = s.get("action", s.get("tool", "unknown"))
            if not isinstance(action, str) or not action.strip():
                continue
            params = s.get("params", s.get("parameters", {}))
            if not isinstance(params, dict):
                params = {}
            steps.append(AgentStep(
                action=action,
                params=params,
                reason=s.get("reason", s.get("rationale", "")),
            ))
        if not steps:
            # 回退: 使用默认的诊断计划
            # 第三轮审查 AGENT-LLM-Y2：移除 generate_report（无效步骤——
            # 工具恒输出"心率: N/A"且结果被综合逻辑忽略）
            # 4A/4J 审查修复：💡-2 本清单仅为"LLM 不可用时的确定性回退计划"，
            # 与 planner 的 few-shot 示例、mock 计划、工具注册清单刻意独立
            # （各自用途不同，不强行单一事实源）。
            logger.warning("LLM 计划解析为空，使用默认计划")
            return [
                AgentStep("extract_r_peaks", {}, "测量心率和基础节律"),
                AgentStep("compute_hrv", {}, "评估自主神经功能"),
                AgentStep("measure_qt_interval", {}, "检查 QT 间期"),
                AgentStep("classify_arrhythmia", {}, "诊断心律失常"),
            ]
        return steps

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
            # 合并步骤参数和记忆上下文（自动注入依赖结果）。
            # 检查报告 1.9 修复：显式步骤参数优先，记忆注入不覆盖同名 key
            params = {**self.memory.get_context_for_tool(tool_name), **step.params}

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
        # 第三轮审查 AGENT-LLM-O3：revise_plan 与 generate_plan 同款容错
        # （旧版无 try/except，LLM 不可用时异常穿透 diagnose 崩溃）
        try:
            new_plan_json = self.planner.revise_plan(context)
        except Exception as e:
            logger.warning(f"计划修订失败（{e}），保留原计划继续执行")
            return plan
        new_plan = self._parse_plan(new_plan_json)

        # 检查报告 1.9 修复：按 action 名回填已完成结果（旧版按下标，
        # LLM 重排/缩短计划时结果挂到错误步骤或静默丢弃）；
        # 未出现在新计划中的已完成步骤追加保留。
        done_by_action = {s.action: s for s in plan if s.completed}
        for ns in new_plan:
            if ns.action in done_by_action:
                ns.result = done_by_action[ns.action].result
                ns.completed = True
        appended = [s for s in plan if s.completed and s.action not in
                    {ns.action for ns in new_plan}]
        return new_plan + appended

    def _synthesize_diagnosis(self, plan: List[AgentStep]) -> DiagnosisResult:
        """从所有已完成的步骤中综合生成最终诊断。

        遍历已完成的工具调用结果，提取心率、节律、间期、
        诊断结论和置信度。

        R3 修复（第二次审查 2D-R1/R2/R3）：失败显式化——
        心率 0 / 节律 insufficient_data / 测量值 None 一律不当作
        真实测量透传；全部工具失败时最终建议必须显式标注"分析未完成"，
        禁止合成"常规随访"式确定性结论。
        """
        hr = None
        rhythm = None
        intervals = None
        diagnoses = []
        confidence = 0.0
        num_results = 0
        n_failed_steps = 0
        n_ok_steps = 0
        classification_failed = False  # 第三轮审查 AGENT-LLM-R1：诊断分类步骤失败标志
        classifier_ran = False  # 5A 审查（🟠-1）：分类步骤是否真正执行过（失败/未计划均 False）

        for step in plan:
            if not step.completed or step.error:
                n_failed_steps += 1
                if "classif" in step.action:
                    classification_failed = True
                continue

            result = step.result
            if not isinstance(result, dict):
                n_failed_steps += 1
                if "classif" in step.action:
                    classification_failed = True
                continue
            # R3 修复（2D-R1/R3）：工具"成功返回但标记测量失败"（insufficient_data/
            # error）同样计入失败步骤，不产出任何测量值
            if (result.get("error")
                    or result.get("insufficient")
                    or result.get("rhythm") == "insufficient_data"
                    or result.get("qt_interpretation") == "insufficient_data"):
                n_failed_steps += 1
                if "classif" in step.action:
                    classification_failed = True
                continue
            n_ok_steps += 1

            # 提取基础测量值（R3：0/None 视为测量失败，不透传为真实值）
            hr_v = result.get("heart_rate")
            # 4A/4J 审查修复：🟡-4 心率以 R 峰检测（extract_r_peaks）口径为
            # 唯一权威来源。QT 工具（measure_qt_interval）的 heart_rate 由 R 峰
            # 反推（qt_analyzer.py:111-114），数值接近但不完全相同；默认计划中
            # QT 排在 R 峰之后，旧版按步骤顺序"最后写入者胜"会被 QT 覆盖。
            # 仅当尚无有效心率时，才回退采纳其他工具的心率。
            if step.action == "extract_r_peaks":
                if hr_v not in (None, 0):
                    hr = hr_v
            elif hr is None and hr_v not in (None, 0):
                hr = hr_v
            rh = result.get("rhythm")
            if rh and rh != "insufficient_data":
                rhythm = rh

            # 提取间期测量值（R3：qt_ms 为 None 表示 QT 测量失败）
            if result.get("qt_ms") is not None:
                intervals = {
                    "qt_ms": result.get("qt_ms"),
                    "qtc_bazett": result.get("qtc_bazett"),
                    "qrs_duration_ms": result.get("qrs_duration_ms"),
                    "pr_interval_ms": result.get("pr_interval_ms"),
                }

            # 收集分类器诊断结果
            # 4A-RED-1 修复：兼容双分类器契约——旧 classify 工具返回
            # diagnoses[{name,probability,snomed_code,evidence}]，新
            # ECGFounderClassifier 返回 top_k/positives[{name,prob,snomed,
            # positive}]。旧版只读 diagnoses，接入新分类器后提取 0 条诊断
            # 并误报"分类步骤未完成"。
            if "diagnoses" in result:
                classifier_ran = True  # 5A 审查（🟠-1）
                for d in result["diagnoses"]:
                    if not isinstance(d, dict):
                        continue
                    try:
                        prob = float(d.get("probability", 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0
                    diagnoses.append({
                        "name": d.get("name", "Unknown"),
                        "snomed_code": d.get("snomed_code", ""),
                        "confidence": prob,
                        "evidence": d.get("evidence", ""),
                        # 5A 审查（🟠-2）：旧 classify 契约无逐类 Youden 阈值
                        # （decode 按固定 0.3 过滤）。旧版无条件 positive=True 把
                        # prob=0.31 标成"高置信度发现"，与新契约 Youden 判定
                        # 结论相反——现以 0.5 中位阈作近似阳性判定并注明口径
                        "positive": prob >= 0.5,
                    })
                    num_results += 1
            elif "top_k" in result or "positives" in result:
                classifier_ran = True  # 5A 审查（🟠-1）
                tk = result.get("top_k") or []
                pos = result.get("positives") or []
                seen = set()
                for d in list(pos) + list(tk):
                    if not isinstance(d, dict):
                        continue
                    name = d.get("name", "Unknown")
                    if name in seen:
                        continue
                    seen.add(name)
                    try:
                        prob = float(d.get("prob", 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0
                    diagnoses.append({
                        "name": name,
                        "snomed_code": d.get("snomed", ""),
                        "confidence": prob,
                        "evidence": ("分类器 top-5: " + "; ".join(
                            f"{p.get('name')}({p.get('prob', 0):.0%})"
                            for p in tk[:5] if isinstance(p, dict))),
                        # 4A/4J 审查修复：🟠-1 阳性标志沿用分类器逐类 Youden
                        # 阈值判定（positive = prob>=该类阈值），建议层据此识别
                        # "阳性发现"，不再依赖硬编码 0.7。
                        "positive": bool(d.get("positive", False)),
                    })
                    num_results += 1

        # 检查报告 1.9 修复：整体置信度取最高概率诊断（旧版累加后平均，
        # 多结果时被稀释且语义错误）
        if diagnoses:
            confidence = max(d["confidence"] for d in diagnoses)

        # R3 修复（2D-R2）：没有任何成功工具结果 → 分析未完成标志
        # （含 plan 为空/全部失败/无诊断但有失败步骤的兜底）
        analysis_failed = n_ok_steps == 0 and (n_failed_steps > 0 or not diagnoses)

        # 第三轮审查 AGENT-LLM-R1：测量成功但诊断分类步骤失败 → 部分失败，
        # 禁止落入"常规随访"（分类器失败被包装成"无异常"是医疗安全反模式）
        classification_failed = classification_failed and not diagnoses

        # 5A 审查（🟠-1）：无阳性时区分三种情况——分类步骤未执行（无法排除
        # 异常，不得"常规随访"）、分类成功但部分测量失败（提示信息不完整）、
        # 完整分析且无阳性（才允许"常规随访"）。旧版判据过窄：分类器未计划
        # 或测量失败但分类器成功且无阳性时均落入"常规随访"。
        return DiagnosisResult(
            diagnosis=sorted(diagnoses, key=lambda x: -x["confidence"]),
            reasoning_chain=plan,
            heart_rate=hr,
            rhythm=rhythm,
            intervals=intervals,
            confidence=confidence,
            recommendations=self._generate_recommendations(
                diagnoses,
                analysis_failed=analysis_failed,
                classification_failed=classification_failed,
                classifier_ran=classifier_ran,
                partial_failure=n_failed_steps > 0),
        )

    def _generate_recommendations(self, diagnoses: List[Dict],
                                  analysis_failed: bool = False,
                                  classification_failed: bool = False,
                                  classifier_ran: bool = False,
                                  partial_failure: bool = False) -> List[str]:
        """根据诊断结果生成临床建议。

        R3 修复（第二次审查 2D-R2）+ 第三轮审查 AGENT-LLM-R1：失败必须显式化，
        禁止把失败包装成"常规随访"——旧版只处理"全部工具失败"，漏掉
        "测量步骤成功但诊断分类器失败"这一最常见模式。
        5A 审查（🟠-1）：分类步骤未执行或部分测量失败时同样不得落入
        "常规随访"（信息不完整时"排除异常"是医疗安全反模式）。
        """
        if analysis_failed:
            return ["分析未能完成：全部工具调用失败或未返回有效测量"
                    "（可能为信号质量问题），以上内容不应作为临床结论，请重试或检查信号。"]
        if classification_failed:
            return ["诊断分类步骤未完成：分类器未返回有效诊断结果（测量数据正常但诊断结论缺失），"
                    "以上内容不应作为'排除异常'的依据，请重试分类或人工复核。"]
        # 4A/4J 审查修复：🟠-1 建议层不再硬编码 confidence>0.7——发现层"阳性"由
        # 逐类 Youden 阈值（0.06~0.66，ecgfounder_classifier.py）判定，固定 0.7
        # 会把 prob∈(Youden, 0.7) 的真实阳性弱化为"常规随访"（医疗安全反模式）。
        # Agent 无逐类阈值表，故以综合层写入的 positive 标志（=高于该类阳性
        # 判定阈值）为准，只把 positive=True 的诊断列为"高置信度发现"。
        recs = []
        for d in diagnoses:
            if d.get("positive", False):
                recs.append(
                    f"高置信度发现: {d['name']} — 建议临床对照确认"
                )
        if not recs:
            if not classifier_ran:
                # 5A 审查（🟠-1）：分类步骤从未执行（未计划/全失败已在上方
                # 拦截），无法排除异常——不得输出"常规随访"
                recs.append("分析未完成：诊断分类步骤未执行，无法排除潜在异常，"
                            "请重新分析或人工复核。")
            elif partial_failure:
                recs.append("未检测到高置信度异常，但部分测量未完成（信号质量或工具失败），"
                            "结论不完整；建议结合临床评估，如症状持续请复测。")
            else:
                recs.append("未检测到高置信度异常。建议常规随访。")
        return recs

    def format_report(self, result: "DiagnosisResult") -> str:
        """公开的报告格式化接口（conversation 层依赖；默认返回诊断列表文本）。"""
        return str(result.diagnosis)
