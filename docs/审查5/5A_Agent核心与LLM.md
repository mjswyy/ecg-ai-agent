# 第五次全面代码审查 — 5A_Agent核心与LLM

> 审查人：独立审查员 5A（未读取任何历史审查产物）
> 范围：`src/agent/core/*`、`src/agent/llm/*`、`src/agent/orchestration/*`、`src/agent/tools/*`、`src/agent/**/__init__.py`
> 验证方式：逐行通读全部范围内文件；只读 Python 验证（`LLMInterface(mock=True)` + 手造 `AgentStep`/`AgentMemory`/`ConversationManager`，未调用真实 LLM API，未修改任何文件）
> 关键背景（中性事实，非结论）：`ECGAIAgent` 框架（含 `ToolRegistry`/`Memory`/`Planner`/`Reflector`/`Reasoner`/`ConversationManager`/`ToolScheduler`）当前**未接入**实际评测与演示主路径——`web_demo/app.py` 用自有 `DemoPipeline`、`scripts/eval_agent_llm.py` 用自有 `AgentEvaluator`，二者均直接调用 `ECGFounderClassifier.predict` + `RPeakDetector`/`HRVAnalyzer`/`QTAnalyzer`，绕过本范围大部分类。本报告仍按"代码本身"审查，但严重度评估已计入该接线现状。

## 统计（🔴 0 / 🟠 2 / 🟡 6 / 💡 4）

## 发现清单

### 🟠-1 推荐层在"分析不完整"时仍输出"未检测到高置信度异常。建议常规随访。"

- **位置**：`src/agent/core/agent.py:490`（`analysis_failed = n_ok_steps == 0 and ...`）、`agent.py:494`（`classification_failed = classification_failed and not diagnoses`）、`agent.py:518-537`（`_generate_recommendations`）、`agent.py:535-536`
- **证据**：`analysis_failed` 仅在"没有任何一步成功"时为真；`classification_failed` 仅在"存在 classify 步骤且该步骤失败"时为真。两种边界态被漏判为"正常"并落入 536 行的"常规随访"：
  1. **计划未含分类器**：LLM 计划只有 `extract_r_peaks`（无 `classify_arrhythmia`），实测 `_synthesize_diagnosis` 输出 `recommendations=['未检测到高置信度异常。建议常规随访。']`（CASE1）。
  2. **测量失败但分类器成功且无阳性**：`extract_r_peaks` 返回 `insufficient`（`heart_rate=0`）被判失败、`classify_arrhythmia` 成功但 `positives=[]`，实测输出 `hr=None, rhythm=None, recs=['未检测到高置信度异常。建议常规随访。']`（CASE5）——心率/节律测量失败被"常规随访"掩盖。
- **修法**：`_generate_recommendations` 增加两个判据：(a) 计划中从未出现 `classify*` 步骤 → 输出"未执行诊断分类步骤，不能据此排除异常"；(b) 存在失败/`insufficient` 的测量步骤（`n_failed_steps>0`）且无任何阳性诊断 → 输出"部分测量失败（心率/节律/间期未测得），建议重试或人工复核"，而非"常规随访"。即"排除异常"结论必须显式建立在"分类器已运行且测量全部成功"之上。

### 🟠-2 旧 `classify_arrhythmia` 契约下所有诊断被无条件标记 `positive=True`（0.3 固定阈值当"高置信度"）

- **位置**：`src/agent/core/agent.py:435-452`（`"positive": True` 硬编码于 451 行）、`src/agent/tools/diagnosis_tools.py:43`（`decode(probs, threshold=0.3)` 固定阈值）
- **证据**：旧契约分支把每个存活条目 `positive` 恒置 `True`，`_generate_recommendations`（`agent.py:530-534`）据此输出"高置信度发现"。实测 prob=0.31 的 AF 被输出为"高置信度发现: AF — 建议临床对照确认"（CASE6）。而新契约的"阳性"由逐类 Youden 阈值（0.06~0.66）判定，0.31 对多数类低于 Youden 阈值，会被判阴性——同一条目在两契约下结论相反。注释（`agent.py:448-451`）已承认"Agent 无阈值表可取"，但未降级该分支的"高置信度"措辞。
- **修法**：旧契约分支不写死 `positive=True`；要么把逐类 Youden 阈值传入旧工具，要么将旧契约条目标记为 `positive=None` 并在推荐层改用"置信度 X%，请结合逐类阈值确认"，避免固定 0.3 被当作临床高置信度。当前该旧分类器仅被 DEPRECATED 的 `scripts/run_agent_demo.py` 接线（且模型为随机权重），故实际影响有限，但反模式仍在。

### 🟡-1 `ConversationManager._handle_info_collection` 从不收集 `history` 字段

- **位置**：`src/agent/orchestration/conversation.py:77-91`（`_info_prompt` 询问 "relevant medical history"）vs `conversation.py:137-172`（解析分支仅 age/sex/symptoms，无 history）
- **证据**：`_info_prompt` 持续把 `relevant medical history` 列入 missing 直到凑满另外 2 项，用户提供的病史被静默丢弃，`patient_info["history"]` 恒缺失。而 `agent.py:212-217` 的 `_collect_patient_info` 交互路径有 history 字段，两条信息收集路径不一致。
- **修法**：在 `_handle_info_collection` 增加 history 关键词分支（如 `history|既往|病史|stent|hypertension|diabetes`），或从 prompt 与解析两处统一收敛。

### 🟡-2 年龄正则 `old` 子串误匹配（told/hold/cold/gold 等 + 数字 → 伪年龄）

- **位置**：`src/agent/orchestration/conversation.py:146-151`
- **证据**：正则 `(?:age|aged|岁|old)[^\d]{0,8}(\d{1,3})` 中 `old` 命中 "told"/"hold"/"cold"/"gold"/"bold"/"sold"/"fold" 等词后随 1-3 位数字即解析为年龄，且 0<age<120 的范围校验放行。实测：`he told 3 doctors about chest pain` → `age='3'`；`please hold 2 tablets daily` → `age='2'`。该伪年龄流入 `patient_info` 与 planner 提示词。
- **修法**：`old` 用词边界 `\bold\b`，或从备选集中移除 `old`（保留 `year/yr/岁/age` 已覆盖常见表述）；对解析出的 0-2 岁等临床不合理的婴幼儿值二次拒绝。

### 🟡-3 症状关键词 `dizzy` 漏配 "dizziness"；且 `symptoms` 存整段用户输入

- **位置**：`src/agent/orchestration/conversation.py:160`（关键词表）、`conversation.py:161`（`self._collected_info["symptoms"] = text`）
- **证据**：`"dizzy" in "dizziness"` 为假，实测 `the patient reports dizziness ...` → `symptoms=None`。另实测 `female, 30 years old, has chest pain` → `symptoms='female, 30 years old, has chest pain'`（整段原文而非症状短语），把年龄/性别噪声混入症状字段并流入 planner 上下文。
- **修法**：关键词补 `dizziness`（或统一用词干）；`symptoms` 只存命中的症状子串，而非整段输入。

### 🟡-4 `ECGFounderClassifier` 从未注册进 `ToolRegistry`（`tool_schema()` 与 agent 新契约综合分支均未接线）

- **位置**：`src/agent/tools/ecgfounder_classifier.py:156-168`（`tool_schema()`）、`src/agent/core/agent.py:454-481`（新契约 `top_k`/`positives` 综合分支）
- **证据**：全仓 grep 显示 `ECGFounderClassifier` 仅在 `web_demo/app.py`、`scripts/eval_agent_llm.py`、`eval_agent_context.py`、`recompute_agent_metrics.py`、`smoke_ecgfounder_tool.py` 中被直接实例化；`register_diagnosis_tools`（`diagnosis_tools.py:96`）仍只注册旧 `classify_arrhythmia`，无任何代码把 `classifier.predict`/`tool_schema()` 挂进 `ToolRegistry`。故 `agent.py:454-481` 为"为未来分类器预留"的双契约分支，当前在 `ECGAIAgent` 实际可达路径中为死代码，存在契约漂移风险。
- **修法**：若 `ECGAIAgent` 框架不再维护，可在文件头标注 DEPRECATED（同 `pipeline.py`/`scheduler.py` 先例）以明示接线现状；若需保留双契约，应提供 `register_ecgfounder_tool(registry, classifier)` 接线函数并配一条冒烟测试。

### 🟡-5 反射器规则层的 `revise:` 反馈触发 LLM 重规划，但重规划无法真正重测（无效重规划、浪费 LLM 预算）

- **位置**：`src/agent/core/agent.py:189-190`（`"revise" in feedback.lower()` 即 `_replan`）、`agent.py:349-353`（`done_by_action` 按 action 名回填已完成结果）、`src/agent/core/reflector.py:80-110`（生理范围/跨工具心率一致性检查均返回 `revise:`）
- **证据**：心率超生理范围、SDNN>300、QT 超范围、HRV 与 R 峰心率差>20 均为**测量**问题，但统一走 `revise` → `_replan`。`_replan` 会按 action 名回填已完成步骤并置 `completed=True`，因此同名步骤不会重测；新计划里新增的同名步骤也被直接回填旧结果。结果是这类"revise"既不能修正测量，又消耗一次 LLM 重规划调用。
- **修法**：区分"计划性 revise"（缺步骤/多步骤）与"测量不一致告警"；后者只记录告警、不进 `_replan`，或仅在计划真正缺失关键工具时触发重规划。

### 🟡-6 `RubricCache` 无任何调用点接线（判官无缓存 → 跨运行不可复现 + 重复计费）

- **位置**：`src/agent/llm/rubric_judge.py:98-128`（`RubricCache` 类定义）、`scripts/eval_agent_llm.py:490`（`rj = RubricJudge(self.llm)` 未传 cache）
- **证据**：`RubricJudge.score` 仅在 `self.cache is not None` 时读写缓存，而主评测以 `RubricJudge(self.llm)` 构造，`cache` 恒为 `None`。故 `RubricCache`（含其线程锁）为死代码，judge 每次运行都重新调用真实 LLM：与主评测的 `ResponseCache`（planner/reflector/report 均已缓存）不一致，judge 评分在 temperature 0.3 下跨运行不可复现且重复计费。
- **修法**：评测构造 `RubricJudge(self.llm, cache)` 并复用主 `ResponseCache` 或独立 `RubricCache`；同时确认"并行评测"若是多进程，`RubricCache` 的 `threading.Lock` 不保护跨进程追加，需改用文件锁或分进程分文件。

### 💡-1 `prompt_templates.py` 的 PLANNING/REASONING/REFLECTION 模板与 planner/reasoner/reflector 内联提示词持续漂移

- **位置**：`src/agent/llm/prompt_templates.py:3-9`（头部已自注）、`prompt_templates.py:21-71` vs `planner.py:89-102/116-122`、`reasoner.py:58-77`、`reflector.py:126-141`
- **证据**：三个内联提示词与模板结构、字段名（如 `available_tools` vs `tools`、`ecg_shape` 等）均不同。文件头已承认"SA4 🟡"漂移并声明"未列入本次修复范围"。此为非新发现，仅提示后续任一侧改动需同步。
- **修法**：抽单一模板源（模板带 `{...}` 占位，planner/reasoner/reflector 统一引用），避免双份漂移。

### 💡-2 `ECGAIAgent.format_report` 为占位实现，与 `AgentPipeline._format_report` 不一致

- **位置**：`src/agent/core/agent.py:539-541` vs `src/agent/orchestration/pipeline.py:73-86`
- **证据**：`format_report` 仅返回 `str(result.diagnosis)`（诊断列表的 repr），而 `pipeline._format_report` 产出含心率/节律/QTc/建议的 Markdown。`conversation._advance_state` 的 FINALIZING 分支调用 `format_report`，最终"生成报告"只打印诊断列表文本。
- **修法**：统一报告格式化入口（如把 `pipeline._format_report` 上移到 agent 或抽公共函数），或明确 `format_report` 的降级语义。

### 💡-3 会话结果摘要不展示 HRV 心率（`mean_hr` 键未被读取）

- **位置**：`src/agent/orchestration/conversation.py:238-253`（`_summarize_results`）
- **证据**：摘要仅在 `"heart_rate" in result` 时打印心率；`compute_hrv` 的成功结果键为 `mean_hr`（`hrv_analyzer.py:76`），故 HRV 步的心率从不进入摘要。非正确性问题，仅展示遗漏。
- **修法**：读取 `result.get("heart_rate", result.get("mean_hr"))` 或在 HRV 分支单独展示 `mean_hr`。

### 💡-4 `_mock_response` 判官分支关键词大小写不一致

- **位置**：`src/agent/llm/llm_interface.py:235`
- **证据**：`"correctness"/"grounding"` 为大小写敏感匹配，`"rubric"` 走 `.lower()`；若调用方 prompt 写 `Correctness`/`Grounding` 则不命中判官分支而落到通用占位文本。当前 `JUDGE_PROMPT` 恒为小写，故不影响现有路径。
- **修法**：统一 `.lower()` 后再匹配三词。

## 阳性结论（实测/逐行验证正确的性质）

1. **Mock 显式化与失败上抛**（`llm_interface.py:100,187-193,264-267`）：`mock=True` 才返回占位回复且 `mock_mode=True`；无 key 且非 mock 时 `chat()` 抛 `RuntimeError`、`is_available=False`。实测 `mock_mode=True, is_available=False`。
2. **Mock 按语义分派**（`llm_interface.py:220-262`）：planner→计划 JSON、reflector→`continue`、judge→三维评分 JSON、reasoner→占位文本。实测四路分派全部正确，且 reflect 分支先于 planner 分支（避免 "计划" 误抢）。
3. **R3 失败显式化贯穿特征工具链**：`extract_r_peaks`（`r_peak_detector.py:85-92,139-141`）、HRV（`hrv_analyzer.py:61-65,201-211`）、QT（`qt_analyzer.py:101-102,198-206`）失败均返回 `insufficient`/`None`/`error`，不伪造测量值；QT 固定窗口常量法 `_simple_delineate` 已停用并抛 `RuntimeError`（`qt_analyzer.py:191-195`）。
4. **分类步骤失败显式化**：实测 `classify_arrhythmia` 失败时 `recommendations` 输出"诊断分类步骤未完成……不应作为'排除异常'的依据"（CASE2），不落入"常规随访"。
5. **跨患者污染防护**：`agent.diagnose` 入口 `self.memory.clear()`（`agent.py:143`）；`conversation.start` 重置状态并 `agent.memory.clear()`（`conversation.py:102-106`）。
6. **心率单一权威口径**：`_synthesize_diagnosis` 仅 `extract_r_peaks` 心率优先，QT 心率仅作回退（`agent.py:406-415`），与注释所述"最后写入者胜"缺陷已修复。
7. **`_parse_plan` 全面容错 + 默认计划回退**（`agent.py:251-297`）：实测空输入 → 4 步默认计划 `[extract_r_peaks, compute_hrv, measure_qt_interval, classify_arrhythmia]`；非 dict 元素/params 非 dict 均跳过不崩溃。
8. **planner/reflector 异常不炸穿主流程**：`generate_plan`/`revise_plan` 均有 try/except 回退（`agent.py:244-248,340-343`）；反射器异常降级为继续执行（`agent.py:176-182`）。
9. **反射器词边界匹配**：`\brevise\b`/`\bcomplete\b`（`reflector.py:146-150`），避免 "incomplete" 命中 "complete" 的提前停止。
10. **分类器逐类 Youden 阈值与坏输入防护**：`positives` 对全 27 类按 `probs[i] >= thresholds[i]` 判定（`ecgfounder_classifier.py:140-148`），NaN/Inf/形状非法显式失败（`ecgfounder_classifier.py:108-116`）。
11. **RubricJudge 故障不伪装为差评**：非对象 JSON/缺必需字段/解析异常均返回 `None` 并跳过（`rubric_judge.py:71-91`），不补 1 分；数值 `round` 截断到 1-5。
12. **记忆层契约一致**：`get_context_for_tool` 正确注入 `ecg_signal`（classify/measure_qt 等）、`r_peaks`/`rr_intervals`（hrv/qt）、`sex`（qt），实测 `measure_qt_interval` 上下文含 `sex='Female'`、`classify_arrhythmia` 上下文含 `ecg_signal`（见验证输出）；`add_observation` 仅缓存不含 `error` 的成功结果（`memory.py:47-52`）。
13. **反射器跨工具心率一致性读取 `mean_hr`**（`reflector.py:96-100`），与 `hrv_analyzer.py:76` 返回契约一致，不再是死代码。
14. **scheduler 环检测/依赖失败传播/缓存**（`scheduler.py:125-130,77-88`）已修复（虽 DEPRECATED 未接入）。
