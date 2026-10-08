# 第四次独立代码审查 — 4A_Agent核心与LLM

> 审查人：独立审查员 A（第四次；未读取任何历史审查产物）
> 范围：src/agent/core/{agent,memory,planner,reasoner,reflector}.py、src/agent/llm/{llm_interface,prompt_templates,rubric_judge}.py、src/agent/orchestration/{conversation,pipeline}.py、src/agent/tools/{diagnosis_tools,ecg_tools,ecgfounder_classifier,registry,scheduler}.py、src/agent 下各 __init__.py
> 验证方式：read 逐行通读全部 16 个文件 + 3 个被依赖契约文件（r_peak_detector.py / hrv_analyzer.py / qt_analyzer.py / label_extractor.py 用于验证工具返回契约）+ 主评测脚本 eval_agent_llm.py（仅用于核对分类器接线，非本范围）；`python -m py_compile` 全量通过；两段只读小验证（mock 判官、反射器一致性）实测复现。未调用任何真实 LLM API（mock=True 只走占位分支）。

## 统计（🔴 2 / 🟠 3 / 🟡 7 / 💡 2）

## 发现清单

### 🔴-1 ECGFounder 升级分类器与 Agent 合成层的输入/输出契约双向不兼容
- **位置**：ecgfounder_classifier.py:91-149（predict 返回 `probs/top_k/positives`，条目键 `name/snomed/prob/positive`）；ecgfounder_classifier.py:151-163（`tool_schema()` 名 `classify_arrhythmia`，参数名 `signal`）；agent.py:419-432（`_synthesize_diagnosis` 只读 `result["diagnoses"]`，条目键 `probability/snomed_code/evidence`）；memory.py:73-75（对 `classify_arrhythmia` 注入 `ecg_signal`）
- **证据**：
  1. `ECGFounderClassifier.predict()` 输出 `{probs, top_k, positives, summary}`，其中条目为 `{"name","snomed","prob","positive"}`，**没有** `diagnoses` 键、没有 `probability/snomed_code/evidence` 键。
  2. `agent.py` 的 `_synthesize_diagnosis` 只处理 `if "diagnoses" in result`（419 行），逐条读 `d.get("probability")`/`d.get("snomed_code")`/`d.get("evidence")`。
  3. 参数名也不一致：`memory.get_context_for_tool("classify_arrhythmia")` 注入的是 `ecg_signal`（memory.py:73-75），而 `ECGFounderClassifier.predict(signal, ...)` 收 `signal`，`tool_schema()` 声明的也是 `signal`（ecgfounder_classifier.py:159）。
  4. 与旧 `diagnosis_tools.classify_arrhythmia` 对照：后者恰好返回 `diagnoses[{name,snomed_code,probability,evidence}]`（diagnosis_tools.py:44-46），是唯一与 `_synthesize_diagnosis` 匹配的契约。
- **后果**：按文档把升级版 ECGFounder 分类器接入 `ToolRegistry` 后，Agent 的核心合成步骤将**提取到 0 条诊断**，并把"分类器其实已成功运行"误判为"诊断分类步骤未完成"（触发 agent.py:474 的失败建议）——即"核心创新模块"与其旗舰发现层无法对接。当前评测脚本 `eval_agent_llm.py` 直接调用 `classifier.predict(signal)`（eval_agent_llm.py:221），**绕过了 Agent 的 registry/synthesize 路径**，故该断裂在现评测中未被触发，属潜伏缺陷。
- **修法**：在 registry 与 `_synthesize_diagnosis` 之间加一层适配（或统一契约）：把 `top_k/positives`（`prob/snomed`）映射为 `diagnoses`（`probability/snomed_code/evidence`）；并把工具参数统一为 `signal` 或 `ecg_signal`（`memory` 注入与 `predict` 签名一致）。

### 🔴-2 主评测 harness 完全绕过 Agent 核心框架，"核心创新模块"未驱动任何评测指标
- **位置**：pipeline.py:1（`DEPRECATED...未接入主流程`）；scheduler.py:1（`DEPRECATED...未接入主流程`）；reasoner.py:3-5（`reason() 未接入主流程`）；eval_agent_llm.py:43-48、183-224、312-403（对照）
- **证据**：
  1. 主评测脚本 `eval_agent_llm.py` 的 import 只包含 `llm_interface`、`prompt_templates.SYSTEM_PROMPT`、`ecgfounder_classifier.ECGFounderClassifier` 与三个特征提取器（43-48 行），**从不 import** `src.agent.core.agent` / `planner` / `memory` / `reflector` / `reasoner`，也不 import `src.agent.tools.registry` / `diagnosis_tools` / `ecg_tools` / `scheduler`。
  2. 该脚本内联重写了工具执行（`run_tools` 183-224）、规则反射（`rule_reflection` 227-235，只查 HR 20-300 与 QTc 200-700）、LLM 复核（345-359）、规划（312-333）、报告（361-403）——与 `ECGAIAgent.diagnose` 的 ReAct 循环是两套并行实现。
  3. 范围文件自述：pipeline.py/scheduler.py 均标注未接入主流程；reasoner 的 `reason()` 未接入。
- **后果**：论文若以"ReAct Agent（规划-记忆-反思-修订）"作为被评测系统，则五维指标实际测得的是"确定性工具 + 三段 LLM prompt"的简化线性管线；`Planner/Memory/Reflector/MedicalReasoner/ToolRegistry/ToolScheduler/ConversationManager` 全部是未参与评测的死代码，且两套实现存在行为漂移（见 🟠-2）。这直接影响"论文结论可靠性"与"核心创新模块"这一主张。
- **修法**：让评测走 `ECGAIAgent.diagnose`（或其工具层），或在论文/文档中明确声明评测对象是"确定性工具 + LLM 报告管线"而非 ReAct Agent，避免主张与实测对象不一致。

### 🟠-1 建议层硬编码 `confidence > 0.7`，与发现层逐类 Youden 阈值（0.06~0.66）脱节，阳性发现会被弱化为"常规随访"
- **位置**：agent.py:477-483（`_generate_recommendations` 中 `if d["confidence"] > 0.7`，否则回退 `未检测到高置信度异常。建议常规随访。`）；ecgfounder_classifier.py:78-84（逐类 Youden 阈值，注释自述区间 0.06~0.66）
- **证据**：发现层的"阳性"由逐类 Youden 阈值判定（阈值可低至 0.06），但建议层用固定 0.7 作为"高置信度"门槛。对任意 prob ∈ (Youden, 0.7) 的阳性发现，`diagnosis` 列表会列出该发现，而 `recommendations` 却输出"未检测到高置信度异常。建议常规随访。"——发现层与建议层自相矛盾，且把真实阳性包装成"常规随访"。
- **后果**：医疗安全反模式（与代码注释自述的"禁止把失败/阳性包装成常规随访"精神相悖）。当前评测绕过该路径（报告由 LLM 生成），属 Agent 路径潜伏缺陷。
- **修法**：建议层阈值与发现层逐类 Youden 阈值对齐（例如按 `d["confidence"] >= 该类阈值` 判"阳性发现"），或直接删除固定 0.7，按"是否存在任何阳性"给出建议。

### 🟠-2 反射器"跨工具心率一致性"检查是死代码（HRV 返回 `mean_hr` 而非 `heart_rate`）
- **位置**：reflector.py:92-102（check4 读 `result.get("heart_rate")` 且仅当 `step.action == "compute_hrv"`）；hrv_analyzer.py:75-79（HRV 成功结果键为 `mean_hr/sdnn/rmssd`，无 `heart_rate`）
- **证据**：`compute_hrv` 的返回字典没有 `heart_rate` 键（只有 `mean_hr`），故 `reflector._quick_check` 第 78 行 `hr = self._num(result.get("heart_rate"))` 恒为 None，第 93 行 `if step.action == "compute_hrv" and hr is not None` 恒不成立。实测：对 `compute_hrv` 结果 `{mean_hr:80, sdnn:50}` 且前一步 `heart_rate=180`（明显矛盾），`Reflector.check` 返回 `(True, "")`，**未产生任何 revise 反馈**。文档（reflector.py:9、eval_agent_llm.py:8-9）声称的"±20 bpm 跨工具一致性"实际不存在。
- **后果**：反射器宣称的一致性验证失效；诊断中 R 峰心率与 HRV 心率矛盾时不会被拦截。eval harness 的 `rule_reflection` 也只剩 HR/QTc 范围检查（无一致性），两处一致地缺失该检查。
- **修法**：check4 改读 `result.get("mean_hr")`（并对 None 防护），或将 HRV 输出补一个与 R 峰检测同口径的 `heart_rate` 字段。

### 🟠-3 RubricJudge 对"合法 JSON 但字段缺失"静默补 1，mock 下恒全 1（实测）
- **位置**：rubric_judge.py:73-78（仅非 dict → None；缺失字段 `scores.get(field, 1)` 默认 1）；rubric_judge.py:85-87（非 None 即写缓存并返回）
- **证据**：判官解析只对"非对象 JSON"返回 None（73-75 行）；对**形状正确但字段缺失/被篡改的 dict** 会走 77-78 行 `scores.get(field, 1)` 把缺失维度补成 1，`total` 也被独立补 1 而非由三维导出。实测（mock=True，不调 API）：`RubricJudge(llm).score(['Atrial Fibrillation'], '报告: 心率 80 bpm')` 返回 `{'plan': [...], 'correctness': 1, 'completeness': 1, 'grounding': 1, 'total': 1}`——mock 回复的计划 JSON 是合法 dict，被判官当成"有效评分"并全维度补 1，`judge_failures` 不会被计数。
- **后果**：与代码自述"判官故障返回 None 由调用方跳过、旧版全给 1 分把 LLM 故障伪装成报告质量极差"相矛盾——部分/畸形判官输出仍会被静默记成 1 分（且污染 `total_mean`），而不是计入 `judge_failures`。mock 模式 `--judge` 会得到无意义的全 1 均值。
- **修法**：对 dict 逐字段校验存在性与数值性，任一必需字段缺失/非法即返回 None（计 judge_failures）；`total` 应由三维聚合而非默认 1；mock 回复为判官单独返回评分 JSON。

### 🟡-1 `_mock_response` 以 "ECG"/"plan" 子串为开关，对所有含 "ECG" 的 prompt 一律返回计划 JSON
- **位置**：llm_interface.py:207-219
- **证据**：`if "ECG" in last_msg or "plan" in last_msg.lower():` 返回 `{"plan": [...]}`。但本项目 planner/reasoner/reflector/rubric_judge 的 prompt 全部包含 "ECG" 字样（reasoner 用 "ECG Findings"、reflector 用 "审查 ECG 分析"、judge 用 "ECG diagnostic reports"）。实测 mock 下 judge 收到计划 JSON（见 🟠-3）。除 planner 外，其余调用方在 mock 模式收到错误类型的回复。
- **后果**：mock 模式仅对规划路径可用；对 reasoner/reflector/judge 的 mock 测试会得到错误类型结果（judge 表现为全 1、reflector 无法产生 revise/complete）。测试口径失真。
- **修法**：`_mock_response` 按调用语义分派（识别 `response_format`/提示词结构或显式传入 kind），为 judge 返回 `{"correctness":..,"completeness":..,"grounding":..,"total":..}`、为 reflector 返回 "continue"、为 planner 返回计划 JSON。

### 🟡-2 `generate_report` 移除不一致（默认计划已删，mock/注册表/reflector 仍引用；格式化无类型防护）
- **位置**：agent.py:285-287（默认计划注释"移除 generate_report"）；llm_interface.py:216（mock 计划仍含 `generate_report`）；diagnosis_tools.py:112-117（`register_diagnosis_tools` 仍注册）；reflector.py:109（`_is_critical` 仍列 `generate_report`）；diagnosis_tools.py:91（`top.get('probability', 0):.1%` 无类型防护）
- **证据**：四处对 `generate_report` 的态度不一致。且 `generate_report` 本身（diagnosis_tools.py:77-93）`summary` 恒含"心率: N/A bpm"，其 `probability` 若非数值（None/str）会抛格式化异常。
- **后果**：mock 模式规划的 `generate_report` 会执行一个"恒 N/A"且被合成逻辑忽略的无效步骤；反射器仍视其为需要 LLM 深验的关键步骤（浪费一次 LLM 调用）。纯一致性问题，无医疗安全影响。
- **修法**：统一移除（mock 计划、注册表、`_is_critical` 同步），或给 `generate_report` 的 `probability` 加类型防护并让它返回真实测量。

### 🟡-3 双分类器并存：旧自训集成（固定阈值 0.3）与 ECGFounder（逐类 Youden）两套实现、两套阈值、两套输出契约
- **位置**：diagnosis_tools.py:17-50（旧 `classify_arrhythmia`，`threshold=0.3`，返回 `diagnoses`）；ecgfounder_classifier.py:78-143（`ECGFounderClassifier`，逐类 Youden，返回 `top_k/positives`）
- **证据**：`diagnosis_tools.classify_arrhythmia` 用单一固定阈值 0.3 + `label_extractor.decode(probs, threshold)`（diagnosis_tools.py:43），而 `ECGFounderClassifier` 用 `probs[i] >= self.thresholds[i]` 的逐类阈值（ecgfounder_classifier.py:141-142）。前者对 Youden 阈值 < 0.3 的类别会系统性漏报；后者是评测实际使用的实现（eval_agent_llm.py:106/221）。两者输出契约互不兼容（见 🔴-1）。
- **后果**：同一"classify_arrhythmia"工具名对应两套语义/阈值/输出，任何误接线都会得到错误诊断；维护者易混淆"27 类官方评分类"到底走哪条路径。
- **修法**：删除/标记 `diagnosis_tools.classify_arrhythmia` 为 legacy（或改为 ECGFounder 的薄封装），明确唯一实现，消除双阈值/双契约。

### 🟡-4 `_synthesize_diagnosis` 心率/节律"最后写入者胜"，QT 工具的 heart_rate 覆盖 R 峰检测值
- **位置**：agent.py:402-407（遍历步骤时 `hr`/`rhythm` 被后续步骤覆盖）；qt_analyzer.py:131（QT 结果也含 `heart_rate`）；agent.py:288-293（默认计划中 `measure_qt_interval` 排在 `extract_r_peaks` 之后）
- **证据**：默认计划顺序 extract_r_peaks → compute_hrv → measure_qt_interval → classify_arrhythmia。`measure_qt_interval` 的返回含 `heart_rate`（QT 分析器由 `np.median(np.diff(r_peaks))` 反推，qt_analyzer.py:105-106），在合成循环中会覆盖 `extract_r_peaks` 的 `heart_rate`（agent.py:403）。最终 `DiagnosisResult.heart_rate` 是 QT 工具的口径，而非 R 峰检测口径。
- **后果**：两个口径数值接近但不完全相同，无显式优先级，最终心率依赖步骤顺序，属隐蔽的一致性隐患。
- **修法**：为 `heart_rate` 指定唯一权威来源（如只取 `extract_r_peaks` 的），或按工具名显式决定覆盖优先级。

### 🟡-5 `intervals` 提取仅判 `qt_ms is None`，未像心率那样同时排除 0（守卫不对称）
- **位置**：agent.py:410（`if result.get("qt_ms") is not None:`）；对照 agent.py:403（心率用 `hr_v not in (None, 0)`）
- **证据**：注释（agent.py:401、410）声称"0/None 视为测量失败"，但间期分支只排除了 None、未排除 0；心率分支则 `(None, 0)` 都排除。若上游异常返回 `qt_ms=0`，会被写入 `intervals`。当前 QT 分析器失败路径返回 None（qt_analyzer.py:199-207），故暂不可触发，属守卫不对称。
- **修法**：与心率对齐，改为 `if result.get("qt_ms") not in (None, 0):` 并同步注释。

### 🟡-6 RubricCache JSONL 追加无锁；`aggregate()` 对缺字段 dict 会 KeyError
- **位置**：rubric_judge.py:111-115（`open(...,"a")` 逐行写，无跨进程锁）；rubric_judge.py:129-134（`aggregate` 中 `s[field]` 直接下标）
- **证据**：`RubricCache.put` 与主评测 `ResponseCache.put` 都是 `open(path,"a")` 追加；`aggregate` 的 `valid = [s for s in scores_list if isinstance(s, dict)]` 只查"是 dict"，不查必需字段存在，`s[field]` 对缺字段 dict 抛 KeyError。eval_agent_llm.py:337-339 自述并行运行时各进程缓存互不可见（且无锁追加）。
- **后果**：并行评测时 JSONL 可能行级交错/污染；`aggregate` 遇缺字段评分 dict 会崩溃而非计入失败。
- **修法**：缓存写入加锁或每进程独立文件；`aggregate` 对字段做 `s.get(field)` 容错并计入失败。

### 🟡-7 ConversationManager 的 REFLECTING 状态为无操作过渡；`handle()` 在中间状态忽略用户输入
- **位置**：conversation.py:39-46（状态机）、124-128（`handle` 对 PLAN/EXEC/REFLECT/FINALIZING 直接 `_advance_state()`）、224-229（FINALIZING 分支）
- **证据**：反射实际在 EXECUTING 分支内逐步完成（conversation.py:210-221），`REFLECTING` 状态只是 `handle()` 下一次调用时被 `_advance_state()` 直接跳过到 FINALIZING，无任何校验逻辑；`handle(user_input)` 在这些状态完全忽略传入文本。
- **后果**：状态机语义与实现不一致（REFLECTING 是空壳），多轮交互的中间状态无法真正接受用户输入，属可维护性/设计一致性问题。
- **修法**：要么移除 REFLECTING 状态（EXECUTING → FINALIZING），要么在该状态真正执行独立反思并处理用户输入。

### 💡-1 mock 模式仍强制 `import openai`；`_init_vllm` 不尊重 mock
- **位置**：llm_interface.py:109-115（`_init_openai_compatible` 先 `from openai import OpenAI` 再判 key）、147-158（`_init_vllm` 无视 `mock_mode`）
- **证据**：mock=True 且无 key 时仍先执行 `import openai`（未安装则抛 ImportError）；backend="vllm" 时 `_init_vllm` 直接构造真实客户端，不理会 mock。
- **修法**：mock 模式应跳过第三方 client 构造与 import（尽早 `return`），降低 mock 测试的环境依赖。

### 💡-2 默认回退计划与工具注册清单/描述漂移，建议单一事实源
- **位置**：agent.py:288-293（默认计划 4 步）；diagnosis_tools.py:96-117 与 ecg_tools.py:113-131（注册工具与 dependencies）；llm_interface.py:211-218（mock 计划）；planner.py:23-54（few-shot 示例）
- **证据**：默认计划含 4 步；mock 计划含 `generate_report`（已从默认计划移除）；few-shot 示例含 `detect_anomaly`（默认计划没有）。四处工具清单不一致。
- **修法**：由 `ToolRegistry.list_tools()` 派生唯一工具清单，few-shot/默认计划/mock 计划统一引用，避免手工漂移。

## 阳性结论（实测验证正确的性质）

1. **跨患者污染已防护**：`ECGAIAgent.diagnose` 入口先 `self.memory.clear()`（agent.py:143）；`ConversationManager.start` 重置 `_collected_info/_results/_turn` 并 `memory.clear()`（conversation.py:97-101）。上一名患者的工具结果不会注入下一名患者的计划。
2. **测量失败显式化、不伪造测量值**（逐工具验证契约一致）：
   - R 峰不足/噪声 → `rhythm="insufficient_data"` + `insufficient=True`（r_peak_detector.py:85-92、101-114；ecg_tools.py:39-42）。
   - QT 未测出 → `qt_ms=None` + `error="insufficient_data"` + `qt_interpretation="insufficient_data"`（qt_analyzer.py:199-207）。
   - HRV 数据不足 → 数值字段 None + `insufficient=True`（hrv_analyzer.py:64-65、197-207）。
   - ECGFounder 坏输入（None/非 ndarray/非 12 通道/NaN/Inf）→ 返回 `error` dict，拒绝产出"可信"概率（ecgfounder_classifier.py:103-111）。
3. **综合层失败语义正确**：`_synthesize_diagnosis` 把 `error`/`insufficient`/`insufficient_data` 结果计入失败步骤（agent.py:391-398）；`analysis_failed`/`classification_failed` 的兜底与"全失败→分析未完成""分类失败→禁止排除异常"的失败文案正确（agent.py:440-475）。
4. **LLM 故障不再静默降级为占位文本**：mock 模式外 API key 缺失/异常一律显式不可用或向上抛（llm_interface.py:133-145、174-180），`is_available` 属性正确区分 mock 与真实可用（llm_interface.py:221-224）。实测 mock=True 时 `chat` 只走占位分支、不碰网络。
5. **无标签泄漏**：planner（planner.py:86-99）、reasoner（reasoner.py:58-77）、reflector（reflector.py:116-131）的 prompt 只含患者信息 + 工具输出 + 可用工具清单，均不含 ground-truth 标签；评测发现层指标用注入前快照 `orig_top5_names`（eval_agent_llm.py:259-260），注入不污染发现层。
6. **反射器 "complete" 词边界匹配修复正确**：`re.search(r"\bcomplete\b", rl)`（reflector.py:139）不再被 "incomplete" 子串误命中（旧版子串匹配会误判提前停止）。
7. **计划解析全面容错**：`_parse_plan` 对非 JSON/None/非 list/数组内含非 dict/params 非 dict 均不崩溃并逐条跳过，空计划回退默认（agent.py:257-294）。
8. **旧 classify 工具的 decode 顺序正确**：`LabelExtractor.decode` 返回 `[(snomed, name, prob)...]` 且按概率降序（label_extractor.py:160-180），与 `diagnosis_tools.classify_arrhythmia` 的 `for snomed, name, prob in decoded[:top_k]` 解包顺序一致（diagnosis_tools.py:44-46）。
9. **ECGFounder 阳性判定覆盖全 27 类**：`positives` 遍历 `range(len(probs))` 按逐类 Youden 阈值判定（ecgfounder_classifier.py:135-143），不再只从 top_k 子集派生，避免"概率≥阈值但排名第 6 名后"的漏报。
10. **py_compile 全量通过**：`python -m py_compile` 对范围内 16 个文件均 exit=0，无语法错误。
