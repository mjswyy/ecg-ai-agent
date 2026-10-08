# 第二轮全面代码审查 — 2D：Agent 框架（src/agent/ 全量 + scripts 验证脚本）

> 审查方式：逐文件完整阅读（禁止抽样），全部结论来自亲自阅读代码与运行验证。
> 禁止阅读第一轮审查产物（`docs/审查/` 整个目录、`检查报告.md`），本报告未参考其中任何内容。
> 禁止修改任何源码/数据文件；LLM 相关验证全部使用 mock 参数，未调用任何付费 API。

## 统计

| 严重度 | 数量 |
|---|---|
| 🔴 影响正确性/安全/医疗输出可信度 | **3** |
| 🟠 潜在缺陷（特定输入/边界触发） | **9** |
| 🟡 可维护性/风格 | **16** |
| 💡 建议 | **4** |
| 合计 | **32** |

其中**确认缺陷（已运行验证）** 13 条：R1/R2/R3、O1/O2/O3/O4/O5/O6/O8、Y1/Y4/Y5；
**疑似问题（读码确认但未构造运行复现）** 19 条，详见各条标注。

---

## 一、运行验证方法（只读，未改任何源码）

验证脚本写入系统临时目录 `%TEMP%\review2_t*.py`，运行后即弃，未触碰仓库内任何文件。

| 测试 | 内容 | 结果 |
|---|---|---|
| T1 | `AgentStep` 是否可下标访问 `step["error"]` | TypeError：`'AgentStep' object is not subscriptable` |
| T1b | `_collect_patient_info` 写入 memory 的字段名 | keys = `['患者的年龄是多少？', '患者的性别是？', '患者有什么症状？', '有什么相关的病史？']` |
| T2 | `conversation.py` 性别解析 "female" | `sex=Male`（错误） |
| T3 | `ecg_tools.py` 节律分类 hr=0.0 | `rhythm='bradycardia (regular)'`（错误） |
| T4 | `reflector.py` 判定子串匹配 | `'The analysis is incomplete.'` → 触发 STOP（错误） |
| A | `_plan` 在 LLM 抛异常时的回退 | `RuntimeError` 直接上抛，docstring 承诺的回退不存在 |
| B | QT 零值 dict 是否进 `_tool_results` 缓存 | 进入（无 "error" 键，绕过缓存守卫） |
| C | `_synthesize_diagnosis` 对 QT 零值 dict | `intervals={'qt_ms':0,'qtc_bazett':0,...}`，recommendations=常规随访 |
| D | mock LLM 端到端 `agent.diagnose`（有正常心率） | 正常完成，但 classify 未加载模型 → confidence=0.0、建议"常规随访" |
| E2E | 端到端：0 个 R 峰（无真实 QRS）的合成信号 | 最终 `heart_rate=0.0, rhythm='bradycardia (regular)'`，diagnosis=[]，recommendations=「未检测到高置信度异常。建议常规随访。」 |
| F1-F4 | `_parse_plan` 畸形输入 | 全部降级不崩溃（设计良好） |
| G | `generate_report` 上下文注入 | 仅注入 `patient_info`，无 features/diagnoses |
| H | `ToolScheduler` 缓存 key 含 numpy 数组 | 不崩溃，但 key 为巨型字符串（性能） |
| I | conversation EXECUTING 中 replan 重绑 `plan` | 旧列表迭代继续，新计划新增步骤永不执行 |
| E | `Planner._extract_json` 边界 | 无代码块时前后夹散文 → 解析失败 → 默认计划 |
| 冒烟 | `python scripts/verify_llm.py --mock` | Reasoner 假阳性通过（输出实为 plan JSON）；Reflector=unknown |

---

## 二、🔴 严重缺陷（影响正确性/安全/医疗输出可信度）

### R1. R 峰检测失败（<2 个峰）时，心率 0 被标记为 "bradycardia (regular)" 并作为真实节律呈现
- **文件:行号**：`src/agent/tools/ecg_tools.py:38-46`（根因），`src/agent/core/agent.py:361-362`（合成路径）
- **确认缺陷（已验证）**：T3 + E2E 全链路复现。
- **证据**：
  ```python
  # ecg_tools.py:38-46
  hr = result.get("heart_rate", 0)
  hr_std = result.get("hr_std", 0)
  if hr < 60:
      rhythm = "bradycardia"
  elif hr > 100:
      rhythm = "tachycardia"
  ...
  rhythm += " (irregular)" if hr_std > 15 else " (regular)"
  ```
  底层 `RPeakDetector.detect`（`r_peak_detector.py:81-87`）在 R 峰不足 2 个时返回 `heart_rate=0.0`。E2E 实测：对无 QRS 的合成信号（真实场景对应"伪差/停搏/导联脱落"），最终 `DiagnosisResult.rhythm == 'bradycardia (regular)'`、`heart_rate=0.0`。窦性心动过缓（40-59 bpm）与"完全检不出心跳"在临床上是天壤之别，此标签会误导"节律"结论；且 `hr=0` 时 `agent.py:361` 直接透传到输出。
- **建议**：`extract_r_peaks` 内对 `hr <= 0` 或 `num_beats < 2` 显式返回 `rhythm="undetectable"` / `heart_rate=None` 并带 `error` 或 `insufficient_data` 标记；`agent.py:359-362` 对 None 值不覆盖默认。

### R2. 全部工具失败时，最终建议仍输出"未检测到高置信度异常。建议常规随访。"
- **文件:行号**：`src/agent/core/agent.py:405-415`（`_generate_recommendations`），触发条件 `agent.py:374-387`（合成时静默跳过 error/无 diagnoses 的结果）
- **确认缺陷（已验证）**：测试 D 与 E2E 均复现。
- **证据**：E2E 中 HRV 返回 `{"error": ...}`、分类器返回 `{"error": "未加载分类模型..."}`、R 峰检测 0 峰，但最终：
  ```python
  recommendations = ['未检测到高置信度异常。建议常规随访。']
  ```
  `_generate_recommendations([])` 在 `recs` 为空时无条件附加该句（agent.py:413-414）。用户看到的是"未检测到异常、常规随访"，而实际是**分析完全失败**——这正是"错误输出被当作真实结论呈现"的医疗安全反模式。
- **建议**：诊断前统计已完成且无 error 的步骤数；若关键步骤（R 峰/分类器）全部失败，recommendations 改为"分析失败/数据不可靠，请人工复核"，并禁止输出"未检测到异常"类措辞。

### R3. QT 测量数据不足时返回全零 dict，被记忆缓存并作为真实测量值（QTc=0 ms）呈现
- **文件:行号**：`src/agent/tools/ecg_tools.py:102-103`（无守卫直接调用 QTAnalyzer）、`src/agent/core/memory.py:49-52`（缓存守卫只挡 "error" 键）、`src/agent/core/agent.py:365-371`（合成提取）
- **确认缺陷（已验证）**：测试 B + C。
- **证据**：`QTAnalyzer.analyze` 在 `<2` 个 R 峰时返回 `{"qt_ms": 0, "qtc_bazett": 0, ..., "qt_interpretation": "insufficient_data", ...}`（`qt_analyzer.py:53-54,180-184`）。该 dict **不含 "error" 键**，因此：
  - memory.py:51 的守卫 `if not (isinstance(result, dict) and "error" in result)` 放行 → 零值 dict 进入 `_tool_results` 缓存；
  - `_synthesize_diagnosis` 检出 `qt_ms` 键 → `intervals = {"qt_ms": 0, "qtc_bazett": 0, ...}`（测试 C 实测）。
  最终报告呈现"QTc(Bazett): 0 ms"。而 `qt_interpretation: "insufficient_data"` 字段被合成逻辑完全忽略。同一问题也影响 `web_demo/app.py:218`（QTAnalyzer 直接调用，0 峰时显示 0 ms）——根因均在工具层。
- **建议**：`measure_qt_interval` 在 `r_peaks` 不足 2 个时返回 `{"error": "R 峰不足..."}` 而非零值 dict；或 memory 缓存守卫扩展为"结果中含 `qt_interpretation == 'insufficient_data'` 等失败标记也跳过"；`_synthesize_diagnosis` 对全零/`insufficient_data` 的 intervals 置 None。

---

## 三、🟠 潜在缺陷

### O1. run_agent_demo.py 每次运行必崩：AgentStep 不可下标
- **文件:行号**：`scripts/run_agent_demo.py:123-125`
- **确认缺陷（已验证）**：T1。
- **证据**：`result.reasoning_chain` 是 `AgentStep` dataclass 列表，而代码用 `step["error"]` / `step["step"]` / `step["reason"]` 下标访问 → `TypeError: 'AgentStep' object is not subscriptable`。`diagnose()` 成功后 `format_result` 必崩（mock 模式同样复现）。该脚本虽标注 DEPRECATED，但仍是仓库内唯一调用 `ECGAIAgent` 的演示入口。
- **建议**：改为属性访问 `step.error` / `step.action` / `step.reason`（参考 `pipeline.py:56-57` 的写法）。

### O2. 性别解析："female" 包含子串 "male" → 女性患者被记为 Male
- **文件:行号**：`src/agent/orchestration/conversation.py:141-142`
- **确认缺陷（已验证）**：T2。
- **证据**：
  ```python
  if any(w in text_lower for w in ("male", "female", "男", "女")):
      self._collected_info["sex"] = "Male" if "male" in text_lower or "男" in text else "Female"
  ```
  输入 "I am a 45 year old female" → `"male" in "female"` 为 True → `sex="Male"`。患者性别直接影响 QT 阈值（`qt_analyzer.py:111`：男 450 / 女 460）与 LLM 情境。**注意**：`ConversationManager` 经 grep 确认无任何调用方（死代码），当前不可达；一旦接入即为 🔴。此条与 O5 同属死代码缺陷，仍按任务要求报告。
- **建议**：`"male" in text_lower and "female" not in text_lower`；或改用正则/关键词列表。

### O3. `_plan` 声称"LLM 不可用时回退默认计划"，实际直接抛异常
- **文件:行号**：`src/agent/core/agent.py:237`（调用点），docstring 承诺在 `agent.py:225-226`
- **确认缺陷（已验证）**：测试 A。
- **证据**：docstring 写"如果 LLM 不可用，回退到预设的默认计划"，但 `self.planner.generate_plan(context)` 无任何 try/except。`LLMInterface.chat` 在无 key 且非 mock 时抛 `RuntimeError`（`llm_interface.py:164-168`）→ `diagnose()` 整体崩溃。默认计划回退仅在"LLM 返回了但 JSON 解析失败"时生效（agent.py:272-281）。网络超时/API 错误同样炸穿主流程。
- **建议**：在 `_plan` 捕获 `Exception`，LLM 不可用时 `logger.warning` 并返回 `_parse_plan("")` 走默认计划；或至少让 `diagnose()` 把 LLM 故障映射为带 `error` 的 `DiagnosisResult` 而非裸抛。

### O4. 交互式患者信息收集双缺陷：字段名是问题文本 + 合并时被原始 dict 覆盖丢弃
- **文件:行号**：`src/agent/core/agent.py:208-217`（收集）、`agent.py:228-230`（合并）
- **确认缺陷（已验证）**：T1b。
- **证据**：
  ```python
  # agent.py:215-217
  response = input(f"[Agent] {q}\n> ")
  if response.strip():
      self.memory.update_patient_info(q, response.strip())   # q 是完整问句！
  ```
  实测 memory 中键为 `['患者的年龄是多少？', ...]`，而非 `age/sex/symptoms/history`。且 `_plan` 的合并逻辑 `patient_info or memory.get_context().get("patient_info") or {}`：调用方传入非空 dict（如 `{"sex": "Male"}`，仅缺 age）时走交互收集，但 `_plan` 因原 dict 为真值而**丢弃** memory 中的交互信息。两条路径都让交互收集的数据丢失/错位。
- **建议**：`update_patient_info` 用规范字段名（如 `q.split("？")[0]` 映射或固定四字段）；合并改为 `{**memory_patient_info, **patient_info}`。

### O5. ConversationManager EXECUTING 中 replan 重绑 `plan`，修订计划的新步骤永不执行
- **文件:行号**：`src/agent/orchestration/conversation.py:179-199`
- **确认缺陷（已验证）**：测试 I。
- **证据**：`for step in plan:` 迭代绑定旧列表对象；循环体内 `plan = self.agent._replan(plan, feedback)`（198 行）重绑局部变量后，后续迭代仍走**旧列表**。EXECUTING 是一次性状态（之后进入 REFLECTING/FINALIZING），修订计划中新增的未完成步骤既不在本次循环执行，也无后续轮次 → 被静默丢弃。模拟复现：旧计划 `[a(完成), b]`，replan 后 `[a, c, b]`，本次只执行了 `b`，`c` 永不执行。当前 `ConversationManager` 无调用方（死代码），接入即触发。
- **建议**：先收集 `plan` 的待执行快照，replan 后 `continue` 或改为 while + 索引循环；replan 后从新计划头部重新扫描。

### O6. Reflector 判定用子串匹配："incomplete/not complete" 会误触发停止
- **文件:行号**：`src/agent/core/reflector.py:133-137`
- **确认缺陷（已验证）**：T4。
- **证据**：
  ```python
  if "revise" in response.lower():
      return True, response.strip()
  elif "complete" in response.lower():
      return False, response.strip()
  ```
  实测 LLM 回复 "The analysis is **incomplete**. Need more data." → 命中 `"complete"` 子串 → `should_continue=False` → 主循环提前 break（agent.py:183-186），剩余计划步骤不执行，`incomplete=True` 的结果被当作最终结论。真实 LLM 输出"报告不完整/analysis is not complete"的概率不低。`"revise"` 同理会被 `"unrevised"` 之类误命中（概率低）。
- **建议**：按词边界匹配（`re.search(r'\b(?:complete|continue|revise)\b', ...)`），并优先要求结构化输出（如 JSON verdict）。

### O7. ECGFounder 分类器将 NaN/Inf 静默清零后照常分类，坏输入产出"可信"概率
- **文件:行号**：`src/agent/tools/ecgfounder_classifier.py:94-96`
- **疑似问题（读码确认）**：
  ```python
  x = torch.from_numpy(np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)).unsqueeze(0)...
  ```
  导联脱落/数值异常的信号被全量替换为 0 后送入冻结骨干，输出一组"看起来正常"的 27 类概率（含 `positive` 标记），无任何告警。调用方（`web_demo/app.py:122-125`）只捕获异常，捕获不到"被清零的坏数据"。
- **建议**：`predict` 前统计 NaN/Inf 比例，超过阈值返回 `{"error": "信号含大量无效数值"}`；至少返回 `warnings` 字段透传。

### O8. generate_report 工具在 Agent 循环中永远拿不到 features/diagnoses，输出恒为"N/A"
- **文件:行号**：`src/agent/tools/diagnosis_tools.py:77-93`、`src/agent/core/memory.py:69-84`（上下文注入）、`src/agent/core/agent.py:299`
- **确认缺陷（已验证）**：测试 G。
- **证据**：`memory.get_context_for_tool("generate_report")` 仅注入 `patient_info`（实测 keys=`['patient_info']`），而工具签名 `generate_report(features=None, diagnoses=None, patient_info=None)`。LLM 计划中的 params 也无法预知工具结果 → 每次执行 `features=None, diagnoses=None` → `report["summary"] == "心率: N/A bpm。"`。该结果既不被 `_synthesize_diagnosis` 采纳（合成只认 heart_rate/qt_ms/diagnoses 键，agent.py:358-388），又出现在 reasoning_chain 中误导展示。即"综合生成报告"步骤在 Agent 路径上是内容为空的装饰步骤。
- **建议**：`get_context_for_tool("generate_report")` 注入 `features`/`diagnoses`（聚合 `_tool_results` 中已有结果）；或让 `_synthesize_diagnosis` 直接以 memory 观测生成 summary。

### O9. LLM 计划 params 无类型/schema 校验即 `**kwargs` 展开注入工具
- **文件:行号**：`src/agent/core/agent.py:299`
- **疑似问题（读码确认）**：
  ```python
  params = {**self.memory.get_context_for_tool(tool_name), **step.params}
  result = self.tools.call(tool_name, **params)
  ```
  `step.params` 来自 LLM 自由生成的 JSON（仅保证是 dict），值可为任意类型：`fs="abc"`、`top_k="3"`、`ecg_signal="SIG"` 等。多数会被工具内部异常捕获（`_execute_step` 的 except），但：① 会覆盖 memory 注入的依赖值（如 LLM 乱写 `r_peaks` 键覆盖真实结果）；② 报错噪音污染日志与 reasoning_chain。registry 侧 `functools.partial` 绑定 model/detector 防住了模型替换（设计亮点），但其余参数裸露。
- **建议**：为每个工具注册参数类型白名单，`_execute_step` 按 schema 过滤/转换后再调用。

---

## 四、🟡 可维护性/风格

### Y1. Planner JSON 解析容错不足：无代码块且前后夹散文时静默回退默认计划
- **文件:行号**：`src/agent/core/planner.py:101-103`（提示词）、`planner.py:128-138`（`_extract_json`）
- **确认缺陷（已验证）**：测试 E。`'以下是计划：\n{...}'` 与 `'{...}\n说明：...'` 均解析失败 → `_parse_plan` 走默认计划（仅 warning）。真实 LLM 夹散文概率高。
- **建议**：`_extract_json` 增加"提取首个 `{...}` 平衡括号"策略；`generate_plan` 调用时传 `response_format={"type":"json_object"}`（与 verify_llm/eval 一致）。

### Y2. Few-shot 示例含未注册工具 query_medical_knowledge
- **文件:行号**：`src/agent/core/planner.py:54`
- **疑似问题**：病例 3 的示例计划包含 `query_medical_knowledge`，registry 未注册该工具 → 照抄 few-shot 的计划在 `_execute_step` 得到 "工具未找到"（有日志，不崩溃，但消耗步骤）。
- **建议**：从 few-shot 移除或注册该工具。

### Y3. memory 工具上下文表含从未注册的 plot_waveform
- **文件:行号**：`src/agent/core/memory.py:72-73`
- **疑似问题**：`get_context_for_tool` 为 `plot_waveform` 注入 `ecg_signal`，但全仓库无任何注册（run_agent_demo 只注册 5 个工具）。LLM 若计划该工具必然 "工具未找到"。
- **建议**：从注入表移除或补齐注册。

### Y4. Mock 回复启发式把"任何含 ECG/plan 关键词的提示词"都当作规划请求
- **文件:行号**：`src/agent/llm/llm_interface.py:197-209`
- **确认缺陷（已验证）**：`verify_llm.py --mock` 实测 Reasoner/Reflector 测试收到的都是 `{"plan": [...]}`。
- **证据**：`if "ECG" in last_msg or "plan" in last_msg.lower(): return plan_json`。Reasoner 提示词含 "ECG Findings"、Judge 提示词含 "ECG 诊断报告"、Reflector 提示词含 "ECG 分析" → 全部返回 plan JSON。mock 模式下任何下游 JSON 解析都可能拿到错误形状。
- **建议**：按调用方/角色区分 mock 回复，或 mock 返回与提示词尾部指令匹配的通用占位。

### Y5. verify_llm.py 的 Reasoner 检查 `len(raw)>50` 是假阳性
- **文件:行号**：`scripts/verify_llm.py:75`
- **确认缺陷（已验证）**：mock 下 Reasoner 实际收到 plan JSON 仍判 "输出长度正常=True"。
- **建议**：校验输出不含 `{"plan"` 且包含医学关键词；或对 reasoner 走 JSON 结构校验。

### Y6. `response.choices[0]` 无空响应守卫
- **文件:行号**：`src/agent/llm/llm_interface.py:185`
- **疑似问题**：异常 API 响应（choices 为空）→ IndexError 向上抛，无上下文。
- **建议**：判空后抛带响应原文的清晰异常。

### Y7. RubricJudge 缓存 key 只含 report 前 1500 字符，共享前缀的报告互相串分
- **文件:行号**：`src/agent/llm/rubric_judge.py:43-46`
- **疑似问题**：`key = sha256(gt|report[:1500])`。两份不同报告若前 1500 字符相同（模板化报告常见）→ 后一份命中前一份的评分，评测结果被污染。截断与判官所见一致（设计意图），但 key 碰撞是真实风险。
- **建议**：key 增加报告长度或全文哈希；或在报告中加入记录 ID 盐。

### Y8. RubricCache/ResponseCache JSONL 追加无文件锁
- **文件:行号**：`src/agent/llm/rubric_judge.py:99-103`（`RubricCache.put`）
- **疑似问题**：多进程/多线程并发 `open(path,"a")` 写入行交错 → 缓存文件损坏（读取端对坏行静默 pass，缓存静默丢失）。`eval_agent_llm.py` 的 `ResponseCache` 同构。
- **建议**：单进程内加线程锁；多进程用独立缓存文件或 SQLite。

### Y9. 判官失败契约不一致：`score()` 返回 None 的修复被调用方绕过
- **文件:行号**：`src/agent/llm/rubric_judge.py:62-71`（返回 None + 注释声明"由调用方跳过"）
- **疑似问题**：两个调用方行为分裂：`scripts/rejudge_with_rubric.py:64-69` 经 `aggregate` 正确跳过非 dict；但 `scripts/eval_agent_llm.py:386-393` 把解析失败转成 `{"correctness":1,...}` 全 1 分——正是 rubric_judge 注释（67-71 行）明言要消除的"LLM 故障伪装成报告质量极差"。同一批次评测两种口径并存。
- **建议**：统一走 `RubricJudge.score` + `aggregate`，删除 eval_agent_llm 内的重复实现。

### Y10. `patient_info.get("age")` 对非 dict 输入崩溃
- **文件:行号**：`src/agent/core/agent.py:152`
- **疑似问题**：调用方误传 list/str 时 AttributeError。类型注解 `Optional[Dict]` 是软约束。
- **建议**：`isinstance(patient_info, dict)` 前置判断。

### Y11. ToolScheduler 缓存 key 内嵌 numpy 数组 repr（性能）
- **文件:行号**：`src/agent/tools/scheduler.py:91`
- **疑似问题**：`cache_key = f"{name}:{str(sorted(step.params.items()))}"` — 含 5000 点数组时 key 达数十 KB，字典哈希/比较开销大；数组 repr 精度敏感导致缓存命中率低。文件已 DEPRECATED。
- **建议**：key 用参数名 + 轻量摘要（如数组形状/哈希）。

### Y12. AgentPipeline `_load_ecg` 对 list 输入无 astype
- **文件:行号**：`src/agent/orchestration/pipeline.py:64`
- **疑似问题**：`ecg_input.astype(np.float32)` — 传入 list 即 AttributeError。DEPRECATED 文件。
- **建议**：`np.asarray(ecg_input, dtype=np.float32)`。

### Y13. .env 在模块导入期加载，且依赖 cwd
- **文件:行号**：`src/agent/llm/llm_interface.py:38-62`
- **疑似问题**：`_load_dotenv()` 在 import 时执行；候选含 `Path.cwd()/.env` — 从不同目录启动行为不同；运行中修改环境变量不生效。
- **建议**：改为 `LLMInterface.__init__` 内按需加载，cwd 候选加日志。

### Y14. `torch.load(weights_only=False)` 存在 pickle 反序列化风险
- **文件:行号**：`src/agent/tools/ecgfounder_classifier.py:51,63`
- **疑似问题**：`weights_only=False` 对恶意/被篡改 checkpoint 可执行任意代码。当前仅加载本地受信文件，风险可控。
- **建议**：升级 torch 后用 `weights_only=True` + 对含 `net.` 前缀的 checkpoint 单独处理（state_dict 可安全加载）。

### Y15. revise 反馈循环：LLM 返回相同计划时重复 replan 烧 max_steps
- **文件:行号**：`src/agent/core/agent.py:161-191`
- **疑似问题**：规则检查命中（如 R1 的 hr=0）→ revise → replan，若 LLM 修订后仍含相同待执行步骤，下一轮同样步骤再次触发 revise → 反复 replan 直至 `max_steps` 耗尽，最终 `incomplete=True`。有界（不挂死），但浪费且结果不可控。
- **建议**：对同一 (step, feedback) 组合去重；规则检查命中时直接标记该步骤结果不可信并重跑，而不是依赖 LLM 重排。

### Y16. `ECGAIAgent` 实例级 memory 非线程安全
- **文件:行号**：`src/agent/core/agent.py:143-148`
- **疑似问题**：`self.memory` 在 `diagnose()` 首行 clear —— 同一实例并发诊断时互相清空/穿插上下文，跨患者串扰回潮。当前 web_demo 用自有 DemoPipeline，未暴露；但作为公共框架类应注明或加锁。
- **建议**：`diagnose` 内新建局部 `AgentMemory`，或文档注明非线程安全。

---

## 五、💡 建议

1. **`agent.py:193-199`**：`incomplete=True` 时，建议 `format_report`/demo 输出显式附加"⚠️ 分析未完成（达到最大步骤数），结论不可靠，请人工复核"免责标记，避免部分执行结果被当完整结论。
2. **`reflector.py:69-103`**：规则检查（生理范围）命中后建议**强制重跑对应工具步骤**并再次校验，而非仅靠 LLM 重排计划（当前 revise 路径实测不能修正 R1 的错误节律标签）。
3. **`src/agent/core/reasoner.py:31-50`**：`MedicalReasoner.reason` 已被 `ECGAIAgent.__init__` 实例化但从未接入主流程（模块 docstring 已知悉）——建议接入 `_synthesize_diagnosis` 生成推理文本，或删除以减少误导性死代码。
4. **`agent.py:272-281`**：默认回退计划应与实际注册工具集合求交集校验，避免计划中出现未注册工具（如当前默认计划与 few-shot 均含未注册工具名）。

---

## 六、审查结论

- **跨会话状态污染**：Agent 主路径防护良好（`agent.py:143` 每次 `memory.clear()`；`conversation.py:97-101` 重置）；残留风险为 O4（交互收集字段错位）、Y16（并发）。**未发现"上一患者工具结果注入本次计划"的活路径。**
- **LLM 失败路径**：`LLMInterface` 不再静默降级（修复方向正确），但 O3 表明调用方（`_plan`）未兑现 docstring 的回退承诺，LLM 故障会炸穿 `diagnose()`；R2 表明**工具层失败**会被 `_generate_recommendations` 包装成"未见异常"的正面结论——这是本轮最危险的医疗可信度问题。
- **JSON 解析容错**：`_parse_plan` 全面健壮（F1-F4 实测不崩溃）；`Planner._extract_json` 仅支持代码块包裹，散文夹 JSON 静默降级（Y1）。
- **类型防护**：工具/记忆层多数防御到位（`_num`、`isinstance` 守卫、partial 绑定 model）；缺口在 O9（LLM params 直注入）、Y10。
- **状态机**：`ConversationManager` 状态图完整可达、无死锁（start 直达 COLLECTING_INFO 已修复）；但 O5 的 replan 重绑使"修订计划"形同虚设；`ToolScheduler` 环检测完备（`scheduler.py:127-130`）。
- **医疗安全**：R1/R2/R3 三条 🔴 均属"失败/空数据被包装为正常或确定性结论"的模式，建议在发布前全部修复。

（全文完）
