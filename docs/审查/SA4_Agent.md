# Agent 框架模块审查报告

> 审查对象：`src/agent/` 下 15 个模块（core 5 + tools 4 + llm 3 + orchestration 2 + prompt_templates）
> 审查方式：全部文件经 read 工具实际读取，并交叉验证了 `label_extractor.py`、全库引用关系
> 结论统计：🔴 严重 5 处 · 🟠 中等 26 处 · 🟡 轻微 33 处 · 💡 建议 5 处（合计 69 条）

---

## 总体评价

框架分层清晰（core/tools/llm/orchestration），数据类与接口定义规整，ReAct 主循环有 `max_steps` 兜底、`AgentStep.completed` 终态保证，基本不会出现传统意义上的"死循环"。但存在四类系统性问题：

1. **跨患者数据污染**（最危险）：`AgentMemory._tool_results`、`ToolScheduler._cache`、`ConversationManager._collected_info/_results` 均不在会话间清理，同一进程连续诊断两名患者时，第二名患者会拿到第一名患者的上一步工具结果（如 R 峰、缓存诊断）。在医疗场景下这是 🔴 级正确性 bug。
2. **LLM 失败静默降级 mock**：`LLMInterface.chat()` 任何异常（缺 key、网络、限流）都静默返回预设占位文本，且 mock 回复的措辞是"心律在正常范围内…建议临床对照确认"——在生产中会被当作真实诊断呈现。这是审查关注点 2 的核心结论。
3. **conversation 状态机存在不可达死锁**：`INIT` 状态无任何转移出口；`EXECUTING` 分支一次性执行全部步骤绕过 ReAct/反思；`_collected_info`/`_results` 跨会话残留。
4. **死代码与内联 prompt 重复**：`prompt_templates.py` 全文件无引用（四个模板与 planner/reflector/reasoner 内联 prompt 内容不一致，存在漂移）；`ToolScheduler`、`MedicalReasoner.reason()`、`ECGFounderClassifier.tool_schema()` 均未接入主流程；`_format_report` 只存在于 pipeline 而 conversation 却调用 `agent._format_report`。

亮点：`_parse_plan` 有默认计划回退、`_quick_check` 零 LLM 成本硬规则、`ecgfounder_classifier` 做了 nan 清洗/居中 padding/`net.` 前缀兼容、`rubric_judge` 有持久化缓存——这些设计方向正确，只是健壮性细节不足（见下文逐条）。

---

## 各文件逐条发现

### 1. src/agent/core/agent.py（ReAct 主循环）

- **agent.py:139-144** [🔴] `diagnose()` 开头只 `add_context` 不 `memory.clear()`：同一实例第二次诊断时，`AgentMemory._tool_results`（memory.py:32,48-49）残留上一次患者的 R 峰/心率结果，`get_context_for_tool` 会把它注入本次计划（例如新计划没有 `extract_r_peaks` 但含 `compute_hrv` 时直接使用旧 R 峰）。**修法**：`diagnose()` 入口调用 `self.memory.clear()`（或至少清 `_tool_results`），并在测试中加"连续两次诊断互不串扰"用例。

- **agent.py:148-152 vs 211-216** [🟠] 交互模式收集的患者信息写入 `memory._context["patient_info"]`，但 `_plan()` 用的是原始 `patient_info` 参数——LLM 制定计划时看不到刚收集的年龄/性别。**修法**：`_plan` 改从 `self.memory.get_context().get("patient_info")` 取，或 `_collect_patient_info` 后回写 `patient_info` 变量。

- **agent.py:171-177** [🟠] `self.reflector.check()` 不在 try/except 内：`_quick_check` 中 `hr < 20`（reflector.py:69）若 `hr` 为字符串（工具返回 str 或 LLM 注入）会抛 `TypeError`，直接炸穿整个 `diagnose()`。**修法**：包一层 try/except，异常时按"继续执行"降级并记录日志。

- **agent.py:220-244** [🟠] `_parse_plan` 只捕获 `json.JSONDecodeError`：① `{"plan": null}` 或 `{"steps": null}` → `for s in None` 抛 `TypeError`；② JSON 数组内含非 dict 元素（字符串/数字）→ `s.get` 抛 `AttributeError`；③ `params` 为字符串 → 后续 `_execute_step` 中 `{**step.params}` 抛 `TypeError`。三者均未捕获，直接崩。**修法**：捕获 `(json.JSONDecodeError, TypeError, AttributeError)`，对每条 step 做类型校验（非 dict 跳过/记 warning）。

- **agent.py:272-293** [🟠] `_replan` 按"下标"回填已完成结果：LLM 重排步骤顺序或缩短计划时，已完成结果会挂到错误的 action 上（`new_plan[i].result = s.result`），超出 `len(new_plan)` 的已完成步骤结果被静默丢弃。**修法**：按 `action` 名匹配回填，未匹配的已完成步骤追加到新计划末尾保留。

- **agent.py:157-182** [🟠] `max_steps` 耗尽（每次迭代只执行 `pending[0]` 一个步骤，若反复 replan 生成大计划，10 次迭代可能不够）时静默返回部分诊断，无任何警告/标记。**修法**：循环结束后检查 `pending` 非空则 `logger.warning` 并在 `DiagnosisResult` 加 `incomplete=True` 字段。

- **agent.py:333-341** [🟠] `confidence += d.get("probability", 0.0)`：① 概率为字符串/None → `TypeError`；② 多个分类器结果直接累加再除以诊断条数，置信度被稀释且语义错误（应取主诊断置信度或按来源加权）。**修法**：逐条 `float()` 转换 + try/except，整体置信度改为取 top 诊断的概率。

- **agent.py:260-261** [🟡] `params.update(self.memory.get_context_for_tool(...))` 会覆盖步骤显式参数（同名 key）。**修法**：先取记忆上下文，再用 `step.params` 覆盖（`{**ctx, **step.params}`）。

- **agent.py:213** [🟡] `ecg_signal.shape` 无 None 保护，`ecg_signal=None` 时 `AttributeError`。**修法**：入口校验。

- **agent.py:346-354 / 356-366** [🟡] 推理链直接暴露 `plan`（含大数组 result），序列化/日志成本高；`_generate_recommendations` 硬编码 0.7 阈值。**修法**：结果入链前裁剪为标量摘要；阈值提为常量。

### 2. src/agent/core/planner.py

- **planner.py:128-138** [🟠] `_extract_json` 只处理 markdown 围栏：LLM 输出"先解释再给 JSON"（无围栏）时原样返回全文，`json.loads` 失败 → 静默回退默认计划（agent.py:235-244），真实的 LLM 计划被丢弃且无 warning 之外的信号。**修法**：无围栏时用正则提取首个 `{...}`/`[...]` 平衡片段再尝试解析。

- **planner.py:23-58** [🟡] Few-shot 示例引用了 `detect_anomaly`、`query_medical_knowledge`，后者未注册在任何 registry——LLM 依样生成的计划会命中 agent.py:253-256 的"工具未找到"分支。**修法**：示例与注册工具对齐，或 `_execute_step` 对未知工具尝试跳过而非标记错误。

- **planner.py:90-103** [🟡] 内联 prompt 与 `prompt_templates.PLANNING_PROMPT` 重复但内容不一致（模板文件无人引用，见文件 12）。**修法**：统一从模板常量渲染。

### 3. src/agent/core/memory.py

- **memory.py:57-79** [🟠] `get_context_for_tool` 硬编码工具名白名单：新增工具（如 ECGFounder 版 classify）静默拿不到任何上下文；且 `r_result.get(...)` 假设 `extract_r_peaks` 结果是 dict，若结果为错误字符串/数组 → `AttributeError`。**修法**：改为从 `ToolRegistry.get_dependencies()` 动态推导前置结果注入，结果非 dict 时防御性跳过。

- **memory.py:51-55** [🟡] `get_observation` 返回的是 result 而非 obs dict（返回类型与 docstring/类型标注 `Optional[Dict]` 不符，且该方法无人调用）。**修法**：删掉或修正签名。

- **memory.py:48-49** [🟡] `_tool_results` 无界增长且被失败步骤的 error dict 污染（`obs.get("result")` 可能为 `{"error":...}`，后续被当作依赖注入）。**修法**：仅缓存成功结果，或缓存时校验。

### 4. src/agent/core/reflector.py

- **reflector.py:68-91** [🟠] `_quick_check` 数值比较无类型防护：`hr < 20` 遇字符串/数组抛 `TypeError`（配合 agent.py:171 的未捕获异常直接崩流程）；`hr_std > 15`、`sdnn > 300` 同理。**修法**：比较前 `isinstance(x, (int, float, np.number))` 校验，失败视为通过并记录。

- **reflector.py:105-110** [🟡] `step.result` 原样拼入 LLM prompt：可能含 numpy 大数组/超长内容，token 爆炸。**修法**：截断/标量化。

- **reflector.py:110** [🟡] `history.get('observations', [])[-3:]` 取的是"最近 3 条观测"，而 memory 中观测按追加序排列（最新在末尾），语义正确但依赖排序约定，建议注释说明。

- **reflector.py:123-126** [🟡] `_llm_verify` 用子串匹配 `"revise"/"complete"`：LLM 若在句子里顺带提到"complete revision"会误判为 complete 提前终止。**修法**：改为首行/关键词+正则，或要求输出固定标记。

### 5. src/agent/core/reasoner.py

- **reasoner.py:27-46** [🟡] `reason()` 全库无调用点（agent.py:117 实例化后从未使用）——"医学推理器"是流内死代码，且 docstring 声称的 CoT 链从未出现在最终输出。**修法**：在 `_synthesize_diagnosis` 中接入，或删除。

- **reasoner.py:54-61** [🟡] 内联 prompt 与 `prompt_templates.REASONING_PROMPT` 重复且结构不同（模板有 7 段、内联有 6 段）。**修法**：统一模板源。

- **reasoner.py:58-61** [🟡] 非 dict 观测被静默跳过，无日志。**修法**：`logger.debug`。

### 6. src/agent/tools/registry.py

- **registry.py:47-50** [🟡] `register()` 同名重复注册静默覆盖（如 diagnosis_tools 与 ecgfounder 都想注册 `classify_arrhythmia` 时无告警，后者覆盖前者）。**修法**：已存在时 `logger.warning` 或抛错。

- **registry.py:57-70** [💡] `call()` 先 log 再 raise 的写法正确，但 `dependencies` 字段在主流程（agent.py 的 `_execute_step`）中完全未被使用——依赖信息只被死代码 scheduler 消费。**建议**：要么让 agent 用 scheduler，要么删除 dependencies 字段，避免"声明了却没人执行依赖"的假安全感。

### 7. src/agent/tools/scheduler.py（DAG 执行器）

- **scheduler.py:59-99** [🔴] **无循环依赖检测**：若依赖成环（A→B→A），两节点 `in_degree≥1` 永不为 0，队列清空后 `while` 直接退出，环内步骤**静默不执行**，`execute_plan` 返回不完整 `results` 且无任何报错。**修法**：执行前做 Kahn 环检测，成环抛 `ValueError` 或记录后跳过并告警。（注：当前 scheduler 未被主流程引用，属死代码；一旦接入即成为严重问题。）

- **scheduler.py:53-57** [🟠] `if dep in name_to_step`：依赖的工具不在计划中时**静默忽略**，依赖者在前置缺失的情况下仍被执行。**修法**：缺失依赖记录 warning 并按"跳过该步骤"或"自动前置插入"策略处理。

- **scheduler.py:68-73** [🟠] 缓存 key = `name + sorted(params)`：① 信号等大对象通常经 memory 注入而非 params，两次不同患者的 `extract_r_peaks` 参数同为 `{}` → 缓存 key 相同 → **跨患者复用上一次诊断结果**（与 agent 记忆污染同源）；② `sorted(step.params.items())` 遇 numpy 数组等不可比值直接 `TypeError`；③ 长数组进 key 导致缓存膨胀。**修法**：key 注入信号指纹（如 `hash(ecg_signal.tobytes())` 或患者 ID），排序前做可哈希化。

- **scheduler.py:76-91** [🟠] 步骤最终失败后，其依赖者仍被入队执行（`graph[name]` 照常减度入队），前置失败被静默传播。**修法**：失败步骤的依赖者标记 skipped 或记录 error 依赖链。

- **scheduler.py:50-51** [🟡] 计划中同名 action 重复时 `name_to_step` 被后项覆盖、`in_degree` 重复累加，图构建失真。**修法**：去重或校验报错。

- **scheduler.py:78** [🟠] `registry.call(name, **step.params)` 直接调用，**不注入 memory 上下文**：主流程中 `extract_r_peaks` 的信号来自 memory（agent.py:260-261），scheduler 独立执行时信号永远缺失 → 全部返回 `{"error": "未提供 ECG 信号"}`。**修法**：与 agent 共用上下文注入逻辑，或明确 scheduler 只负责排序、执行仍走 agent。

### 8. src/agent/tools/ecg_tools.py

- **ecg_tools.py:38-47** [🟠] `result.get("heart_rate", 0)`：检测器返回 error dict 时 `hr=0` → 被静默分类为 `"bradycardia"`——把一次失败伪装成一个诊断结论。**修法**：`hr` 为 None/0 且结果含 `error` 时直接返回 error，不做节律分类。

- **ecg_tools.py:93-96** [🟠] `ecg_signal.ndim`：若信号以 list 传入（JSON/工具参数常见）→ `AttributeError`（被 agent 捕获记为步骤错误，但信息误导）。**修法**：`np.asarray` 后再取 ndim。

- **ecg_tools.py:32,71,102** [🟡] `from src.ecg_models...` 依赖 cwd 在 `sys.path` 上；与 ecgfounder_classifier.py:20 的路径引导不一致（后者显式 insert，且 insert 本身有 bug，见文件 10）。**修法**：统一在包入口做一次 sys.path 引导，或按包安装。

### 9. src/agent/tools/diagnosis_tools.py

- **diagnosis_tools.py:72** [🟠] `detector._fitted`、`detector.threshold.item()` 直接访问私有属性：任何未实现这两个属性的检测器实现都会 `AttributeError`（被 agent 记为步骤错误，但把"检测器不兼容"伪装成"检测失败"）。**修法**：用 `getattr` + 公开接口/鸭子类型兜底。

- **diagnosis_tools.py:77-93** [🟠] `generate_report` 在 agent 流中**永远拿到空数据**：`memory.get_context_for_tool`（memory.py:57-79）只注入 `patient_info/ecg_signal/r_peaks`，从不注入 `features`/`diagnoses`，而默认计划（agent.py:238-244）里 `generate_report` 的 params 为空 → 报告恒为"心率 N/A、无主要发现"。`_synthesize_diagnosis` 又不使用该工具输出，等于这个工具在主流程里纯属摆设。**修法**：在 `_execute_step` 中对 `generate_report` 特殊注入已收集的 findings/diagnoses，或让 `_synthesize_diagnosis` 直接调用它。

- **diagnosis_tools.py:50** [🟡] `zip(label_extractor.class_names, probs)` 长度不符时静默截断（class_names 少于 27 时丢概率）。**修法**：长度校验 + warning。

- **diagnosis_tools.py:91** [🟡] `top.get('probability', 0):.1%`：probability 为字符串时 `ValueError`。**修法**：`float()` 转换兜底。

### 10. src/agent/tools/ecgfounder_classifier.py

- **ecgfounder_classifier.py:20** [🟠] `sys.path.insert(0, str(Path(__file__).parent.parent.parent))` **少一级 `.parent`**：文件位于 `src/agent/tools/`，三级 parent 只到 `.../ecg-ai-agent/src`，而下一行 `from src.ecg_models...` 需要项目根在 path 上。仅当 cwd 恰为项目根（cwd 在 sys.path 中）时碰巧可用；从其他目录 `python src/agent/tools/ecgfounder_classifier.py` 直接 `ImportError`（`__main__` 冒烟块会挂）。**修法**：改为 `.parent.parent.parent.parent`。

- **ecgfounder_classifier.py:49,61** [🟠] `torch.load(..., weights_only=False)`：pickle 反序列化可执行任意代码（权重文件被投毒 = RCE）；且构造期无 try/except，checkpoint 缺失时抛原始 traceback。**修法**：生产用 `weights_only=True`；构造失败转成带路径提示的清晰异常。

- **ecgfounder_classifier.py:62-64** [🟡] `net.` 前缀兼容只覆盖一种包装名；若 checkpoint 键含 `head.` 前缀或混合键（部分 `net.`、部分裸键），strip 后 `load_state_dict` 失败且报错信息不含原因。**修法**：打印首尾键样本的调试日志，支持 `net.`/`head.` 双前缀。

- **ecgfounder_classifier.py:68-75** [🟡] 阈值加载：`metrics.thresholds` 键名/形状不符时**静默回退 0.5**，无日志——用户无法得知阈值未生效。**修法**：加载/回退都 `logger.info/warning`。

- **ecgfounder_classifier.py:92-99** [🟡] `predict` 未校验 `signal.ndim == 2`：1D 信号 → backbone 输入通道不匹配报错（信息不友好）。**修法**：入口 `np.asarray` + 维度检查。

- **ecgfounder_classifier.py:123-135** [💡] `tool_schema()` 声称工具名 `classify_arrhythmia`，但该类从未注册进任何 ToolRegistry——agent 实际用的是 diagnosis_tools.py:99-105 的旧 partial（模型可能为 None → 恒返回"未加载分类模型"）。**建议**：写一个 `register_ecgfounder_tools(registry, classifier)` 统一接线，否则 M2.2 升级在 agent 流中不生效。

### 11. src/agent/llm/llm_interface.py（含 .env 加载）

- **llm_interface.py:160-162** [🔴] **LLM 失败静默降级 mock 掩盖真实错误**：任何异常（缺 key、网络、限流、超时）都 `logger.error` 后返回 `_mock_response`——mock 文案"心律在正常范围内。未检测到急性异常。建议临床对照确认"会被当作真实诊断输出。医疗场景下这是"看起来正常、实则全是占位符"的最危险模式；同时它掩盖了配置错误（key 拼错/额度耗尽），让故障长期潜伏。**修法**：① 降级必须显式：`AGENT_MOCK=1` 环境变量才启用 mock，否则异常向上抛并由调用方决策；② 返回 mock 时在结果对象上加 `"mock": True` 标记并在日志/报告抬头显著提示"模型未连接，输出为占位内容"。

- **llm_interface.py:96-100** [🟠] API key 缺失/为空时不校验，直接 `OpenAI(api_key="")` 构造必然失败的 client，最终落到上一条的 mock 分支。**修法**：`_init_openai_compatible` 里 `if not api_key: raise RuntimeError("缺少 DEEPSEEK_API_KEY/OPENAI_API_KEY")`（或显式进入 mock 并告警），别把配置错误和运行时错误混为一谈。

- **llm_interface.py:30-48** [🟡] `_load_dotenv`：① `return` 在第一个存在的候选后——cwd 的 `.env` 永远不会被读到（若项目根已有）；② 不处理 `export KEY=...` 前缀（k 会变成 `"export KEY"`）；③ 不处理 BOM。**修法**：遍历所有候选并合并，`line.lstrip("export ")`，`utf-8-sig` 解码。

- **llm_interface.py:178-181** [🟡] `is_available` 恒为 True（`OpenAI()` 从不返回 None，即使 key 为空），名称具有误导性。**修法**：改为"构造成功且非 mock"或干脆删除该属性。

- **llm_interface.py:135-145** [🟡] 无显式 timeout/retry 参数，依赖 SDK 默认（openai 默认 600s 超时+2 次重试）——长请求卡死时 ReAct 循环整体悬挂。**修法**：暴露 `timeout`/`max_retries` 到构造参数。

- **llm_interface.py:17-18** [💡] docstring 建议 `llm._client = None` 手动切 mock——与上一条"显式 mock 开关"建议合并为构造参数 `mock: bool = False`。

- **密钥处理结论**：未发现打印/记录 api_key 的路径（`logger.info` 只记录 .env 路径，异常信息不含 key），此点安全；真正的问题是降级路径掩盖了凭据错误（见第一条 🔴）。

### 12. src/agent/llm/prompt_templates.py

- **prompt_templates.py:7-79** [🟡] **全文件死代码**：grep 全 `src/` 无任何 `prompt_templates` import。四个模板与 planner.py:90-103、reasoner.py:54-73、reflector.py:105-120 的内联 prompt 内容不一致（如 REASONING_PROMPT 是 7 段结构、内联是 6 段；PLANNING_PROMPT 无 few-shot 而 planner 有）。**修法**：二选一——删除该文件并删掉内联重复（保持内联），或让各模块 import 模板渲染（推荐，单一事实来源）。

### 13. src/agent/llm/rubric_judge.py

- **rubric_judge.py:47-55** [🟠] **缓存 key 与判官实际所见不一致**：key 用**完整** report（`_key` 拼接全文），prompt 却 `report[:1500]` 截断。同一份长报告：命中缓存返回按全文评的分数，未命中则按截断文本重评——同一输入两种结果；而共享前 1500 字符的不同报告又无法命中缓存。**修法**：先 `trunc = report[:1500]`，再对 trunc 做 key。

- **rubric_judge.py:60-68** [🟠] 解析失败 → 三维+总分全给 1 分：把"判官/LLM 故障"伪装成"报告质量极差"，系统性拉低评测基准且无法区分。**修法**：失败时返回 None/重试一次，由调用方决定跳过该样本并单独统计 judge 失败率。

- **rubric_judge.py:63-64** [🟡] `int(scores.get(field, 1))`：LLM 返回 `"4.5"`（字符串小数）→ `ValueError` → 四字段全部重置为 1，无部分抢救。**修法**：`float()` 再取整，逐字段 try。

- **rubric_judge.py:93-100** [🟡] JSONL 缓存追加写：多进程并发评测时行间交错损坏；读侧只捕 `JSONDecodeError`。**修法**：写前加锁/写临时文件原子改名，或记录损坏行数。

- **rubric_judge.py:103-116** [🟡] `aggregate` 中 `s[field]` 直接取 key，任一评分 dict 缺字段即 KeyError。**修法**：`.get(field, 0)`。

- **rubric_judge.py:49-52** [🟡] `self.cache.get(key, 0.0)` 的 `0.0` 参数是摆设（`RubricCache.get` 签名是 `(self, key, _t)`，忽略默认值），易误导。**修法**：去掉默认值参数。

### 14. src/agent/orchestration/pipeline.py

- **pipeline.py:66-70** [🟠] `.hea` 分支：`ECGLoader(...).load_record(...)` 返回 `None` 时（记录缺失/解析失败）直接 `sample.signal` → `AttributeError`，掩盖真实加载错误。**修法**：判空后抛带路径的 `ValueError`。

- **pipeline.py:75** [🟡] `result.heart_rate or 'N/A'`：心率恰为 0（合法边界值，如停搏）也显示 N/A。**修法**：`if result.heart_rate is not None`。

- **pipeline.py:49** [💡] `self.preprocessor(ecg_signal, 500.0)` 假设预处理器 `__call__(signal, fs)` 签名，无契约文档。**建议**：类型/接口注释明确。

### 15. src/agent/orchestration/conversation.py（状态机）

- **conversation.py:103-120 + 150-155** [🔴] **INIT 状态死锁**：`start()` 把状态置为 `INIT`（L95），但 `handle()` 既没有 INIT 分支、也不会对 INIT 调用 `_advance_state()`（L116-118 只覆盖 PLANNING/EXECUTING/REFLECTING/FINALIZING）→ 首次 `handle()` 直接 `return None`，状态永远停在 INIT；按 docstring 的用法循环（L7-11）将**无限循环**（`prompt` 对 INIT 返回空串，无任何提示）。**修法**：`start()` 直接置 `COLLECTING_INFO`，或 `handle()` 增加 INIT → `_advance_state()` 分支，并给 INIT 加 prompt 文案。

- **conversation.py:88-101** [🔴] `start()` 不重置 `self._collected_info` 和 `self._results`：第二名患者会继承第一名患者的年龄/性别/症状和工具结果——跨患者数据污染（与 agent.py:139 同源）。**修法**：`start()` 中一并清空并重置 `_turn`。

- **conversation.py:168-178** [🟠] EXECUTING 分支一次执行**全部**未完成步骤，完全绕过 agent 的 ReAct 循环与 Reflector（无反思、无 revise）；`_results` 跨轮累积，`_summarize_results` 每轮重放全部。**修法**：EXECUTING 按 `agent.diagnose` 的循环语义逐步执行，或明确该模式为"批量执行"并文档化。

- **conversation.py:180-183** [🟠] `hasattr(self.agent, '_format_report')` 恒为 False——`_format_report` 只定义在 `AgentPipeline`（pipeline.py:72），`ECGAIAgent` 没有 → 永远走 `str(result.diagnosis)` 兜底，脆弱的鸭子类型判断掩盖了契约缺失。**修法**：在 agent 上提供 `format_report()` 公开方法，或 conversation 直接接收 pipeline 注入的格式化函数。

- **conversation.py:194-198** [🟠] `_summarize_results` 中 `result['sdnn']:.1f`/`rmssd`/`qtc_bazett`：值为 None（工具返回 error dict 时极常见）或字符串 → `TypeError` 崩溃。**修法**：`isinstance(v, (int,float))` 校验再格式化。

- **conversation.py:114-120** [🟡] REFLECTING 轮次返回 None（`_advance_state` 无 REFLECTING 分支），用户多输入一轮却得不到任何回复。**修法**：REFLECTING 轮返回"验证完成"过渡文案。

- **conversation.py:144** [🟡] `len(self._collected_info) >= 2` 即放行：收集到 age+sex 后 symptoms/history 永不再问，与 `_info_prompt` 的"缺什么问什么"逻辑矛盾。**修法**：按 agent 需要的字段集判断完整性。

- **conversation.py:72-86,122-148** [🟡] 信息提取用英文关键词+中文混杂硬编码；`re.findall(r'\d+', text)` 会捕获非年龄数字（如"2 个孩子"）。**修法**：简单正则即可但需限定范围，或改为 LLM 提取。

- **conversation.py:21-29 / 39-46** [💡] 状态机本质是线性链，`STATE_TRANSITIONS` 字典 + 分支派发是手写状态机；REFLECTING 分支缺失、INIT 无出口都源于此。**建议**：改用显式"状态 → (事件, 动作, 下一状态)"表驱动，杜绝不可达状态。

---

## 修复优先级清单

**P0（正确性/数据安全，建议本周内）**
1. `conversation.py` INIT 死锁（🔴，文件 15 条 1）——按 docstring 用法即死循环。
2. `agent.py:139` 诊断前清空 memory + `conversation.start()` 重置 `_collected_info/_results`（🔴，跨患者污染，文件 1 条 1 / 文件 15 条 2）。
3. `llm_interface.py` mock 降级显式化 + API key 校验（🔴，文件 11 条 1-2）——医疗输出不可静默降级。

**P1（健壮性，两周内）**
4. `scheduler.py` 环检测 + 缺失依赖告警（🔴/🟠，接入前必修）。
5. `agent.py` reflector 异常防护（🟠）、`_parse_plan` 全面容错（🟠）、`_replan` 按 action 回填（🟠）、max_steps 耗尽告警（🟠）。
6. `rubric_judge.py` 缓存 key 与截断对齐（🟠）、解析失败不伪装成 1 分（🟠）。
7. `ecgfounder_classifier.py:20` sys.path 少一级 parent（🟠）、`torch.load` 安全检查（🟠）。
8. `generate_report` 上下文注入（🟠）；`diagnosis_tools` 私有属性访问改公开接口（🟠）。
9. `conversation.py` EXECUTING 纳入反思、`_summarize_results` 格式化防护（🟠）。

**P2（一致性/维护，后续）**
10. 死代码清理或接线：`prompt_templates.py`（全文件）、`MedicalReasoner.reason()`、`ToolScheduler`、`ECGFounderClassifier` 注册。
11. prompt 单一事实来源（模板化），消灭内联重复。
12. 🟡 级风格项：类型防护、日志、`.env` 加载细节、Few-shot 与工具对齐等按上文逐条处理。
