# 第四次独立代码审查 — 4G_评测harness

> 审查人：独立审查员 G（第四次；未读取任何历史审查产物）
> 范围：`scripts/eval_agent_llm.py`、`scripts/eval_agent_context.py`、`scripts/eval_agent.py`、`scripts/build_agent_evalset.py`、`scripts/build_context_subset.py`、`scripts/recompute_agent_metrics.py`、`scripts/recompute_context_metrics.py`、`scripts/recompute_hallucinations.py`、`scripts/rejudge_with_rubric.py`、`scripts/analyze_context_inconsist.py`
> 验证方式：逐行 read 全部 10 个文件（无抽样）；为核实交叉引用另读 `src/data_pipeline/label_extractor.py`、`src/agent/tools/ecgfounder_classifier.py`、`src/ecg_models/feature_extraction/r_peak_detector.py`、`src/ecg_models/feature_extraction/hrv_analyzer.py`、`src/agent/llm/llm_interface.py`、`src/agent/llm/rubric_judge.py`；运行 `python scripts/eval_agent_llm.py --mock --limit 2 --output <TEMP>` 端到端管道测试（mock 不花 token、不写真实缓存，产物写临时目录并已清理）；用只读内联 python 复现两处口径缺陷并得到量化证据。
> 与"待重跑中间态"的关系：下述缺陷均为代码口径/守卫问题，独立于磁盘上 n<130 的中间产物，与"三臂待重跑"状态不冲突。

## 统计（🔴 2 / 🟠 2 / 🟡 7 / 💡 2）

## 发现清单

### 🔴-1 幻觉率分子分母口径不一致：PR间期/QRS时限/电轴只进分子、不进分母，幻觉率可 >1.0
- 位置：`scripts/eval_agent_llm.py:411-447`（8 个 claim_patterns，其中 PR间期/QRS时限/电轴 `tool_val=None` 在 `442-443` 计入 `hallucinated_claims`）vs `563-578`（`n_claims` 分母正则只有 心率/HR/QTc/QT/SDNN 5 个模式）
- 证据：`aggregate_metrics` 中 `n_hallu = sum(len(r["hallucinated_claims"]))`（第 563 行）包含全部 8 类声明，而分母 `n_claims`（第 568-570 行）只统计 5 类。只读复现实测：报告文本 `"心率 88 次/分，QTc 440 ms。PR间期 160 ms，QRS 120 ms，电轴 -45 度。"` → `n_hallu=3`（PR/QRS/电轴）、`n_claims=2`（心率、QTc），幻觉率 = 3/2 = 1.5 > 1.0。即"工具未提供测量值却声明数值"这类最强幻觉信号（PR/QRS/电轴）被计入分子却不进分母，系统性抬高并可能使 headline 指标"幻觉率"超过 100%，直接损害论文结论可靠性。
- 修法：让分母 `n_claims` 与分子同口径——把 PR间期/QRS时限/电轴 三条声明模式也纳入分母统计（它们 tool_val 恒为 None，任何具体数值声明都应既进分子又进分母）；或反之将这三类从分子移除。任选其一，但分子分母必须对齐。

### 🔴-2 情境评测 mock 模式无缓存守卫：占位回复写入共享 context_cache.jsonl，会静默污染真实评测
- 位置：`scripts/eval_agent_context.py:96-106`（`ask()` 无条件 `self.cache.put`，无 mock 分支）
- 证据：同仓库 `eval_agent_llm.py:159-162` 明确在 mock 模式跳过缓存并注释"防止占位回复污染"，而 `eval_agent_context.py` 的 `ask()` 无任何 mock 判断；缓存文件 `context_cache.jsonl`（第 327 行）不区分 mock/真实（记录文件 `records.jsonl` 有 `_mock` 后缀，缓存无）。`llm_interface.py:207-219` 的 `_mock_response`：仅当消息含 "ECG" 或 "plan" 才返回 JSON，情境 prompt（第 155-171 行）不含这两个词 → 返回非 JSON 占位文本 `【占位回复 · LLM 未连接】...`。mock 与真实同 key（messages 确定性相同 + temperature=0.3 + model=deepseek-chat）→ 先 `--mock` 后真实运行时，真实请求命中占位文本 → `json.loads` 失败 → 全部情境 `parse_error` → 情境评测指标静默归零/失效。这是"无 key 管道测试"文档化入口与正式评测共用同一缓存导致的真实数据完整性风险。
- 修法：在 `ask()` 中加 mock 守卫（`if getattr(self.llm,"mock_mode",False): return self.llm.chat(...)`，不读不写缓存），或给缓存文件加 `_mock` 后缀隔离；与 `eval_agent_llm.py` 对齐。

### 🟠-1 发现层-GT 重合率：部分情境 findings 为空时，空情境被整体跳过，重合率被高估
- 位置：`scripts/eval_agent_context.py:259-268`
- 证据：口径注释（第 254-258 行）声明"按记录×情境逐情境平均"。但代码 `if nonempty: for s in nonempty:` 只遍历**非空** findings 集合，空 findings 的情境既不进分子也不进分母。只读复现：GT={A,B}，三情境 findings 分别为 {A}、{}、{A} → 代码算得 `0.5`（gt_total=2，仅两个非空情境），而正确的逐情境均值应为 `(0.5+0+0.5)/3 = 0.3333`。即"部分情境为空"的记录被系统性高估（注释只修复了"全空"分支，未覆盖"部分空"）。
- 修法：改为遍历全部 `valid` 情境：`for s in f_sets.values(): findings_gt_overlap += len(s & gt)/len(gt); gt_total += 1`（空集自然贡献 0）。

### 🟠-2 recompute_hallucinations.py 与现役 harness 口径不同步：只扫 5 类声明，重算会丢掉 PR/QRS/电轴幻觉
- 位置：`scripts/recompute_hallucinations.py:29-35`（`CLAIM_PATTERNS` 仅 5 条）vs `scripts/eval_agent_llm.py:411-423`（8 条，含 PR间期/QRS时限/电轴）
- 证据：现役 `eval_agent_llm.py` 已把 PR/QRS/电轴 声明计为幻觉（3D 💡-1 扩展），但后置重算脚本 `scan_claims` 只有 5 条模式并直接 `r["hallucinated_claims"] = new_claims` 覆盖旧值（第 98 行），会把已正确标记的 PR/QRS/电轴幻觉静默抹掉，且与 `aggregate_metrics` 的分子口径脱节。
- 修法：同步 8 条 `CLAIM_PATTERNS`（含 tool_val 恒 None 的 PR/QRS/电轴），并让 `tool_values` 至少保持一致；否则重算后的 metrics 与在线评测不可比。

### 🟡-1 TOOLS_DESC 广告 `read_report` 工具，但 golden_tools 永不包含它 → 工具选择 precision 系统性偏低
- 位置：`scripts/eval_agent_llm.py:54-58`（`TOOLS_DESC` 第 58 行含 `read_report`）vs `scripts/build_agent_evalset.py:77-88`（`golden_tools` 只产出 classify_arrhythmia/extract_r_peaks/compute_hrv/measure_qt_interval）
- 证据：规划 prompt 把 `read_report` 作为可用工具展示给 LLM，但报告臂实际是直接把 `item['report_text']` 注入报告 prompt（`eval_agent_llm.py:380`），harness 从不产出 `read_report` 工具结果；golden 计划也永远不含 `read_report`。LLM 若据实规划 `read_report`，将按 `planned - golden` 记为 FP，拉低工具选择精确率，且该工具是"幻影"（不可执行）。
- 修法：要么从 `TOOLS_DESC` 移除 `read_report`（报告臂以直接注入文本实现，不暴露为工具），要么在 golden 计划中为报告臂纳入 `read_report` 并真正实现该工具返回。二选一，保证工具清单与黄金计划严格对齐。

### 🟡-2 build_agent_evalset 超量裁剪用随机 shuffle 截断，破坏"保持类覆盖"（注释与实际不符）
- 位置：`scripts/build_agent_evalset.py:125-129`
- 证据：注释（第 125 行）称"保持类覆盖：先按类去重后的记录裁剪"，但代码 `random.shuffle(picked_list); picked_list = picked_list[:args.max_total]` 是全量随机裁剪，可能把某稀有类整类裁掉，破坏分层覆盖。默认参数下 27 类×5 条≈135 < max_total=200 不触发，但调大 n-per-class 或调小 max-total 即触发。
- 修法：按类轮流取（round-robin）裁剪，或直接在抽样阶段按类配额限流，保证裁剪后各类仍有 ≥1 条。

### 🟡-3 build_agent_evalset docstring 声明不存在的 "ST/心梗/缺血/T波→plot_waveform" 规则
- 位置：`scripts/build_agent_evalset.py:13`（docstring 规则）vs `46-47`（注释明确 plot_waveform 不在工具清单）与 `77-88`（`golden_tools` 无任何 ST/MI/T 波规则）
- 证据：27 类官方评分类已不含 ST 段改变/心肌梗死/缺血（`label_extractor.py:76-78`），且 `golden_tools()` 实际只有 rhythm/QT/conduction 三条规则，`T Wave Abnormal/T Wave Inversion` 归入 `MORPH_CLASSES` 后不触发任何额外工具。docstring 第 13 行是陈旧描述，与实际实现矛盾，易误导后续维护者。
- 修法：删除/改写 docstring 该行，与实际 `golden_tools` 规则一致。

### 🟡-4 注入排除表引用不存在的类名 "Ventricular Tachycardia"，且漏了同为缓性诊断的 "Bradycardia"
- 位置：`scripts/eval_agent_llm.py:278`（`top1 not in ("Sinus Bradycardia",)`）、`283`（`top1 not in ("Sinus Tachycardia","Ventricular Tachycardia")`）
- 证据：27 类官方列表（`label_extractor.py:40-73`）无 "Ventricular Tachycardia"（该表有 "Premature Ventricular Contractions" 与 "Ventricular Premature Beats"，无 VT），故 `283` 行对 VT 的排除永假；同时 `278` 行只排除 "Sinus Bradycardia"，未排除同为缓性诊断的独立类 "Bradycardia"（`label_extractor.py:45`），导致 HR<60 且 top1 已是正确 "Bradycardia" 的记录仍被强制改写成 "Sinus Tachycardia" 注入。
- 修法：`283` 行去掉不存在的 VT（或换成真实存在的室性类）；`278` 行排除集补上 "Bradycardia"，使"top1 已与心率方向一致"的记录不被迫注入。

### 🟡-5 冲突捕获词表缺 "冲突"，"与心率冲突"式显式标记不被计入捕获率
- 位置：`scripts/eval_agent_llm.py:505`（`CONFLICT_MARK = re.compile(r"矛盾|不一致|不符|存疑")`）
- 证据：`_report_flags_conflict` 要求同一句内同时出现冲突词与快/慢诊断词。LLM 报告最常见的自然表述是"心率与诊断**冲突**"或"**相冲突**"，但词表只有 矛盾|不一致|不符|存疑，缺"冲突"，导致这类显式标记被判为未捕获，注入纠错 catch_rate 被低估（且对照组同口径，判别力也受影响）。
- 修法：`CONFLICT_MARK` 加入 `冲突`（以及可选的 `违背|相反`），并建议对"快/慢诊断词"同句命中做一次标注校验。

### 🟡-6 recompute_hallucinations 按 `evalset[i]` 位置索引对齐记录，evalset 版本/顺序变化即静默错配工具值
- 位置：`scripts/recompute_hallucinations.py:91`（`item = evalset[i] if i < len(evalset) else None`）
- 证据：脚本用 `records` 的行号 `i` 直接取 `evalset[i]`，再按该 item 的 `signal_file` 重算工具值（`tool_values`）。若 records 由旧版 evalset（不同抽样/顺序/条数）生成，或记录被删/重排，位置对齐即错位——用**错误信号**的工具值重判幻觉，`hallucinated_claims` 被静默改错。脚本无任何 record_id/signal_file 校验。
- 修法：用 `r["record_id"]`/`signal_file` 建立记录↔item 映射，并对齐失败时显式报错，不做位置索引假设。

### 🟡-7 eval_agent.py（已废弃）把 HRV 测量失败显示为 "SDNN=0ms"
- 位置：`scripts/eval_agent.py:84-88`
- 证据：`hrv_result.get("sdnn", 0)` 在 `sdnn=None`（数据不足，`hrv_analyzer.py:47-49/64` 返回 None）时取默认 0，随后 `result: f"SDNN={sdnn:.0f}ms"` 打印 `SDNN=0ms`，把"未测量"伪装成有效数值 0。该脚本已 DEPRECATED（第 1 行），影响有限，但属同类"失败伪装成数值"的口径残留。
- 修法：None 时显示"未测量"；或直接删除该废弃脚本。

### 💡-1 mock 的 `_mock_response` 硬编码 "generate_report" 计划动作，不在 TOOLS_DESC/golden_tools 内
- 位置：`src/agent/llm/llm_interface.py:211-218`（证据）→ 影响 `scripts/eval_agent_llm.py` mock 管道
- 证据：本次 mock 实测（--limit 2）两条记录的 `planned_tools` 均含 `generate_report`（如 `["extract_r_peaks","classify_arrhythmia","generate_report"]`），而 `generate_report` 既不在 `TOOLS_DESC` 也不在 `golden_tools`，导致 mock 模式工具选择指标含幻影 FP（实测 precision=0.5 部分由此产生）。真实 LLM 因不广告该工具不会出现此问题。
- 修法：让 mock 占位计划的动作名与 `TOOLS_DESC` 一致（去掉 generate_report）。

### 💡-2 计划动作用精确字符串集合比较，大小写/空白/中文译名差异会误计 FP/FN
- 位置：`scripts/eval_agent_llm.py:529-535`（`set(r["planned_tools"])` vs `set(r["golden_tools"])`）
- 证据：工具选择 P/R 依赖 LLM 输出的 `action` 字符串与 `TOOLS_DESC` 中的蛇形名逐字符相等；LLM 若输出 `Extract_R_Peaks`、`extract_r_peaks `（尾空格）或中文译名，会同时产生 FP 与 FN。属评测鲁棒性风险，非正确性错误。
- 修法：对 action 做规范化（strip/小写/别名映射到工具 id）后再比较。

## 阳性结论（实测验证正确的性质）

1. **三臂注入公平性成立**：注入决策 `self.rng.rand()` 仅此一处消费（`eval_agent_llm.py:103,271`），`np.random.RandomState(42)` + 确定性工具 + 同 evalset 顺序 → 三臂（各自独立进程）注入记录集合一致；注入发生在规划之后、报告之前，规划阶段三臂输入完全一致。
2. **无 GT 泄漏进 agent 推理提示**：规划/反射/报告 prompt 只含 `patient_info + TOOLS_DESC + feature_summary`（工具输出），`dx_names`（GT）仅用于 `top5_hit` 与 `RubricJudge.score`，`report_text` 仅报告臂注入、`kb_text` 仅知识库臂注入，均按设计。
3. **注入前快照口径正确**：`orig_top5_names` 在注入前捕获（第 259-260 行），`top5_hit` 用快照计算（第 452-454 行），注入不污染发现层 Top-5；`_inject_dx` 同步改写 name/prob/positive/snomed 四字段（第 237-250 行）并重建 `positives`（第 289-297 行），注入条目内部一致。
4. **缓存键正确**：`ResponseCache._key` 含 消息哈希+真实温度+json_mode+模型（第 77-80 行）；planning/reflection 三臂 key 相同可共享缓存，report 请求因 report_text/kb_text 不同而隔离；`eval_agent_llm.py` mock 模式不读不写缓存（第 159-162 行）。
5. **工具失败显式透传**：r_peaks/qt 的 None/insufficient 透传（第 194/202-206/213-216 行），`round(None)` 崩溃已防；HRV 空 RR（<3）→ None 由 `hrv_analyzer.py:47-49` 守卫，不会伪造 sdnn 数值。
6. **幻觉排除规则生效**：范围/百分比（尾随 `-…/%`）与参考值（前 10 字符含"平均/正常/参考…"）排除实测有效；工具值缺失仍声明数值计幻觉（第 442-443 行）。
7. **判官失败口径正确**：`RubricJudge.score` 解析失败返回 None → 调用方跳过并计 `judge_failures`（`eval_agent_llm.py:484-487`），不把 LLM 故障记 1 分；`score` 内部 1500 截断（`rubric_judge.py:53`），`rejudge_with_rubric.py` 传全文与在线评测输入口径一致。
8. **端到端 mock 管道实测跑通**：`--mock --limit 2` 成功加载分类器/工具、处理 2 条记录、落盘 `metrics_main_mock.json`/`records_main_mock.json`（产物 `_mock` 后缀隔离、mock 不写缓存）；注入逻辑（HR 60-100 区间不注入，`n_injected=0`）、工具值、top5 命中均正确。
