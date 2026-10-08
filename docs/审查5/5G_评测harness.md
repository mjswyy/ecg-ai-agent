# 第五次全面代码审查 — 5G_评测harness

## 审查人说明

- 审查员：5G（范围：评测 harness）
- 独立性声明：本次审查从零开始，未读取任何历史审查产物（检查报告 1–4、docs/审查*、docs/代码审查报告.md）；代码注释中的历史修复说明按"代码一部分"处理。
- 范围文件（逐行通读，非抽样）：
  - `scripts/eval_agent_llm.py`（670 行）、`scripts/eval_agent_context.py`（352 行）、`scripts/eval_agent.py`（231 行）
  - `scripts/build_agent_evalset.py`（208 行）、`scripts/build_context_subset.py`（55 行）
  - `scripts/recompute_agent_metrics.py`（63 行）、`scripts/recompute_context_metrics.py`（23 行）、`scripts/recompute_hallucinations.py`（139 行）
  - `scripts/rejudge_with_rubric.py`（87 行）、`scripts/analyze_context_inconsist.py`（25 行）
- 交叉验证只读文件（非审查对象，用于核实口径）：`src/data_pipeline/label_extractor.py`、`src/agent/tools/ecgfounder_classifier.py`、`src/agent/llm/llm_interface.py`、`src/agent/llm/rubric_judge.py`、`scripts/relabel_official_27.py`。
- 验证方法：只读 Python 命令（加载 evalset/records/metrics 并核对）；正则单元测试；`--mock --limit 2` 冒烟（mock 不花 token）。未修改任何源代码，未调用真实 LLM API。
- 中性背景事实核对结果（供参考，非发现）：
  - evalset 130 条（docstring 目标 ~150–200，实际 130）；27 类全覆盖；`dx_names` 全部落在 27 官方类名内；golden 工具分布 classify=130 / r_peaks=101 / hrv=74 / qt=13。
  - source 分布：ptb-xl 58 / georgia 47 / cpsc_2018 11 / cpsc_2018_extra 13 / incart 1；report_text 仅 ptb-xl 58 条非空。
  - 磁盘 `outputs/agent_eval/records_{main,report_arm,kb_arm}.jsonl` 各 130 条，**均无 `record_id` 字段**（记录键为 planned_tools/golden_tools/…/top5_hit），即 4G-YELLOW-10（补 record_id）之前的旧产物；`outputs/context_eval/records.jsonl` 40 条**均有 record_id**。三臂注入记录集合完全一致（index 4/14/15/32/40/71/98，共 7 条，7/130≈5.4%）。
  - 情境评测：40 条 × 3 情境，`n_abnormal=15`（与 build_context_subset 的 serious=15 一致）。

## 统计

| 严重度 | 数量 |
|---|---|
| 🔴 严重 | 0 |
| 🟠 中等 | 1 |
| 🟡 轻微 | 3 |
| 💡 建议 | 8 |
| 合计 | 12 |

## 发现清单

### 🟠-1 幻觉检测存在结构性漏报：QRS时限 与 电轴 两类模式匹配不完整，实测 2 例编造数值漏检

- 位置：`scripts/eval_agent_llm.py:430-431`（claim_patterns 的 QRS/电轴 两条）、`scripts/eval_agent_llm.py:582`（n_claims 分母同款正则）；`scripts/recompute_hallucinations.py:46-47`（同款 CLAIM_PATTERNS，与在线口径一致故同样漏报）。
- 证据（实测）：
  1. 正则单元测试：`QRS时限 120ms`、`QRS 时限 120ms`、`QRS时限120` 均**不命中** `QRS[时限宽]?[为是]?\s*[:：]?\s*(\d{2,3})`——`[时限宽]?` 只允许 QRS 后**单个**字符，而规范写法"时限"是 2 个字符，故"QRS时限 120ms"漏报（只有 `QRS 120ms`、`QRS宽 130ms` 能命中）。
  2. 正则单元测试：`电轴左偏 -30°`、`电轴右偏 +110°` 均**不命中** `电轴[为是]?\s*[:：]?\s*[-+]?\s*(\d{1,3})`——"左偏/右偏"夹在"电轴"与数字之间（`电轴: -30°`、`电轴 -30度`、`心电轴 +60°` 能命中）。
  3. 实际报告扫描：main 臂存在 **1 例** `QRS 时限 100`（工具不测 QRS，属编造数值）与 **1 例** `电轴左偏 -45`（工具不测电轴，属编造数值），均被现有 8 类模式**漏检**，未进入 `hallucinated_claims`。
- 影响：PR/QRS/电轴 三类"工具无测量、声明数值即幻觉"的扩展（3D 💡-1）实际只对部分写法生效；这两条漏报使三臂 `n_hallucinated` 被低估（当前 metrics 显示 0/99、0/120、0/135，即幻觉率恒 0.0，但真实至少 2 例编造值漏检）。因分子分母用同一正则，**率值本身不越界**，但绝对计数（n_hallucinated）被低估，幻觉率 0.0 的结论不严谨。
- 修法：把 `QRS[时限宽]?` 改为 `QRS(?:时限|宽度|宽)?`（覆盖 2 字"时限"）；把电轴模式改为 `电轴(?:左偏|右偏)?[为是]?\s*[:：]?\s*[-+]?\s*(\d{1,3})`。分母 `n_claims`（`eval_agent_llm.py:582`）与 `recompute_hallucinations.py:46-47` 同步修改，保持分子分母同口径。

### 🟡-1 `recompute_agent_metrics.py` 仍按位置索引对齐，未按 record_id（与 4G-YELLOW-10 修复口径不一致）

- 位置：`scripts/recompute_agent_metrics.py:23-39`（`assert len(items)==len(records)` + `zip(items, records)` 位置对齐）。
- 证据：`for i, (item, r) in enumerate(zip(items, records))` 用评估集位置 `i` 对应记录位置 `i`，对 `r.get("injected")` 的记录用 `item["signal_file"]` 重跑分类器并覆写 `r["classifier_top5"]`、重算 `r["top5_hit"]`。而 `eval_agent_llm.py:401`（4G-YELLOW-10）与 `recompute_hallucinations.py:102-109` 均已改为按 `record_id` 对齐。该脚本是唯一仍用位置对齐的后置重算脚本。
- 影响：一旦 records 产物带 record_id 且顺序与 evalset 不一致（并行运行、记录被丢/重排、未来 limit 非从头切片），`classifier_top5` 与 `top5_hit` 会用**错误的信号**重判，污染发现层 Top-5 与命中口径；且 `assert` 只能拦"条数不等"，拦不住"顺序错位"。当前旧产物（无 record_id）下位置对齐是唯一可行路径，故未在现状数据中显形，属隐患。
- 修法：与 `recompute_hallucinations.py` 一致，改按 `ev_by_id = {it["record_id"]: it}` 用 `r.get("record_id")` 对齐，缺失时告警跳过；并对 `report_arm`/`kb_arm` 两臂一并重算（当前只处理 main）。

### 🟡-2 `eval_agent_context.py` 的 `abnormal_ed_urgent_rate` 分子分母不对齐（ed 情境解析失败但其余情境成功时被计进分母）

- 位置：`scripts/eval_agent_context.py:283-286`。
- 证据：`n_abnormal += 1` 只要求 `gt & SERIOUS_LABELS` 非空（且该记录有 ≥2 个 valid 情境，见 `:248-252` 的 `len(valid)<2 → continue`）；而分子 `abnormal_ed_urgent += 1` 额外要求 `"ed_chest_pain" in u and u["ed_chest_pain"] in ("urgent","emergent")`。若某严重记录 ed_chest_pain 情境解析失败（返回 parse_error 或 urgency=None），但 routine+af 两个情境成功，该记录仍计进 `n_abnormal` 分母，却不进分子 → 率被系统性低估。
- 影响：分母口径应为"严重疾病且 ed 情境有效"的记录，实际是"严重疾病且 ≥2 个任意情境有效"。当前数据 `parse_failures=0`、`ed 缺失=0`（实测），故未显形；属口径隐患。
- 修法：`n_abnormal += 1` 与 `abnormal_ed_urgent += 1` 同置于 `"ed_chest_pain" in u and u["ed_chest_pain"] is not None` 分支内，使分子分母同口径（ed 情境无效的严重记录从分母剔除或单列统计）。

### 🟡-3 `eval_agent_context.py` 的 `ask()` 缺空响应重试与"空结果不写缓存"守卫（与 `eval_agent_llm.ask` 的 2026-08-29 修复不一致）

- 位置：`scripts/eval_agent_context.py:111-113`。
- 证据：`raw = self.llm.chat(...)` 后**无条件** `self.cache.put(messages, t, raw, model)` 再返回；而 `eval_agent_llm.py:172-181` 对空响应重试一次、两次皆空返回空串且**不写缓存**（该修复源于"deepseek-v4-flash 对长 prompt 返回空 content 导致 main 臂 60/130 条空报告"）。情境评测的 prompt 同样很长（feature_summary + 3 情境要求），暴露在同类故障下。
- 影响：空响应会被写入 `context_cache.jsonl` 并永久命中 → 对应记录/情境 parse_error 固化，且无重试机会。当前 `context_eval/records.jsonl` 显示 parse_failures=0，说明本次真实运行未触发；属与主评测 harness 不对称的隐患。
- 修法：与 `eval_agent_llm.ask` 对齐——空响应重试一次；两次皆空返回空串且不写缓存（下游 json.loads 落到 parse_error 分支即可）。

### 💡-1 `recompute_hallucinations.py` 存 `tool_values` 用原始值，与 `eval_agent_llm` 存的 round(…,1) 值不一致

- 位置：`scripts/recompute_hallucinations.py:69-89`（`tv["hr"] = r.get("heart_rate")` 等取原始值）对比 `scripts/eval_agent_llm.py:192/458`（`hr = round(float(...),1)` 存入 tool_values）。
- 证据：在线评测 `rec["tool_values"]` 存的是 `round(…,1)`（如 74.5），重算脚本覆写为原始值（如 74.53），两处口径不一致；`scan_claims` 内 `round(tool_val)` 与 `abs(claimed-round(tool_val))>5` 的 ±5 容差可吸收 ≤1 的四舍五入差，故**指标结果基本不受影响**，仅持久化字段与幻觉提示串里的"实际{tool_val}"显示值有出入。
- 修法：重算时与在线一致地 `round(…,1)` 后再入 `tool_values`，保持字段语义统一。

### 💡-2 rubric 判官全失败时，`aggregate()` 返回无维度字段的 dict，调用方取 `["total"]["mean"]` 会 KeyError

- 位置：`src/agent/llm/rubric_judge.py:140-141`（`if not valid: return out`，`out` 只有 `n`/`judge_failures`）＋调用方 `scripts/eval_agent_llm.py:503-507`（`agg["total"]["mean"]`）、`scripts/rejudge_with_rubric.py:78-82`（`agg["correctness"]["mean"]`）。
- 证据：`aggregate([None, None])` 返回 `{"n":0,"judge_failures":2}`（无 correctness/completeness/grounding/total），随后 `agg["total"]["mean"]` 抛 KeyError。`eval_agent_llm.py` 的兜底 `agg = rub_agg(scores) if scores else {...}` 只防"scores 空列表"，防不住"非空但全 None"（`[None]` 为真值 → 进入 rub_agg → 返回无字段 dict）。
- 影响：仅当某臂判官**全部**调用失败（LLM 全故障）时触发；概率低但会中断评测而非记为 judge_failures=全部。
- 修法：调用方改用 `agg.get("total", {}).get("mean")` 或先判 `agg.get("n")==0` 再取均值。

### 💡-3 `ContextCache` 缓存 key 不含 `json_mode`（与 `ResponseCache` 口径不一致）

- 位置：`scripts/eval_agent_context.py:70-82`。
- 证据：`ContextCache.get/put` 的 key 只含 `messages + t + model`；`eval_agent_llm.ResponseCache._key`（`:79-82`）额外含 `json_mode`。当前情境评测 `ask(json_mode=True)` 恒真，故无实际碰撞；若未来某调用传 `json_mode=False`，将命中同 key 的 json 响应。
- 修法：key 中补 `json_mode` 分量，两处缓存口径统一。

### 💡-4 情境评测 findings 仅靠 prompt 约束"只能从 classifier_positives 选取"，无程序校验

- 位置：`scripts/eval_agent_context.py:171`（prompt 约束）对比 `:211-215`（`get_findings` 直接取 `f["name"]`，不校验 ∈ classifier_positives）。
- 证据：`get_findings` 对 findings 列表中任意 dict 的 name 一律入集，无"是否属于分类器阳性集"的过滤。LLM 若在 findings 层编造一个不在 classifier_positives 的诊断名，该名字会进入 `f_sets`，参与 `findings_consistency_rate` 与 `findings_gt_overlap`，且不会像主评测那样被当作幻觉计数。
- 影响：发现层"必须由工具锚定"这一临床诚实原则未被执行层兜底；风险层/建议层的编造同理未被检测。
- 修法：`get_findings` 增加可选白名单参数（classifier_positives），对不在集合内的 findings 名标记/剔除，并可在 metrics 中单列"越界 findings 数"。

### 💡-5 report_arm 的幻觉校验只对照工具值，忽略"医生报告"作为合法依据来源（跨臂可比性风险）

- 位置：`scripts/eval_agent_llm.py:386`（report 臂注入医生原始报告）＋`:414-455`（幻觉校验只对 `tools_out` 的 hr/qtc/qt/sdnn 验证）。
- 证据：report_arm 的报告生成额外注入 `item['report_text']`（医生原始报告，含其自身测量的心率/QTc 等数值）；但幻觉校验的 `tool_val` 只来自确定性工具。若 LLM 忠实引用医生报告中的数值（与算法工具值差 >5），会被计为"幻觉"，尽管其有提供材料作为依据；main/kb 臂无此信息源，故 report_arm 的幻觉率口径与其他臂不完全可比。
- 影响：当前 report_arm `n_hallucinated=0` 未触发；属"注入公平性/跨臂可比性"口径说明，建议在报告中明确 report_arm 幻觉判定仅对工具值锚定（医生报告数值差异另行标注）。
- 修法：至少在 docstring/metrics 注明 report_arm 幻觉口径的额外信息源；若要严格可比，可把医生报告中的数值声明单列一档"报告臂附加数值"而非与工具值幻觉混计。

### 💡-6 注入捕获 `_report_flags_conflict` 口径较窄：需同句内同时出现矛盾词与快/慢诊断词

- 位置：`scripts/eval_agent_llm.py:514-523`。
- 证据：`CONFLICT_MARK = 矛盾|不一致|不符|存疑|冲突`，`RATE_WORDS = 心动过速|心动过缓|窦速|窦缓|Sinus Tachycardia|Sinus Bradycardia`，仅当二者在**同一句**（按 。；;\n 切分）共现才判捕获。实测 main 臂 7 条注入记录 3 条命中（0.4286），report/kb 臂 5 条（0.7143），对照组冲突标记率 0.0163~0.0244——判别力成立；但跨句表达、或无显式矛盾词（如"心率 55 与窦速诊断不相符"若"不相符"未入词表）会漏判。
- 影响：`catch_rate_report` 是注入纠错主指标，偏窄的正则可能低估真实捕获率（下限）。
- 修法：扩充矛盾词表（如"不相符/矛盾"等），或允许相邻两句内共现；同时保留当前口径作为可复现基线。

### 💡-7 `_inject_dx` 后的 positives 重建未去重（注入名若已在 positives 中会产生重复条目）

- 位置：`scripts/eval_agent_llm.py:295-303`。
- 证据：`tools_out["classifier"]["positives"] = [d for d in pos_list if d.get("name") != top1] + [{...injected_name...}]`。若注入名（如 hr<60 时注入的 "Sinus Tachycardia"）本就已在 positives（分类器对慢心率记录给窦速阳性概率≥阈值的罕见情形），剔除的是 top1、追加的是同名条目，得到重复的 "Sinus Tachycardia"。该 positives 用于 kb_excerpts 检索与 feature_summary 展示。
- 影响：仅在慢心率记录原本就窦速阳性的罕见组合下出现重复；影响 kb 臂上下文与报告输入的整洁性。
- 修法：追加前按 name 去重，或仅在 name 不存在时追加。

### 💡-8 `eval_agent.py` 已废弃但含硬编码对比基线（若被误用会误导）

- 位置：`scripts/eval_agent.py:1`（DEPRECATED 标记）、`:229-231`。
- 证据：`:229-231` 硬编码 `Model Top-1: 70.7%`、`Model Top-5: 93.1%` 与 `gap = top1_hits/total - 0.707`，对应的是旧自训集成，与现役 ECGFounder 分类器（test macro_auc 0.9456）无关。文件头已注明废弃（指向 eval_agent_llm.py），但脚本无运行时拦截。
- 影响：无（现役评测不走此脚本）；仅为防误用建议。
- 修法：可加 `raise SystemExit` 或 `assert False, "deprecated"` 阻止误跑，或直接删除。

## 阳性结论

1. **三臂注入公平性成立（实测）**：main/report_arm/kb_arm 三臂 `records_*.jsonl` 的注入记录 index 完全一致（4/14/15/32/40/71/98，各 7 条），`n_injected=7`、7/130≈5.4%；注入逻辑无 `use_report/use_kb` 分支条件，三臂同 seed 同序 → 注入集合一致，跨臂可比。
2. **规划/反射 prompt 三臂完全一致，共享缓存**：`tool_selection` 三臂完全相同（precision 0.5991 / recall 0.8459 / exact_match 1）；report_text 仅注入报告生成阶段（`:386`），kb_text 仅注入报告阶段（`:387`），均不进入 planner/reflector。
3. **GT 无泄漏**：规划 prompt 只含情境卡（age/sex/setting/chief_complaint/history），不含 `dx_names`/`report_text`；发现层 Top-5 指标用注入前快照 `orig_top5_names`（`:261-262,410`），`top5_accuracy` 三臂同为 0.9308，注入不污染发现层。
4. **幻觉率分子分母同口径、率值域正确**：8 类模式（心率/HR/QTc/QT/SDNN/PR/QRS/电轴）在 `eval_one`（`:420-432`）与 `aggregate_metrics` 分母（`:579-583`）逐字一致，分母同做范围/百分比/参考值排除，`rate=n_hallu/n_claims` 恒 ∈[0,1]（4G-RED-1 修复到位）；实测三臂 0/99、0/120、0/135。
5. **缓存键正确、mock 隔离有效**：`ResponseCache._key` 含消息哈希+真实温度(0.3)+json_mode+模型名（与 `llm_interface.temperature=0.3` 及 chat 采样温度一致）；mock 模式不读不写缓存。实测 `--mock --limit 2` 冒烟运行后 `llm_cache.jsonl` 未被污染，mock 产物落 `*_mock` 后缀。
6. **注入纠错主指标判别力成立（实测）**：`catch_rate_report` 0.4286(main)/0.7143(report/kb) vs `conflict_flag_rate_control` 0.0163~0.0244；verdict 对照组基率高（revise_rate_control≈0.3577）代码已注明"仅作参考不作主指标"（`:533`）。
7. **27 类名与排除表全部真实存在（实测核对）**：注入排除表（Sinus Bradycardia/Bradycardia/Sinus Tachycardia）、SERIOUS_LABELS、GROUPS（serious/blocks/normal）全部落在 `LabelExtractor` 的 27 官方类名内；evalset `dx_names` 无一越界，27 类全覆盖；`labels`（27 维）与 `dx_names`（由 dx_codes 映射）同源一致（relabel_official_27 用 `le.encode(dx_codes)` 生成 labels）。
8. **情境 findings_gt_overlap 分母含全部 valid 情境**：`eval_agent_context.py:268-272` 对 GT 非空记录遍历**全部** valid 情境、空 findings 计 0（4G-ORANGE-3 口径），率 ∈[0,1]（实测 0.6457）；parse_error dict 被 `"parse_error" not in v` 排除（3D-Y4），不再当空集参与比较。
9. **mock 管道端到端可跑**：`eval_agent_llm.py --mock --limit 2` 与 `eval_agent_context.py --mock --limit 2` 均成功加载工具、跑 2 条记录、产出指标并隔离写盘（实测 n_records=2 等输出正常）。
10. **两处后置重算脚本共用 `aggregate_metrics` 单一口径**：`recompute_agent_metrics.py`、`recompute_hallucinations.py` 均 `from eval_agent_llm import aggregate_metrics`，并透传旧 metrics 的 `judge` 结果（2E-O5），避免判官分数被静默抹掉。
