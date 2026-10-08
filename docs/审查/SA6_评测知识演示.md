# Agent评测与知识脚本审查报告

审查范围：scripts/ 下 18 个评测/知识库/演示脚本 + web_demo/app.py + 4 个 configs，并逐一与 outputs/ 下实际产物、paper/main.md 论文数字交叉核对。
审查日期：2026（与论文工作期一致）。
说明：任务描述中的行数（如 eval_agent_llm.py 398 行）与实际文件不符，本报告所有行号均为**实际读取到的行号**。

---

## 总体评价

项目工程质量"两头分化"：**知识库流水线（crawl→distill→compile）整体健壮、合规意识强**（限速、来源分级、防幻觉约束、缓存断点续跑都做对了）；**但 Agent 五维评测 harness 存在多处直接影响论文数字成立性的缺陷**，且三臂评测产物明显来自不同代码版本，跨臂可比性存疑。

最需要警惕的三件事：
1. **"注入错误纠错率 100%" 是伪指标**——反射器提示词把"注入错误"四个字直接写进了规则告警并喂给 LLM，catch 判定又包含"告警文本含'注入'字样"这一条件，无论 LLM 表现如何恒为 100%（代码同义反复 + 答案泄漏）。
2. **工具选择 P=0.52 / R=0.76 被 golden 计划定义 bug 主导**——88/133 条记录的 golden 计划含 `plot_waveform`，但该工具根本不在 LLM 的可选工具清单里，recall=0.7609 恰好等于 280/(280+88)，完全由这个不可选工具决定；另有 18 条传导阻滞记录因集合用缩写（LBBB/RBBB/WPW…）导致 golden 计划漏掉 `extract_r_peaks`。
3. **论文 §4.3"三臂 rubric 总分 2.61/2.71/2.60"与 outputs/agent_eval/judge_rubric_results.json（仅存 main 臂，总分 3.02）和雷达图（主分支 3.02）三者互相矛盾**；报告辅助/知识库两臂的 rubric 分数在磁盘上无任何 json 来源，属于图内硬编码。

其余正确核对的数字（0.9214、幻觉率 0.34%、Top-5 97.0%、情境基准五项、跨源 AUC、MIMIC 域分析、知识库 350 来源）与代码/outputs 一致，详见文末核对表。

---

## 各文件逐条发现

### 1. scripts/build_agent_evalset.py

- 🔴 **39-40 行：CONDUCTION_CLASSES 使用缩写，永远匹配不上真实类名**。LabelExtractor 的 27 类名是 `Left Bundle Branch Block` / `Right Bundle Branch Block` / `Incomplete Right Bundle Branch Block` / `Left Anterior Fascicular Block` / `Wolff-Parkinson-White`，而集合里写的是 `LBBB / RBBB / IRBBB / LAFB / WPW`。经对 evalset.json 实测：18 条"纯传导阻滞"记录的 golden_tools 均缺 `extract_r_peaks`（如 WPW 记录 golden 只有 `classify_arrhythmia`）。→ 修法：改用 `LabelExtractor.class_names` 中的全名，或直接按 SNOMED 码/类别索引判断。
- 🟠 **71-83 行：golden 计划含 `plot_waveform`，但评测侧（eval_agent_llm.py:47-51）工具清单里没有该工具**。88/133 条记录 golden 含 plot_waveform（实测分布：classify 133 / extract 74 / hrv 62 / qt 11 / plot_waveform 88）。→ 修法：golden 生成器与 TOOLS_DESC 共用同一工具注册表，并在构建期断言 `golden ⊆ 工具空间`。
- 🟡 41 行：`QT_CLASSES` 中 `QT Prolongation`、`Prolonged QT Interval` 是死条目（真实类名仅 `QT Prolonged`）。
- 🟡 131 行：`json.load(open(...))` 未关闭文件句柄（同 99 行用 with 不一致）。

### 2. scripts/eval_agent.py（旧版端到端评测，非论文主评测）

- 🟠 **188 行："All tools successful" 名不副实**。`errors` 只在 R 峰检测失败时追加（74 行），HRV/QT 失败仅写 `steps[].ok=False`（84-94 行）不进 errors → 该指标只反映 R 峰成功率。
- 🟠 224-225 行：对比基线 "Model Top-1 70.7% / Top-5 93.1%" 硬编码（经核对与 outputs/comprehensive_eval.json 的"集成 Ensemble" top1=0.7074/top5=0.9312 一致，但硬编码有漂移风险）。
- 🟡 31 行：`load_one(..., dropout=0.3)` 与 model_config.yaml 的 dropout 0.1 不一致（eval 模式 dropout 不生效，无实质危害，属配置漂移）。
- 🟡 60 行：`ecg_id` 参数未使用；162-165 行 shape 兜底可能静默保留错误形状。

### 3. scripts/eval_agent_llm.py（五维评测 harness，核心）

- 🔴 **239-240 + 356-359 行：注入纠错率是"告警文本泄漏 + 判定同义反复"**。
  - 239-240 行：注入后规则告警被写入 `"分类器与特征测量不一致（注入错误）"`——"注入错误"四个字直接进入 prompt（246 行把 alerts 原样喂给 LLM 反射器）。
  - 356-359 行：`caught` 判定为 `verdict=="revise"` **或** `rule_alerts 含"注入"`。由于 239 行保证每个注入记录都含"注入"字样，`catch_rate` 恒为 1.0（实测 35/35=1.0；35 条注入记录 verdict 全部是 revise，正是被提示词剧透的结果）。
  - 结论：论文"注入错误纠错率 100%"（abstract、§4.3 35/35）不成立，是答案泄漏下的伪指标。→ 修法：告警文本改为中性描述（如"分类器与特征测量不一致"），caught 只以 `verdict=="revise"` 判定，且不可在反射提示词中给任何注入线索。
- 🔴 **47-51 行（TOOLS_DESC）与 golden 计划（build_agent_evalset.py:71-83）工具空间不一致**。`plot_waveform` 不在 TOOLS_DESC，88 条记录 golden 含它 → 每条恒产生 1 个 fn。实测 fn=88 全部来自 plot_waveform，R=280/(280+88)=0.7609 与 metrics_main.json 完全吻合。**工具选择 recall 是代码 bug 决定的，不是"LLM 计划偏全面"**（论文 §4.3 的解读不成立）。precision 也被"LLM 对每条记录都计划全部 4 个工具"（实测 planned 分布：extract/classify/hrv/qt 各 133 次）拉低到 0.5214。
- 🟠 **191-197 行：注入只发生在主分支（`not self.use_report`），report_arm 无注入**。论文"报告辅助对照"主 vs 报告臂的 rubric/幻觉比较混入了"主分支 20% 记录被注入错误、报告臂 0%"这一混淆变量，跨臂不可比。
- 🟠 **305-320 行幻觉检查的判定条件 `tool_val is None` 时跳过，但 362-369 行 n_claims 分母照常统计这些声明** → 不可验证的声明只进分母不进分子，rate 系统性偏低。另：±5 容差偏宽（心率差 4bpm 不算幻觉）。
- 🟠 **362-369 行：三臂 n_claims 严重失衡（main=587 / report_arm=0 / kb_arm=8）**，实测 report_arm 的报告为"### 结构化心电图诊断报告 / **心率信息：**"句式，正则 `心率[为是]?...` 全部落空。三臂产物明显由不同代码版本生成，**幻觉率、judge 分数跨臂不可比**，论文 §4.6"KB 分支声明 8 条全锚定"与 §4.3 三臂对照的成立性存疑。
- 🟠 128-138 行 + 453 行：缓存 key = 消息哈希 + temperature，**不含模型名与 json_mode**；换 `--model` 或切换 json_mode 会命中旧响应。且 `ask()` 会把 mock/异常响应也写入缓存——配合 llm_interface.py:160-162 的静默降级，一次 API 抖动会永久污染缓存。
- 🟠 18 行 docstring 声称 "temperature=0"，但 LLMInterface 默认 temperature=0.3（llm_interface.py:72），实际 API 调用为 0.3 → 可复现性声明与实际不符（agent_config.yaml 的 0.3 倒是"歪打正着"）。
- 🟡 322-325 行：Top-5 命中不受注入影响（交换只动 top1/top2 顺序，top5 集合不变）——0.9699 与原始 0.9775 的差距来自评估集抽样，非注入污染；但论文若把"发现层 Top-5"写成注入混合口径需注明。
- 🟡 379-397 行：`RubricJudge(self.llm)` 实例创建后未使用；`from ... import` 写在循环内（import 会提升，属风格问题）。
- 🟡 200 行：情境卡固定取 `context_cards[0]`（急诊胸痛），三张卡只用了第一张。

### 4. scripts/eval_agent_context.py（情境敏感性评测，核心）

- 🟠 **229-239 行：parse-error 字典被当成"空 findings"参与一致性判定**。229-231 行只在**三个**情境全部非 dict 时才计入 parse_fail 并跳过；单个情境 `{"parse_error": ...}`（169 行产物）是 truthy dict → 进入 f_sets 得空集 → 236 行 `all(s == nonempty[0] for s in f_sets.values())` 会把空集与非空集比较 → 误判"不一致"。且 233 行注释"空集视为一致"与 236 行实现矛盾（全空才一致，部分空判不一致）。本批次恰好未触发（36/36 一致），但属潜伏 bug。
- 🟠 **247-261 行：None 值虚增敏感性/风险变化率**。`u = {c: get_urgency(...)}` 中 parse 失败的上下文返回 None，252 行 `len(set(u.values())) > 1` 会把 None 当独立取值 → 一个情境解析失败即触发 sensitivity/risk_change（实测本次 0 个 None，未发作）。259-261 行 risk 同理。
- 🟠 252 行 + 16 行注释：docstring 称"异常记录上……是否变化"，实现却对所有记录计数。
- 🟠 242-245 行：`gt_total += len(gt) if gt else 1`——无真实标签（dx_names 空）的记录每上下文给分母 +1、分子 +0，系统性拉低 findings_gt_overlap。
- 🟡 20 行用法示例含 `--judge`，278-283 行 argparse 未定义该参数（运行会报 unrecognized arguments）。
- 🟡 266-273 行：三个率的统一分母 `n - parse_fail` 与"单情境 parse 失败仍计入 n"的判定不一致。

### 5. scripts/rejudge_with_rubric.py（rubric 重打分）

- 🟠 **47 行：判官缓存 key（rubric_judge.py:43-45，sha256(gt|report)[:20]）不含模型名/提示词版本** → 换模型或改 JUDGE_PROMPT 后重跑会命中旧分数（无失效机制）。
- 🟠 **64 行：对全部记录（含 20% 注入记录）打分，未剔除/标注 injected**；而 report_arm 无注入 → 三臂 judge 分数可比性受损（同发现 3）。
- 🟡 27-31 行：ARMS 字典硬编码文件名，与 eval_agent_llm.py 的 suffix 命名强耦合。
- ✅ 任务点 3 专项结论：eval 的 `--judge` 走 `ask()` → 缓存在 `llm_cache.jsonl`（ResponseCache，key=消息哈希）；rejudge 走 `RubricJudge.score` → 缓存在 `judge_rubric_cache.jsonl`（RubricCache，key=gt|report）。**两个缓存文件、两种 key，不会交叉污染**；但 judge_rubric_results.json 当前只含 main 臂，report_arm/kb_arm 缺失（见 make_figures 发现）。

### 6. scripts/verify_llm.py

- 🟠 97-100 行：mock 模式 `llm._client = None` 后照常调用 `llm.chat`——依赖 llm_interface.py:132-133 的静默回退，若该回退被移除则 mock 直接崩。
- 🟡 86-88 行：verdict 三态判定 OK（优先级经核实正确）；68-75 行 reasoner 仅验证长度>50，判定过弱（冒烟可接受）。

### 7. scripts/run_agent_demo.py

- 🟠 **59-65 行：分类模型"未训练即使用"**——`ArrhythmiaClassifier(inception_time(...))` 从不加载任何权重，日志仅提示 "untrained"，demo 的诊断结果（除 LLM 推理外）无意义。
- 🟠 34 行：默认模型 `deepseek-v4-pro` 与评测脚本/配置的 `deepseek-chat` 不一致（llm_interface.py:69 默认也是 deepseek-v4-pro，若该模型名不存在则 API 报错并触发静默 mock）。
- 🟡 77 行：`np.load` 后无 NaN/shape 校验；117-124 行对 `result.confidence/diagnosis[i]['confidence']/step['error']` 的取值依赖 agent 实现，缺失即 KeyError。

### 8. scripts/crawl_knowledge.py

- 🟠 **73-77 行：save_doc 非原子写**（直接 open("w")），中断会留半截 json；doc_id 冲突时后写覆盖前写（如同一类两次查询命中同一 PMID，或 ecgpedia 同页被多类抓到——实测 pubmed_40273320 同时存在于 atrial_fibrillation/ 与 atrial_flutter/，跨类目录各自独立缓存无碍，但同类内重复会覆盖）。
- 🟠 **无重试/退避**：esearch/efetch 网络异常（214-215、237-238 行）直接 warning 跳过该来源；NCBI 偶发 429/超时会静默少数据。
- 🟠 15 行 docstring 的 `--limit-sites` 参数在 316-320 行 argparse 中不存在（照抄会报错）。
- 🟡 177-192 行 wikitext 清理：正则均为无嵌套量词的安全模式，无灾难回溯/注入面（slugify 对文件名的清洗也安全）；但单 `=` 标题不转换、嵌套模板残留（影响有限）。
- 🟡 246 行 LITFL 仅 200 时抓取，403 静默；合规性（UA 声明、1 req/s 限速 70 行）做得对。

### 9. scripts/distill_knowledge.py

- 🟠 **182 行：main 循环未包 try**——`distill_one` 内 API 异常（非 JSONDecodeError，如网络/限流）直接抛出，整个流程中断；只有 JSONDecodeError 才重试一次（126-130 行）。
- 🟠 **91-93 行 doc_key（内容哈希）是死代码**：179 行按 `doc_id` 去重（106 行 done 集合）。doc_id 是来源标识（pubmed_pmid / ecgpedia_slug），若重爬更新了文档内容，旧提炼条目会永久保留（缓存失效问题，任务点 5）。
- 🟡 136 行 schema 校验只查 `condition` 键，不校验 ecg_criteria 等字段类型（compile 侧 70-71 行做了 str/list 兜底，尚可）。
- 🟡 84 行：`len(text)>=150` 过滤短文档（合理但会把高质量短摘要滤掉）。

### 10. scripts/compile_knowledge.py

- 🟠 **94-98 行：definition 选择违反"grade B 优先"设计**——按 grade 升序迭代但取全局最长，grade C 的更长的定义会覆盖 grade B。→ 改为按 grade 分组取最长。
- 🟠 106/108 行：measurements/clinical_significance 只保留 `text`，**丢弃 citations**，与 14 行"保留全部引用清单"矛盾（to_markdown 对这两类渲染为无引用条目）。
- 🟡 46-55 行：`merge_unique` 定义后从未调用（死代码，去重实际在 join_list 内完成）。
- 🟡 42-43 行 norm 只去空白/小写/首尾破折号，"ST elevation" 与 "ST elevation." 不去重（标点残留）；118-145 行 markdown 不对文本做转义，含换行/`|` 的 LLM 输出会破坏列表渲染。
- 🟡 202 行：master.md 总论标题取 `_general_`（crawl 写入的 class 字段），展示不友好。

### 11. scripts/build_context_subset.py

- 🟡 13-20 行：GROUPS 与 eval_agent_context.py:48-50 的 SERIOUS_LABELS 重复定义（单点真源缺失）；实测 context_subset.json 实际 36 条（当前脚本会生成 40 条）——脚本与产物版本漂移。
- 🟡 27 行 `for/else` 分组逻辑正确，但 "normal" 组仅按 `Sinus Rhythm` 判定，窦律+其他异常记录会落入 first-match 组而非 normal。

### 12. scripts/recompute_context_metrics.py

- 🟡 17 行：print 硬编码 "36 条 × 3 情境"，与实际 n 解耦；14 行 `ContextEvaluator.__new__` 绕过 __init__ 取 aggregate（能跑，但依赖 aggregate 不碰 self 的实现细节）。
- 🟠（联动）该脚本用**当前代码**的 aggregate 重算历史记录——若 aggregate 有 bug（见 eval_agent_context 发现），重算只是把 bug 复算一遍。

### 13. scripts/analyze_context_inconsist.py

- 🟡 15 行：`f.get("name")` 可能为 None 混入集合；纯诊断脚本，无大碍。

### 14. scripts/audit_m0_m1.py / 15. scripts/audit_supplement.py

- 🟡 audit_m0_m1.py:34 行条件表达式嵌套可读性差；37 行 `json.load(open())` 未关闭；40 行对缺失键直接 KeyError。
- 🟡 audit_supplement.py:17 行 `[0]` 越界风险（E09680 不在旧 manifest 时 IndexError）。

### 16. scripts/make_figures.py（论文四图）

- 🟠 **72-76 行：report_arm / kb_arm 的 rubric 四维分数（2.414/2.150/4.008/2.707 与 2.301/2.075/3.850/2.602）硬编码，outputs/agent_eval/judge_rubric_results.json 中不存在对应数据**（该文件仅 main 臂）→ 图内数字不可溯源（任务点 9）。
- 🟠 33-43 行：成绩单 AUC 全部硬编码；其中 0.791/0.808/0.822 与 comprehensive_eval.json 吻合、0.905/0.9214 与 ecgfounder_mlp/metrics.json 吻合，但 **0.590（InceptionTime）、0.841（SimCLR）、0.850（3模型集成）在 outputs/ 下找不到来源 json**。
- 🟠 80 行：五维能力值硬编码（0.619/1.0/0.997/0.970/3.023÷5）；"纠错率 1.0"即第 3 节伪指标的复现；"0.997/0.970"与 metrics_main.json 一致。
- 🟡 104/156/193 行：n=133、n=36、KB 规模（"12-49 条"）硬编码——实测 ecg_criteria 范围是 **10-49**（paper/main.md:64 的 "12-49" 偏小）；"每类 6-13 来源"实测 6-13 ✓。

### 17. scripts/smoke_demo.py / 18. scripts/smoke_ecgfounder_tool.py

- 🟡 smoke_demo.py:13-15 行：record_id="HR13177" 硬编码（该记录若不在 manifest 则直接报错退出）；24 行 `fig` 可能为 None。
- ✅ smoke_ecgfounder_tool.py 无实质问题。

### 19. web_demo/app.py（演示主程序）

- 🔴 **78 行 vs 259 行：测试集下拉框主路径必然失败**。259 行下拉选项值 = `"source/record_id"`（如 `"ptb-xl/HR13177"`），78 行却用 `m["record_id"] == record_id` 匹配（manifest 中 record_id 只是 `"HR13177"`，实测 test_manifest.json 首条 record_id="A0017"）→ `next()` 恒为 None → GUI 上"从测试集选择"永远提示"记录 xxx 不存在"。smoke_demo.py 能跑是因为直接传了纯 record_id。
- 🟠 108-118 行：`rpeak.detect`（108 行）与 `classifier.predict`（118 行）不在 try 内，坏输入/模型加载失败直接抛给 gradio；96-100 行上传预处理链也无异常兜底。
- 🟠 160-163 行：LLM 降级只覆盖"无 key"（123 行），**API 调用异常不降级**（网络错误直接崩页面）；结合 llm_interface.py 的静默 mock，用户可能拿到占位回复而不自知。
- 🟠 91-92 行：上传文件用 `Path.replace()` 移入临时目录（可行但依赖 gradio 临时文件可移动）；.hea 由用户提供，wfdb 按 header 定位数据文件，恶意 .hea 可诱导读取指定路径（本地演示工具风险低，仍建议校验 sig_name/记录名）。上传无大小/时长上限。
- 🟡 210/265/273 行："test macro AUC 0.9214""6,472 条"硬编码（当前与 manifest 一致，但属脆性耦合）。
- ✅ 44 行 `import scripts.preprocess_data_5k as p5` 无模块级副作用（仅 logging/常量/函数），安全；上传后临时目录 with 块自动清理，临时文件处理基本合格。

### 20-23. configs（YAML 与代码默认值一致性，任务点 8）

- 🟠 agent_config.yaml:28 `qt_plausible_range: [200,600]` 与 eval_agent_llm.py:179 `not (200 <= qtc <= 700)` 不一致（600 vs 700）。
- 🟠 agent_config.yaml:6 `temperature: 0.3`：与代码实际行为一致，但与 eval 脚本 docstring "temperature=0"（eval_agent_llm.py:18）矛盾。
- 🟡 agent_config.yaml:35 / data_config.yaml:23 `target_length: 4096`：processed_5k 实际为 5000 样本（10s@500Hz），配置过期。
- 🟡 agent_config.yaml:20 `classifier.threshold: 0.3`：评测/演示实际用 metrics.json 的逐类 Youden 阈值（ecgfounder_classifier.py:68-75），该配置项无人读取（死配置）。
- 🟡 model_config.yaml:8/17 dropout 0.1 vs eval_agent.py:31 的 0.3；train_config.yaml:46-48 OBS bucket 为占位符（模板性质，可接受）。

### 关联依赖源码（影响上述结论的核实点）

- **src/agent/llm/llm_interface.py:160-162**：`chat()` 捕获所有异常并返回 mock——评测中任何 API 故障都会**静默混入假回复并被缓存**；:179-181 `is_available` 只查 client 是否非空，空 key 也返回 True → eval 脚本的"无 key 退出"防线形同虚设。这是所有 LLM 评测脚本共用的隐患。
- **src/agent/tools/ecgfounder_classifier.py:115**：`positives` 只从 top_k（默认 5）中取——超过 5 个阳性类别时发现层被截断。
- **src/knowledge/kb_loader.py:41**：双向子串匹配可能误中（如查询 "Sinus" 命中 "Sinus Arrhythmia"），对精确类名场景影响小。
- **src/agent/llm/rubric_judge.py:64**：`int(scores.get(field,1))` 遇 "4.5" 字符串抛 ValueError 落入全 1 兜底（评分被低估的边界）。

---

## 修复优先级清单

| 优先级 | 事项 | 影响 |
|---|---|---|
| P0 | 修复注入纠错率伪指标（eval_agent_llm.py:239-240/356-359）：告警中性化、去掉"注入"字样、caught 仅看 verdict | 论文"纠错率 100%"需重测 |
| P0 | 统一工具空间：golden 生成器与 TOOLS_DESC 同源，去掉 plot_waveform 不可选矛盾；CONDUCTION_CLASSES 换全名 | 工具选择 P/R 需重测；"LLM 计划偏全面"结论需改写 |
| P0 | web_demo/app.py:78：下拉值拆分为 source/record_id 再匹配 | 演示主入口当前不可用 |
| P0 | llm_interface.py:160-162：评测模式禁用静默 mock（或打 `mock_fallback` 标记并排除出指标），is_available 校验 key 非空 | 防止假回复混入论文数字 |
| P1 | 三臂数据溯源：重新用同一代码版本跑 report/kb 臂；补齐 judge_rubric_results.json 三臂；修正 paper §4.3 "2.61" 与 json "3.02" 的矛盾 | 三臂对照/图 2 数字可信度 |
| P1 | eval_agent_context.py 空集/None 处理：parse 失败的上下文剔除出 f_sets/u/rk；敏感度仅对异常记录计数 | 情境基准鲁棒性（当前数字未受影响） |
| P1 | 缓存 key 加模型名+提示词版本；ask() 不缓存异常/mock 响应 | 可复现性 |
| P2 | crawl 重试/退避 + 原子写；distill main 循环 try/except；compile definition 按 grade 取最长、measurements/significance 保留引用 | 流水线健壮性 |
| P2 | 幻觉率分母只统计可验证声明；三臂口径统一 | 幻觉率跨臂可比性 |
| P3 | 配置漂移（qt 600/700、target_length 4096/5000、temperature 0/0.3）；make_figures 从 json 读数而非硬编码 | 可维护性 |

---

## 论文数字一致性核对表

| 论文引用（paper/main.md） | 代码/产物来源 | 一致性 |
|---|---|---|
| 发现层 ECGFounder+MLP macro AUC **0.9214**（摘要/L85） | outputs/ecgfounder_mlp/metrics.json:4、per_source_auc.json overall=0.9214 | ✅ 一致 |
| 成绩单 0.791/0.808/0.822/0.905/0.9214 | comprehensive_eval.json（0.7908/0.8082/0.8224）、ecgfounder_mlp/metrics.json（0.905/0.9214） | ✅ 一致 |
| 成绩单 0.590（InceptionTime）/0.841（SimCLR）/0.850（3模型集成） | outputs/ 下未见来源 json（make_figures.py:34-43 硬编码） | ⚠️ 无法溯源 |
| 工具选择 P=0.52 / R=0.76（L102） | metrics_main.json:4-5（0.5214/0.7609） | ⚠️ 数字吻合，但**指标被 plot_waveform 不可选 + 全计划四工具主导，结论不成立**（🔴） |
| 注入纠错 **35/35=100%**（L103） | metrics_main.json:8-9 catch_rate=1.0；records_main.jsonl 实测 35 条全 revise、全带"注入"告警 | ⚠️ **伪指标**：告警文本泄漏答案 + 判定同义反复（🔴） |
| 幻觉率 **0.34%**（587 声明中 2 条）（L104） | metrics_main.json:12-14；独立重算 n_claims=587 / n_hallu=2 ✓ | ✅ 数值一致（分母含不可验证声明，rate 略低估） |
| Top-5 发现准确率 **97.0%**（L105） | metrics_main.json:16 top5_accuracy=0.9699；重算 129/133 ✓（交换不改 top5 集合，无注入污染） | ✅ 一致 |
| Rubric 报告质量 3.02/2.51/2.74/4.08（L106） | judge_rubric_results.json main（3.023/2.511/2.737/4.075） | ✅ 一致 |
| 三臂 rubric 总分 **2.61**/2.71/2.60（L108） | 2.71/2.60 无 json 来源（图内硬编码）；**2.61 ≠ judge_rubric_results.json 的 3.02** | ❌ 矛盾且不可溯源（🔴） |
| 情境基准：一致性 100%、违例 0、严重疾病 100%(15/15)、敏感性 97.2%、风险变化 69.4%（L115-119） | outputs/context_eval/metrics.json（1.0/0/1.0/0.9722/0.6944，n_abnormal=15）；36 条记录与 subset 完全对应 | ✅ 一致（代码存在潜伏边界 bug，未影响本次数字） |
| 跨源 AUC（CPSC 0.940…PTB-XL 0.897，总体 0.921，n=6472）（L127-133） | per_source_auc.json 全项吻合；test_manifest 实测 6472 条 | ✅ 一致 |
| MIMIC 域分析（0.9970/3.402/5.51/ABNORMAL 229/窦速 15 vs 8/平导 0.002）（L138-143） | mimic_domain.json 全项吻合 | ✅ 一致 |
| 逐类最强 T Wave Abnormal 0.850、Sinus Arrhythmia 0.834(n=3)（L146） | per_class_auc.json（0.8496 / 0.8344, n=3） | ✅ 一致 |
| 混淆模式 缺血→窦律 244、窦律→缺血 134（L147） | per_class_auc.json top_confusion_pairs（244/134） | ✅ 一致 |
| 知识库 **350 来源**/27 类、每类 6-13 来源（L64） | crawl_summary.json total=350；kb/classes.json n_sources 6-13 | ✅ 一致 |
| 每类 12-49 条 ECG 特征（L64/图5） | kb/classes.json 实测 ecg_criteria **10-49** | ⚠️ 下限不符（10 vs 12） |
| 62 条记录 8-9s 零填充（L156） | processed_5k 三 manifest 实测 d<9.5 → 62 条 | ✅ 一致（"8-9s"表述略宽于"<9.5s"标准） |
| 公平消融 A=0.9214/B=0.9126/C2=0.9126/D2=0.9159/C=0.9388/D=0.9458（L94-97） | multimodal_fair/metrics.json 全项吻合 | ✅ 一致 |
| 评测集 133 条 / 情境 36 条 | evalset.json items=133；context_subset.json items=36 | ✅ 一致 |

---

## 结语

代码整体可运行、可复现（产物齐全），但**论文 §4.3 的两项核心指标（工具选择、注入纠错）被评测 harness 自身缺陷污染，必须修复后重测**；三臂对照数字存在数据溯源缺口与文本/图/产物三方矛盾；情境基准与幻觉率（主臂）数字经逐条复核与代码/产物一致。建议按 P0 清单处理后再投稿。
