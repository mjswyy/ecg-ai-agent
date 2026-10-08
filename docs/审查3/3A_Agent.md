# 第三轮代码审查 — Agent 核心与 LLM 层（3A_Agent）

> 审查人：独立审查员 A（第三轮，未读取任何历史审查产物）
> 范围：`src/agent/` 全部 20 文件 + 相关工具/特征提取/标签映射（21 文件，约 1833 行）
> 验证方式：逐行通读 + mock LLM 端到端运行 + RubricJudge/记忆注入/年龄解析边界实测

## 统计

| 严重度 | 数量 |
|---|---|
| 🔴 | 1 |
| 🟠 | 6 |
| 🟡 | 4 |
| 💡 | 2 |
| **合计** | **13** |

## 发现清单

### 🔴-1 分类器失败被包装成"未检测到高置信度异常→常规随访"（医疗安全反模式）
- **位置**：`src/agent/core/agent.py:426-460, 199, 379-384`
- **问题**：诊断分类器失败/空结果时输出 diagnosis=[]、incomplete=False、recommendations=['未检测到高置信度异常。建议常规随访。']。analysis_failed 仅在 n_ok_steps==0（全部步骤失败）时触发，漏掉"测量步骤成功但诊断分类器失败"这一最常见失败模式。
- **证据（confirmed）**：端到端复现（mock LLM + ToolRegistry，classify_arrhythmia 返回 `{"error":"未加载分类模型"}`，extract_r_peaks 正常）→ 输出 diagnosis=[]、confidence=0.0、heart_rate=72.0、rhythm='normal rate (regular)'、incomplete=False、recommendations=['未检测到高置信度异常。建议常规随访。']。根因：`_execute_step` 异常路径也置 completed=True（agent.py:319）；`_synthesize_diagnosis` 中 classify 的 error dict 仅计入 n_failed_steps，因 n_ok_steps=1 使 analysis_failed=False。
- **修法**：失败判定扩展为"诊断步骤失败或 diagnoses 为空且存在 n_failed_steps>0"时显式标注"诊断/分类步骤未完成"；incomplete 或新增 failure 字段反映 step.error。

### 🟠-1 RubricJudge 对非对象 JSON 抛 AttributeError 而非返回 None
- **位置**：`src/agent/llm/rubric_judge.py:66-75`
- **证据（confirmed）**：FakeLLM 依次返回 '[1,2,3]'/'null'/'4'/"text"，score() 全部抛 AttributeError（list/NoneType/int/str has no attribute 'get'），未走到 return None，会中断评测循环。
- **修法**：json.loads 后校验 isinstance(scores, dict)，非 dict 时 return None。

### 🟠-2 判官缓存 key 用 report[:1500] 截断 → 不同报告碰撞
- **位置**：`src/agent/llm/rubric_judge.py:50-54, 43-48`
- **证据（confirmed）**：rep1=1500个A+'X-report-1'、rep2=1500个A+'Y-report-2'，_key 返回相同 sha256。
- **修法**：key 用完整 report（或全文哈希），仅对送入 LLM 的 prompt 做 1500 截断。

### 🟠-3 _replan 无 try/except，LLM 不可用时穿透崩溃
- **位置**：`src/agent/core/agent.py:333`
- **证据（suspected）**：revise_plan→llm.chat 在无 key 且非 mock 时 raise RuntimeError（llm_interface.py:164-168），_replan 无捕获（对比 _plan 有容错回退）。
- **修法**：revise_plan 包 try/except，失败保留原计划并告警。

### 🟠-4 ConversationManager EXECUTING while True 无迭代上限
- **位置**：`src/agent/orchestration/conversation.py:187-212`
- **证据（suspected）**：循环仅靠 pending 为空或 should_continue=False 退出；每次 _replan 生成全新 AgentStep（新 id 不入 executed 集），反射器反复 revise 时可无限续跑。
- **修法**：加 max_steps 上限或 replan 计数硬顶。

### 🟠-5 ECGFounderClassifier 默认阈值 0.5 与 Youden 阈值脱节；positives 仅从 top_k 派生
- **位置**：`src/agent/tools/ecgfounder_classifier.py:72, 111-120`
- **证据（confirmed）**：metrics.json thresholds=[0.13,0.10,0.11,...,0.66]，与 0.5 差异显著；所有入口以 ECGFounderClassifier() 无参构造、从未加载该阈值 → 大量真实阳性（0.1~0.5）被判 negative，系统性漏报；positives 仅从 top_k(5) 派生，第 6 名后的阳性被丢弃。
- **修法**：默认按 metrics.json 加载 Youden 阈值；positives 对全部 27 类按阈值判定。

### 🟠-6 性别从未流入 QT 解读；sex 特异性上界是死参数
- **位置**：`src/agent/core/memory.py:61-84; src/agent/tools/ecg_tools.py:82; src/ecg_models/feature_extraction/qt_analyzer.py:116,143-148`
- **证据（confirmed）**：memory 对 measure_qt_interval 只注入 ecg_signal/r_peaks/rr_intervals，sex present=False；工具默认 sex='Unknown'；qt_upper（450/460）计算后从未被 _interpret_qt 使用（函数只用类常量 QTC_NORMAL_UPPER=440）。
- **修法**：注入 patient_info['sex'] 并在 _interpret_qt 真正使用 sex 特异性上界（或删除死常量）。

### 🟡-1 年龄解析取第一个数字（含病史数字时解析错误）
- **位置**：`src/agent/orchestration/conversation.py:135-139`
- **证据（confirmed）**：'the patient had 3 stents placed, now age 65'→age=3；'2 episodes of chest pain, age 65'→age=2。
- **修法**：正则锚定 age/岁 附近数字 + 0<age<120 校验。

### 🟡-2 generate_report 是无效步骤（恒输出"心率: N/A"）
- **位置**：`src/agent/tools/diagnosis_tools.py:77-93; src/agent/core/memory.py:61-84`
- **证据（suspected）**：get_context_for_tool 不为其注入 features/diagnoses → summary 恒为 '心率: N/A bpm'；其返回被 _synthesize_diagnosis 忽略；但 FEW_SHOT 计划（planner.py 病例1/2/3）均含该步骤。
- **修法**：让 generate_report 真正聚合 memory 观测/诊断，或从默认计划与 FEW_SHOT 移除。

### 🟡-3 工具 schema/提示与实际注册不一致
- **位置**：`src/agent/core/planner.py:54; src/agent/core/memory.py:73; scripts/run_agent_demo.py:52-69`
- **证据（suspected）**：FEW_SHOT 建议 query_medical_knowledge 但 run_agent_demo 未注册（必然"工具未找到"）；memory 注入分支含 plot_waveform 但从未注册；detect_anomaly detector=None 恒 error。
- **修法**：统一可用工具清单（注册或移除）。

### 🟡-4 置信度语义模糊；失败无结构化字段表达
- **位置**：`src/agent/core/agent.py:421-424, 74, 199`
- **证据（confirmed）**：全失败与分类器失败场景均输出 confidence=0.0、diagnosis=[]、incomplete=False；incomplete 只反映 max_steps 耗尽，不反映 step.error。
- **修法**：新增 analysis_status ∈ {complete, partial, failed}；避免 confidence=0.0 同时表示"无异常"与"无结果"。

### 💡-1 判官打分 int() 截断而非四舍五入
- **位置**：`src/agent/llm/rubric_judge.py:70`
- **证据（confirmed）**：{"correctness":4.9,...,"total":4.5} → 4/4。
- **修法**：round() 后 clamp。

### 💡-2 predict 不校验形状/None
- **位置**：`src/agent/tools/ecgfounder_classifier.py:96-104`
- **证据（suspected）**：仅校验 NaN/Inf；1D 或非 12 通道输入在 backbone Conv1d 崩溃；signal=None 时 np.isfinite(None) 先抛 TypeError。
- **修法**：校验 ndarray 且 shape[0]==12，否则返回 {"error":...}。

## 阳性结论（本次验证正确的性质）

- R 峰失败（<2 峰）→ insufficient_data + insufficient=True 已正确落地并端到端验证
- QT 数据不足 → 数值 None + error 标记已正确落地
- "全部工具失败"分支的失败显式化已正确（verify_r3_safety 实测通过）
- Reflector 词边界匹配、分类器 NaN 显式拒绝、性别"female"误判修复均正确
- rubric_judge 的 None 口径 + judge_failures 单列（对合法 dict 输出）工作正常
