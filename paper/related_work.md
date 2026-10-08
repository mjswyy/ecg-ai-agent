# 相关工作调研：ECG 多模态大模型与智能体（12 篇精读对比）

> 调研方式：以 arXiv 摘要页 / ar5iv / arXiv HTML / IEEE Xplore / Nature Scientific Data / 官方 GitHub 为准，逐一精读并提取信息。所有数字均来自检索到的原文或官方仓库，未臆造；个别摘要未公开的数字已明确标注"未在公开摘要给出"。

---

## 0. 检索与勘误说明（重要）

| 用户清单条目 | 勘误结果 |
|---|---|
| 1. ECG-Chat "Stern et al. 2024, arXiv:2410.18973" | **ID 与作者均不符**。arXiv:2410.18973 是一篇 MCMC 统计学论文（Tuning-Free Coreset MCMC，stat.CO）。ECG 领域的 ECG-Chat 实为 **Yubao Zhao 等 (arXiv:2408.08849)，ICME 2025**，作者为 Zhao/Kang/Zhang/Han/Chen，无 Stern。本报告按真实论文收录。 |
| 2. ECG-QA "arXiv:2503.14608" | **ID 不符**。arXiv:2503.14608 是量子物理论文。ECG-QA 数据集论文实为 **arXiv:2306.15681（Jungwoo Oh 等，NeurIPS 2023 Datasets & Benchmarks）**。本报告按真实论文收录。 |
| 其余 10 篇 | 编号/ID 全部核实无误（GEM=2503.06073；ECG-Agent=arXiv:2601.20323/ICASSP 2026；Cardiologent=2607.25340；CARE-ECG=2604.10420；UniECG=2509.18588；ECG-R1=2602.04279/ICML 2026；MEETI=s41597-026-06796-1；ECGFounder=2410.04133/NEJM AI 2025；ECG-FM=2408.05178）。 |

---

## 1. 十二篇论文要点精读

### 1.1 ECG-Chat：A Large ECG-Language Model for Cardiac Disease Diagnosis（arXiv:2408.08849，ICME 2025）
- **数据**：对比学习预训练用 MIMIC-IV-ECG + Chapman-Shaoxing-Ningbo(CSN) + Shandong Provincial Hospital(SPH) 共 805K 对；预训练指令 619K（MIMIC）；指令微调 ECG-Instruct 19K 诊断 + 25K 多轮对话（4–15 轮）；数据用 GPT-4o 生成。
- **模态**：12 导联信号 + 文本（报告 + NeuroKit2 提取的波形参数）。无图像输入。
- **LLM**：Vicuna-13B，LoRA 微调；ECG 编码器为 CoCa 式对比学习（ECG-文本），经两层 MLP 适配器接入 LLM（LLaVA 风格）。
- **工具/Agent**：非开放式 Agent；含 **GraphRAG（7 本心脏病学教材构建知识图谱）+ DSPy 自动提示调优 + LaTeX 报告流水线**；Diagnosis-Driven Prompt（线性分类器把预测标签注入 prompt）。
- **情境条件化**：弱-中——报告流水线会整合患者信息与诊断标签。
- **双层架构**：弱——"分类器→标签注入 prompt→LLM 生成"的隐式分层，非显式发现/决策两层。
- **评测**：ECG 分类、零样本报告检索、报告生成（基准自建）、多模态对话；摘要称"best performance in classification, retrieval, report generation"（**具体 ACC/ROUGE 数值未在摘要公开**）。
- **局限**：依赖 GPT-4o 合成指令数据（可能引入医学错误）；仅信号-文本，不处理图像；摘要明确承认 LLM 在心内科存在显著幻觉问题。

### 1.2 ECG-QA：A Comprehensive Question Answering Dataset Combined With Electrocardiogram（arXiv:2306.15681，NeurIPS 2023 D&B）
- **定位**：首个 ECG 专用 QA 数据集/基准，非模型。
- **数据**：70 个经 ECG 专家验证的问题模板；原始版基于 PTB-XL，扩展版（v1.0.2）基于 MIMIC-IV-ECG（155 个 SCP 代码，机器生成语句人工标注）；含单 ECG 与双 ECG 对比类问题（comparison-consecutive / comparison-irrelevant）。
- **模态**：信号 + 模板化文本问答。
- **LLM**：不训练模型，论文中评测通用 LLM 的 ECG QA 能力。
- **工具/Agent / 情境 / 双层**：均无。
- **评测维度**：模板化 QA 准确率（single/choose/query/verify 等 7 种题型）、对比分析任务、LLM 基准实验。
- **局限**：模板固定、可扩展性有限；MIMIC-IV-ECG 版机器生成语句未经心脏病学家质量验证（README 自述可能存在标签-真值不匹配）。

### 1.3 GEM：Empowering MLLM for Grounded ECG Understanding with Time Series and Images（arXiv:2503.06073，NeurIPS 2025）
- **数据**：首个高粒度 ECG grounding 数据集 **ECG-Grounding：30,000 指令对，心跳级生理特征标注**（QRS/PR 间期等）；训练基于 MIMIC-IV-ECG/CSN 等；评测含 CSN 分类与自建 Grounded ECG Understanding 基准。
- **模态**：**时间序列信号 + 12 导联图像 + 文本三模态统一（首个）**。
- **LLM**：MLLM（LLaVA 式）；时间序列编码器用 ECG-CoCa、图像编码器用 LLaVA 的 CLIP；GPT-4o 做知识引导指令生成。
- **工具/Agent**：无。
- **情境条件化**：无——Cardiologent 论文专门指出 GEM"没有临床上下文时会高估其发现的严重性"。
- **双层架构**：无。
- **评测**：CSN 预测性能 **7.4%↑**、可解释性 **22.7%↑**、grounding **25.3%↑**。
- **局限**：模态耦合（信号 token 塞进 \<image\> 占位符、复用图像-语言投影器），单模态缺失时性能明显下降、跨模态输出矛盾（ECG-R1 实测其跨模态 BLEU-4 仅 0.33）；无患者情境条件化。

### 1.4 ECG-Expert-QA：A Benchmark for Evaluating Medical LLMs in ECG（IEEE BIBM 2025；预印本 arXiv:2502.17475）
- **定位**：医疗 LLM 的 ECG 专家级评测基准（数据集+评测协议）。
- **数据**：预印本版（2502.17475）：真实 + 合成 ECG，12 个诊断任务，**47,211 条专家验证 QA 对**；BIBM 版基于 MIMIC-IV-ECG；覆盖基础心电图知识、复杂诊断、跨模态诊断（ECG+文本）、长文本诊断、风险评估、患者预后、实体抽取、记忆纠正、医患多轮问诊等 11 类 JSON 数据。
- **模态**：文本为主 + ECG 相关信息（跨模态临床推理）。
- **LLM**：评测对象（医疗 LLM），不训练模型。
- **工具/Agent**：无。
- **情境条件化**：部分——多轮问诊模拟（医生-患者对话）。
- **双层架构**：无。
- **评测维度**：ECG 解读、临床推理、多轮 QA、跨模态推理、安全性/风险感知、鲁棒性。
- **局限**：被评测模型的**具体准确率数字未在公开摘要/GitHub 给出**（在论文正文）；问题多为文本-标签形式，弱于信号级 grounding。

### 1.5 ECG-Agent：On-Device Tool-Calling Agent for ECG Multi-Turn Dialogue（arXiv:2601.20323，IEEE ICASSP 2026）
- **数据**：自建 **ECG-MTD**：21,837 个多轮 ECG 对话（12 导联/Lead I/Lead II 三种配置，每配置 98K 训练实例，平均 7.68 轮/对话）；基于 PTB-XL（医生确认各导联可检测的 FDA 类）；Gemini-2.5-Flash 生成 + Gemini-2.5-Pro 精化主题；7 个主题类别 × CEFR 三级语言水平 × 20 个动作序列。
- **模态**：信号（经工具转成结构化输出）→ 文本对话；LLM 本身是文本-only。
- **LLM**：Llama-3.2-1B/3B、Llama-3.1-8B、Qwen3-32B，LoRA（rank 16）微调。
- **工具/Agent**：**首个 ECG 多轮对话工具调用 Agent**——三个工具：classification（W2V+CMSC+RLM 自监督预训练模型）、measurement（Neurokit2，PQRST 测量）、explanation（SpectralX 时频解释，仅单导联）；每轮"thought→工具调用→回复"循环。
- **情境条件化**：仅对话上下文；无患者病史/症状/元数据。
- **双层架构**：无（agent 循环而非发现/决策分层）。
- **评测**：响应 Accuracy/Completeness（1–5 分，Gemini-2.5-Pro 作 judge，与人类 Spearman ρ=0.70 / 0.46）、下一动作预测、faithfulness、幻觉、工具 TIoU。
- **局限**：训练对话由 LLM 生成（真实性上限）；解释工具无法用于 12 导联；on-device 模型仍存在幻觉（论文评估了但未消除）；无情境条件化。

### 1.6 Cardiologent：Multi-Agent Clinical Decision Support for Patient-Level Arrhythmia Assessment, Urgency, and Management（arXiv:2607.25340，2026-07）
- **数据**：VitalDB（6,388 例术中监护）+ VitalDB Arrhythmia Database：734,528 秒 ECG 注释、482 患者、661,894 心搏、4 搏类型/10 心律类别（≥2/5 名麻醉医师标注，κ=0.930）；426 患者双信号（lead II ECG + PPG）→ 166,026 窗口；测试 65 患者 / 18,873 窗口（按患者划分）。
- **模态**：lead II ECG + PPG + **患者元数据（年龄/性别/合并症/病例记录）** + 检索的临床指南文本。
- **LLM**：每信号 specialist = **LLaVA-7B 微调**；窗口融合与患者级推理用 LLM 辩论（analyst / devil's advocate）+ 指南检索 + critic。
- **工具/Agent**：每信号确定性测量工具（ECG 7 个、PPG 9 个），所有读数基于实测值；明确的多智能体 + 检索 + 验证循环。
- **情境条件化**：**强**——"identical signal, opposite decision"：同段房颤在健康成人是次要发现、在老年高血压患者是抗凝依据；患者数据与指南检索参与决策。
- **双层架构**：**明确的双层**——窗口级（发现：逐窗口 ECG/PPG 读数→融合裁决）→ 患者级（决策：心律画像→综合诊断、临床意义、紧急度与管理）。
- **评测**：评估临床决策本身（非报告流畅度）：综合诊断、临床意义、紧急度与管理三个轴；心脏病学家 + 大规模 LLM judge 双评；LLM judge 与心脏病学家 ICC 0.74/0.66 ≈ 心脏病学家互评 0.67；**"highest on every axis"**。
- **局限**：术中监护数据（麻醉场景，非急诊/门诊）；测试集仅 65 患者；PPG 只能区分 7 类（无法区分室性/房性早搏、VT/MAT）。

### 1.7 CARE-ECG：Causal Agent-based Reasoning for Explainable and Counterfactual ECG Interpretation（arXiv:2604.10420，2026-04）
- **方法**：因果结构化 ECG-语言推理：多导联 ECG 编码为**时序组织的潜在生物标志物** → 因果图推断概率诊断 → 结构因果模型（SCM）反事实评估；语言输出经 **causal RAG** 接地 + **模块化 agentic pipeline（history→diagnosis→response→verification）**。
- **数据/评测**：多个 ECG 基准 + 专家 QA：**Expert-ECG-QA 0.84、SCP-mapped PTB-XL 0.76（GPT-4 底座）**。
- **模态**：多导联信号 + 文本（病史、诊断、问答）。
- **LLM**：GPT-4（评估与生成底座）。
- **工具/Agent**：agentic 流水线 + 验证环节；causal RAG。
- **情境条件化**：部分——history 进入 pipeline。
- **双层架构**：表征学习→诊断→解释统一流水线 + 验证，非显式双层，但含 agentic 流程。
- **评测维度**：诊断准确率、解释忠实度（faithfulness）、幻觉减少（显著降低）、反事实 what-if 分析。
- **局限**：潜在生物标志物提取质量决定 grounding 上限；反事实结论依赖 SCM 因果假设（难以在真实数据上验证）。

### 1.8 UniECG：Understanding and Generating ECG in One Unified Model（arXiv:2509.18588，2025-09 / v2 2026-06）
- **定位**：**交互式 ECG 教学**工具，明确声明"非临床验证的诊断系统"。
- **方法**：两阶段——①从 ECG 信号-图像-文本学习 grounded 解释（理解）；②引入 ECG 生成 token，将其隐表示与预训练**文本条件 ECG 扩散模型**对齐，实现可控信号级 ECG 生成（给定学习目标生成示例心电图）。
- **模态**：信号/图像/文本（理解）+ 文本→信号（生成）。
- **LLM**：MLLM + 扩散模型（摘要未指明 backbone）。
- **工具/Agent / 情境 / 双层**：无（情境仅"教学目标文本"）。
- **评测**：grounded ECG 解释 + 生成质量**定性分析**；无临床指标。
- **局限**：教育辅助定位；无临床验证；生成评估以定性为主。

### 1.9 ECG-R1：Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation（arXiv:2602.04279，ICML 2026）
- **数据**：**Protocol-Guided Instruction Data Generation**：FeatureDB（确定性、不可训练）从 12 导联提取 14 类逐搏特征序列 → 教科书（*ECG from Basics to Essentials* 第 23 章）五阶段协议（速率/节律→传导/轴/间期→肥厚/电压→缺血/梗死→电解质/QT）构造 prompt → DeepSeek-V3.1-Terminus 生成 30,000 条协议化六步思考+摘要+诊断指令（MIMIC-IV-ECG）；RL 子集 3,948；评测 ECG-Grounding 测试集 2,381 例 + 100 例 4 名持证心脏病学家盲评。
- **模态**：信号 + 图像 + 文本；**模态解耦双编码器**（Qwen3-VL-8B 作为 LLM+视觉编码器，ECG-CoCa 作为时间序列编码器，独立投影器、\<ecg\> 标签）；Interleaved Modality Dropout（IMD）训练。
- **LLM**：Qwen3-VL-8B；数据生成 DeepSeek-V3.1-Terminus；RL 用 DAPO + **EDER（ECG 诊断证据过程奖励）**。
- **工具/Agent**：FeatureDB 为确定性特征工具；**无开放式 agent 循环**；RL 过程奖励而非工具调用。
- **情境条件化**：无（单次记录解读，无患者背景）。
- **双层架构**：无显式双层（协议化 think→answer 推理流程）。
- **评测**：诊断准确率、分析完整性、导联证据有效性、临床诊断忠实度（七项 rubric，LLM judge）+ 心脏病学家；跨模态一致性（BLEU-4/ROUGE-L/SBERT）；幻觉定量评估。
- **局限**：FeatureDB 设定 grounding 上限（提取不到的特征=覆盖不到）；30K 语料仍属"LLM 生成"范式（协议化抑制而非消除）；跨模态一致性保证依赖 ECG 特有的 Δview≈0 假设；RL 子集仅 3,948。

### 1.10 MEETI：A Multimodal ECG Dataset from MIMIC-IV-ECG（Scientific Data 2026，s41597-026-06796-1）
- **定位**：数据集（MIMIC-IV-Ext ECG-Text-Image，Zenodo v2 2026-02-12）。
- **数据**：基于 MIMIC-IV-ECG 约 **800,000** 条 12 导联记录（10 s / 500 Hz），四组件按 study_id 对齐：①原始信号（HIPAA Safe Harbor 脱敏）；②300 dpi 高分辨率 12 导联图（25 mm/s、10 mm/mV，3×4/6×2/12×1 布局）；③FeatureDB 逐搏定量参数（P/QRS/T、PR/QRS/QT/QTc 等）；④**GPT-4o 生成的解释文本**。
- **模态**：信号 + 图像 + 参数 + 文本（首个四组件同步数据集）。
- **LLM**：GPT-4o（文本生成）。
- **评测/用途**：支持多模态 transformer 训练与可解释分析；论文未做模型基准（纯数据贡献）。
- **局限**：单中心（BIDMC）；GPT-4o 生成文本未经心脏病学家系统验证（讨论中承认需后续验证）。

### 1.11 ECGFounder：An Electrocardiogram Foundation Model Built on over 10 Million Recordings（arXiv:2410.04133，NEJM AI 2025, 2:AIoa2401033）
- **数据**：Harvard-Emory ECG Database **>1000 万条**心电图、**150 个标签类别**；外部验证跨多域；扩展至低秩/任意单导联。
- **模态**：信号（多导联/单导联）。**无 LLM、无文本输出**。
- **性能**：内部验证 **80 个诊断 AUROC>0.95（专家级）**；外部验证强泛化；微调后优于基线（人口统计分析、临床事件检测、跨模态心律诊断）。
- **定位**：判别式基础模型（"开箱即用 + 可微调"），与生成式 ECG-LLM 互补。
- **局限**：非对话/生成模型；无解释文本；训练数据来自单一数据库。

### 1.12 ECG-FM：An Open Electrocardiogram Foundation Model（arXiv:2408.05178）
- **数据**：**150 万**条 ECG 自监督预训练；MIMIC-IV-ECG 上发布 LVEF 降低 + 解释标签基准任务。
- **方法**：Transformer + 混合对比/生成式自监督；开放权重与代码。
- **模态**：信号。
- **性能**：**房颤 AUROC 0.996、LVEF≤40% AUROC 0.929**；小-中规模数据下显著优于任务特定模型；跨数据集泛化；标签高效。
- **局限**：判别式分类模型，无语言/解释/对话能力；评估以二分类为主。

---

## 2. 中文对比表

| 论文（年份/Venue） | 训练与评测数据 | 输入模态（信号/图像/文本/元数据） | 是否用 LLM（哪个） | 工具调用/Agent 循环 | 情境条件化（患者背景/症状/病史） | 双层"发现 vs 决策" | 评测维度 | 主要局限 |
|---|---|---|---|---|---|---|---|---|
| **ECG-Chat**（2024 arXiv / ICME 2025） | 对比预训练 805K（MIMIC-IV-ECG+CSN+SPH）；指令 619K+19K 诊断+25K 对话（GPT-4o 生成） | 信号 + 文本（报告、波形参数）；无图像 | 是：Vicuna-13B（LoRA），编码器 ECG-CoCa；GPT-4o 造数据 | 弱：GraphRAG(7 教材)+DSPy+LaTeX 流水线+诊断注入 prompt；非开放式 agent | 弱-中：报告流水线整合患者信息 | 弱：分类器→标签注入→LLM 生成的隐式分层 | 分类、零样本检索、报告生成（ROUGE 类）、多轮对话 | 摘要未公开具体 ACC/ROUGE；GPT-4o 合成数据可能引入错误；无图像模态 |
| **ECG-QA**（2023 / NeurIPS 2023 D&B） | 70 个专家验证模板；PTB-XL 版 + MIMIC-IV-ECG 扩展版（155 SCP 码） | 信号 + 模板化文本 QA | 评测通用 LLM；不训练模型 | 无 | 无 | 无 | 7 类题型 QA 准确率、双 ECG 对比任务 | 模板固定；MIMIC 版机器标签未经专家验证 |
| **GEM**（2025 / NeurIPS 2025） | ECG-Grounding 30K 心跳级标注指令对；MIMIC-IV-ECG/CSN 等 | **信号+图像+文本**（三模态统一） | 是：LLaVA 式 MLLM；TS 编码器 ECG-CoCa，图像编码器 CLIP；GPT-4o 造数据 | 无 | 无（被 Cardiologent 指出无上下文会高估严重性） | 无 | CSN 分类（7.4%↑）、可解释性（22.7%↑）、grounding（25.3%↑） | 模态耦合，单模态缺失性能骤降、跨模态不一致（BLEU-4 0.33）；无情境 |
| **ECG-Expert-QA**（2025 / IEEE BIBM） | 预印本版 47,211 专家验证 QA 对、12 任务；BIBM 版基于 MIMIC-IV-ECG | 文本为主 + ECG 相关信息 | 评测医疗 LLM；不训练模型 | 无 | 部分：医患多轮问诊模拟 | 无 | ECG 解读、临床推理、多轮 QA、跨模态、风险/安全、鲁棒性 | 被评模型准确率未公开在摘要；文本-标签为主，弱 grounding |
| **ECG-Agent**（2026 / ICASSP） | ECG-MTD：21,837 对话、每导联配置 98K 实例、7.68 轮/对话；PTB-XL（12/Lead I/Lead II） | 信号（工具输出）→ 文本 | 是：Llama-3.2-1B/3B、Llama-3.1-8B、Qwen3-32B（LoRA） | **是**：classification/measurement(Neurokit2)/explanation(SpectralX) 三工具，thought→调用→回复 | 仅对话上下文；无病史/症状/元数据 | 无 | 响应 Accuracy/Completeness(1–5)、下一动作预测、幻觉、工具 TIoU、faithfulness | LLM 生成对话数据；解释工具不支持 12 导联；on-device 仍有幻觉；无情境 |
| **Cardiologent**（2026-07 / arXiv） | VitalDB+VitalDB-ADB：734,528s、482 患者、661,894 心搏（κ=0.930）；测试 65 患者 18,873 窗口 | lead II ECG + PPG + **患者元数据** + 指南文本 | 是：LLaVA-7B 微调 specialist + LLM 辩论/critic | **是**：每信号 7/9 个确定性测量工具 + 指南检索 + critic 验证 | **强**：年龄/性别/合并症+指南参与决策（同信号异决策） | **有**：窗口级(发现)→患者级(决策:诊断/意义/紧急度/管理) | 临床决策三轴（诊断/意义/紧急度-管理），心脏病学家+LLM judge（ICC 0.74/0.66≈医生互评 0.67） | 术中监护数据；测试仅 65 患者；PPG 仅 7 类；分数细节在正文 |
| **CARE-ECG**（2026-04 / arXiv） | 多 ECG 基准 + Expert-ECG-QA、SCP 映射 PTB-XL | 多导联信号 + 文本（病史/问答） | 是：GPT-4 | **部分**：causal RAG + 模块化 agentic pipeline（history→diagnosis→response→verification） | 部分：history 进 pipeline | 部分：表征→诊断→解释统一流水线+验证 | 诊断准确率（0.84/0.76）、解释忠实度、幻觉减少、反事实 | 依赖 SCM 因果假设；生物标志物提取质量设上限 |
| **UniECG**（2025-09 / arXiv） | 信号-图像-文本 grounding 数据 + 文本条件 ECG 扩散模型 | 信号/图像/文本 + 文本→信号生成 | 是：MLLM+扩散模型（backbone 未指明） | 无 | 无（仅教学目标文本） | 无 | grounded 解释 + 生成质量**定性**分析 | 教育辅助定位；无临床验证；定性为主 |
| **ECG-R1**（2026 / ICML） | 30K 协议化指令（FeatureDB+教科书 5 阶段协议，DeepSeek 生成）；RL 3,948；评测 ECG-Grounding 2,381+100 例医生盲评 | 信号 + 图像 + 文本（模态解耦） | 是：Qwen3-VL-8B；生成 DeepSeek-V3.1-Terminus；RL：DAPO+EDER | 部分：FeatureDB 确定性特征工具；无开放式 agent | 无 | 无（协议化 think→answer） | 诊断准确率 80.29、临床忠实度 84.20、跨模态一致性（BLEU-4 0.69）、幻觉定量 | FeatureDB 上限；30K 仍 LLM 生成；Δview≈0 假设 ECG 特异；RL 集小 |
| **MEETI**（2026 / Sci. Data） | MIMIC-IV-ECG 约 80 万条；四组件对齐（信号/图像/参数/GPT-4o 文本） | 信号+图像+参数+文本 | 用 GPT-4o 生成文本（数据贡献，非模型） | 无 | 无 | 无 | 数据质量/对齐（无模型基准） | 单中心；GPT-4o 文本未系统专家验证 |
| **ECGFounder**（2024 arXiv / NEJM AI 2025） | Harvard-Emory >1000 万条、150 标签；内外验证多域 | 信号（多/单导联） | 无（判别式基础模型） | 无 | 无 | 无 | 内部 80 诊断 AUROC>0.95；外部泛化；下游微调 | 无语言/解释/对话；单一训练库 |
| **ECG-FM**（2024 / arXiv） | 150 万条自监督预训练；MIMIC-IV-ECG 基准（LVEF、标签） | 信号 | 无 | 无 | 无 | 无 | AFib AUROC 0.996、LVEF≤40% 0.929、标签高效 | 判别式；无语言能力 |

---

## 3. 与我们的定位差异（ECGFounder 冻结特征 + DeepSeek Agent + 双层情境架构）

**我们的工作**：ECGFounder 冻结特征（0.9456，官方评分类，患者级划分）+ DeepSeek Agent + 双层情境架构（发现层无越界编造 100% / 决策层 97.5% 敏感（39/40））+ 分级可信知识库（官方 27 类全覆盖）+ 五维评测 + 情境敏感性基准。

对照上述 12 篇，以下维度**现有工作没有做或没有评测**：

1. **"冻结基础模型特征 + LLM Agent"的模块化组合**。现有路线只有三类：(a) 端到端微调 ECG-MLLM（ECG-Chat/GEM/ECG-R1/PULSE 系）；(b) 判别式基础模型（ECGFounder/ECG-FM，无语言）；(c) 文本-only LLM + 外部工具（ECG-Agent）。**没有工作**把"冻结、不微调的 ECG 基础模型特征"直接作为 Agent 的感知输入，再叠加 LLM 推理/工具调用——我们的 0.9456 冻结特征即属此类，现有 12 篇无一对应。
2. **显式双层"发现层（信号→结构化发现，与情境无关） vs 决策层（发现+情境→临床决策）"架构**。仅 Cardiologent 有窗口级/患者级双层，但它是单任务心律失常分类 + 指南应用，且 specialist 是微调的 LLaVA-7B；GEM/ECG-R1/ECG-Chat 都是单层端到端。**无人把"发现层"设计为情境无关、可单独验证（无越界编造 100%，严格一致性 77.5%）的独立层**，也无人把"决策层"单独评测其情境敏感性（97.5% 敏感，39/40）。
3. **情境敏感性基准（同一信号 × 不同患者背景/症状/病史 → 决策应当且必须改变）**。Cardiologent 只提出了"identical signal, opposite decision"的动机并做了真实病例评测，**未构建系统性"同信号异情境"基准**；GEM 被指出无上下文会高估严重性但没有相应评测；ECG-Expert-QA 的"风险感知"是安全维度而非情境敏感性。**该基准是空白**——这正是我们的"决策层 97.5% 敏感（39/40）"所评测的能力。
4. **分级可信知识库（27 类全覆盖、174 篇去重来源文档、来源权威性分级 A/B/C）**。最接近的是 ECG-Chat 的 GraphRAG（7 本教材）、Cardiologent 的指南检索+critic、CARE-ECG 的 causal RAG；**无人构建大规模、按可信度分级、面向多病种（27 类）的知识库，也无人评测"引用正确率随来源分级的衰减"**。
5. **五维评测的组合（分类精度 / 幻觉 / 工具正确率 / 情境敏感性 / 可解释性-引用）**。现有覆盖情况：ECG-Agent（响应质量+工具+幻觉，无情境敏感性）；ECG-R1（准确率+幻觉+一致性+医生盲评，无工具/情境）；Cardiologent（决策+医生/LLM judge，无工具正确率/幻觉量化）；GEM（分类+可解释+grounding，无幻觉/工具/情境）；CARE-ECG（准确率+忠实度+幻觉，无工具/情境）。**没有一篇一次性覆盖五维，尤其"工具调用正确率 × 情境敏感性 × 幻觉"同测**。
6. **"发现层一致率 + 决策层敏感率"这种双层指标**（发现层无越界编造 100%（严格一致性 77.5%）、决策层 97.5% 敏感（39/40））：现有工作无此指标设计；ECG-R1 的跨模态一致性（BLEU-4）测的是"同信号两种模态输出是否一致"，与"发现层在情境扰动下是否保持稳定"是不同概念。
7. **情境条件化的广度**：现有工作中只有 Cardiologent 真正把患者背景（年龄/性别/合并症）送进决策；**没有一篇使用"症状/主诉"文本做条件化**，也没有一篇把"患者背景-症状-病史"三要素组合进双层架构。

---

## 4. 可引用的关键数字（均标注来源）

| 论文 | 关键数字 | 来源 |
|---|---|---|
| ECG-Chat | 训练数据 805K（对比）/619K（预训练指令）/19K 诊断+25K 对话；报告称分类、零样本检索、报告生成全部最优（摘要未公开具体 ACC/ROUGE） | arXiv:2408.08849 摘要与全文 |
| ECG-QA | 70 个专家验证模板；MIMIC 版 155 个 SCP 代码；7 类题型含双 ECG 对比 | arXiv:2306.15681 摘要 + GitHub README |
| GEM | CSN 预测 +7.4%、可解释性 +22.7%、grounding +25.3%（相对提升）；ECG-Grounding 30,000 指令对 | arXiv:2503.06073 摘要 |
| ECG-Expert-QA | 47,211 条专家验证 QA 对、12 个诊断任务（预印本版）；BIBM 2025 正式出版 | arXiv:2502.17475 摘要；BIBM DOI 10.1109/BIBM66473.2025.11356744 |
| ECG-Agent | 12 导联 Accuracy（1–5 分）：1B 3.44 / 3B 3.45 / 8B 3.50 / 32B 3.54 vs PULSE 2.27、GEM 2.48、Gemini-2.5-Flash 1.85；Lead I：1B 3.76=32B 3.76；Lead II：3B 3.88 最高；直接回复 Accuracy 1B 3.71 vs PULSE 1.86/GEM 1.75；judge 与人类 Spearman ρ=0.70（accuracy）/0.46（completeness）；解释工具 TIoU：PAC 64.5–67.3%、PVC 72.3–74.1%、ST↓ 71.9–76.0%；数据集 21,837 对话/98K 实例每配置/7.68 轮 | arXiv:2601.20323 全文 Table 1–3（ar5iv） |
| Cardiologent | LLM judge 与心脏病学家 ICC 0.74 / 0.66，心脏病学家互评 0.67；所有评测轴第一；数据 734,528 s 注释（κ=0.930）、测试 65 患者/18,873 窗口 | arXiv:2607.25340 摘要+全文 |
| CARE-ECG | Expert-ECG-QA 准确率 0.84；SCP-mapped PTB-XL 0.76（GPT-4 底座） | arXiv:2604.10420 摘要 |
| UniECG | 无量化临床指标；两阶段理解+生成（文本条件 ECG 扩散）；明确非临床验证系统 | arXiv:2509.18588 摘要 |
| ECG-R1 | 诊断准确率：ECG-R1(RL) 80.29 / (SFT) 79.33 vs GEM 74.70、PULSE 66.13、MedGemma-27B 25.23、GPT-5.1-Instant 31.48；临床诊断忠实度 84.20 vs GEM 62.90；跨模态一致性 BLEU-4 0.69 vs GEM 0.33、ROUGE-L 0.73 vs 0.43、SBERT 0.97 vs 0.92；30,000 协议化指令；2,381 测试 + 100 例 4 医生盲评 | arXiv:2602.04279 摘要/全文；数字经第三方解读页（papernotes）转述论文表格 |
| MEETI | MIMIC-IV-ECG 约 800,000 条；四组件（信号/300 dpi 图像/FeatureDB 逐搏参数/GPT-4o 文本） | Nature Scientific Data s41597-026-06796-1 全文 |
| ECGFounder | >1000 万条 ECG、150 标签；内部验证 80 个诊断 AUROC>0.95；外部多域验证；支持单导联 | arXiv:2410.04133 摘要（NEJM AI 2025） |
| ECG-FM | 150 万条预训练；AFib AUROC 0.996；LVEF≤40% AUROC 0.929 | arXiv:2408.05178 摘要 |

> 注：ECG-R1 的表格数字转引自第三方论文解读页（en.papernotes.org，见文末链接），已尽量与 arXiv 摘要/全文交叉核对；ECG-Chat、ECG-Expert-QA 的模型级指标未在公开摘要/仓库给出，如需引用建议直接查论文正文表格。

---

## 5. 来源链接

1. ECG-Chat：https://arxiv.org/abs/2408.08849 ；https://ar5iv.labs.arxiv.org/html/2408.08849 ；https://github.com/YubaoZhao/ECG-Chat
2. ECG-QA：https://arxiv.org/abs/2306.15681 ；https://github.com/Jwoo5/ecg-qa
3. GEM：https://arxiv.org/abs/2503.06073 ；https://ar5iv.labs.arxiv.org/html/2503.06073 ；https://github.com/lanxiang1017/GEM
4. ECG-Expert-QA：https://github.com/Zaozzz/ECG-Expert-QA ；https://arxiv.org/abs/2502.17475 ；https://ieeexplore.ieee.org/abstract/document/11356744
5. ECG-Agent：https://arxiv.org/abs/2601.20323 ；https://ar5iv.labs.arxiv.org/html/2601.20323 ；https://github.com/gustmd0121/ECG-Agent ；https://ieeexplore.ieee.org/document/11464123
6. Cardiologent：https://arxiv.org/abs/2607.25340 ；https://arxiv.org/html/2607.25340v1 ；https://github.com/sukjuoh/Cardiologent
7. CARE-ECG：https://arxiv.org/abs/2604.10420 ；https://arxiv.org/html/2604.10420v1
8. UniECG：https://arxiv.org/abs/2509.18588 ；https://ar5iv.labs.arxiv.org/html/2509.18588
9. ECG-R1：https://arxiv.org/abs/2602.04279 ；https://arxiv.org/html/2602.04279v3 ；https://github.com/PKUDigitalHealth/ECG-R1 ；https://en.papernotes.org/ICML2026/multimodal_vlm/ecg-r1_protocol-guided_and_modality-agnostic_mllm_for_reliable_ecg_interpretatio/
10. MEETI：https://www.nature.com/articles/s41597-026-06796-1 ；https://zenodo.org/records/18523205
11. ECGFounder：https://arxiv.org/abs/2410.04133 ；https://github.com/PKUDigitalHealth/ECGFounder
12. ECG-FM：https://arxiv.org/abs/2408.05178 ；https://github.com/bowang-lab/ECG-FM
