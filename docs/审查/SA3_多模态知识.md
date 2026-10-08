# 多模态与知识模块审查报告

审查范围：`src/context_modeling/`（分词器、文本/元数据编码、融合、对比学习）、`src/evaluation/metrics/classification.py`（评估协议）、`src/knowledge/`（知识库检索）、`src/utils/torchvision_stub.py`。
审查方式：13 个目标文件逐一 `read` 深读，并交叉核对 `data/knowledge/kb/classes.json` 数据格式与 `scripts/eval_*.py`、`src/ecg_models/trainer.py` 等调用链。

## 总体评价

模块整体结构清晰、职责划分合理，评估协议统一入口的设计（M0.4 固化版）值得肯定，论文用指标的调用链（验证集 Youden 阈值 → 测试集评估）在已审查的 `train_multimodal_fair.py` / `train_ecgfounder_head.py` / `eval_per_class.py` 中均正确执行。但存在 **5 个会直接影响论文数字正确性或训练有效性的严重问题**：VQ-VAE 的 perplexity 恒为 1、解码器与编码器采样率不对称、SupCon 分母把自身相似度计入负项、InfoNCE 温度无正性约束、ECGTextCLIP 双温度死参数。另有 13 个中等隐患（bootstrap CI 偏置、mAP 全零类 NaN、max pooling padding 泄漏、KB 模糊匹配误伤等）与 10 个轻微项。**结论：指标模块的 AUC/F1/topk 主路径正确，但对比学习与 VQ-VAE 部分存在会污染训练/监控信号的实质性 bug，论文复现前必须修复 🔴 项。**

---

## 各文件逐条发现

### 1. src/context_modeling/ecg_tokenizer.py（138 行）

- 🔴 `ecg_tokenizer.py:53` — **perplexity 公式错误，恒等于 ≈1，完全无信息量**。`F.one_hot` 每行仅一个 1，`one_hot * log(one_hot + 1e-10)` 只在 1 处 ≈ `1e-10`，其余 0×(-23)=0；总和对 B×L 个 token 求和却只除以 B（应除以 B×L 或直接按频数直方图），最终 `exp(-L·1e-10)≈1.0`。**codebook 使用率监控（论文常报的 VQ perplexity）恒为 1，无法发现 codebook 退化（死码）**。修法：`p = torch.bincount(indices, minlength=K).float() / indices.numel(); ppl = torch.exp(-(p * torch.log(p + 1e-10)).sum())`。
- 🔴 `ecg_tokenizer.py:86-92` — **解码器上采样 4 层 vs 编码器下采样 3 层，严重不对称**。`rev_dims = reversed(hidden_dims) + [input_channels]` 产生 4 个 `ConvTranspose1d(4,2,1)`，而编码器只有 3 个 stride-2 卷积（第 4 层是 stride-1 的 3×3）。输入 4096 → 编码 512 → 解码 8192，`forward`（104-105 行）被迫截断掉后一半输出。后果：重构"拉伸"2 倍后裁剪，前一半承载全部监督信号，后一半无梯度，重建质量与文档声明（L/2³）不符；`decode()` 独立调用时输出长度是 `encode` 的 2 倍，契约不一致。修法：decoder 只遍历 `reversed(hidden_dims)`（3 个上采样层），最后一层输出通道单独设为 `input_channels`（如 `ConvTranspose1d(in_dim, input_channels, 4, 2, 1)`），使 2³=8× 对称；截断/pad 分支恢复为纯防御代码。
- 🟡 `ecg_tokenizer.py:38` — codebook 初始化 `uniform(-1/1024, 1/1024)` 范围过小（±0.001），与潜在向量初始距离远，早期训练易死码；建议 `kaiming_uniform_` 或 `±1/sqrt(embedding_dim)`。
- 🟡 `ecg_tokenizer.py:94/103` — `encode` 实际返回 `(z_q 量化特征, indices, loss, ppl)`，与顶部文档"`tokens, features, vq_loss, ppl = vq.encode(ecg)`"及 `forward` 中 `recon = ...` 命名不符（`recon` 先装量化特征又被 decode 覆盖），易误导调用方。
- 🟡 `ecg_tokenizer.py:51` — VQ 损失两项顺序正确（第一项梯度流向 codebook，第二项 commitment 流向 encoder），无 bug；但若未来有人改成"先直通再算 loss"会破坏该性质，建议加注释。
- 💡 `ecg_tokenizer.py:97-100` — `encode_to_features` 用全局平均池化，会把 ST 段、QRS 形态等瞬态特征平均稀释；建议学习权重池化或注意力池化。
- 🟡 `ecg_tokenizer.py:90` — 末层 `BatchNorm1d + Tanh` 输出范围 [-1,1]，要求输入 ECG 已归一化到 [-1,1]，否则重构 MSE 有系统性尺度偏差，需在预处理侧确认。
- 注：`_enc_seq_len = input_length // 2**len(hidden_dims)` 与编码器实际输出 `floor(L/2^k)` 一致（恒等），无 bug。

### 2. src/context_modeling/text_encoder.py（98 行）

- 🟠 `text_encoder.py:90` — **max pooling 的 padding 泄漏**：`hidden * attention_mask` 把 padding 位置置 0 后再取 max；若某样本所有真实 token 的激活均为负值，max 会选中 padding 位置的 0 → 特征错误。修法：`hidden.masked_fill(mask==0, -1e9).max(1).values`。
- 🟠 `text_encoder.py:95-97` — 占位编码返回 **CPU 全零向量**：① 与 GPU 上的 ecg 特征拼接时 device mismatch；② 全零向量经 `F.normalize` 后为 NaN（0/0），接入 `ECGTextCLIP` 对比学习会直接污染整个 batch。修法：占位返回可学习/随机向量并放到调用方 device，或文档明确禁止占位模式接入对比学习。
- 🟠 `text_encoder.py:82` — 空列表 `texts=[]` 时 tokenizer 返回空 batch，pooled 形状 (0,D)，下游 concat/融合维度错乱；建议对空输入显式报错或短路。
- 🟡 `text_encoder.py:45` — `_output_dim=768` 硬编码仅在加载失败时生效，若真实模型非 BERT-base 会维度错配；建议占位时也从配置读取或抛错。
- 💡 `text_encoder.py:82` — 未处理 `tokenizer.pad_token is None` 的模型（如部分 GPT 系）；建议 `if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token`。
- 🟠（跨模块）`text_encoder.py:63` — 模型可能落在 CPU（冻结模型常见），输出 pooled 在 CPU，与 GPU 端 ecg 特征融合时无设备对齐；`multimodal_model.py` 也未做搬运，训练脚本必须自行 `.to(device)`，模块自身脆弱。

### 3. src/context_modeling/metadata_encoder.py（92 行）

- 🟠 `metadata_encoder.py:82` — **sex=None 时默认用索引 0（Female）**，缺失性别被当作女性，引入系统偏差；语义上应默认 Unknown(2)。修法：`torch.full((batch_size,), 2, ...)`。
- 🟠 `metadata_encoder.py:79` — `sex.long().clamp(0, 2)` 静默把负索引/越界值钳到 0（Female）：若数据用 -1 表示未知会被错误编码为女性；建议先映射未知值到 2 再 clamp 或校验。
- 🟠 `metadata_encoder.py:81-90` — device 推断 `features[0].device if features else None`：当 age、sex 均为 None 时 features 为空 → CPU 零张量，与 GPU 特征拼接失败；建议显式接受 `device` 参数。
- 🟠 `metadata_encoder.py:70` — `age < 0` 判定未知：NaN 年龄 `NaN<0=False` 会流入 `age_proj` 产生全 NaN；建议 `torch.isnan(age) | (age < 0)`。
- 🟡 `metadata_encoder.py:29` — `DEFAULT_SOURCES` 与数据管线中的源命名（`cpsc_2018_extra` 等）硬编码耦合，若命名漂移所有样本静默落入 unknown 嵌入；建议与数据管线共享常量或从配置注入。
- 🟡 `metadata_encoder.py:87` — 未知 source 的 fallback 逻辑正确，但 `known_sources` 不含 `"unknown"` 时静默落到索引 0，语义不透明。

### 4. src/context_modeling/multimodal_model.py（92 行）

- 🟠 `multimodal_model.py:78-81` — texts=None 用全零向量占位：在 GatedFusion 中 `text_proj(0)=bias` 尚可，但 CrossAttention 中 text 全零使 K/V=0、注意力退化；且占位向量无梯度信号。建议可学习占位参数或复用 text_encoder 的 unknown 向量。
- 🟠 `multimodal_model.py:84-89` — 未做设备对齐（见 text_encoder 问题），meta_feat/text_feat 与 ecg_feat 设备不一致时融合直接崩溃。
- 🟡 `multimodal_model.py:72` — 类型契约脆弱：文档称 ecg_encoder 用 SimpleECGProjector，但若误传 `ECGTokenizer`，`forward` 返回 4 元组（recon, indices, loss, ppl）→ 融合层崩溃；建议入口加输出形状断言。

### 5. src/context_modeling/fusion/cross_attention.py（76 行）

- 🟠 `cross_attention.py:21` — `head_dim = hidden_dim // num_heads` 无整除校验：传 `num_heads=7` 或 `hidden_dim=300` 时静默截断 head_dim，随后 `view` 直接崩；建议 `assert hidden_dim % num_heads == 0` 并给出清晰报错。
- 🟠 `cross_attention.py:67-69` — **无 attention mask**：当前调用方传 pooled 向量（序列长 1）+ meta（长 1），无 padding 问题；但若接入 token 级文本序列（变长 + padding），padding 位置会参与 softmax 污染注意力。建议 forward 增加可选 `attn_mask` 参数。
- 🟡 `cross_attention.py:52-75` — 变量 `ecg` 被投影后覆盖、再用于残差（74-75 行），可读性差，建议改名 `ecg_proj_out`。
- 注：残差形状（B,1,hidden）+ LayerNorm + FFN 正确；`metadata_features.unsqueeze(1)` 后与 text 拼接维度正确。

### 6. src/context_modeling/fusion/gated_fusion.py（108 行）

- 🟡 `gated_fusion.py:99` — metadata=None 时用 `zeros_like(ecg)`：gate 网络必须自行学到 g3≈0，浪费一个门自由度且初期可能把权重分给零向量；建议 meta 缺失时在 gate 输入中显式标记。
- 🟡 `gated_fusion.py:96-97` — 若 `metadata_features` 为 3D（如 (B,1,D)），与 2D 的 ecg/text 在 `torch.cat` 处维度不匹配直接报错；建议与 late_fusion 一样先做 3D→2D 池化。
- 注：softmax 门 + 加权求和公式正确。

### 7. src/context_modeling/fusion/late_fusion.py（82 行）

- 🟡 `late_fusion.py:75-79` — metadata=None 用零占位（device 处理正确，比 multimodal_model 严谨），但同样无"缺失标记"，信息上把缺失当作全零。
- 🟡 `late_fusion.py:71-74` — 3D 输入无条件 mean 池化，若传入的 3D 是带 padding 的序列会池化进 padding；当前调用方传 2D，仅为防御性代码。
- 注：其余实现正确，无可指摘。

### 8. src/context_modeling/contrastive/losses.py（97 行）— ⚠️ 论文重点关注

- 🔴 `losses.py:21/40` — **InfoNCE 温度是可学习 Parameter 且无正性约束**：优化器可直接把 `temperature` 推到 0 或负值 → 除以 0/负数 → logits 爆炸 → NaN。CLIP 原版用 `log(1/t)` 指数化保证正。修法：`self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))`，forward 中 `temp = self.log_temperature.exp().clamp(min=1e-4)`；若论文要固定温度 0.07，需 `requires_grad_(False)`。
- 🔴 `losses.py:94` — **SupCon 分母把自身相似度计入负样本项**：`pos_mask.fill_diagonal_(False)` 后对角为 False，`(~pos_mask)` 对角为 True → `exp_sim` 的对角项（normalize 后 sim_ii=1，τ=0.07 时 exp(14.3)≈1.6e6）**主导分母**，而正样本项通常仅 ~1e3，损失被严重压小、梯度被稀释，训练近乎失效。SupCon 原文分母排除自身（A(i)=I\{i}）。修法：`eye = torch.eye(B, device=..., dtype=torch.bool); neg_mask = (~pos_mask) & ~eye`，或 `neg_sum = exp_sim.sum(1) - exp_sim.diag()`。
- 🟠 `losses.py:92-96` — `torch.exp(sim)` 未做数值防护：温度小时 sim/τ 可超 20，fp32 尚可但结合可学习温度无下界即爆炸（与上条联动）；建议 logsumexp 形式或 clamp。
- 🟠 `losses.py:96` — 全 batch 无正样本时 `loss[pos_sum > 0].mean()` 为 NaN（empty mean）；多标签且 batch 内无同类时可能触发，建议 `if mask.any() else 0`。
- 🟡 `losses.py:60-62` — SupCon 温度是普通 float 而 InfoNCE 是 Parameter，两处语义不一致，建议统一为可学习（或统一固定）。

### 9. src/context_modeling/contrastive/ecg_text_clip.py（99 行）

- 🔴 `ecg_text_clip.py:51,53,98` — **双温度死参数**：`self.logit_scale = nn.Parameter(log(1/temperature))` 定义后从未参与 loss（`forward` 里 `loss_fn` 用自己的独立温度参数），logit_scale 是死参数（无梯度、永不更新），且与 loss_fn 内部温度**不一致**——`ECGTextCLIP(temperature=0.1)` 时 logit_scale 用 0.1、loss_fn 仍用默认 0.07。论文若报告"温度 0.07"，实际训练的是另一个温度。修法：删除 logit_scale，把 `InfoNCELoss` 的温度作为唯一温度源（可学习），或把 logit_scale 传给 loss_fn 复用。
- 🟠 `ecg_text_clip.py:41/64` — 依赖 `ecg_encoder.feature_dim`，但 `ECGTokenizer` 没有 `feature_dim` 属性（只有 `seq_len`）→ 误传即 AttributeError；建议在 ECGTokenizer 补 `feature_dim`（即 codebook_dim）或文档限定。
- 🟠 `ecg_text_clip.py:64/77` — `encode_ecg` 假设特征为 (B,D)；若 backbone 输出 (B,seq,D)，投影后仍是 3D，`loss_fn` 中 `z1 @ z2.T` 崩溃。建议显式池化或断言。
- 🟡 `ecg_text_clip.py:76-79` — `encode_text` 的 if/else 两分支完全相同，死分支。
- 🟠（跨模块）占位文本编码器（全零 CPU 向量）经本项目 normalize → NaN，会静默毒化对比预训练（与 text_encoder 问题联动）。

### 10. src/evaluation/metrics/classification.py（128 行）— ⚠️ 论文评估协议

- 🟠 `classification.py:32-44` — **bootstrap 中退化类被整体跳过**：抽样后某类全正/全负即 `continue`，导致每个 bootstrap 的 macro 均值基于**不同类集合**，CI 系统性偏窄且有偏（高方差类被剔除）。修法：bootstrap 内对退化类按 AUC=0.5 计入（保守）或按全数据该类 AUC 计（插值），并固定参与类集合；或在报告中披露"仅对非退化类平均"的口径。
- 🟠 `classification.py:44` — 全退化时 `boot` 为空回退 `(macro, macro)`，逻辑正确，但无任何提示，读者会把 0 宽度 CI 当真；建议打 warning。
- 🟠 `classification.py:29/90` — `roc_auc_score` / `average_precision_score` 未防护**常量概率列**（sklearn 新版会抛 ValueError）与**全零标签列**（mAP 返回 NaN 或告警）。AUC 已跳过退化类但 mAP 没有 → 口径不一致且可能整表 NaN。修法：mAP 逐类计算并对退化列跳过/置 0，统一防护。
- 🟠 `classification.py:81` — **evaluate() 缺省时在给定数据上现算 Youden 阈值**：若调用方直接 `evaluate(test_probs, test_labels)`，阈值在测试集上拟合 → F1 乐观（测试集泄漏）。已审查的论文脚本（`train_multimodal_fair.py:126-127`、`train_ecgfounder_head.py:226-228`、`eval_per_class.py:54-55`）均正确传入验证集阈值，但 API 本身是陷阱。修法：`thresholds` 改为必填或加 `allow_fit_on_eval=False` 守卫。
- 🟠（跨文件观察）`src/ecg_models/trainer.py:431-445` — `evaluate()` 在**测试集上搜索 PR 最优 F1 阈值**再报 macro F1，属于同类泄漏模式（不在本次审查清单内，但论文若引用 trainer 指标需一并处理）。
- 🟡 `classification.py:65` — `topk_hits` 的 `total` 只对含正类样本计数（等价于题目所述"无正类样本 total-=1"），逻辑正确；但 `k > num_classes` 时 `argsort(...)[:, :k]` 全取，无告警，建议 clamp。
- 注：`macro_auc_with_ci` 主路径（逐类 AUC → 宏平均 → 2.5/97.5 百分位）正确；`youden_thresholds`（`t[np.argmax(tpr-fpr)]`，全退化类保持 0.5）正确；`retrieval_recall` 正确（qid 不在 key 中跳过、rank 计算无误）。

### 11. src/knowledge/kb_loader.py（103 行）

- 🟠 `kb_loader.py:39-42` — **双向子串模糊匹配误匹配风险高**：`low in cname.lower() or cname.lower() in low` 对短查询灾难性——查询 "in" 会命中所有含 "in" 的类名（Sinus…、Atrial…），查询 "a" 命中几乎全部；且 `find` 返回**第一个**匹配项，dict 顺序取决于 JSON 文件写入顺序，结果不稳定。修法：① 最小查询长度（如 ≥3）；② 词边界匹配（正则 `\b`）；③ 用 `difflib.get_close_matches` 或 Levenshtein 相似度阈值；④ 多个候选时返回列表由上层消歧。
- 🟡 `kb_loader.py:70` — 截断在任意字符边界：可能把 markdown 行/列表项/中括号切半，且实际返回长度 = max_chars+7（截断标记）；建议按行截断并完整保留最后一行。
- 🟡 `kb_loader.py:62-64` — `it.get("citations", [])` 后直接 `c['source']/c['grade']`，数据缺字段会 KeyError；建议 `c.get(...)` 防御。
- 注：`KB_PATH` 上溯 3 级解析正确；精确匹配 → 大小写不敏感包含 → 无结果的查找链符合设计文档；`items[:8]` 每节截断是有意设计。

### 12. src/knowledge/sources.yaml（46 行）

- 💡 纯配置无代码 bug。建议：① 为 `sources:` 条目加 schema 校验（`kind`/`grade`/`enabled` 必填枚举），避免采集脚本静默缺字段；② `general_topics` 未标注 grade，若注入 LLM 需说明其可信度等级；③ litfl/utah 的 `enabled: false` 与注释（403/类名不稳定）记录良好，保持。

### 13. src/utils/torchvision_stub.py（74 行）

- 🟠 `torchvision_stub.py:43-50` — **检测逻辑存在边缘漏洞**：先查 `"torchvision" in sys.modules` 直接 return，再 try 顶层 `import torchvision`。若同进程此前某库已顶层导入成功但 `torchvision.io` 子模块缺失/损坏（顶层 OK、子模块坏的半损坏安装），桩不会安装，随后 `transformers.image_utils` 的 `from torchvision.io import ...` 依旧崩溃。修法：判据改为"顶层导入成功 **且** `import torchvision.io` 成功"才让位；或捕获子模块 ImportError 后装桩。
- 🟠（真实修复后的行为）`torchvision_stub.py:46-48` — 环境修复后顶层 `import torchvision` 成功 → 正确让位，**主路径行为正确**；但注意顶层 import 会触发 torchvision 全量初始化（含 ops），导入失败时也会留下部分初始化的副作用，建议 import 放在 try 内并记录 warning。
- 🟡 `torchvision_stub.py:58-61` — `InterpolationMode` 成员按 transformers 版本可能还需 `AREA`/`NEAREST_EXACT`（0.22+ 已有）等；建议按 `transformers.image_utils` 实际引用的成员补齐，或对缺失成员动态补充。
- 🟡 `torchvision_stub.py:35-36` — `_fail` 抛 RuntimeError 但未说明是桩被误用，错误信息已足够清晰，无问题；`ModuleSpec(loader=None)` 使 `importlib.reload` 不可用，可接受。

---

## 修复优先级清单

**P0（论文数字/训练有效性，必须立即修）**
1. `losses.py:94` — SupCon 分母排除自身（否则对比预训练近乎失效）。
2. `losses.py:21/40` — InfoNCE 温度指数化+下界（防 NaN 爆炸）。
3. `ecg_text_clip.py:51/53` — 删除死参数 `logit_scale`，统一单一温度源。
4. `ecg_tokenizer.py:53` — perplexity 改为频数直方图计算（否则监控信号恒 1）。
5. `ecg_tokenizer.py:86-92` — 解码器改为 3 层上采样，与编码器 8× 对称。

**P1（指标正确性/边界，论文报告前修）**
6. `classification.py:32-44` — bootstrap 退化类插值 0.5，统一类集合口径。
7. `classification.py:90` — mAP 全零类防护，与 AUC 口径对齐。
8. `classification.py:81` — `evaluate()` 强制显式传验证集阈值（防测试集 Youden 泄漏）；`trainer.py:431-445` 同模式一并处理。
9. `kb_loader.py:39-42` — 模糊匹配加最小长度/词边界/相似度阈值。
10. `text_encoder.py:90` — max pooling 改 `masked_fill(-1e9)`。
11. `text_encoder.py:95-97` — 占位编码去全零（设备+NaN 隐患）。
12. `metadata_encoder.py:82` — sex 缺失默认 Unknown(2) 而非 Female；负索引映射未知。
13. `cross_attention.py:21` — head_dim 整除断言；后续支持 attn_mask。
14. `torchvision_stub.py:43-50` — 判据升级为顶层+io 子模块双验证。

**P2（健壮性与可维护性）**
15. `ecg_text_clip.py:41` — ECGTokenizer 补 `feature_dim` 或加断言。
16. `multimodal_model.py:78-81` — 文本缺失改可学习占位；模块内统一设备搬运。
17. `metadata_encoder.py:70/79` — NaN 年龄、sex 越界显式处理。
18. `classification.py:65` — topk 的 k clamp；`classification.py:44` 空 boot 打 warning。
19. `kb_loader.py:70` — 按行截断；`kb_loader.py:62-64` citation 字段防御。
20. `ecg_tokenizer.py:38` — codebook 初始化范围修正；文档/返回命名统一。
