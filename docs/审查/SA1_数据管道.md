# 数据管道模块审查报告

审查范围: `src/data_pipeline/`（loader / preprocessor / augmentor / dataset / label_extractor / metadata_parser / __init__）+ `tests/test_loader.py`、`tests/test_pipeline.py`
对照基准: `scripts/preprocess_data_5k.py`（主用 5000 样本协议）、`scripts/preprocess_data.py`（4096 版）、`scripts/repair_manifest_5k.py`、`scripts/train_backbone.py`
审查日期: 2025（本次会话）

## 总体评价

模块分层清晰、docstring 详尽、27 类 SNOMED 映射与 PhysioNet 2020 官方定义核对无误，工程骨架是合格的。但存在 **2 个 🔴 级静默数据损坏缺陷**：重采样因子钳制循环会破坏采样率比例（257Hz 等含大素因子的采样率下内容被压缩约一半），`apply_time_warp` 的插值方向与文档相反且会在约半数增强样本尾部制造长达 20% 的平直线伪迹。中等问题集中在三处：**多进程 DataLoader 下增强随机性根本不按 `_worker_init_fn` 工作**（Generator 在父进程播种后 fork，所有 worker 同态、逐 epoch 重复）；**模块默认协议（0.5–45Hz 带通+逐导联 z-score+5σ 裁剪+4096 长度）与项目主用的 5000 样本协议（[0.67,40]Hz+0.4s 中值基线+全局 z-score+不裁剪+5000 长度）不一致**；**划分只按记录不按患者、且只包含处理成功的记录**，存在同患者跨划分泄漏与静默丢记录风险。测试方面，两个测试文件在数据缺失时"静默通过"（假绿色），且 dataset.py 三个数据集零测试、增强只验形状不验内容，关键缺陷全部漏网。

---

## 各文件逐条发现

### 1. src/data_pipeline/loader.py

| # | 行号 | 严重级 | 问题 | 建议修法 |
|---|------|--------|------|----------|
| 1 | 283–285 | 🟠 | `load_record` 捕获所有异常并 `return None`，文件损坏、磁盘错误、wfdb 解析失败与"形状异常"被一视同仁地静默丢弃。43K 条数据中只要有一批坏文件，train/val/test 划分会悄悄缺条目（5k 脚本仅统计 `skipped` 总数，无法定位原因）。 | 区分异常类型：校验类跳过记 warning+计数，IO/解析类异常记录 `record_ref` 与异常栈；提供 `strict=True` 开关直接抛出；`iter_records` 返回 (样本, 跳过原因) 统计。 |
| 2 | 258 | 🟠 | 信号不做 NaN/inf 清洗直接进入下游。若 .hea 记录含 NaN（部分数据源存在），NaN 会穿透 `sosfiltfilt` → mean/std 变 NaN → 整条记录毒化（见 preprocessor #10）。5k 脚本的 `filter_bandpass` 做了 `nan_to_num`，而 4096 协议路径（ECGPreprocessor）没有。 | 在 `load_record` 中 `signal = np.nan_to_num(signal, nan=0.0)`，或至少在 preprocessor 入口统一清洗。 |
| 3 | 230–232 | 🟡 | 裸记录名（无路径）用 `rglob` 返回**第一个**匹配；同名记录存在于多个数据源时（跨源重复 ID），来源归属与加载结果不确定。 | 返回匹配列表；多于一个时记 warning 并支持 `source` 参数消歧。 |
| 4 | 6–8 | 🟡 | docstring 声称 ".mat 不是 MATLAB 格式"、并描述"一阶差分编码需 cumsum 解码"。事实：Challenge 2020 的 `.mat` 是 MATLAB v5 文件（int16 `val`），由 wfdb 内部解析；本文件没有任何 cumsum 逻辑，依赖 wfdb 行为，注释与实现脱节且有误导。 | 修正注释；建议加一条"用 wfdb 读出的数值与官方 helper_code 比对"的冒烟断言，防止 wfdb 版本行为差异静默改变数值。 |
| 5 | 367–379 | 🟡 | `get_record_count()` 统计包含隐藏 `.hea` 文件，而 `iter_records()` 跳过隐藏文件 → 5k 脚本的 tqdm 总进度条与真实处理数不一致。 | 两处统一跳过隐藏文件的过滤逻辑。 |
| 6 | 430 | 🟡 | `nan_rate` 实为"年龄未知率"（`1 - 有效年龄数/总数`），命名误导；且性别未知未纳入统计。 | 改名 `age_unknown_rate`，或按年龄缺失/性别缺失分别统计。 |
| 7 | 40–60 | 💡 | `ECGSample.__init__` 不校验导联数=12（`lead_names` 默认 12 个名字但信号可能是 6/8 导联）；`fs` 仅校验 >0 未校验异常大值。 | 增加 shape/lead 数一致性断言；补充基于 mock WFDB 数据的单测（`_parse_metadata` 正则、`_find_record_path` 三分支、`_infer_source`、age/sex/dx 属性边界）。 |

### 2. src/data_pipeline/preprocessor.py

| # | 行号 | 严重级 | 问题 | 建议修法 |
|---|------|--------|------|----------|
| 8 | 222–225 | 🔴 | 重采样因子钳制循环 `up=(up+1)//2; down=(down+1)//2` 只对偶数保持比例。对含大素因子的采样率（如 257Hz）：500/257 → 63/65，比例从 1.9455 变成 0.9692，**内容被压缩约一半**（QRS 被压扁），随后按 target_len 补零，长度正确但内容损坏，全程无警告。500/1000Hz 因 gcd 约简后因子小不触发，属潜伏缺陷。 | 用浮点比例近似替代：`scipy.signal.resample`（FFT 法）按 `target_len` 直接输出，或先算 `new_len=round(L*ratio)` 再对整段做一次 `resample_poly` 而不拆因子；至少检测 `abs(up/down - target_fs/orig_fs) > 1e-3` 时报警/抛错。**5k 脚本 `resample_to_500`（L118–121）同款 bug 需一并修复**。 |
| 9 | 37–46 | 🟠 | 模块默认协议与项目主用 5k 协议不一致：(a) 带通 0.5–45Hz vs 官方 [0.67,40]Hz；(b) 无 0.4s 中值滤波基线漂移去除步骤（5k 有，且这是官方配方核心）；(c) 逐导联 z-score vs 官方全导联合并全局统计；(d) 默认 `outlier_threshold=5.0` 做 5σ 裁剪，而 5k 协议明确"不做 5σ 裁剪"；(e) 默认 `target_length=4096` vs 5000。任何人用本模块重新生成 5k 数据会得到分布不一致的结果。 | 提供 `protocol="ecgfounder"` 参数一键对齐 5k 配方（含中值基线步骤与全局 z-score 选项），或把 5k 配方抽到模块内供脚本复用，消除两套配方漂移。 |
| 10 | 133/137 | 🟠 | `sosfiltfilt` 前无 NaN/inf 清洗，NaN 一路传播到归一化产生 NaN 输出（配合 loader #2）。 | `__call__` 入口 `ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)`（与 5k `filter_bandpass` 一致）。 |
| 11 | 73 | 🟡 | 滤波器缓存仅以 `fs` 为键；同一实例中途修改 `bandpass_high/notch_freq` 等参数后缓存不失效，静默沿用旧系数。 | 缓存键改为 `(fs, bandpass_low, bandpass_high, notch_freq, filter_order)` 元组。 |
| 12 | 353–359 | 🟡 | `_clip_outliers` 对**含零填充的整条导联**重算 mean/std：填充零拉低 std → 裁剪界变紧，可能误裁真实 R 波峰。 | 复用 `_normalize` 的 pad 区域信息，只对信号区统计裁剪界（或裁剪时排除 pad 区）。 |
| 13 | 79–83 | 🟡 | `bandpass_high >= 0.99×Nyquist` 直接抛 ValueError：fs<90.4Hz 时 45Hz 上限即超界，整条记录被判失败。当前数据源无此采样率，但模块声称通用。 | 自动降阶/降上限并 warning，而不是抛错；或在文档中明确支持的 fs 下限。 |
| 14 | 363–407 | 💡 | `segment()` 不校验 `ecg.ndim`；超短信号（<padlen）下 `sosfiltfilt` 会因 padlen 限制崩溃（loader 的 100 样本下限已缓解但仅限该路径）。测试未覆盖重采样（orig_fs≠target_fs）、短信号对称填充、带填充归一化三条路径。 | 增加 ndim/长度校验；补齐上述三条路径的断言测试（含内容级断言，如重采样后 QRS 峰值位置误差 < 2 样本）。 |

### 3. src/data_pipeline/augmentor.py

| # | 行号 | 严重级 | 问题 | 建议修法 |
|---|------|--------|------|----------|
| 15 | 216–221 | 🔴 | `apply_time_warp` 两处缺陷：(a) 语义与文档相反——`result[n]=ecg[n·scale]`，scale>1 时内容被**压缩**（文档称"拉伸"）；(b) scale<1 时定义域只有 [0, scale·L]，查询到 L-1 的区间全部落入 `fill_value=(ecg[0], ecg[-1])` 的上界 → 尾部 (1−scale)·L 被平直填充为常数 `ecg[-1]`。scale 均匀取 [0.8,1.2]，约半数增强样本尾部出现最长 20% 的平直线伪迹，模型会学到虚假模式。 | 用正确的时间扭曲：`y[n] = interp(orig_positions, ecg, n/scale)`（拉伸时输出更长再裁剪回 L），或 `resample_poly` 拉伸后居中裁剪；去掉 fill_value 平直尾，改为两端镜像/回绕填充；同步修正 docstring 的 scale 语义。 |
| 16 | 86 + dataset.py 392–399 | 🟠 | "多进程安全/可复现"不成立：`default_rng(seed)` 在父进程播种，DataLoader 各 worker fork 到**相同初始态**；`worker_init_fn` 只设 `np.random.seed`（全局），而增强用的是 `self.rng`（Generator），完全不受影响。后果：4 个 worker 只有 4 条相同起点、逐 epoch 重复的增强流（每 epoch 重新 fork 同一父态）；对比学习 DataLoader（train_backbone.py L121）连 `worker_init_fn` 都没传。 | 把 `worker_init_fn` 改为实例方法，在 worker 内 `self.augmentor.rng = np.random.default_rng(worker_seed)`（每个 worker 有独立副本，仅影响自己）；shuffle 可复现需另传 `generator=torch.Generator().manual_seed(...)`。 |
| 17 | 288–308 | 🟠 | `apply_segment_shuffle` 最后一段宽度为 `L−(num_segments−1)·seg_len`，非整除时比前段长 → `np.concatenate` 形状不匹配直接 **ValueError 崩溃**（如 L=5000, num_segments=3）。当前默认 chunk=1.0s 恰好整除（4096/8、5000/10）才未触发，属潜伏崩溃。 | 各段等长切分（丢弃尾部余数），或分段后每段统一 `resize` 到 seg_len；补充非整除用例测试。 |
| 18 | 160 | 🟡 | `add_baseline_wander` 幅度用 `np.std(ecg[i])`——若在 dataset 中增强发生在 padding 之后，统计含填充零，短信号漂移幅度被低估。 | 增强放到 padding 之前执行，或传入信号有效区掩码。 |
| 19 | 42–44 | 🟡 | docstring 宣称 Generator "在 DataLoader 多进程环境下更安全"——恰好相反，fork 复制的是同一状态（见 #16），表述误导。 | 修正注释，并落实 #16 的修复。 |
| 20 | — | 💡 | 测试只验形状：time_warp 的语义反转与平直尾、segment_shuffle 的非整除崩溃、多 worker 种子一致性均无测试。 | 增加内容级断言（扭曲方向、无长平直段）、非整除分段、`num_workers=4` 与 `=0` 的增强可复现性对照测试。 |

### 4. src/data_pipeline/dataset.py

| # | 行号 | 严重级 | 问题 | 建议修法 |
|---|------|--------|------|----------|
| 21 | 146–153 | 🟠 | `labels` 长度/维度零校验：`record.get("labels", [])` 为空或长度≠27 时，产出 (0,) 或错误维度的张量，批量 collate 中途崩溃（报错位置远离根因）；仅当 `label_extractor` 存在且 dx_codes 在时才在线编码，两条路径并存易漂移。 | 统一入口：恒走 `label_extractor.encode(dx_codes)`（manifest 的 labels 仅作缓存），并断言 `labels.shape == (27,)`，异常时给出 record_id 与文件路径。 |
| 22 | 155–163 | 🟠 | `return_metadata=True` 返回含 `dx_codes`（字符串 list）、`source`（str）的 dict，默认 `collate_fn` 会对 str list 调用 `torch.as_tensor` 而崩溃；代码未提供/未文档化配套 collate_fn，`ECGDatasetForAgent` 同理。 | 内置/导出默认 `collate_fn`（dx_codes 保持 list、标量转 tensor），并在 docstring 写明用法。 |
| 23 | 66 / 326 / 128–137 | 🟠 | 默认 `target_length=4096`：把主用协议的 processed_5k（5000 样本）喂进来会被**静默居中裁剪**掉首尾各 452 点（约 0.9s），与 5000 协议（10s 完整保留）不一致，且无任何提示；`ECGPreprocessor`/`ECGContrastiveDataset` 默认同值。 | 默认值改为 5000 对齐主协议，或从 manifest 的 `signal_shape` 自动读取长度；长度不匹配时发 warning。 |
| 24 | 392–399 | 🟠 | `_worker_init_fn` 只种 `np.random` 全局，对使用独立 Generator 的 augmentor 无效（详见 augmentor #16），等于"多进程可复现"的实现缺失。 | 见 augmentor #16 修复方案；对比学习 DataLoader 也需接入。 |
| 25 | 113 | 🟡 | `nan_to_num(..., nan=0.0)` 把 NaN 替换为 0——在已 z-score 归一化的数据里 0 是合法取值，若上游残留 NaN 会被伪装成真实 0 值样本。 | 加载后检测 NaN 并 `logger.warning`（带文件名），再由调用方决定丢弃或修复；上游 loader/preprocessor 已清洗后此处仅为兜底。 |
| 26 | 141 | 🟡 | `self.augmentor(signal)` 未传 fs，augmentor 默认 500Hz；若处理 1000Hz 数据，基线漂移/片段打乱的秒数换算错误。 | 从 manifest 读 `fs_target` 传入。 |
| 27 | — | 💡 | 三个 Dataset + DataModule 在 test_pipeline.py 中**零测试**；padding 区域未暴露给下游（注意力掩码/评估无法区分填充与真实信号）。 | 补 dataset 单测（形状校正、长度对齐、标签路径、collate）；考虑返回 `valid_mask`。 |

### 5. src/data_pipeline/label_extractor.py

| # | 行号 | 严重级 | 问题 | 建议修法 |
|---|------|--------|------|----------|
| 28 | 121–124 | 🟠 | `num_classes<27` 时按字典序截取排序前缀，等价映射中指向被截断规范码的条目被静默丢弃（`if canonical in self.snomed_to_idx` 守卫吞掉），无警告。 | 截断时 warning；或改为仅支持 27（其它值抛 ValueError）。 |
| 29 | 121 | 🟡 | 类索引按 SNOMED 码字典序排列，**不是官方 Challenge 的 27 类顺序**；与官方评测脚本或官方预训练权重对接时标签向量会整体错位。项目内部自洽，跨生态不兼容。 | 在模块内固化官方类顺序表（与 CHALLENGE_CLASSES 定义顺序一致），或在 save_mapping 中记录顺序来源；与官方 evaluation-2020 对照测试。 |
| 30 | 249–254 | 🟡 | `get_unmapped_codes()` 返回 `{code: 0}`——计数值恒为 0，与"用于诊断"的用途不符（文档暗示统计）。 | 实现计数（`_unmapped_codes` 改为 Counter）或改名 `get_unmapped_codes()` 返回纯集合。 |
| 31 | 213–216 | 🟡 | `get_category` 只查 `CHALLENGE_CLASSES`，等价码（如 428750005）返回 None，与 `get_class_name` 的行为不一致。 | 先经 `_full_mapping` 归一化再查类别。 |
| 32 | 36–102 | 💡 | 27 类定义与官方核对**无误**（9 节律 + 8 传导 + 10 形态），等价映射表设计合理——这是本模块最扎实的部分。缺测试：等价码编码（428750005→VEB）、27 类完整性断言（`len(CHALLENGE_CLASSES)==27`）、round-trip encode/decode。 | 补上述测试；`load_mapping` 后校验 `snomed_to_idx` 与内置表一致，防止旧 JSON 静默降级。 |

### 6. src/data_pipeline/metadata_parser.py

| # | 行号 | 严重级 | 问题 | 建议修法 |
|---|------|--------|------|----------|
| 33 | 129–131 | 🟠 | 关键词匹配为**无词边界的子串**匹配："af" 命中 "afternoon/safe"、"dm" 命中 "admin"、"mi" 命中 "minimal" → 病史/症状标志静默假阳性。 | 对英文用正则词边界（`\baf\b` 或先分词），中文保持子串；"af" 等 2 字母缩写优先要求上下文或长度≥3 才匹配。 |
| 34 | 55 | 🟡 | `age_default=60.0` 参数从未被使用（死参数）。 | 删除或实现"未知年龄回填默认值"开关。 |
| 35 | 80 | 🟡 | 归一化除 120，但 `_parse_age_raw` 允许到 122 → 输出可 >1.0，突破 docstring 声称的 [0,1] 区间。 | 归一化分母改 122 或上限改 120，二者统一。 |
| 36 | 30–31 | 🟡 | `"1"→Male`、`"0"→Female` 的映射假设了特定编码约定，跨数据源存在反转风险（部分库 1=Female）。 | 仅对明确文本值（Male/Female/M/F）做映射，数字编码交给调用方显式配置。 |
| 37 | 134–138 | 🟡 | `encode_metadata_vector` 的 `include_dx` 参数标注"尚未实现"却暴露在签名中（死参数）。 | 实现或移除。 |

### 7. src/data_pipeline/__init__.py

| # | 行号 | 严重级 | 问题 | 建议修法 |
|---|------|--------|------|----------|
| 38 | 3–8 | 🟡 | 只导出 2 个 Dataset（ECGDataset、ECGDataModule），遗漏 `ECGContrastiveDataset` 与 `ECGDatasetForAgent`——与 dataset.py 文档"3 种 Dataset"不一致，调用方被迫深路径导入。 | 补齐导出并更新 `__all__`。 |

### 8. tests/test_loader.py

| # | 行号 | 严重级 | 问题 | 建议修法 |
|---|------|--------|------|----------|
| 39 | 19–21/35–37/60–62/82–84 | 🟠 | 数据目录不存在时 `print("SKIP")` + `return`——**测试"通过"但什么都没测**。CI 上无数据也全绿，loader 实际零覆盖；且 `if __name__=="__main__"` 照样打印 "All loader tests passed!"，误导性极强。 | 改为 `pytest.mark.skipif`（带原因）；用合成 WFDB 文件（写入临时目录的假 .hea/.mat）做真正单测，不依赖真实数据集。 |
| 40 | 15–17 | 🟡 | 硬编码机器绝对路径 `c:/Users/llyun/Desktop/ecg资料/...`，换机器即失效（且正是 SKIP 分支的诱因）。 | 数据路径走环境变量/夹具参数。 |
| 41 | — | 💡 | 测试断言全是形状/存在性，`_parse_metadata`、`_find_record_path`、`_infer_source`、`ECGSample` 属性边界（age NaN/122 上限、sex 变体、dx 过滤）均未覆盖。 | 补充基于合成数据的单元测试。 |

### 9. tests/test_pipeline.py

| # | 行号 | 严重级 | 问题 | 建议修法 |
|---|------|--------|------|----------|
| 42 | — | 🟠 | dataset.py 的 ECGDataset / ECGContrastiveDataset / ECGDatasetForAgent / ECGDataModule（含 worker 种子逻辑）**完全没有测试**，而这是训练数据主入口。 | 用临时目录 + 合成 .npy + manifest 补全四个组件的测试。 |
| 43 | 15–47 | 🟡 | `test_preprocessor` 只验形状与 dtype：未覆盖重采样路径（orig_fs≠target_fs）、短信号对称填充、带填充归一化；未断言滤波/归一化的内容正确性（如 50Hz 分量是否被抑制）。 | 补内容级断言（频谱能量、填充区不参与统计）。 |
| 44 | 49–94 | 🟡 | `test_augmentor` 仅查形状；time_warp 只测 `scale=0.9` 且不查内容（缺陷 #15 完全漏网）；segment_shuffle 未测。 | 按 augmentor #20 建议补内容与边界测试。 |

### 10. 跨文件：与 preprocess_data_5k.py 的协议一致性（特别关注项）

| # | 位置 | 严重级 | 问题 | 建议修法 |
|---|------|--------|------|----------|
| 45 | 5k L157–175 vs 模块 | 🟠 | 导联重排（`reorder_leads`）只存在于 5k 脚本；模块管线（ECGPreprocessor→ECGDataset）从不校验/重排导联顺序。loader 注释声称 lead_names "用于下游导联顺序校验/重排"，但下游没有任何消费方（死承诺）。若未来用模块管线重新生成数据，导联排列将取决于 .hea 文件内顺序 → 与已训练模型静默错位。 | 把 `reorder_leads` 下沉到模块（loader 或 preprocessor），并在 ECGDataset 加载时校验 manifest 记录的标准导联序；删除或兑现 loader 的死承诺。 |
| 46 | 旧 preprocess_data.py L148–161 + 5k L16 | 🟠 | **划分按记录不按患者**：旧版按数据源内随机打散记录，PTB/Chapman 等多记录同患者数据源存在同患者跨 train/test 泄漏；5k 脚本复用旧划分并宣称"防泄漏"，但保证的只是"与旧版一致"，不是"患者隔离"。 | 若数据源含患者 ID（如 PTB 记录名前缀），按患者分组后整组划分；无法获得患者 ID 时在文档中明示该限制并评估影响。 |
| 47 | 5k L253–264 + 旧 L137–142 | 🟠 | 划分只包含**处理成功**的记录：任何偶发加载/预处理失败（含缺陷 #8 在 257Hz 下的静默损坏）都会让该记录永久缺席所有划分（train/val/test 都没有），静默改变数据构成与类别分布。 | 失败记录单独记录清单；划分前对失败记录重试或显式排除并生成报告，保证"每条记录要么进划分要么有明确原因"。 |
| 48 | 5k L118–121 | 🟡 | 与 preprocessor #8 相同的重采样因子钳制缺陷（257Hz 内容压缩约一半），同源复制粘贴，需同步修复。 | 见 #8 修复方案。 |

---

## 修复优先级清单

### 🔴 严重（立即修复，涉及静默数据损坏）
1. **preprocessor.py L222–225（+5k 脚本 L118–121）**: 重采样因子钳制破坏采样率比例（257Hz 等含大素因子 fs 下内容压缩约一半，无警告）。
2. **augmentor.py L216–221**: `apply_time_warp` 语义与文档相反 + scale<1 时尾部 (1−scale)L 平直伪迹（最长 20% 信号，约半数增强样本受影响）。

### 🟠 中等（尽快修复，影响正确性/一致性/可复现性）
3. **dataset.py L146–153**: labels 长度/维度零校验 → collate 中途崩溃或错误维度。
4. **dataset.py 默认 target_length=4096 vs 5000 协议**: 对 processed_5k 静默裁剪 0.9s。
5. **preprocessor 默认协议与 5k 协议不一致**（滤波配方/基线去除/z-score 方式/5σ 裁剪/长度）。
6. **augmentor + dataset 多进程随机性**: `_worker_init_fn` 对 Generator 无效，增强流跨 worker/epoch 重复，声称的可复现性落空。
7. **augmentor.py L288–308**: `apply_segment_shuffle` 非整除时 concat 崩溃。
8. **loader.py L283–285 + 5k L253–264**: 异常全吞 → 记录静默丢失且无定位信息。
9. **loader L258 + preprocessor L133**: NaN 信号无清洗，可毒化整条记录。
10. **划分按记录不按患者**（旧 L148–161 / 5k L16）: 同患者跨划分泄漏风险。
11. **导联重排仅存在于 5k 脚本**，模块管线无校验/重排，混用管线会静默错位。
12. **metadata_parser L129–131**: 无词边界子串匹配 → 病史/症状假阳性。
13. **dataset.py L155–163**: return_metadata/Agent 数据集的 dict 无配套 collate_fn。
14. **label_extractor L121–124**: num_classes<27 静默截断 + 等价码丢弃。
15. **test_loader.py 数据缺失时假绿色**、**test_pipeline.py dataset 零测试**（测试可信度）。

### 🟡 轻微（择机修复）
16. loader: .mat 格式注释错误、裸记录名歧义、get_record_count 隐藏文件计数不一致、nan_rate 命名。
17. preprocessor: 滤波器缓存键、_clip_outliers 含填充统计、低 fs 抛错策略。
18. augmentor: baseline wander 含填充 std、docstring "多进程更安全" 误导。
19. dataset: nan_to_num→0 语义、augmentor fs 硬编码 500、teardown 空实现。
20. label_extractor: 类顺序非官方顺序、get_unmapped_codes 计数恒 0、get_category 不支持等价码。
21. metadata_parser: age_default 死参数、归一化分母 120 vs 上限 122、数字性别映射脆弱、include_dx 死参数。
22. __init__.py: 缺 2 个 Dataset 导出。
23. 测试: 硬编码绝对路径、覆盖缺口（重采样/填充/增强内容/等价映射/多 worker 种子）。

---

*附注: 本报告所有行号基于审查时文件快照；`label_extractor.py` 27 类 SNOMED 定义经与 PhysioNet 2020 官方类表核对一致（这是本模块的正确性亮点）。*
