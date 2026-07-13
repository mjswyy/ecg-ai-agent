# Phase 1: 数据管道 — 完成报告

> **日期**: 2026-07-07
> **状态**: ✅ 完成（经过 5 轮全面审查，7 项技术债务已清零）
> **代码量**: ~1,700 行 Python + ~250 行配置/测试
> **预处理数据**: 43,101 条 12 导联 ECG → (12, 4096) float32 .npy 文件（~4 GB）

---

## 1. 目录结构

### 1.1 当前实际文件

> **说明**: 以下为 Phase 1 完成后的实际文件。Phase 2-7 的目录已创建但仅有 `__init__.py` 占位。

```
ecg-ai-agent/
├── Phase1_数据管道完成报告.md          # 本报告
├── requirements.txt                    # 全部 Python 依赖
│
├── configs/
│   └── data_config.yaml                # 数据管道完整配置（采样率、滤波参数、增强开关等）
│
├── data/
│   ├── physionet2020/
│   │   ├── raw/                        # 原始 PhysioNet 数据（指向 training/ 目录）
│   │   ├── processed/                  # 43,101 个 .npy 预处理文件 + 3 个 manifest JSON
│   │   │   ├── train_manifest.json     # 训练集 30,167 条
│   │   │   ├── val_manifest.json       # 验证集 6,462 条
│   │   │   └── test_manifest.json      # 测试集 6,472 条
│   │   ├── splits/                     # 预留（split 逻辑在 preprocess_data.py 中）
│   │   └── labels/
│   │       └── snomed_to_class.json    # 27 类 SNOMED CT 映射 + 42 个等价代码
│   └── external/                       # 预留（其他数据集）
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_pipeline/                  # ✅ Phase 1 完成
│   │   ├── __init__.py                 # 模块导出
│   │   ├── loader.py                   # ECG 数据加载器（365 行）
│   │   ├── preprocessor.py             # 信号预处理管道（356 行）
│   │   ├── augmentor.py                # 6 种数据增强（320 行）
│   │   ├── label_extractor.py          # 27 类标签编码 + 等价映射（392 行）
│   │   ├── metadata_parser.py          # 患者元数据解析（193 行）
│   │   └── dataset.py                  # 3 种 Dataset + DataModule（340 行）
│   │
│   ├── ecg_models/                     # ⏳ Phase 2
│   │   ├── backbone/
│   │   ├── classifiers/
│   │   └── feature_extraction/
│   │
│   ├── context_modeling/               # ⏳ Phase 3
│   │   ├── fusion/
│   │   └── contrastive/
│   │
│   ├── agent/                          # ⏳ Phase 4
│   │   ├── core/
│   │   ├── tools/
│   │   ├── llm/
│   │   └── orchestration/
│   │
│   ├── integration/                    # ⏳ Phase 5
│   └── evaluation/                     # ⏳ Phase 7
│       └── metrics/
│
├── tests/
│   ├── __init__.py
│   ├── test_loader.py                  # Loader 单元测试
│   └── test_pipeline.py                # Preprocessor/Augmentor/Label 集成测试
│
├── scripts/
│   └── preprocess_data.py              # 端到端预处理脚本（~210 行）
│
├── notebooks/                          # Phase 2+ 使用
├── paper/sections/                     # Phase 11-12 使用
└── web_demo/                           # Phase 6 使用
```

---

## 2. 核心模块详解

### 2.1 数据加载器 (`loader.py`)

**两个核心类**: `ECGSample`（数据容器）+ `ECGLoader`（批量加载器）

#### ECGSample — 单条 ECG 记录

```
ECGSample
├── 属性
│   ├── record_id: str          # 记录标识（如 "A0001"）
│   ├── signal: np.ndarray      # 12 导联信号 (12, L)，单位 mV
│   ├── fs: float               # 原始采样率 Hz
│   ├── metadata: Dict          # 原始元数据 {"Age": "74", "Sex": "Male", ...}
│   ├── source: str             # 数据来源（如 "cpsc_2018"）
│   └── lead_names: List[str]   # 导联名称
│
├── 计算属性
│   ├── duration → float        # 信号时长（秒）
│   ├── num_leads → int         # 导联数
│   ├── num_samples → int       # 样本数
│   ├── age → Optional[float]   # 解析后的年龄（NaN/inf/越界 → None）
│   ├── sex → str               # 标准化性别 "Male" / "Female" / "Unknown"
│   └── dx_codes → List[str]    # 诊断码列表
```

**关键设计**:
- Age 安全解析: `NaN`、`inf`、负数、>120 岁全部返回 `None`，防止毒化下游统计
- Sex 标准化: 处理 `Male/Female/M/F/NaN` 等多种输入格式
- 导联名默认值采用 `or` 短路而非 `is None` 检查（空列表 `[]` 会被替换为默认 12 导联）
- `duration` 属性: `fs ≤ 0` 时安全返回 `0.0`（防止损坏头文件导致 ZeroDivisionError）

#### ECGLoader — 批量数据加载

```
ECGLoader(raw_dir)
├── sources: List[str]            # 自动发现的 6 个数据源
├── load_record(ref) → ECGSample  # 单条加载
├── iter_records(sources, max)    # 批量迭代（生成器，内存友好）
├── get_record_count() → int      # 记录计数（~43,101）
└── get_statistics(n) → Dict      # 采样统计
```

**数据读取流程**:
```
record_ref (如 "cpsc_2018/g1/A0001")
  → _find_record_path()          # 解析为 .hea 路径
  → wfdb.rdrecord()              # 自动读取 .hea + .mat 对
  → record.p_signal.T            # (L, 12) → (12, L) 转置
  → _parse_metadata(comments)    # 正则提取 Age/Sex/Dx/Rx/Hx/Sx
  → _infer_source(path)          # 从路径推断数据源
  → ECGSample(...)
```

**关键设计**:
- 使用 `wfdb` 库自动处理 WFDB format 16（差分编码→原始信号）
- 正则匹配 `^field:` 而非 `# field:`（wfdb 会去掉 `# ` 前缀）
- `iter_records()` 是生成器，43K 条记录内存占用恒定
- 加载失败时返回 `None` 而非抛异常，保证批量处理不被单条坏数据中断

---

### 2.2 信号预处理器 (`preprocessor.py`)

**处理管道**（6 步）:

```
原始 ECG (12, L_orig) @ fs_orig
  │
  ├─ Step 1: 带通滤波 0.5-45 Hz
  │   └─ Butterworth 4阶, SOS 形式, sosfiltfilt (零相位, 数值稳定)
  │
  ├─ Step 2: 陷波滤波 50 Hz
  │   └─ IIR Notch → SOS, Q=30（fs 过低时自动跳过）
  │
  ├─ Step 3: 重采样 → 500 Hz
  │   └─ scipy.signal.resample_poly (多相滤波 + 抗混叠, 保真 ECG 波形)
  │
  ├─ Step 4: 分段 → 4096 样本
  │   ├─ 过长: 居中裁剪
  │   ├─ 过短: 对称零填充（记录 pad_left/pad_right）
  │   └─ 等长: 不变
  │
  ├─ Step 5: 逐导联 z-score 归一化
  │   └─ 仅对非填充区计算统计量（防零填充失真）
  │
  └─ Step 6: 异常值裁剪 5σ
      └─ 先 copy 再 clip（不修改原数组）
```

**关键修复 — 零填充归一化失真**:

```
修复前: 对整个 (12, 4096) 计算 mean/std
  → 1000 样本信号 + 3096 零填充 → std 被稀释 ~50%
  → 归一化后信号幅度放大 ~2x ❌

修复后: 仅对信号区域 [pad_left : L-pad_right] 计算统计量
  → mean/std 基于真实信号
  → 归一化后信号 std ≈ 1.0 ✓
```

**边界安全**:
- Nyquist 检查: `bandpass_high ≥ 0.99*Nyquist` → 抛异常
- 陷波跳过: `notch_freq ≥ 0.95*Nyquist` → 自动跳过
- 短信号: `orig_len < 2` → 跳过重采样
- 零长度: `round()` 确保 `new_len ≥ 1`

**滤波设计**:
- 不同 fs 自动重新计算系数（缓存避免重复设计）
- 4 阶 Butterworth 带通 + 2 阶 IIR 陷波
- **SOS（Second-Order Sections）形式**: 比传统 `ba` 形式数值更稳定，尤其对窄带/低频滤波
- `sosfiltfilt` 零相位滤波（前后向各一次，有效阶数翻倍）
- 陷波滤波器自动 `tf2sos` 转换为 SOS 保持一致性

**重采样设计**:
- `scipy.signal.resample_poly` 多相滤波重采样（替代线性插值）
- 内置抗混叠滤波，保留 ECG 波形中 QRS 波群等高频成分的形态
- `up/down` 因子自动化简（`math.gcd`），过大因子逐步约减防内存溢出
- 输出长度自动对齐：`resample_poly` 可能产生略多/略少样本 → pad/crop 精确对齐

---

### 2.3 数据增强器 (`augmentor.py`)

**6 种 ECG 专用增强**（均来自文献）:

| # | 增强 | 方法 | 参数 | 文献依据 |
|---|------|------|------|----------|
| 1 | 基线漂移 | 每导联独立正弦波 | freq 0.1-0.5 Hz, amp 0.3×std | Clifford et al. |
| 2 | 高斯噪声 | 每导联独立缩放 | σ 0.01-0.05×signal_std | 标准 |
| 3 | 时间扭曲 | 线性插值 + 边界填充 | scale 0.8-1.2 | Um et al. 2017 |
| 4 | 幅度缩放 | 每导联独立随机倍数 | 0.8-1.2× | 标准 |
| 5 | 导联丢失 | 随机置零 1-2 导联 | max_drop=2 | Strodthoff et al. |
| 6 | 片段打乱 | 分 4 段随机排列 | num_segments=4 | 对比学习专用 |

**关键修复**:
- `time_warp`: `fill_value="extrapolate"` → `fill_value=(边界值, 边界值)`，防止 scale<1 时产生直线外推伪影
- `lead_dropout`: 添加 `min(num_drop, n_leads-1)` 钳制，防止单导联信号崩溃
- `apply_amplitude_scaling`: 参数名 `pre_lead` → `per_lead`（拼写修正）
- `shuffle_chunk_seconds` + `fs`: 参数联动生效，`num_segments = L / (fs * chunk_seconds)`
- `RandomState` → `Generator`: 新版 NumPy RNG，fork-safe，所有 API 同步更新（`random()`/`standard_normal()`/`integers()`）

**使用方式**:
```python
aug = ECGAugmentor(random_seed=42)
augmented = aug(ecg_signal)              # 默认全部增强
augmented = aug(ecg_signal, apply_prob=0.5)  # 50% 概率
noisy = aug.add_gaussian_noise(ecg, sigma=0.02)  # 单独调用
```

---

### 2.4 标签提取器 (`label_extractor.py`)

**27 个规范类别**（基于 PhysioNet 2020 Challenge 官方评分标准）:

| 分类 | 数量 | 示例 |
|------|------|------|
| 节律类 (Rhythm) | 9 | 窦性心律、房颤、房扑、窦缓、窦速、窦性心律失常、PAC、PVC、VT |
| 传导类 (Conduction) | 8 | LBBB、RBBB、I°AVB、II°AVB、III°AVB、IRBBB、LAFB、WPW |
| 形态类 (Morphology) | 10 | ST压低、ST抬高、T波倒置、T波异常、心梗、心肌缺血、VEB、QT延长、RVH、低电压 |

**架构**:
```
SNOMED CT Code
  │
  ├─ 规范映射 (CHALLENGE_CLASSES)
  │   └─ 27 个主要代码 → 直接映射到 0-26 索引
  │
  ├─ 等价映射 (SNOMED_CT_EQUIVALENTS)
  │   └─ 15 个非规范代码 → 映射到对应规范类
  │   └─ 例: 39732003 → 164909002 (LBBB)
  │
  └─ 完整映射 (_full_mapping)
      └─ 42 个 SNOMED CT 代码 → 类索引
      └─ save/load 持久化
```

**等价映射表**（42 个代码覆盖 99.6% 数据）:

| 等价代码 | 规范代码 | 含义 |
|----------|----------|------|
| 428750005 | 164884008 | 室性早搏 → VEB |
| 39732003 | 164909002 | LBBB 变体 → LBBB |
| 164873001 | 164865005 | 心肌缺血变体 → MIsch |
| 445118002 | 164865005 | 急性心梗 → MIsch |
| 713426002 | 164861001 | 陈旧心梗 → MI |
| 713427006 | 164861001 | 陈旧心梗变体 → MI |
| 164930006 | 164865005 | ECG: 心肌缺血 → MIsch |
| 164867002 | 164865005 | ECG: 侧壁缺血 → MIsch |
| 164917005 | 164909002 | ECG: LBBB → LBBB |
| 164951009 | 164865005 | ECG: 缺血 → MIsch |
| 55930002 | 164865005 | 缺血变体 → MIsch |
| 47665007 | 164861001 | 心梗发现 → MI |
| 425623009 | 164861001 | 心梗发现变体 → MI |
| 67741000119109 | 164865005 | 异常Q波 → MIsch |
| 428417006 | 164861001 | 心梗变体 → MI |

**编码格式**:
```python
extractor.encode(["426783006", "164889003"])
# → np.array([1, 1, 0, 0, ...])  # multi_hot (27,)

extractor.encode(["426783006"], format="indices")
# → [0]  # 类索引列表

extractor.encode(["426783006"], format="names")
# → ["Sinus Rhythm"]  # 类名列表

extractor.decode(prob_vector, threshold=0.5)
# → [("426783006", "Sinus Rhythm", 0.95), ...]  # 解码
```

**关键修复**:
- 重复代码 `251146004` → 同时映射到 IRBBB 和 LVH ❌ → 保留 IRBBB，删除 LVH
- 缺失第 27 类 → 添加 VEB (164884008)
- 等价映射表可序列化 → `save_mapping` 保存 `_full_mapping`，`load_mapping` 恢复

---

### 2.5 元数据解析器 (`metadata_parser.py`)

**处理功能**:

| 字段 | 输入格式 | 输出 |
|------|----------|------|
| Age | "74" / "NaN" / "inf" / "unknown" | 归一化 [0,1] 或 -1.0（未知标记） |
| Sex | "Male"/"M"/"Female"/"F"/"NaN" | 0=Female, 1=Male, 2=Unknown |
| Dx | "59118001,270492004" | ["59118001", "270492004"] |
| Rx/Hx/Sx | "Unknown" / 自由文本 | 症状/病史关键词标志字典 |

**关键词匹配** — 中英文双语:
```python
SYMPTOM_KEYWORDS = {
    "chest_pain": ["chest pain", "chest tightness", "胸痛", "胸闷"],
    "palpitations": ["palpitation", "心悸"],
    ...
}
HISTORY_KEYWORDS = {
    "hypertension": ["hypertension", "htn", "高血压"],
    "diabetes": ["diabetes", "dm", "糖尿病"],
    ...
}
```

> ⚠️ 注意: Rx/Hx/Sx 在 PhysioNet 2020 数据集中几乎全为 "Unknown"，关键词匹配的实际命中率很低。Phase 3 将用 LLM 生成合成临床文本补充。

**元数据特征向量**（5 维，供下游模型使用）:
```python
encode_metadata_vector(parsed)
# → [age_norm, age_unknown_flag, sex_female, sex_male, sex_unknown]
```

---

### 2.6 PyTorch Dataset 模块 (`dataset.py`)

**三种 Dataset**:

| 类 | 用途 | 返回 | 特殊处理 |
|----|------|------|----------|
| `ECGDataset` | 有监督分类训练 | `(signal, labels)` | 形状校正 + padding/crop + 增强 |
| `ECGContrastiveDataset` | SimCLR 对比预训练 | `(view1, view2)` | 同一信号两个增强视图 |
| `ECGDatasetForAgent` | Agent 推理/评估 | `dict(含元数据)` | 完整病人上下文 |
| `ECGDataModule` | 统一 DataLoader 管理 | train/val/test 三件套 | 独立训练循环 + Lightning 兼容 |

**形状校正逻辑**（防止错误方向的数据）:
```python
# 正确的: (12, L)     → 不处理
# 转置的: (L, 12)     → 自动转回来
# 异常的: (1, L) 等   → 抛 ValueError
if signal.ndim == 2 and signal.shape[1] == 12 and signal.shape[0] != 12:
    signal = signal.T
elif signal.ndim != 2 or signal.shape[0] != 12:
    raise ValueError(...)
```

**关键修复**:
- `ECGContrastiveDataset`: 添加短信号 padding（之前只 crop 不 pad）
- `ECGDatasetForAgent`: 添加形状校正 + padding/crop + `target_length` 参数
- 删除死代码: `HAS_LIGHTNING`、`return_sample`
- `teardown()`: 添加空实现，PyTorch Lightning 兼容
- `_worker_init_fn`: 每 worker 独立 RNG 种子，多进程增强可复现
- 所有 DataLoader 传入 `worker_init_fn`

---

### 2.7 预处理脚本 (`preprocess_data.py`)

**完整流程**:
```
ECGLoader 读取原始数据
  ↓
ECGPreprocessor 预处理每条记录
  ↓
LabelExtractor 编码标签
  ↓
保存 .npy 文件 + 构建 manifest
  ↓
按数据源分层划分 train/val/test（70/15/15）
  ↓
保存 3 个 manifest JSON + 标签映射 JSON
```

**关键修复**:
- `sources = sorted(set(...))` 固定顺序 → 数据划分可复现
- `(output_dir.parent / "labels").mkdir()` → 防止 save_mapping 崩溃
- `except (KeyboardInterrupt, SystemExit): raise` → Ctrl+C 正常中断
- Age 字段清理: NaN/inf → None → JSON 安全序列化

---

## 3. 数据集统计

### 3.1 来源分布

| 来源 | 记录数 | 占比 | 采样率 | 典型时长 |
|------|--------|------|--------|----------|
| CPSC 2018 | 6,877 | 16.0% | 500 Hz | 10-15s |
| CPSC 2018 Extra | 3,453 | 8.0% | 500 Hz | 10-15s |
| Georgia | 10,344 | 24.0% | 500 Hz | ~10s |
| PTB | 516 | 1.2% | 1000 Hz | 38-115s |
| PTB-XL | 21,837 | 50.7% | 500 Hz | ~10s |
| St Petersburg INCART | 74 | 0.2% | 257 Hz | ~30min |
| **总计** | **43,101** | 100% | | |

### 3.2 数据划分

| 划分 | 数量 | 占比 | 策略 |
|------|------|------|------|
| 训练集 | 30,167 | 70% | `sorted(set(sources))` + `RandomState(42)` |
| 验证集 | 6,462 | 15% | 同上 |
| 测试集 | 6,472 | 15% | 同上 |
| 数据泄漏 | **0** | | 三集合 record_id 零重叠 |

### 3.3 标签分布（27/27 类全部有数据）

| # | SNOMED CT | 类别 | 样本数 | 分类 |
|---|-----------|------|--------|------|
| 1 | 426783006 | Sinus Rhythm | 20,846 | 节律 |
| 2 | 164865005 | Myocardial Ischemia | 14,291 | 形态 |
| 3 | 164909002 | Left Bundle Branch Block | 7,584 | 传导 |
| 4 | 164861001 | Myocardial Infarction | 6,103 | 形态 |
| 5 | 164884008 | Ventricular Ectopic Beats | 5,464 | 形态 |
| 6 | 164934002 | T Wave Inversion | 4,673 | 形态 |
| 7 | 164889003 | Atrial Fibrillation | 4,430 | 节律 |
| 8 | 59118001 | Right Bundle Branch Block | 3,286 | 传导 |
| 9 | 429622005 | Low QRS Voltages | 2,962 | 形态 |
| 10 | 284470004 | Premature Atrial Contraction | 2,514 | 节律 |
| 11 | 270492004 | First Degree AV Block | 2,394 | 传导 |
| 12 | 427084000 | ST Elevation | 2,314 | 形态 |
| 13 | 427172004 | Premature Ventricular Contractions | 2,175 | 节律 |
| 14 | 698252002 | Left Anterior Fascicular Block | 2,147 | 传导 |
| 15 | 164931005 | ST Depression | 2,101 | 形态 |
| 16 | 111975006 | QT Prolonged | 1,707 | 形态 |
| 17 | 426177001 | Sinus Bradycardia | 1,645 | 节律 |
| 18 | 59931005 | T Wave Abnormal | 1,496 | 形态 |
| 19 | 164890007 | Atrial Flutter | 1,464 | 节律 |
| 20 | 427393009 | Sinus Tachycardia | 1,235 | 节律 |
| 21 | 195042002 | Second Degree AV Block | 889 | 传导 |
| 22 | 446358003 | Right Ventricular Hypertrophy | 864 | 形态 |
| 23 | 10370003 | Wolff-Parkinson-White | 804 | 传导 |
| 24 | 17338001 | Ventricular Tachycardia | 759 | 节律 |
| 25 | 27885002 | Complete AV Block | 514 | 传导 |
| 26 | 251146004 | Incomplete Right BBB | 491 | 传导 |
| 27 | 713422000 | Sinus Arrhythmia | **43** | 节律 |

> ⚠️ 类别严重不平衡: Sinus Rhythm (20,846) vs Sinus Arrhythmia (43) ≈ 485:1
> Phase 2 需使用 Asymmetric Loss / Focal Loss 处理。

### 3.4 标签覆盖率演进

| 审查轮次 | 规范类 | 等价映射 | 零标签率 | 说明 |
|----------|--------|----------|----------|------|
| 初版 | 26/27 | 26 代码 | ~9% | 重复代码 + 缺失 VEB |
| 第 1 轮修复 | 27/27 | 31 代码 | 3.3% | 修复重复 + 添加 VEB + 基础等价 |
| 第 2 轮修复 | 27/27 | 42 代码 | **0.4%** | 扩展等价映射（15 个新代码） |
| 最终 | 27/27 | 42 代码 | **0.4%** | 36 个低频代码待补充 |

### 3.5 预处理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 目标采样率 | 500 Hz | 统一不同来源的采样率 |
| 目标长度 | 4096 样本 (~8.2s) | 居中裁剪/对称零填充 |
| 带通滤波 | 0.5-45 Hz | Butterworth 4 阶，零相位 |
| 陷波滤波 | 50 Hz | 自动跳过 fs 过低的数据源 |
| 归一化 | 逐导联 z-score | 仅对信号区计算统计量 |
| 异常值裁剪 | 5σ | 先 copy 再 clip |
| 输出格式 | float32 .npy | 96 KB/条，总计 ~4 GB |

---

## 4. 质量审查历史

Phase 1 经过 **5 轮全面审查**，共发现并修复 **40+ 个问题**，7 项技术债务全部清零。

### 审查方法
- 3 个并行 Explore Agent 逐文件深度审查
- 手动逐行代码走查
- 43,101 条全量数据预处理验证
- 边界情况单元测试

### 修复清单

#### 第 1 轮（初始审查）

| # | 严重度 | 文件 | 问题 | 状态 |
|---|--------|------|------|------|
| 1 | 🔴 | `label_extractor.py` | `251146004` 重复映射到 IRBBB + LVH | ✅ |
| 2 | 🔴 | `label_extractor.py` | 缺失第 27 类 VEB | ✅ |
| 3 | 🔴 | `loader.py` | 正则匹配 `#\s*field` 不匹配 wfdb 输出 | ✅ |
| 4 | 🟡 | `preprocessor.py` | `_clip_outliers` 原地修改输入数组 | ✅ |
| 5 | 🟡 | `augmentor.py` | `time_warp` 外推产生零值伪影 | ✅ |

#### 第 2 轮（全量数据验证）

| # | 严重度 | 文件 | 问题 | 状态 |
|---|--------|------|------|------|
| 6 | 🔴 | `label_extractor.py` | SNOMED CT 等价映射缺失（75% 代码未覆盖） | ✅ |
| 7 | 🔴 | `label_extractor.py` | `save_mapping` 不保存 `_full_mapping` | ✅ |
| 8 | 🔴 | `label_extractor.py` | `load_mapping` 不恢复 `_full_mapping` | ✅ |
| 9 | 🔴 | `preprocess_data.py` | `labels/` 目录未创建导致崩溃 | ✅ |
| 10 | 🔴 | `preprocess_data.py` | NaN age 导致 JSON 序列化失败 | ✅ |

#### 第 3 轮（并行 Agent 深度审查）

| # | 严重度 | 文件 | 问题 | 状态 |
|---|--------|------|------|------|
| 11 | 🔴 | `preprocessor.py` | 零填充后归一化失真（统计量包含填充区） | ✅ |
| 12 | 🔴 | `preprocessor.py` | Nyquist 边界无检查 | ✅ |
| 13 | 🔴 | `preprocessor.py` | 陷波频率 ≥ Nyquist 时崩溃 | ✅ |
| 14 | 🔴 | `loader.py` | Age 返回 NaN 而非 None | ✅ |
| 15 | 🔴 | `label_extractor.py` | 等价映射试图合并 6 个规范类 | ✅ |
| 16 | 🟠 | `augmentor.py` | `lead_dropout` 导联不足时崩溃 | ✅ |
| 17 | 🟠 | `augmentor.py` | `time_warp` scale<1 时外推垃圾数据 | ✅ |
| 18 | 🟠 | `dataset.py` | `ContrastiveDataset` 缺少 padding | ✅ |
| 19 | 🟠 | `dataset.py` | `ForAgent` 缺少形状校正 | ✅ |
| 20 | 🟠 | `dataset.py` | 转置启发式在 (1,L) 单导联时错误 | ✅ |
| 21 | 🟠 | `preprocess_data.py` | `set()` 非确定性 → 数据划分不可复现 | ✅ |
| 22 | 🟠 | `preprocess_data.py` | `except Exception` 吞掉 Ctrl+C | ✅ |
| 23 | 🟡 | `metadata_parser.py` | 短关键词 ("af","mi","hf") 大量误匹配 | 📝 记录 |

#### 第 4 轮（最终逐行审查）

| # | 严重度 | 文件 | 问题 | 状态 |
|---|--------|------|------|------|
| 24 | 🟡 | `preprocessor.py` | 注释称 "SOS" 但实际用 `ba` 格式 | ✅ |
| 25 | 🟡 | `preprocessor.py` | `int()` 浮点截断风险 → `round()` | ✅ |
| 26 | 🟡 | `preprocessor.py` | `segment()` 短信号 pad 到错误长度 | ✅ |
| 27 | 🟡 | `label_extractor.py` | 重复的 section header 注释块 | ✅ |
| 28 | 🟡 | `metadata_parser.py` | `_parse_age_raw` NaN 逃逸 | ✅ |
| 29 | ⚪ | `dataset.py` | `HAS_LIGHTNING` 死代码 | ✅ |
| 30 | ⚪ | `dataset.py` | `return_sample` 死参数 | ✅ |
| 31 | ⚪ | `augmentor.py` | `from scipy import signal` 未使用 | ✅ |

#### 第 5 轮（技术债务清零）

| # | 严重度 | 文件 | 问题 | 状态 |
|---|--------|------|------|------|
| 32 | 🟡 | `preprocessor.py` | `ba` 滤波 → 升级为 `sos` 形式 | ✅ |
| 33 | 🟡 | `preprocessor.py` | `interp1d` 线性插值 → `resample_poly` | ✅ |
| 34 | 🟡 | `augmentor.py` | `shuffle_chunk_seconds` 参数生效 | ✅ |
| 35 | 🟡 | `augmentor.py` | `apply_segment_shuffle` 的 `fs` 参数生效 | ✅ |
| 36 | 🟡 | `augmentor.py` | `RandomState` → `Generator`（fork 安全） | ✅ |
| 37 | 🟡 | `dataset.py` | `teardown()` + `worker_init_fn` | ✅ |
| 38 | 🟡 | `loader.py` | `duration` 属性 `fs ≤ 0` 保护 | ✅ |

> 第 5 轮修复过程中发现并修复 1 个新 bug: `resample_poly` 输出长度与目标长度不完全一致，添加 pad/crop 对齐逻辑。

---

## 5. 已知限制（非阻塞）

### 5.1 标签覆盖
- **0.4% 记录**（118 条/30,167 训练集）有 SNOMED CT 代码但无法映射到 27 类
- 涉及 36 个低频罕见代码（最多出现 14 次）
- 需要医学专业知识逐一确认等价关系
- Phase 2-3 可逐步补充

### 5.2 类别不平衡
- Sinus Arrhythmia 仅 43 样本（vs Sinus Rhythm 20,846）
- Phase 2 需 Asymmetric Loss / Focal Loss / SMOTE

### 5.3 元数据稀缺
- Rx/Hx/Sx 字段几乎全为 "Unknown"
- Phase 3 将用 LLM 生成合成临床文本

### 5.4 技术债务 — 全部已清零 ✅

| # | 原问题 | 修复 | 文件 |
|---|--------|------|------|
| 1 | `ba` 滤波形式 | → `sos`（二阶节级联，数值更稳定） | `preprocessor.py` |
| 2 | `interp1d` 线性插值 | → `resample_poly`（多相滤波+抗混叠） | `preprocessor.py` |
| 3 | `shuffle_chunk_seconds` 未使用 | → 联动 `fs` 计算 `num_segments` | `augmentor.py` |
| 4 | `apply_segment_shuffle` 的 `fs` 未使用 | → 与 #3 联动生效 | `augmentor.py` |
| 5 | `RandomState` fork 不安全 | → `Generator`（新版 NumPy RNG） | `augmentor.py` |
| 6 | `ECGDataModule` 缺 Lightning 兼容 | → `teardown()` + `worker_init_fn` | `dataset.py` |
| 7 | `duration` 未处理 `fs=0` | → `fs ≤ 0` 返回 `0.0` | `loader.py` |

> ⚠️ 性能提示: `resample_poly` 保真度更高但速度较慢（全量预处理 ~45 分钟 vs 之前 ~20 分钟）。仅需对不同来源重采样（~20% 数据），500Hz 同频数据无影响。

---

## 6. 使用方式

### 6.1 安装依赖

```bash
cd ecg-ai-agent
pip install -r requirements.txt
```

### 6.2 全量预处理

```bash
python scripts/preprocess_data.py
# 输出: data/physionet2020/processed/*.npy (43,101 文件, ~4 GB)
# 耗时: ~20 分钟
```

### 6.3 快速测试（100 条）

```bash
python scripts/preprocess_data.py --max-records 100
```

### 6.4 代码使用

```python
from src.data_pipeline import (
    ECGLoader, ECGPreprocessor, ECGAugmentor,
    LabelExtractor, MetadataParser, ECGDataset, ECGDataModule,
)

# === 原始数据加载 ===
loader = ECGLoader("path/to/training")
sample = loader.load_record("cpsc_2018/g1/A0001")
print(sample.signal.shape)  # → (12, 7500)

# === 信号预处理 ===
prep = ECGPreprocessor(target_fs=500, target_length=4096)
clean = prep(sample.signal, original_fs=sample.fs)
print(clean.shape)  # → (12, 4096)

# === 标签编码 ===
extractor = LabelExtractor()
labels = extractor.encode(sample.dx_codes)       # → np.array (27,)
names = extractor.encode(sample.dx_codes, format="names")  # → ["Right BBB"]

# === PyTorch 训练 ===
dataset = ECGDataset(
    data_dir="data/physionet2020/processed",
    split="train",
    augment=True,
    augmentor=ECGAugmentor(),
    label_extractor=extractor,
)
signal, labels = dataset[0]
# signal: torch.Tensor (12, 4096)
# labels: torch.Tensor (27,)

# === DataLoader ===
dm = ECGDataModule(
    data_dir="data/physionet2020/processed",
    batch_size=128,
    augmentor=ECGAugmentor(),
    label_extractor=extractor,
)
dm.setup()
for batch in dm.train_dataloader():
    signals, labels = batch
    # signals: (128, 12, 4096), labels: (128, 27)
    break

# === 对比学习 ===
from src.data_pipeline.dataset import ECGContrastiveDataset
cs = ECGContrastiveDataset(
    data_dir="data/physionet2020/processed",
    augmentor=ECGAugmentor(segment_shuffle=True),
)
view1, view2 = cs[0]  # 同一信号两个增强视图
```

---

## 7. 文件清单

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/data_pipeline/__init__.py` | ~15 | 模块导出 |
| `src/data_pipeline/loader.py` | ~370 | ECGLoader + ECGSample（含 fs≤0 保护） |
| `src/data_pipeline/preprocessor.py` | ~380 | 6 步预处理管道（SOS + resample_poly） |
| `src/data_pipeline/augmentor.py` | ~325 | 6 种数据增强（Generator RNG） |
| `src/data_pipeline/label_extractor.py` | ~392 | 27 类 + 15 等价映射 |
| `src/data_pipeline/metadata_parser.py` | ~195 | 元数据解析 + 特征向量（含 NaN 检查） |
| `src/data_pipeline/dataset.py` | ~360 | 3 Dataset + DataModule（teardown + worker_init_fn） |
| `tests/test_loader.py` | ~100 | Loader 测试 |
| `tests/test_pipeline.py` | ~150 | 集成测试 |
| `scripts/preprocess_data.py` | ~210 | 端到端预处理 |
| `configs/data_config.yaml` | ~60 | 配置 |
| `requirements.txt` | ~30 | 依赖 |
| **总计** | **~2,600** | |

---

## 8. 待办：Phase 2

- [ ] ECG 特征提取（R-peak, HRV, QT）— `feature_extraction/`
- [ ] xResNet1D-101 基线复现 — `backbone/xresnet1d.py`
- [ ] ECG Transformer 骨干网络 — `backbone/transformer_encoder.py`
- [ ] 心律失常分类器（27 类多标签 + Asymmetric Loss）— `classifiers/`
- [ ] 异常检测器（VAE + Deep SVDD）— `classifiers/anomaly_detector.py`
- [ ] SimCLR 对比预训练 — `trainer.py`
- [ ] 训练脚本 — `scripts/train_backbone.sh`
