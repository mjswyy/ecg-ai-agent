# GitHub ECG 相关项目分析报告

> 分析日期：2026-08-04
> 共分析 10 个 GitHub 项目，评估其对 ECG AI Agent 项目的借鉴价值

---

## 总览

| # | 项目 | Stars | 论文 | 框架 | 核心价值 | 推荐度 |
|:---:|------|------|------|------|------|:---:|
| 1 | **ECGFounder** | — | NEJM AI 2025 | PyTorch | 150类基础模型，可下载权重 | ⭐⭐⭐⭐⭐ |
| 2 | **ECG-FM** | — | arxiv 2024 | fairseq | PatternLabeler知识图谱 | ⭐⭐⭐⭐⭐ |
| 3 | **torch_ecg** | — | Physiol Meas 2022 | PyTorch | 全栈ECG框架，25+架构 | ⭐⭐⭐⭐⭐ |
| 4 | **ecg_ptbxl_benchmarking** | — | IEEE JBHI 2021 | FastAI | XResNet1d最佳架构+评估框架 | ⭐⭐⭐⭐ |
| 5 | **automatic-ecg-diagnosis** | — | Nat Commun 2020 | TensorFlow | ResNet+统计评估范式 | ⭐⭐⭐⭐ |
| 6 | **braindecode** | — | Hum Brain Mapp 2017 | PyTorch+skorch | 50+架构+滑窗训练+数据增强 | ⭐⭐⭐⭐ |
| 7 | **ecg (Stanford)** | — | Nat Med 2019 | Keras/Py2.7 | 经典34层ResNet，段级分类 | ⭐⭐⭐ |
| 8 | **HeartPy** | — | Transp Res 2019 | 纯NumPy | 自适应R峰检测+HRV完整指标 | ⭐⭐⭐ |
| 9 | **BrainFlow** | — | — | C++/Python | 65+传感器硬件抽象+信号处理 | ⭐⭐⭐ |
| 10 | **ECGFounder** 与 **ECG-FM** 对比 | — | — | — | 两个基础模型的优劣 | — |

---

## 1. ECGFounder — ECG基础模型（最推荐）

**论文**：NEJM AI, 2025 | **作者**：Jun Li, Shenda Hong 等（北大）
**HuggingFace**：https://huggingface.co/PKUDigitalHealth/ECGFounder

### 架构

Net1D：7阶段1D CNN + 残差Bottleneck + Squeeze-and-Excitation注意力 + Swish激活

```
输入 (B, 12, 5000) @ 500Hz, 10s
  → Conv1d(16) stride2 + BN + Swish
  → 7 Stages: [2,2,2,3,3,4,4] blocks, 通道 64→160→400→1024
  → 全局均值池化 → Linear(1024, 150)
```

**关键设计**：预训练时不用 BatchNorm 和 Dropout——只靠 Swish + 残差 + SE。

### 数据与训练

- **预训练**：HEEDB 1000万+条ECG（私有），仅开源权重和推理代码
- **验证**：PTB-XL 150类零样本分类，逐类Youden最优阈值
- **微调**：MIMIC-IV-ECG做LVEF分类/回归，支持linear probing和全微调

### 对我们项目的价值

1. **可以直接下载权重**：12-lead和1-lead两种预训练权重在HuggingFace
2. **对PTB-XL做零样本评估**：和我们的模型直接对比
3. **预处理管道可以直接复用**：0.67-40Hz带通+50Hz陷波+中值滤波去基线漂移
4. **逐类动态阈值**（Youden's J）：比我们当前的P-R curve方法更简洁
5. **150类标签体系**（tasks.txt）：比我们的27类更细粒度

---

## 2. ECG-FM — 开源基础模型 + 标签器（最推荐）

**论文**：arxiv 2408.05178 | **架构**：wav2vec 2.0 适配 ECG（90.9M参数）
**HuggingFace**：https://huggingface.co/wanglab/ecg-fm

### 架构

```
输入 (B, 12, 2500) @ 500Hz, 5s
  → 7层CNN特征编码器
  → Transformer Encoder（多头自注意力）
  → 量化模块（掩码token预测）
  → 分类头 → 18类多标签
```

### 预训练方法（WCR）

- **W2V**（wav2vec 2.0）：掩码预测量化潜在表示
- **CMSC**（对比多段编码）：相邻5秒ECG段对比学习
- **RLM**（随机导联掩码）：训练时随机mask导联，使模型对缺失导联鲁棒

### 核心创新：PatternLabeler 标签系统

**这是我们最应该借鉴的组件**。将自由文本ECG报告转为结构化标签：

- `EntityTem`：疾病实体（如心动过速、梗死）
- `DescriptorTem`：描述属性（如急性、可能、前壁）
- `Connective`：连接词（如和、与、提示）
- `CompoundTem`：复合实体（双束支阻滞=RBBB+分支阻滞）
- `UncertaintyMap`：不确定性映射（"可能"=0.5，"很可能"=0.7）
- **知识图谱**：实体间有父子关系DAG，不确定性递归向上传播
- **1000+** 医学缩写/同义词/拼写纠正规则

### 对我们项目的价值

1. **PatternLabeler 直接用于 Phase 3**：将PTB-XL报告文本结构化
2. **RLM导联掩码预训练**：可移植到我们的Transformer训练中
3. **标签特定预测聚合**：瞬态事件用max，持续状态用mean
4. **Memmap大数据IO**：处理大规模嵌入的实用方案
5. **知识图谱JSON**：entity_tem.json可直接用于构建诊断知识库

---

## 3. torch_ecg — 全栈ECG深度学习框架（最推荐）

**论文**：Physiological Measurement, 2022 | **作者**：Hao Wen, Jingsu Kang
**PyPI**：`pip install torch-ecg`

### 架构

旗舰模型 `ECG_CRNN`——完全模块化的分类管道：

```
CNN Backbone（可插拔，25+变体）
  → RNN（可选：LSTM/GRU/Linear/None）
  → Attention（可选：SE/GC/NL/Transformer/None）
  → Global Pooling（Max/Avg/None）
  → Classifier MLP
  → 输出 (B, n_classes)
```

### 核心模块

| 模块 | 功能 |
|------|------|
| `ECG_CRNN` | 主分类模型，CNN→RNN→Attention→分类头 |
| `ECG_SEQ_LAB_NET` | 序列标注（QRS检测、波形定位） |
| `ECG_UNET` | U-Net分割（P/QRS/T波定位） |
| 18+数据库包装器 | PhysioNet、CPSC各挑战赛数据 |
| `BaseTrainer` | 完整训练循环（~1500行），checkpoint/早停/日志 |
| `ClassificationMetrics` | 20+指标（敏感度、特异度、F1、AUC等） |
| `AugmenterManager` | 8种ECG增强（cutmix、mixup、导联丢失等） |
| Registry模式 | `@MODELS.register()` 装饰器，字符串即可组装模型 |

### 对我们项目的价值

1. **可以直接 pip install** 作为依赖
2. **25+ CNN变体**直接可用，包括我们用的xResNet和Nature Comm ResNet
3. **Registry模式**天然适配LLM Agent：`"build CRNN with resnet_nature_comm backbone + SE attention + LSTM layer"`
4. **CFG配置系统**：层次化字典，LLM可增量构建配置
5. **全栈工具**：数据库→预处理→增强→模型→训练→评估，一个库搞定
6. **FocalLoss/AsymmetricLoss实现**：可比对我们自己的实现

---

## 4. ecg_ptbxl_benchmarking — PTB-XL基准测试

**论文**：IEEE JBHI, 2021 | **作者**：Strodthoff et al.

### 架构

7种架构在PTB-XL上对比，**XResNet1d101表现最佳**：

| 模型 | Macro AUC (71类) |
|------|:---:|
| xresnet1d101 | **0.925** |
| inception1d | 0.925 |
| resnet1d_wang | 0.919 |
| fcn_wang | 0.918 |
| lstm_bidir | 0.914 |
| Wavelet+NN | 0.849 |

### 核心组件

- **`SCP_Experiment`**：`prepare()→perform()→evaluate()` 三步实验框架
- **`TimeseriesDatasetCrops`**：分块数据集，支持stride、随机裁剪、三种数据模式
- **`create_head1d`**：AdaptiveConcatPool1d（avg+max拼接）+BN+Dropout+Linear
- **Bootstrap CI**：100次bootstrap + 多进程（n_jobs=20），逐类置信区间

### 对我们项目的价值

1. **XResNet1d实现**可对照我们的版本，特别是AdaptiveConcatPool1d
2. **标签层次化**（diagnostic→subdiagnostic→superdiagnostic）参考
3. **`create_head1d`**直接复用：avg+max池化拼接是时间序列分类的验证设计
4. **实验框架模式**：prepare→perform→evaluate 三步法可移植

---

## 5. automatic-ecg-diagnosis — Nature Comm 2020

**论文**：Nature Communications, 2020 | **作者**：Ribeiro et al.
**数据集**：CODE-15%（Zenodo公开）

### 架构

4阶段1D ResNet，~530K参数：

```
输入 (B, 4096, 12) @ 400Hz, 10.24s
  → Conv1D(16→64) + BN + ReLU
  → 4 ResidualUnits: 4096→1024→256→64→16
  → Flatten → Dense(6, sigmoid)
```

### 评估框架

**这是我们见过的最完善的统计评估**：
- Bootstrap置信区间
- Cohen's Kappa评估者间一致性
- McNemar检验（DNN vs 人类标注者统计显著性）
- 逐异常类别F1/精确率/召回率/特异度
- DNN vs 心内科医生 vs 住院医 vs 医学生对比

### 对我们项目的价值

1. **统计评估框架**：generate_figures_and_tables.py 可以直接参考
2. **多标注者对比方法**：论文级评估模板
3. **输入格式标准**：4096样本/400Hz/12导联顺序参考
4. **HDF5+CSV数据加载模式**：简单高效

---

## 6. braindecode — 脑电解码工具箱（ECG可适配）

**论文**：Human Brain Mapping, 2017 | **PyPI**：`pip install braindecode`

### 核心能力

- **50+深度学习架构**：EEGNet（2.5K参数）、BDTCN（扩张因果卷积）、EEGConformer（CNN+Transformer）、BENDR（157M基础模型）
- **滑窗训练+试次评估**：`trial_preds_from_window_preds()` 处理变长信号
- **20+信号增强**：TimeReverse, SignFlip, ChannelsDropout, GaussianNoise, FrequencyShift
- **skorch集成**：scikit-learn兼容API，自动评分/回调/超参搜索

### 对我们项目的价值

1. **EEGNet→ECGNet**：时序+空间（导联）卷积天然映射到12导联ECG
2. **BDTCN扩张因果卷积**：直接可用于ECG
3. **滑窗训练评估范式**：处理长程ECG的核心方案
4. **数据增强库**：15+针对生理信号（非图像）的增强方法
5. **`BaseConcatDataset`**：元数据丰富的多患者数据集管理

---

## 7. ecg (Stanford) — Nature Medicine 2019 经典

**论文**：Nature Medicine, 2019 | 引用：2000+

### 架构

34层1D ResNet，单导联，变长ECG：

- 16个残差块，交替subsample [1,2,1,2,...]
- 每256样本做一步预测（TimeDistributed Dense+Softmax）
- 最终标签=所有段预测的mode

### 对我们项目的价值

1. **段级分类+投票聚合**：经典的变长ECG处理策略
2. 但已严重过时（Python 2.7 + Keras）
3. 仅作架构参考，不能直接复用

---

## 8. HeartPy — 心率变异性分析

**论文**：Transportation Research Part F, 2019

### 核心能力

- **自适应R峰检测**：18个阈值候选（5-300%移动平均），选RR变异最小且BPM∈[40,180]的
- **完整HRV指标**：SDNN, RMSSD, pNN50, LF/HF, Poincare SD1/SD2
- **截断插值**：三次样条，利用截断段前后100ms数据
- **QRS模板卷积增强**：15个合成模板（3种形态×5种宽度）

### 对我们项目的价值

1. **自适应R峰检测**可作为Agent工具的备选方案
2. **截断插值**直接处理传感器饱和
3. **HRV完整指标库**可丰富Agent的诊断特征

---

## 9. BrainFlow — 生物传感器硬件抽象

**核心能力**：65+设备统一API（C++核心 + 9语言绑定）

- 信号处理：Butterworth/Chebyshev/Bessel滤波、FFT/PSD、小波去噪、ICA
- 数据处理：陷波滤波、移动平均、降采样、峰值检测
- ML推理：ONNX Runtime集成

### 对我们项目的价值

如需接入实际ECG硬件（Holter、贴片等），BrainFlow提供开箱即用的硬件抽象层和实时信号处理管线。

---

## 综合建议：优先借鉴清单

### 立即可用

| 来源 | 借什么 | 用于 |
|------|------|------|
| **ECGFounder** | 预训练权重 | 零样本对比、微调基线 |
| **ECG-FM** | PatternLabeler知识图谱 | Phase 3 报告文本结构化 |
| **torch_ecg** | pip install 整个框架 | 替换自研训练器 |
| **ecg_ptbxl_benchmarking** | `create_head1d` (AdaptiveConcatPool) | 改进分类头 |

### Phase 3 关键借鉴

| 来源 | 借什么 |
|------|------|
| **ECG-FM** | PatternLabeler：将PTB-XL德语/英语报告→结构化18类标签 |
| **ECG-FM** | RLM导联掩码：训练时随机mask导联提升鲁棒性 |
| **torch_ecg** | `ECG_CRNN` 模块化架构：CNN→RNN→Attention→分类头 |

### 评估提升

| 来源 | 借什么 |
|------|------|
| **automatic-ecg-diagnosis** | McNemar检验 + Cohen's Kappa + Bootstrap CI |
| **ECGFounder** | 逐类Youden最优阈值（更简洁） |
| **braindecode** | 滑窗试次评估（长程ECG） |
