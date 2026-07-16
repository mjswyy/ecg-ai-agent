# 在华为云平台上训练心电图（ECG）识别基础模型的实践报告

> **实验日期**：2026年7月
> **平台**：华为云 ModelArts + Ascend NPU 910B
> **数据**：PhysioNet/CinC Challenge 2020，43,101 条 12 导联心电图

---

## 摘要

本实验在华为云 ModelArts 平台上，使用 Ascend NPU 910B 加速芯片，对 PhysioNet 2020 Challenge 数据集进行了深度学习模型训练。实验选取了两种不同架构的骨干网络——xResNet1D-101（7.6M 参数，卷积架构）和 ECG Transformer（6.5M 参数，自注意力架构），从头训练 27 类心律失常多标签分类模型。训练过程中解决了数据 NaN 污染、混合精度兼容性、NPU 算子适配、过拟合等一系列工程问题。最终通过模型集成，在测试集上取得 macro_auc = 0.822 的成绩。

---

## 1. 实验背景

心电图（ECG）是心血管疾病诊断最常用的无创检查手段。12 导联心电图记录了心脏在不同方向上的电活动，可用于诊断心律失常、心肌缺血、传导阻滞等多种心脏疾病。传统 ECG 判读依赖心内科医生的专业经验，深度学习在 ECG 自动分析领域取得了显著进展。

PhysioNet/CinC Challenge 2020 提供了一个包含 43,101 条 12 导联 ECG 的公开数据集和 27 类心律失常的标准化评估基准，是验证 ECG 识别模型的理想平台。

---

## 2. 实验环境

### 2.1 硬件

| 项目 | 配置 |
|------|------|
| 平台 | 华为云 ModelArts Notebook |
| 芯片 | Ascend 910B (Snt9B)，单卡，~61 GB HBM |
| 存储 | 100 GB 云硬盘 + OBS 对象存储（7.93 GB 数据） |

### 2.2 软件

| 组件 | 版本 |
|------|------|
| 操作系统 | EulerOS 2.10 (aarch64) |
| Python | 3.10 |
| PyTorch | 2.1.0 |
| torch_npu | 2.1.0.post10 |
| CANN | 8.0.0 |
| NumPy | < 2.0（兼容性要求） |

---

## 3. 数据集

### 3.1 规模与划分

| 划分 | 样本数 | 占比 |
|------|--------|------|
| 训练集 | 30,167 | 70% |
| 验证集 | 6,462 | 15% |
| 测试集 | 6,472 | 15% |
| **总计** | **43,101** | 100% |

### 3.2 标签体系

27 个规范类别，分为三大类：
- **节律类（9 类）**：窦性心律、房颤、房扑、窦缓、窦速、窦性心律失常、房性早搏、室性早搏、室性心动过速
- **传导类（8 类）**：左/右束支阻滞、I°/II°/III° 房室阻滞、不完全性右束支阻滞、左前分支阻滞、预激综合征
- **形态类（10 类）**：ST 段压低/抬高、T 波倒置/异常、心肌梗死、心肌缺血、QT 间期延长、右室肥厚、低电压

### 3.3 类别不平衡

最常见类别"窦性心律"（20,846 条）与最稀有类别"窦性心律失常"（43 条）比例为 **485:1**，对损失函数设计和评估指标选择有重要影响。

---

## 4. 数据预处理

原始 ECG 信号来源多样（6 个数据源），采样率 257-1000 Hz 不等，时长从 10 秒到 30 分钟。统一预处理为 6 步管道：

| 步骤 | 方法 | 参数 | 目的 |
|------|------|------|------|
| 带通滤波 | Butterworth 4阶，SOS 形式 | 0.5-45 Hz | 去除肌电干扰和基线漂移 |
| 陷波滤波 | IIR Notch | 50 Hz | 去除工频干扰 |
| 重采样 | `resample_poly`（抗混叠） | 500 Hz | 统一采样率 |
| 分段对齐 | 居中裁剪/对称零填充 | 4096 (~8.2s) | 统一长度 |
| 归一化 | 逐导联 z-score | — | 消除幅值差异 |
| 异常值裁剪 | 5σ 裁剪 | — | 抑制极端噪声 |

**关键技术选择**：
- **SOS 滤波**：比 ba 形式数值更稳定，配合 `sosfiltfilt` 零相位滤波
- **resample_poly**：带抗混叠滤波，保留 QRS 波群高频形态（线性插值会模糊 QRS）
- **填充感知归一化**：仅对信号区域计算统计量，避免零填充稀释 std

---

## 5. 模型架构

### 5.1 xResNet1D-101（7.6M 参数）

PhysioNet 2020 Challenge 冠军方案采用的卷积骨干。

```
输入 (B, 12, 4096)
  → Stem Conv1d + BN + ReLU + MaxPool
  → ResNet Block × N (Bottleneck 残差)
  → AdaptiveAvgPool1d
  → Linear(512→256) + Dropout
  → Linear(256→27) → Sigmoid
```

### 5.2 ECG Transformer（6.5M 参数）

本实验自研的自注意力骨干，结合导联卷积投影与多头注意力。

```
输入 (B, 12, 4096)
  → Lead-wise Conv1d (49→24 stride) → (B, 256, 170)
  → Transpose → (B, 170, 256)
  → Position Encoding + Transformer Encoder × 8
  → Global Mean Pooling → (B, 256)
  → Linear(256→27) → Sigmoid
```

**NPU 适配问题**：CANN 8.0.0 未适配 `nn.TransformerEncoderLayer` 和 `nn.MultiheadAttention` 的融合算子，PyTorch 原生实现会回退到 CPU 执行。本实验手写 attention（QKV 投影 → 多头分拆 → batch matmul → softmax → 合并），全部用基础算子（Linear, matmul, softmax, LayerNorm），确保全在 NPU 上执行。

---

## 6. 训练策略

### 6.1 损失函数与优化器

| 组件 | 配置 | 说明 |
|------|------|------|
| 损失函数 | BCEWithLogitsLoss | 数值稳定，与标签平滑兼容 |
| 优化器 | AdamW | weight_decay=1e-3 |
| 学习率 | 5e-5 | 大模型避免尖锐极小值 |
| 调度器 | CosineAnnealing + LinearWarmup (5 epochs) |
| 混合精度 | FP32（Transformer 因 AMP NaN 禁用） |

**关键发现**：AsymmetricLoss（负类聚焦）与 Label Smoothing（标签软化）的梯度方向相反，叠加后大模型训练崩溃（test_auc=0.544）。改回 BCE + 强正则化后正常。

### 6.2 正则化

| 方法 | 参数 | 原理 |
|------|------|------|
| Dropout | 0.3 | 随机丢弃 30% 神经元 |
| 标签平滑 | 0.1 | 硬标签 → 软标签，防过自信 |
| 权重衰减 | 1e-3 | L2 正则化 |
| 数据增强 | 80% 概率 | 基线漂移、高斯噪声、时间扭曲、幅度缩放、导联丢失 |
| 早停 | patience=10, min_delta=1e-4 | val_auc 改善不够不重置 |

---

## 7. 实验过程

### 7.1 实验一：InceptionTime 轻量基线

**配置**：291K 参数，lr=1e-4, batch_size=256, AsymmetricLoss

**训练过程**：

```
Epoch 1:  train_loss=0.076, val_auc=0.557
Epoch 2:  train_loss=0.023, val_auc=0.567
Epoch 3:  train_loss=0.003, val_auc=0.583  ← train_loss 快速归零
Epoch 5:  val_auc=0.593  ← 峰值
Epoch 6+: val_auc 微跌，持续下滑
```

**结果**：峰值 val_auc=0.593，test macro_auc=0.590。train_loss 在 3 轮内从 0.076 骤降至 0.003，此后 val_auc 不再上升。

**分析**：尝试 weight_decay=1e-3 + dropout=0.5 后 val_auc 反而降到 0.50。291K 参数是表示能力的硬瓶颈，与正则化强度无关。结论：小模型天花板 ~0.60。

### 7.2 实验二：xResNet1D-101（两轮调优）

**第一轮（失败）**：

配置：lr=1e-5, batch_size=128, AsymmetricLoss, weight_decay=5e-4, dropout=0.2

```
Epoch 1:  train_loss=0.097, val_auc=0.540
Epoch 3:  train_loss=0.015, val_auc=0.583  ← 峰值
Epoch 4+: train_loss→0.000, val_auc 持续下跌至 0.544
Test macro_auc: 0.544（比 InceptionTime 还差）
```

**根因**：① lr=1e-5 过低，7.6M 参数模型快速收敛到尖锐极小值而非平坦极小值，泛化极差；② AsymmetricLoss 的 `(1-p)^γ` 负类聚焦与标签平滑 `y→y·(1-α)+α/2` 的梯度方向相反，叠加后优化方向混乱。

**第二轮（成功）**：

配置：lr=5e-5, batch_size=128, BCEWithLogitsLoss, weight_decay=1e-3, dropout=0.3, label_smoothing=0.1

```
Epoch 1:  loss=0.561, val_auc=0.541
Epoch 5:  loss=0.319, val_auc=0.664
Epoch 10: loss=0.309, val_auc=0.688
Epoch 15: loss=0.303, val_auc=0.744  ← 加速提升
Epoch 20: loss=0.298, val_auc=0.767
Epoch 25: loss=0.295, val_auc=0.791  ← 接近峰值
Epoch 36: loss=0.291, val_auc=0.793  ← 最终峰值
Epoch 46: 早停触发
```

**结果**：test macro_auc=0.791，train_loss 全程稳在 0.29-0.56 未归零，val_auc 持续上升无崩溃。BCE + 强正则化是正确策略。

### 7.3 实验三：ECG Transformer（NPU 适配 + 两版训练）

**NPU 算子适配**：PyTorch `nn.TransformerEncoderLayer` 使用融合算子 `aten::_transformer_encoder_layer_fwd`，`nn.MultiheadAttention` 使用 `aten::_native_multi_head_attention`。CANN 8.0.0 均未适配，trigger CPU 回退。拆分为基础算子：

```
QKV = Linear(src) → chunk → reshape → Q, K, V
attention = softmax(Q @ K^T / sqrt(d_k)) @ V
output = attention.reshape + Linear
```

全部在 NPU 上执行，无需 CPU 回退。

**第一版（AMP，失败）**：

配置：lr=5e-5, batch_size=64, AMP 开启

```
Epoch 1-29: 无 NaN，val_auc 持续升至 0.799
Epoch 30:   LossScale→8,388,608，开始批量 NaN
Epoch 31:   几乎每 batch 都 NaN，val_auc 从 0.804 跌至 0.789
Epoch 32+:  持续退化
Best (epoch 30): test macro_auc=0.802
```

**根因**：手写 attention 在 FP16 下 `Q @ K^T` 数值范围不受融合算子的精度保护，LossScale 不断加倍以补偿梯度下溢，最终溢出为 NaN。

**第二版（FP32，成功）**：

配置：全部同上 + `--no-amp`

```
Epoch 1:  loss=0.516, val_auc=0.566
Epoch 7:  loss=0.306, val_auc=0.700
Epoch 10: loss=0.303, val_auc=0.735
Epoch 15: loss=0.299, val_auc=0.732
Epoch 20: loss=0.295, val_auc=0.779
Epoch 27: loss=0.291, val_auc=0.791
Epoch 38: loss=0.288, val_auc=0.794  ← 峰值
Epoch 48: 早停触发
```

**结果**：test macro_auc=0.808，全程无 NaN。FP32 略慢但数值安全。

### 7.4 模型集成

xResNet（卷积，擅长局部形态特征）与 ECG Transformer（自注意力，擅长全局节律依赖）概率平均：

```
ensemble_prob = (xResNet_prob + Transformer_prob) / 2
```

**结果**：test macro_auc=**0.822**，相比最强单模型（Transformer 0.808）提升 +0.014。两类异质架构互补验证有效。

---

## 8. 实验结果

### 8.1 测试集评估

| 模型 | macro_auc | macro_f1 | mAP | 参数量 | 训练时间 |
|------|:---:|:---:|:---:|------|------|
| InceptionTime | 0.590 | 0.290 | — | 291K | 40 min |
| xResNet1D-101 | 0.791 | 0.291 | 0.249 | 7.6M | 83 min |
| ECG Transformer | 0.808 | 0.321 | 0.286 | 6.5M | 85 min |
| **集成** | **0.822** | **0.328** | **0.292** | 14.1M | — |

### 8.2 指标解读

- **macro_auc = 0.822**：随机挑一个阳性患者和一个阴性患者，模型有 82.2% 概率给阳性患者更高分数。不依赖分类阈值，不受类别不平衡影响，是本实验最可信的指标。
- **macro_f1 = 0.328**：精确率与召回率的调和平均，受 485:1 极端不平衡和阈值选择影响。逐类最优阈值优化后从 0.139（阈值 0.5）提升至此。
- **mAP = 0.292**：27 类平均精度均值，稀有类（43 样本）严重拖累。

### 8.3 与文献对比

| 方案 | macro_auc | 备注 |
|------|:---:|------|
| 随机基线 | 0.50 | — |
| InceptionTime（本实验） | 0.59 | 轻量基线 |
| xResNet1D-101（本实验） | 0.791 | 单模型 |
| ECG Transformer（本实验） | 0.808 | 单模型 |
| **集成（本实验）** | **0.822** | 两模型概率平均 |
| Challenge Top 10 单模型 | 0.80-0.83 | 文献报告 |
| Challenge 冠军集成 | 0.85+ | 多模型 + 预训练 |

---

## 9. 关键问题与解决

| 问题 | 严重度 | 解决方法 |
|------|:---:|------|
| 8 个 .npy 文件含 NaN，静默污染模型权重 | 🔴 | Dataset 层 `np.nan_to_num()` 清洗 |
| NumPy 2.x 与 CANN 8.0.0 不兼容 | 🔴 | `pip install "numpy<2"` |
| `--quick-test` 模式下 T_max 为负数崩溃 | 🔴 | `T_max = max(epochs - warmup, 1)` |
| minmax 归一化静默归零 | 🔴 | 补充 minmax 分支实现 |
| CANN 8.0.0 不支持 Transformer 融合算子 | 🟠 | 手写 attention，拆分为基础算子 |
| AMP LossScale 膨胀致梯度 NaN | 🔴 | 切换 FP32 |
| AsymmetricLoss + Label Smoothing 冲突 | 🟠 | 改回 BCEWithLogitsLoss |
| xResNet lr=1e-5 陷入尖锐极小值 | 🟠 | lr 提高到 5e-5 |
| 梯度噪声公式 bug（放大 1e5 倍） | 🟠 | 修正衰减公式 |
| F1 硬编码阈值 0.5 低估模型能力 | 🟡 | 逐类 P-R curve 搜索最优阈值 |
| 早停无 min_delta 致无效延长 | 🟡 | `> best + 1e-4` |

---

## 10. 模型能力与局限

### 10.1 能力

- 常见心律失常（窦性心律、房颤、束支阻滞等，样本 > 1000）的识别已较可靠
- 可作为辅助筛查工具——将概率排序后交给医生确认
- 推理速度满足实时性要求（单条 ECG < 1 秒）

### 10.2 局限

- **稀有类极弱**：窦性心律失常（43 样本）等几乎无法预测，需更多数据
- **相近疾病混淆**：房颤 vs 房扑、左束支 vs 右束支容易互报
- **无预训练**：未利用大量无标签 ECG 数据，泛化上限未达到
- **单模态**：未结合临床文本和患者元数据

### 10.3 改进方向

1. **SimCLR 对比预训练**：预期 +0.02-0.03 AUC
2. **长尾类增强**：对稀有类进行针对性过采样或合成
3. **多模态融合**：结合 LLM 生成的临床症状描述
4. **多模型集成扩展**：加入 InceptionTime、ResNet1D-34

---

## 11. 演示程序

本实验附带一个 Web 演示程序（`demo.py`），基于 Gradio 构建：

- 从测试集 6,472 条 ECG 中任选一条
- 实时显示 12 导联波形图
- 对比 xResNet、ECG Transformer、集成模型的预测结果
- 与真实诊断标签并列对比

**运行方式**：
```bash
cd 大作业演示
pip install gradio matplotlib torch
python demo.py
# 浏览器打开 http://127.0.0.1:7860
```

---

## 12. 结论

1. 在 Ascend NPU 910B 上完成了 xResNet1D-101 和 ECG Transformer 的从头训练，集成宏平均 AUC 达 **0.822**，接近 PhysioNet 2020 Challenge 前 10 名单模型水平。

2. BCE + 强正则化（dropout 0.3, weight_decay 1e-3, label_smoothing 0.1）是 27 类极端不平衡任务的有效训练策略，优于 AsymmetricLoss。

3. CANN 8.0.0 的算子覆盖有限（Transformer 融合算子未适配），但通过手写 attention 拆分基础算子可绕过限制，确保全在 NPU 执行。

4. 模型对常见心律失常的识别已具备临床参考价值，但稀有类仍是主要瓶颈，需通过预训练、数据增强和多模态融合进一步改进。
