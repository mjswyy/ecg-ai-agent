# ECG 模型模块审查报告

审查范围：`src/ecg_models`（feature_extraction ×4、backbone ×5、classifiers ×2、trainer）、`src/utils/device_utils.py`、`tests/test_ecg_models.py`，共 14 个文件。
审查方式：逐文件用 read 工具实际读取全量源码后人工核对（含与官方 Net1D / ASL 论文公式 / SimCLR 标准实现的比对）。

---

## 总体评价（3-5句）

整体架构清晰、模块划分合理，NPU 适配思路（手写 attention、AMP 适配器、NaN 防护）方向正确，backbone 形状链路 `(B,12,L)→(B,feature_dim)` 在 ResNet1D / xResNet1D-101 / InceptionTime / ECGTransformer 四者间一致且经测试验证。但存在 **5 处 🔴 级正确性问题**，集中在"算法公式与实现不一致"上：InfoNCE 正样本泄漏进负样本集、AsymmetricLoss 负类聚焦权重与论文方向相反、`_challenge_score` 实参错位、QT 波形索引的 NaN→int 崩溃、VAE 阈值跨设备比较崩溃。此外特征提取三个模块对"空/平/脏"信号的鲁棒性总体尚可（有长度与过滤防护），但异常捕获范围偏窄、时间戳传递链路存在静默失效；trainer 是逻辑最复杂却**零测试覆盖**的文件，上述算法级 bug 均未被现有测试捕获。ecgfounder_net1d 与官方 Net1D 的移植核对结论：forward/BasicBlock/SE/SAME padding 均忠实一致，未发现移植引入的结构性改动，但保留了 einsum 与 groups 整除性两个 NPU/配置隐患。

---

## 各文件逐条发现

### 1. `src\ecg_models\feature_extraction\r_peak_detector.py`

- 🟠 **L76**：`except (ImportError, ValueError, KeyError)` 捕获范围过窄。neurokit2 在异常信号上可能抛 `TypeError`、`IndexError` 或 numpy 内部异常（如 `np.linalg` 类），此时回退逻辑不生效、整个 `detect()` 崩溃。修法：回退路径改为 `except Exception`（该分支本就是降级路径，宽捕获合理）。
- 🟠 **L141-145**：`detect_multi_lead` 对每个导联完整跑一遍 `nk.ecg_process`（每导联秒级），默认 3 个导联 = 3 倍全管道耗时，且共识投票为 O(P²·L)。对长记录或批量推理是明显性能瓶颈。建议：只对 2-3 个导联并行复用单次 `ecg_peaks`（信号拼接 batch 化），或先用快速阈值法粗筛再精调。
- 🟡 **L81-87**：`len(r_peaks) < 2` 早退分支返回的 dict **缺少 `"rhythm"` 键**（正常路径有），且不更新 `self.last_result`（L121 只在末尾赋值）→ 返回 schema 不一致、缓存失效。修法：早退分支补 `"rhythm": "unknown"` 并设置 `last_result`。
- 🟡 **L111-120**：返回的 `rr_intervals` 是**未过滤**的原始间期（L91 的 diff 全集），而 `heart_rate`/`hr_std` 基于过滤后的 `rr_valid`（L94）计算 → 下游若直接用 `rr_intervals` 统计会与 HR 口径不一致（下游 HRVAnalyzer 会再过滤，影响被掩盖）。建议返回前也过滤，或文档注明。
- 🟡 **L90 / L44**：`fs` 无校验。`fs <= 0` 时 L90 除零产生 Inf、`int(fs)` 截断 0.5Hz 级误差。建议入口校验 `fs > 0`。
- 🟡 **L197-202**：`_simple_peak_detect` 中 `fs < 10Hz` 时 `5.0/nyquist > 1` 使 butter 抛 `ValueError`，被宽 except 静默吞掉后**无滤波直接找峰**——平信号/强基线漂移下易产生大量伪峰。建议对 nyquist 越界显式告警而非静默降级。
- 💡 **L35-36**：`VALID_METHODS`/`LEAD_II_INDEX` 类常量设计良好；但 `detect` 的 docstring 返回键清单缺 `"rhythm"`（L55-56），建议同步。
- 💡 **L161**：`agreement = len(consensus) / max(len(p) for p in all_peaks) if all_peaks else 0.0` 的三元优先级正确（`X if cond else Y`），无 bug，但可读性差，建议加括号。

### 2. `src\ecg_models\feature_extraction\hrv_analyzer.py`

- 🟠 **L85 + L93-98**：**时间戳静默失效**。L85 先对 `rr` 过滤（长度变小），再在 L93-98 用"过滤后"的 `len(rr)` 与 `len(timestamps_ms)`（未过滤的 R 峰时间戳，长度 = 原始间期数+1）比对。只要过滤掉 ≥1 个间期（有早搏/伪差时几乎必然），两个分支都不命中 → 落到 L98 的 `cumsum` 近似，**调用方传入的 `r_peaks_ms` 被静默丢弃**，频域时基失真（压缩了被剔除间期的时间）。修法：先按 `timestamps_ms` 对齐过滤（同步剔除对应 R 峰），再判断长度分支；或在 docstring 明确"时间戳仅在全间期有效时使用"。
- 🟠 **L146-165**：`_sample_entropy` 为 O(N²) 的 Python 双层循环 + 每模板构造 `templates` 数组（内存 O(N·m)）。对长记录（>1000 个 RR，如 24h 心电）耗时可到分钟级，且不可在 NPU 加速。建议：向量化（一次广播计算全部成对 Chebyshev 距离）或改用 `nolds`/neurokit2 的现成实现；同时给 `N` 设上限。
- 🟡 **L117-121**：`np.trapz` 在 NumPy ≥2.0 已弃用（改名 `np.trapezoid`），建议替换以避免告警/未来移除。
- 🟡 **L102**：`np.arange(t[0], t[-1], 1/fs)`——若 `t` 范围不足 1 个采样步长（极端短数据）会得到空数组，后续 interp/welch 空输入报错。虽然有 `len(rr)>=10` 前置（≥10 个 300-2000ms 间期时跨度 ≥2.7s 足够），仍建议防御性判空。
- 🟡 **L69**：`pnn50 = nn50 / len(diff) * 100`——`len(diff)` 恒 ≥1（有 `len(rr_clean)>=2` 前置），逻辑安全，但表达式顺序易误读为 `nn50 / (len(diff)*100)`，建议加括号。
- 💡 **L47**：`len(rr_intervals) < 3` 即返回空——时域只需 ≥2，频域需 ≥10，非线性需 ≥3，三档阈值不一致且未在 docstring 说明；建议按需分级返回而非一刀切全空。

### 3. `src\ecg_models\feature_extraction\qt_analyzer.py`

- 🔴 **L141-149 `_safe_wave_index`**：**NaN → int() 崩溃**。neurokit2 `ecg_delineate` 返回的 waves 列（pandas Series）中漏检的波值为 `np.float64('nan')`。L147 的守卫 `not (isinstance(val, float) and np.isnan(val))` 对 `np.float64` 恒为 True（`np.float64` 不是 `float` 子类）→ 执行 `int(np.float64('nan'))` 抛 `ValueError`，而 except 元组只有 `(IndexError, TypeError, KeyError)` → **异常逃逸，整个 `analyze()` 崩溃**。只要任意一个 beat 漏检 Q/T/P 波即触发。修法：`isinstance(val, (float, np.floating)) and np.isnan(val)`，并把 `ValueError` 加入 except 元组。
- 🟠 **L85**：只捕获 `ImportError`。`ecg_delineate` 对短信号/异常 r_peaks 可能抛 `ValueError`（如 R 峰越界）→ 直接崩溃而非走 `_simple_delineate` 回退。建议与 RPeakDetector 一致改为宽捕获。
- 🟠 **L91-96 + L137**：**测量全部失败时误报 "shortened"**。当 `qt_intervals` 为空（delineate 全失败）→ `qt_ms=0` → `_interpret_qt(0, 0, upper)`：两个 QTc 阈值都不满足 → `qt_ms < 300` 命中 → 返回 `"shortened"`，而 `num_beats_analyzed=0`——临床输出自相矛盾。修法：`qt_intervals` 为空时直接返回 `insufficient_data` 语义（interpretation 与 QTc 均置 0/unknown）。
- 🟡 **L71/78/83**：间期无下限校验。若 delineate 给出 `t_offset < q_onset`（负 QT）或 `s_offset < q_onset`，产生负间期 → 中位数可能为负 → 误判 "shortened"。建议过滤 `>0` 的间期。
- 🟡 **L111**：`sex == "Male"` 大小写敏感，`"male"`/`"M"` 会落入女性阈值（460ms），边界行为不一致。建议 `sex.lower() == "male"`。
- 🟡 **L162**：`_simple_delineate` 的 PR 间期恒为 `min(200, r)`ms 常量（`r - max(0, r-0.2fs)` 恒等于 min(200ms, 到信号起点)），无测量意义，属于回退方案中的"占位值"，建议文档注明或直接返回 None 语义。
- 💡 **L59**：`ecg_delineate` 第一参数应为**清洗后**信号（`ecg_clean`），此处传原始信号，含基线漂移/工频时 delineate 精度下降。建议先 `nk.ecg_clean`。

### 4. `src\ecg_models\feature_extraction\feature_bank.py`

- 🟠 **L75-87 + L93-99**：与 hrv_analyzer 的 L85 时间戳问题联动——`rr_ts = r_peaks_ms`（全 beat）与过滤后的 `rr` 长度不匹配，实际几乎总是落入 hrv 的 cumsum 回退（见 hrv L85/L93-98）。频域 HRV 的时间基准在所有含剔除间期的信号上都是近似值。修法：在 FeatureBank 层完成"过滤+对齐"后再传入，或给 hrv 增加"同步剔除"接口。
- 🟡 **L60**：`min(self.lead_for_analysis, ecg.shape[0]-1)`——`ecg.shape[0]==0` 时得 `-1`，`ecg[-1]` 在空数组上抛 `IndexError`。极端输入防护缺失（与 qt 的空信号处理不对称）。
- 🟡 **L134-138**：`to_vector` 中 `isinstance(val, bool)` 分支是死代码——`bool` 是 `int` 子类，先被 L134 的 `(int, float)` 分支捕获（结果为 1.0/0.0，行为正确）。建议删除或调整分支顺序。
- 🟡 **L87/L99**：直接调用 `_empty_result()` 私有方法（同类内 OK），但两条空路径（HRV <3 间期 / QT <2 beat）的触发条件不一致（3 vs 2），建议统一常量。
- 💡 **L143-158**：`_default_feature_keys` 与 `_empty_result` 的键集合手工维护，新增指标易漏同步导致 `to_vector` 静默补 0。建议让 `feature_dim`/键列表从同一数据源派生，并加一个"键全集一致性"单测。

### 5. `src\ecg_models\backbone\resnet1d.py`

- 🟡 **L8-9/L141-142**：docstring 声称 base_channels=32 的 ResNet-18 "~5M params"——实际约 2-3M（取决于 layers），且 ResNet-34 注释 "~5M" 在 base=32 时也不准。参数数量与配置脱钩，建议按实际计算或删去具体数字。
- 🟡 **L163-171（及各 backbone 通病）**：无最短输入长度校验。`L < in_kernel_size` 时 stem 卷积输出长度 ≤0 → 后续 BN/池化崩溃，报错信息晦涩。建议在各 backbone `forward` 入口对 `L` 做防御性校验并给出清晰错误。
- 🟡 **L157**：`zero_init_residual` 默认 `False`，与 ResNet 官方建议（True，加速深层收敛）相反；xResNet1D 则恒为 True，两文件默认值不一致。建议统一为 True。
- 💡 **L220-233**：初始化仅覆盖 Conv1d/BN，`downsample` 内的 Conv1d/BN 也被 `self.modules()` 遍历覆盖 ✓；但 `dropout` 参数没有同步用于 stem，若需 stem 正则需手动加。可维护性建议：补一个"两阶段初始化（pretrain/finetune）"的公共入口。

### 6. `src\ecg_models\backbone\xresnet1d.py`

- 🟡 **L58-83**：Stage 通道链路核对结论：stage1 `32→128`（downsample 因 32≠32×4 而创建，正确）、stage2 `128→256`、stage3 `256→512`、stage4 `512→1024`，GAP 后 `Linear(1024,512)` → `feature_dim=512`，与注释一致、形状链路正确。唯一疑点：**Stage2 从 128 降到 64 再扩回 256** 的结构是"瓶颈中道"设计（middle=64），与 Ribeiro 原版 `filter_list` 结构对应关系建议在注释里给出官方出处，便于后续对照权重。
- 🟡 **L127-141**：`zero_init_residual` 硬编码为 True（无开关），与 resnet1d 的默认 False 不一致（同上）；`Linear` 初始化用 `normal_(0,0.01)` 而 head 用 xavier，风格不统一（影响小）。
- 🟡 **L166-168**：工厂函数 `xresnet1d_101(in_channels, dropout=0.1, **kwargs)`——`**kwargs` 可再传 `dropout` 造成重复参数冲突（`TypeError: got multiple values`）。建议 `kwargs` 中剔除已显式列出的参数。
- 💡 **L96-125**：`_make_stage` 与 resnet1d 的 `_make_stage` 逻辑重复（仅 block 类型/kernel 不同），建议抽公共 stage 工厂，减少两处演化不同步的风险。

### 7. `src\ecg_models\backbone\inception_time.py`

- 🟡 **L35/L54-59**：`kernel_sizes` 假定全为奇数（padding=ks//2 才对称）；MaxPool1d(3,1,1) 无 stride 保长 ✓。若用户传偶数 kernel，输出长度偏移 1 且 `combine` 后 BN 仍能跑但语义错位。建议构造时校验奇偶。
- 🟡 **L113-129**：残差投影用 `Conv1d(1x1, bias=False)` 无 BN，模块输出含 BN——恒等路径与残差路径的数值尺度略有差异，属可接受设计；但 `proj` 在 `current_channels != module.out_channels` 时无 ReLU，与主路 ReLU 后相加，建议注释说明（当前仅靠"原版如此"）。
- 💡 **L136-154**：模块内无 dropout（原版 InceptionTime 默认 dropout=0 亦如此）✓ 与官方一致；`depth=3` 时 `feature_dim=128` 与 docstring 一致 ✓。建议补一个 `depth`/`n_filters` 非默认配置的形状测试（当前测试只跑默认值）。

### 8. `src\ecg_models\backbone\transformer_encoder.py`

- 🟡 **L41-48**：`PositionalEncoding` 假设 `d_model` 为偶数——`pe[:,0::2]` 列数与 `div_term` 长度（`d_model//2`）在奇数 d_model 下不匹配（broadcast 报错或错位）。建议 `assert d_model % 2 == 0` 或对奇数分支处理。
- 🟡 **L66**：`assert d_model % nhead == 0`——`python -O` 下 assert 被剥离，`head_dim = d_model // nhead` 变非整除 → 后续 `view` 崩溃且报错信息不明。建议改为显式 `if ... : raise ValueError`。
- 🟡 **L159-160 + L51**：`max_seq_len=3000` 硬上限——当输入长度使 `L' > 3000` 时，`pe[:, :x.size(1), :]` 切片只取前 3000 行，`x + pe` broadcast 直接 RuntimeError（报错信息不含"序列超长"提示）。建议入口校验并给出计算式（`compute_output_length` 已有，可直接复用）。
- 🟡 **L181-187**：`_init_weights` 只初始化 `lead_proj` 与 encoder 层参数；`lead_bn` 由 PyTorch 默认初始化 ✓；`PositionalEncoding` 固定正弦（docstring 写"sinusoidal + learnable"实为纯 sinusoidal）——文档与实现不符。
- 💡 **L55-113（NPU 手写 attention）**：拆分算子思路正确（避开 `_transformer_encoder_layer_fwd`/`_native_multi_head_attention` 的 CANN 回退）；但**没有与 `nn.MultiheadAttention` 的输出一致性/梯度一致性测试**，NPU 上算子行为差异无法被现有测试捕获。建议加"同权重下手写层 vs nn.TransformerEncoderLayer 输出误差 < 1e-4"的回归测试。
- 💡 **L228-232**：`compute_output_length` 静态方法无任何调用方/测试——建议在 `forward` 内用它做长度校验，消除"注释里的公式"与"实际卷积"漂移风险。

### 9. `src\ecg_models\backbone\ecgfounder_net1d.py`（与官方 Net1D 移植核对）

**移植一致性结论**：与官方 `Net1D`（Shenda Hong 原版）逐段比对——`MyConv1dPadSame` 的 SAME padding 公式（L55-59）、`MyMaxPool1dPadSame`（L85-88）、`BasicBlock` 的 conv1→convk→conv1 + SE 结构（L175-228）、`is_first_block` 跳过 bn1/activation1/do1 的逻辑（L181-186）、通道 padding 恒等捷径（L218-223）、`BasicStage` 首块 downsample 逻辑（L258-265）、`Net1D` 的 first_conv(bn/activation) → stages → `mean(-1)` → dense（L373-395）**均忠实一致，未发现移植改动引入的结构性 bug**。以下为移植保留/环境相关隐患：

- 🟠 **L211**：`torch.einsum('abc,ab->abc', out, se)` 与官方一致，但 einsum 在 Ascend NPU 上算子支持不完整、且不在 AMP autocast 的算子清单内（fp16 输入下可能按 fp32 执行或回退 CPU）。**等价替换**为 `out * se.unsqueeze(-1)` 可完全消除该风险，且数值等价。
- 🟠 **L361 + L131/L148-153**：`groups = out_channels // groups_width`（L361）——当 `out_channels % groups_width != 0` 或 `ratio` 使 `middle_channels` 不被 `groups` 整除时，`Conv1d(groups=...)` 直接 RuntimeError。默认配置（filter_list 均为 16 的倍数）安全，但换成任意 `filter_list`/`groups_width`/`ratio` 组合即炸。建议构造时校验 `out_channels % groups_width == 0` 且 `middle_channels % groups == 0`。
- 🟡 **L7-15**：死代码导入：`numpy as np`（文件内未用）、`Counter`、`matplotlib.pyplot`、`torch.optim` 均未使用；`Dataset` 仅被 MyDataset 使用。移植残留，建议清理。
- 🟡 **L17-26**：`MyDataset`（数据加载）定义在 backbone 模块内，与训练器职责混杂；本仓库已有 trainer，建议删除或移到 data 模块。
- 🟡 **L397-411**：`__main__` 调试块用 `use_bn=False, use_do=False` 且 n_classes=150，与模块默认（use_bn/use_do=True）不一致，易误导阅读者；建议删除或改为与默认一致的 smoke 配置。
- 🟡 **L321-336**：`self.verbose`/`print` 调试路径保留（L288-292），生产代码应改为 `logging`。
- 💡 **L371/L389-395**：Net1D 输出 `(B, n_classes)` 而非 `(B, feature_dim)`，且**无 `feature_dim` 属性**——不能直接作为 backbone 接入 `ArrhythmiaClassifier`（后者访问 `backbone.feature_dim` 会 AttributeError）。若作为 backbone 使用需 `return_features=True` 并自行包装。建议增加 `feature_dim` 属性（`in_channels` 末段宽度）与包装类。
- 💡 **测试缺口**：移植代码在 `tests/test_ecg_models.py` 中**零覆盖**——建议至少加一个"默认配置 forward 形状 + 与官方权重加载"的冒烟测试。

### 10. `src\ecg_models\classifiers\arrhythmia_classifier.py`

- 🔴 **L100-106（AsymmetricLoss 负类聚焦权重方向错误）**：docstring 声称"从 Ben-Baruch et al. (ICCV 2021) 论文公式正确实现"，但负类权重 `neg_weights = pt_neg ** gamma_neg`（`pt_neg = 1 - p`）与论文公式 **相反**。论文：`L_neg = −(1−y)·(p_m)^γ⁻·log(1−p_m)`，其中 `p_m = max(p−m, 0)`——权重随"困难程度"（p 增大）**增大**，简单负样本（p→0）权重为 0。当前实现：困难负样本（p→0.9）权重 `(0.1)^4 ≈ 1e-4` ≈ 0，简单负样本（p→0.1）权重 `(0.9)^4 ≈ 0.66`——**恰好把 ASL 要保留的困难负样本梯度抹除、简单负样本保留**，假阳性将无法被纠正，与 ASL 设计目标相反。修法：`neg_weights = torch.clamp(p - self.clip, min=0.0) ** self.gamma_neg`（配合 L96-98 已有的 `log(1−p+clip)` 即 `log(1−p_m)`）。建议同时补一条"困难负样本损失 > 简单负样本损失"的数值断言单测。
- 🟡 **L89**：`clamp(logits, -50, 50)` 注释为"防 NPU sigmoid 溢出"——合理；但注意 clamp 使 |logit|>50 处梯度为 0（对损失影响可忽略），建议注释说明。
- 🟡 **L96-98**：`loss_neg` 对"已正确分类的简单负样本"（p→0）给出 `−log(1+clip) < 0` 的**负损失贡献**（小幅拉低总损失）。论文此处应为 0（p_m=0 → log(1)=0）。数值影响小（~−0.05/样本），但属公式偏差，建议修正为对 `p_m==0` 直接置 0。
- 💡 **L110-124（FocalLoss）**：实现正确（`pt=exp(−bce)` 等价 `p^y(1−p)^(1−y)`）；`alpha` 的 device 迁移在 forward 内完成 ✓。建议加 `alpha` 的单元测试（当前无）。

### 11. `src\ecg_models\classifiers\anomaly_detector.py`

- 🔴 **L128**：`self.threshold = torch.tensor(float(...))` 创建的是 **CPU 张量**并赋给注册 buffer。当模型在 cuda/npu 上时，`is_anomaly` 的 `self.anomaly_score(x) > self.threshold`（L111）发生**跨设备比较 RuntimeError**。修法：`self.threshold = torch.tensor(..., device=next(self.parameters()).device)`（或 `x.device`）。现有测试未调用 `fit_threshold`/`is_anomaly`，故未暴露。
- 🟠 **L95-96**：`anomaly_score` 内无条件 `self.eval()` 且**不恢复训练状态**——若在训练循环中调用（如定期记录分数），模型会永久停留在 eval，BN 统计冻结、dropout 关闭，后续训练被静默破坏。修法：`training = self.training; ... ; if training: self.train()`。
- 🟠 **L123-128**：`fit_threshold` 对**空 loader** 执行 `np.concatenate([])` 直接抛 `ValueError`；且 `_fitted` 是普通属性（不在 state_dict）——加载 checkpoint 后即使 `threshold` buffer 已保存也必须重新 fit，否则 `is_anomaly` 报错。建议：空 loader 返回错误/默认阈值；`_fitted` 改为按 `threshold` 是否已非默认推断或存入 checkpoint。
- 🟡 **L73-77**：`reparameterize` 在 eval 时返回 `mu`（确定性评分）✓ 合理，但 `anomaly_score` 依赖此行为而 `forward`（训练）用采样——同一 API 两种语义，建议文档写明。
- 💡 **L58-66**：`beta=1.0`（标准 VAE）；β-VAE 下 KL 权重可调 ✓。建议补 `fit_threshold`/`is_anomaly` 的端到端测试（含 CPU 与加速器两条路径，可暴露 L128 的跨设备 bug）。

### 12. `src\ecg_models\trainer.py`

- 🔴 **L213-228（InfoNCE 正样本泄漏）**：`mask = torch.eye(z.size(0))` 只剔除主对角线（自身相似度），`negatives = sim[~mask]` 中**包含正样本对** `sim[i][B+i]`（z1[i]·z2[i]，即 L217 `diag(sim, B)` 取出的同一元素）。同一相似度既被当作正样本（logits 第 0 列）又被当作负样本（negatives 中）→ softmax 分母里正样本出现两次，交叉熵对"推高正样本"和"压低正样本"同时施压，对比梯度自相矛盾，InfoNCE 语义错误，预训练表征质量受损。修法（标准 SimCLR）：`labels = torch.arange(B)` 拼接为 (2B,) 并 `logits = sim`，将主对角线与配对位置置 `-inf` 后 `cross_entropy`；或构造 `neg_mask = ~(eye | diag-offset-B 的配对掩码)` 后取 negatives。
- 🔴 **L454（`_challenge_score` 实参错位）**：调用 `self._challenge_score(labels, preds_optimal, probs, best_thresholds)` 与签名 `(labels, probs, preds_binary=None, thresholds=None)` 逐位对齐后：`probs` 形参收到**二值矩阵** `preds_optimal`、`preds_binary` 形参收到**连续概率** `probs`。后果：① F_beta 用连续概率算"软计数"（`tp = (preds_binary * labels).sum(...)` 不再是计数）；② G_beta 的 `np.argsort(-probs[:, c])` 对二值矩阵排序，同值内部顺序任意（quicksort 不稳定）→ DCG 退化为随机排列求和。**Challenge Score 指标整体失真**，且该指标用于早停依据（L364）。修法：改为 `self._challenge_score(labels, probs, preds_optimal, best_thresholds)`。
- 🟠 **L141-146 / L299-301（NaN 跳过未清梯度）**：`optimizer.zero_grad()` 在 NaN 检查**之后**（L149/L303）。当某 batch loss 为 NaN 被 `continue` 时，上一 batch 累积的梯度残留，下一 batch `backward()` 会**累加**到旧梯度上 → 梯度污染。修法：NaN 检查前先 `optimizer.zero_grad()`，或 `continue` 前显式清空。
- 🟠 **L406-413（evaluate 空 loader）**：`np.concatenate(all_logits)` 在 loader 为空时抛 `ValueError`。训练中途用空验证集评估会直接崩溃。建议判空返回全 0 指标。
- 🟡 **L328-333（梯度噪声与 AMP 顺序）**：`grad_noise` 的噪声加在 `clip_grad_norm_` **之后**（L319-323），且当 `grad_clip==0` 时（L319 不进入）梯度仍是 **scaler 缩放后**的（未 `unscale_`），噪声标准差与梯度尺度不一致（NPU/CUDA 上差一个 scale factor）。建议：先 `unscale_`（若未 clip）→ 加噪声 → 再 clip（可选）→ step。
- 🟡 **L560-571 + L379**：`_save_checkpoint` 保存的 `self.best_metric` 在训练过程中**恒为 0.0**（L379 在训练结束后才赋值，而保存发生在 L368 早停判定内）→ checkpoint 元数据错误，加载后 `best_metric` 失真。修法：L364-368 更新 `best_val_auc` 时同步 `self.best_metric`。
- 🟡 **L7/L265-267**：类 docstring 声称"多标签微调（Asymmetric Loss）"，实际默认 `BCEWithLogitsLoss`（L267，注释说"ASL 可在确认收敛后切换"）——文档与行为不符，两处二选一统一。
- 🟡 **L31**：`from src.utils.device_utils import ...` 绝对导入依赖项目根在 `sys.path`（测试靠 `sys.path.insert` 兜底），打包安装后 `src` 命名空间不可用。建议改相对导入或包内 `utils` 导入。
- 🟡 **L548-556**：`_cosine_schedule` 当 `warmup >= epochs` 时 `T_max = max(epochs-warmup, 1) = 1`，且 SequentialLR 的 milestone 永不触发——预热期直接跨过整个训练，cosine 段完全没用。边缘参数应加校验或告警。
- 🟡 **L291-292**：标签平滑 `labels * (1-s) + 0.5*s` 对 0/1 二值标签正确 ✓；但与 ASL 混用时会破坏其二值假设（`(1-targets)` 变分数）——当前默认 BCE 无碍，若切换 ASL 需注意。
- 💡 **整体**：trainer 是 579 行、含 InfoNCE/早停/AMP/梯度噪声/Challenge Score 等复杂逻辑的文件，**tests 中零覆盖**——本文件的两个 🔴 均为"标准实现一眼可验"的错位/泄漏，若有一条 InfoNCE 与 Challenge Score 的数值单测即可拦截。强烈建议补：① InfoNCE 在 2 样本人工输入下的闭式解断言；② `_challenge_score` 与官方 Challenge 脚本（PhysioNet 2020）数值一致性测试；③ NaN 跳过后梯度确实被清空的测试；④ 梯度噪声/标签平滑的梯度形状冒烟。

### 13. `src\utils\device_utils.py`

- 🟡 **L122-129**：`_init_amp` 只要 `import torch_npu` 成功就选用 NPU AMP，**不检查是否有 NPU 设备**（装了 CANN 驱动但无 910B 的开发机也会命中）。当前靠 trainer 的 `use_amp = use_amp and is_accelerator(device)`（enabled=False）兜底，但 `device_utils` 自身的行为与 `detect_device`（用 `_has_npu()`）不一致。建议：`if _has_npu():` 再选 NPU AMP，逻辑统一。
- 🟡 **L135-139**：`from torch.cuda.amp import GradScaler, autocast` 在 torch ≥2.4 已弃用（建议 `torch.amp.GradScaler("cuda")` / `torch.amp.autocast("cuda")`），新版本下产生 DeprecationWarning。NPU 侧 `torch.npu.amp` 同理建议跟进 torch_npu 新 API。
- 🟡 **L61-66**：用户显式传 `"npu:1"`/`"cuda:2"` 等带索引字符串时，仅 `== "npu"` 不匹配 → 原样透传不校验，设备不存在时崩溃而非回退。建议对 `startswith("npu")`/`startswith("cuda")` 校验。
- 💡 **L150-179**：`_DummyGradScaler`/`_DummyAutocast` 无测试；`step(optimizer)` 直通 `optimizer.step()` 的行为（配合 trainer 的 `unscale_` no-op + `clip_grad_norm_`）建议加一条 CPU 端到端冒烟（trainer 在 CPU 上跑 1 个 batch）。
- 💡 **L33-67**：`detect_device` 检测顺序 NPU>CUDA>CPU 与注释一致 ✓；`_has_npu` 的 `(ImportError, AttributeError)` 双捕获正确 ✓。建议补充"显式指定不可用设备回退 CPU"的测试。

### 14. `tests\test_ecg_models.py`

- 🟡 **L77**：`test_feature_bank` 用纯随机噪声（`randn(12,5000)*0.5`）做 R 峰检测——结果不确定且若 neurokit2 在噪声上抛非捕获异常（见 r_peak_detector L76 的窄捕获）会 flaky。建议改用与 `test_r_peak_detector` 相同的合成 QRS 信号。
- 🟡 **L63-70**：`test_qt_analyzer` 只测"空输入"一条路径——`ecg_delineate` 主路径、QTc 三公式数值、`_safe_wave_index` 的 NaN 分支（对应 🔴 崩溃）均无覆盖。建议补：构造含漏检波的 waves 桩、负 QT 过滤、三公式手算断言。
- 💡 **L85-106**：backbone 测试只覆盖默认配置的前向形状；未测 `feature_dim` 随 `base_channels/layers/depth` 变化的配置矩阵，也未测反向传播（NaN 梯度检查）。建议加 `loss.backward()` 冒烟。
- 💡 **ecgfounder Net1D 零覆盖**（移植代码无任何测试）；trainer 零覆盖（见 trainer 💡）；`fit_threshold`/`is_anomaly` 零覆盖（device mismatch 未被捕获）。
- 💡 **L12-48**：R 峰测试质量较好（合成 QRS + 心率区间断言）；建议补"平信号/全零信号 → num_beats=0 不崩溃"的负例，验证 🔴 类边界。
- 💡 **整体**：无 CI 配置说明、无 pytest 标记（若 neurokit2/scipy/sklearn 缺失时部分测试应 skip 而非失败）。建议在文件头按依赖打 `pytest.importorskip`。

---

## 修复优先级清单

### P0 — 正确性（🔴，必须尽快修复）
1. **trainer.py L213-228**：InfoNCE 正样本泄漏进负样本集 → 改标准 SimCLR 标签式实现。
2. **trainer.py L454**：`_challenge_score(labels, probs, preds_optimal, best_thresholds)` 实参错位 → 交换第 2、3 实参（该指标用于早停，失真影响训练决策）。
3. **arrhythmia_classifier.py L104**：ASL 负类聚焦权重 `(1−p)^γ⁻` 与论文 `(max(p−m,0))^γ⁻` 方向相反 → 改为 `clamp(p−clip, min=0)**gamma_neg`，并加数值断言单测。
4. **qt_analyzer.py L141-149**：`_safe_wave_index` 对 `np.float64('nan')` 守卫失效 → `int(nan)` 抛未捕获的 ValueError → 崩溃。改用 `(float, np.floating)` 判断并把 `ValueError` 纳入 except。
5. **anomaly_detector.py L128**：threshold 为 CPU 张量，加速器上 `is_anomaly` 跨设备比较崩溃 → 用模型参数设备创建。

### P1 — 健壮性/数值（🟠，尽快）
6. trainer.py L141-146/L299-301：NaN 跳过前先 `zero_grad`，防梯度残留累加。
7. hrv_analyzer.py L85+L93-98：时间戳与过滤后 RR 对齐，杜绝 `r_peaks_ms` 静默失效。
8. hrv_analyzer.py L146-165：样本熵 O(N²) 向量化或换库；长记录加长度上限。
9. anomaly_detector.py L95：`anomaly_score` 恢复训练状态；L123-128 空 loader 防护。
10. trainer.py L406-413：evaluate 空 loader 判空返回。
11. qt_analyzer.py L85 / r_peak_detector.py L76：异常捕获放宽到回退路径（回退即降级，宽捕获合理）。
12. qt_analyzer.py L137：QT 测量全失败时返回 `insufficient_data` 而非误报 "shortened"。
13. ecgfounder_net1d.py L211：einsum 改 `out * se.unsqueeze(-1)`；L361 增加 groups 整除性校验。
14. feature_bank.py L75-87：与 hrv 时间戳问题联动修复（见第 7 条）。

### P2 — 质量/维护（🟡💡，择机）
15. 统一 schema：r_peak_detector 早退/多导联结果补 `rhythm` 键与 `last_result` 缓存。
16. trainer：`best_metric` 保存时机、grad_noise 与 unscale 顺序、类 docstring 与默认损失统一、`src.` 绝对导入改包内导入。
17. 各 backbone 增加最小长度校验与 `feature_dim` 配置矩阵测试；transformer 补 d_model 奇偶/assert 显式化。
18. 清理 ecgfounder 死导入、`MyDataset` 职责、`__main__` 调试残留。
19. 测试补强（P0/P1 修复后立即跟进）：trainer（InfoNCE/Challenge Score/NaN 跳过）、ecgfounder Net1D、ASL 公式断言、fit_threshold 端到端、QT NaN 分支、平信号负例。

---

*审查日期：本报告基于当前工作区源码逐文件读取完成。行号以读取时的实际文件内容为准。*
