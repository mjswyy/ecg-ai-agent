# 数据与模型脚本审查报告

> 审查范围：`ecg-ai-agent/scripts/` 下 25 个"数据与模型训练"类脚本（预处理、特征提取、训练、评估、MIMIC 域分析、ModelArts 入口）。
> 审查基准：`src/evaluation/metrics/classification.py`（M0.4 协议）、`src/data_pipeline/*`、`src/ecg_models/trainer.py`。
> 结论格式：`文件:行号 [严重级] 问题 → 修法`。

## 总体评价

**架构层面是健康的**：旧/新预处理管线分离（4096 vs 5000）、5k 版复用旧 manifest 划分防泄漏、CLIP 训练对只取 train、fair 评测走 `src` 统一指标入口——这些设计都是正确的。但存在 **5 个 🔴 级问题**会在用户不知情时产出错误结果或违反泄漏红线，以及一批系统性的 🟠 级问题：

1. **重采样比率被破坏**（`preprocess_data_5k.py`）——"防爆内存"循环把 `up/down` 同时约减，改变了 257Hz 数据源的真实重采样比率，信号被静默时间压缩并产生尾部零填充，且这些零进入 z-score 统计。
2. **对比预训练包含测试集**（`train_backbone.py` / `train_simclr.py`）——`ECGContrastiveDataset` 用 `rglob("*.npy")` 扫全目录，val/test 信号进入 SimCLR 预训练，违反泄漏红线。
3. **多模态"测试评估"实为验证集**（`train_multimodal.py` L317）——用 `val_loader` 计算并打印 `Test macro_auc`，且最优模型由同一验证集选出，双重乐观。
4. **`.hea` 直接分析路径是坏的**（`analyze_ecg.py`）——`np.loadtxt` 读 WFDB format-16 二进制会得到垃圾信号并静默输出错误诊断。
5. **指标口径三套并存**：`trainer.evaluate`（PR 曲线最优阈值 macro-F1）、`eval_comprehensive`（测试集上 Youden）、M0.4 协议（验证集 Youden，`classification.py`）。同一批模型报告的 macro-F1 互不可比。

**复现性**：`train_backbone` / `train_simclr` / `train_ecgfounder_ft` / `train_multimodal` 均未设置 `torch.manual_seed`（DataLoader shuffle、dropout、梯度噪声均不可复现）；`eval_comprehensive` 的 bootstrap 无种子。

**路径假设**：约半数脚本默认参数依赖 ModelArts 风格的 `/cache/...` 或用户绝对路径（`C:\Users\llyun\Desktop\ECG`、`D:\ecg\mimic_iv_ecg`），且新旧预处理默认 raw-dir 还不一致（`ecg资料` vs `ECG`）。换个机器/盘符即崩溃，或静默写到错误目录。

**"YAML 从未被加载"问题仍存在**：`configs/train_config.yaml`、`model_config.yaml`、`data_config.yaml` 三个文件存在，但本次审查的 25 个脚本 **没有一个**读取它们（全仓仅 `crawl_knowledge.py` 读 `sources.yaml`）。参数真值全部散落在 argparse 默认值里，YAML 已漂移为死文件。

**内存**：无致命隐患。最大的是 `extract_ecgfounder_features` 全量特征驻留（≈176MB）与 fair/CLIP 的多 split 全载（≈500MB），可接受但可优化。

---

## 各文件逐条发现

### 1. scripts/preprocess_data.py（旧 4096 版）

- `45-47` [🟠] `--raw-dir` 默认值硬编码用户绝对路径（`c:/Users/llyun/Desktop/ecg资料/...`），换环境必崩 → 改为必填参数或读环境变量。
- `53` [🟠] `--output-dir` 默认相对路径 `data/physionet2020/processed`，依赖 CWD=项目根目录 → 脚本内用 `Path(__file__).parent.parent` 锚定。
- `98-101` [🟡] tqdm `total` 用 `get_record_count()`（含隐藏 `.hea` 与加载失败记录），与实际产出数可能不符，进度条会漂移。
- `137-142` [🟡] 异常只记 `logger.warning` 且 `failed_sources` 截断 200 字符、不落盘——事后无法重试失败清单 → 把失败 `(source, record_id, err)` 写入 `failures.jsonl`。
- `144-161` [💡] 按 source 分层 + `RandomState(42)` 划分正确（防跨源泄漏、可复现），是好的实践，5k 版复用它是对的。

### 2. scripts/preprocess_data_5k.py（主用）

- `113-121` [🔴] **重采样比率被破坏**：`while up > 100 or down > 100: up=(up+1)//2; down=(down+1)//2` 在 257Hz 源（st_petersburg_incart）上把 `500/257≈1.9455` 变成 `63/33≈1.909`。结果：输出 4907 样本而非 5000，信号被时间压缩 ≈1.9%，尾部 93 个样本静默为 0；且 `official_zscore`（L148-154）的 mean/std 把这些内部零计入统计。数据被系统性、静默地写坏 → 修法：去掉该 hack，直接用 `resample_poly` 原始比率（257 已是素数分母，不会爆内存），或改用 `scipy.signal.resample` 固定输出长度；改后需重跑 st_petersburg 源并更新所有下游特征缓存。
- `186-187` [🟠] 默认 `--raw-dir` 硬编码 `C:\Users\llyun\Desktop\ECG\...`（与旧版默认根 `ecg资料` 不一致，两处互为副本/软链不明）→ 参数化。
- `269-276` [🟠] `--skip-existing` 直接 `np.load` 复用，不校验形状 `(12,5000)`、不校验文件是否由本管线产出——若输出目录混入旧 4096 管线同名 npy，会被静默当作 5k 数据 → 复用前校验 shape 并写 `provenance`（管线版本+参数哈希）。
- `290-306, 326-339` [🟠] 循环内 manifest 条目**不含 `labels`**，靠事后从旧 manifest 回填；旧条目缺 `labels` 时静默写成 `[0]*27`（repair 脚本路径就是这种来源）→ 回填时对缺失值显式告警或直接整体复制旧条目。
- `242` [🟡] pbar `total` 与 `unknown_split` 跳过计数不一致（纯显示问题）。
- `316-318` [🟡] `skipped` 混合了"加载失败"与"处理异常"两类，日志无法区分 → 分开计数。
- `348-352` [💡] 只复制 `ptbxl_reports.json`，若旧目录还有其它附属产物会丢——建议显式列出待复制清单。

### 3. scripts/repair_manifest_5k.py

- `17-20` [🟠] `OLD_DIR/NEW_DIR` 相对 CWD、`RAW_DIR` 用户绝对路径硬编码 → 参数化。
- `50-67` [🟠] 文件已存在时补录，`signal_shape` **硬编码 `[12,5000]` 且不读盘校验**——若已有 npy 实为旧 4096 产物，manifest 说谎 → 补录前 `np.load` 校验 shape。
- `104-108` [🟠] **非幂等**：重复运行会向 `new[s]` 追加重复条目 → 写前按 `(source, record_id)` 去重。
- `73-75` [🟡] 运行期再插 `sys.path` 并 import 兄弟脚本，顺序敏感——建议模块顶部统一处理。

### 4. scripts/verify_processed_5k.py

- `17, 72` [🟠] 硬编码期望 `43101` 个 npy；正确做法是从旧 manifest 统计总数比对，魔数会让任何合法变化直接 FAIL。
- `27-33` [🟡] 形状抽查每 split 只抽 60 条，抽样可接受，但 `bad_shape` 仅打印前 3 个——够了。
- `70` [🟡] 数值健康度只打印 mean/std 无断言（z-score 后应 ≈0/1），建议加阈值断言。

### 5. scripts/restore_e09680.py（一次性恢复脚本）

- `7-8` [🟡] ZIP/OUT 硬编码绝对路径（一次性脚本可接受，但无 `__main__` guard）。
- `16-18` [🟡] `parts[1:]` 依赖 zip 存在顶层目录；若 zip 根结构不同会写错位置 → 按条目名匹配 `training/` 前缀更稳。

### 6. scripts/reprocess_e09680.py

- `48-51` [🟠] `next(...)` 找不到旧记录时抛未捕获 `StopIteration` 裸崩 → 显式判断并友好报错。
- `65-70` [🟠] **非幂等**：重复运行向 `train_manifest.json` 追加重复条目 → 写前查重。
- `19-20` [🟡] `TMP/NEW_DIR` 相对 CWD、`ZIP` 硬编码。

### 7. scripts/extract_ecgfounder_features.py

- `51-94` [🟠] 全量特征驻留内存（train 30K×1024×4B≈123MB，test 另计），43K 全跑 ≈176MB——可接受，但建议按 split 流式写盘或用 memmap。
- `86` [🟠] `item.get("labels", [0]*27)`——若 manifest 缺 labels（见 5k 预处理回填问题）会静默产出全零标签特征缓存，下游全被污染 → 缺失时硬失败。
- `38-39` [🟡] `torch.load(weights_only=False)`——非可信 checkpoint 存在反序列化风险（`train_ecgfounder_ft.py:39`、`train_multimodal.py:47`、`eval_*` 同）。
- `30-42` [💡] 无 `--device` 参数，全 CPU 推理 43K 条很慢 → 加 device 参数。

### 8. scripts/train_backbone.py

- `112-116` [🔴] **泄漏红线**：`ECGContrastiveDataset(data_dir=args.data_dir)` 内部 `rglob("*.npy")` 扫描**全部** npy（train+val+test），对比预训练把测试集信号纳入训练（`dataset.py:191`）→ 修法：给 `ECGContrastiveDataset` 增加 `split`/`manifest_file` 参数，只读 `train_manifest.json`。
- `55, 70` [🟠] `--data-dir`/`--output-dir` 默认 `/cache/data`、`/cache/output`（ModelArts 风格），本地直跑必然失败或写错目录 → 默认项目内相对路径。
- 全文件 [🟠] 无 `torch.manual_seed`/`random.seed`：DataLoader shuffle、dropout、梯度噪声均不可复现（augmentor 有种子但不够）。
- `102, 114` [🟡] `--augment-prob` 只作用于微调 augmentor；预训练路径内部另建 `ECGAugmentor(segment_shuffle=True, random_seed=42)`（默认 apply_prob=1.0），参数语义不一致。
- `142` [🟡] `pretrain_only` 双重判断冗余（L110/L142），无逻辑错误。

### 9. scripts/train_simclr.py

- `83-87` [🔴] 同 `train_backbone`：对比预训练数据集含 val/test，泄漏红线（且日志明示"43K unlabeled ECGs"，即作者有意用全量——仍应在报告/文档中声明，或改为 train-only）。
- `40-42` [🟠] `/cache` 默认路径。
- 全文件 [🟠] 无 torch 种子，不可复现。
- `128-130` [🟠] `--skip-pretrain` 且 `simclr_pretrained.pt` 不存在时**静默从随机初始化继续**微调，无任何警告 → 检查存在性并 warn/exit。
- `66, 110` [💡] 预训练 checkpoint 直接存裸 `state_dict`，与 trainer 的 `{model_state_dict,...}` 格式不一致，易混用——建议统一格式。

### 10. scripts/train_ecgfounder_head.py

- `38-114` [🟡] `macro_auc_with_ci`/`youden_thresholds`/`topk_hits`/`evaluate` 与 `src/evaluation/metrics/classification.py` **逐字重复**（当前口径一致，但属漂移风险源）→ 直接 import。
- `104` [🟠] `average_precision_score(average="macro")`：若测试集某类零正样本，sklearn 行为不确定（可能 nan 传播，拖垮 mAP）——与 `classification.py` 同缺陷；建议用 `evaluate` 前校验各类正样本数。`eval_ensemble.py:98` 同。
- `147, 176` [🟡] 若验证集上所有类 AUC 始终不改善，`best_state` 保持 `None` → `model.load_state_dict(None)` 崩溃（边界：验证集类别全退化时）。
- `210` [🟡] `args.device if torch.cuda.is_available() else "cpu"`：显式传 `cuda` 但无 GPU 时静默降级，日志已打但易被忽略。
- `202-207` [🟡] 六个 npy 缺失时裸崩，无友好提示。

### 11. scripts/train_ecgfounder_ft.py

- `75, 77` [🟠] `--data-dir`/`--output-dir` 默认 `/cache/...`。
- 全文件 [🟠] 无随机种子。
- `39` [🟡] `weights_only=False`。
- `44-54, 64` [💡] 头替换（dense→Identity + 新 head）与 4096→5000 对称 padding 实现正确。

### 12. scripts/train_multimodal.py

- `317` [🔴] **"Test evaluation" 实际用 `val_loader`**，日志/结果打印 `Test macro_auc`（L331），且 best model 由同一验证集选出（L298-304）→ 测试指标实为验证指标、双重乐观；注释"test doesn't have reports"不成立（`ptbxl_reports.json` 覆盖全部 PTB-XL 记录，test split 同样有报告）→ 改用 `test_loader`。
- `192-194, 233` [🟠] **`--no-text` 基本无效**：只跳过预检（L200-210），模型构造仍总是尝试加载 TextEncoder（L233→`ECGFounderMultimodal.__init__` 内 L59-66），仅当加载失败才 fallback None。"ECG-only 消融"语义与实现不符 → 把 `no_text` 传入模型构造。
- 全文件 [🟠] 无随机种子。
- `184` [🟡] `--output-dir` 默认 `/cache/output/multimodal`。
- `260-266` [🟡] 自定义循环无标签平滑，与 head 实验（smooth=0.1）口径不一致——对比时需注明。
- `298-304` [🟡] `best_model.pt` 保存完整 `model.state_dict()`（含冻结 ECGFounder 权重），文件偏大——可只存可训练部分。

### 13. scripts/train_multimodal_fair.py

- `56-58` [🟠] `meta_mode="full"` 含数据源 one-hot：不同源诊断分布差异大，数据源是强"捷径"特征，B/C/D 组的增益可能主要来自"认出数据源"而非情境理解——脚本有 C2/D2 对照是对的，但结果文件（`metrics.json`）未标注此解读陷阱 → 在输出中附加说明字段。
- `145-148` [🟡] 三个 split 特征全量载入（≈500MB），可接受。
- `171-211` [💡] CLIP 投影加载与 `train_ecg_text_clip` 的 `ecg_proj` 结构一致，`load_state_dict` 正确。
- `28` [💡] 走 `src` 统一指标入口，与 M0.4 口径一致——所有新脚本应以此为准。

### 14. scripts/extract_text_embeddings.py

- `74` [🟠] `report_splits.json` 的 `split` 取自 `ptbxl_reports.json` 内嵌字段，而该文件**非本仓库脚本生成**（来源未审计）；若其 split 与 manifest 划分不一致，下游 CLIP 红线检查只 warning 不阻断（见 15）→ 生成时直接用 manifest 交叉校验。
- `62` [🟡] 空报告替换为 `"none"`——合理，但 `"none"` 的嵌入会进入缓存，下游应知晓。
- `24` [🟡] `import src.utils.torchvision_stub` 绕过本机损坏的 torchvision——环境 hack，换机器可能不需要且掩盖真实问题。

### 15. scripts/train_ecg_text_clip.py

- `111-112` [🟠] 红线检查（报告 split vs manifest split）**仅 warning**：系统性不一致时配对照常进行，泄漏红线形同虚设 → 不一致即 hard fail，或把不一致清单写入 `meta.json` 供审计。
- `127-130` [🟡] train/val 特征与文本嵌入全量载入（≈120MB），可接受。
- `139-145` [🟡] 若 `x_train` 为空（train 无 ptb-xl），`np.mean(losses)` 对空列表崩溃——边界。
- `9-11, 103-120` [💡] **设计正确**：训练对仅 train、val 仅检索、test 完全旁观；种子固定（L80-81）。这是泄漏红线的正面范例。

### 16. scripts/eval_model.py

- `25, 27` [🟠] checkpoint/data-dir 硬编码相对路径，CWD 依赖。
- `46` [🟠] 指标来自 `trainer.evaluate`：macro-F1 用 **PR 曲线最优阈值**、另有 challenge_score——与 M0.4 协议（Youden 阈值 macro-F1）**口径不同**，报告数字与 `train_ecgfounder_head`/`eval_per_source` 不可比 → 统一改走 `classification.evaluate`。

### 17. scripts/eval_topk.py

- `52-76` [🟠] 四个 checkpoint 路径硬编码相对 CWD（data_dir 用 `__file__` 锚定是对的，checkpoint 却没有）。
- `105` [🟡] `models["集成 Ensemble"] = None` 占位 + `all_probs` 另存，逻辑正确但易误读。
- `28-40` [💡] `topk_hit_rate` 与 `classification.topk_hits` 逻辑一致（无漂移）。
- [💡] 无结果落盘——只打印，无法审计复现。

### 18. scripts/eval_ensemble.py

- `26-50` [🟠] checkpoint 硬编码相对路径，且**无 try/except**——任一成员缺失直接崩溃（`eval_topk`/`eval_comprehensive` 都有容错，此处不一致）。
- `83-95` [🟠] macro-F1 用 PR 曲线最优阈值（与 `trainer.evaluate` 同口径），≠ M0.4 Youden——与其它评估不可比。
- `98` [🟠] mAP nan 风险（测试集某类零正样本）。
- [💡] 无输出文件，建议落盘 JSON。

### 19. scripts/eval_comprehensive.py

- `30-39` [🟠] bootstrap 用 `np.random.choice` **无种子**——CI 每次运行不同，不可复现（`train_ecgfounder_head`/`classification.py` 都用 seed=42）。
- `165-168` [🟠] **Youden 阈值在测试集上计算**再用于 macro-F1——阈值选择泄漏测试信息，F1 乐观；M0.4 协议要求验证集定阈值（`train_ecgfounder_head.py:226` 做对了，此脚本做错了）。且 `macro_auc` 报告的是 **bootstrap 分布均值**而非测试集点估计，与协议口径有偏差。
- `94-95, 101-117` [🟠] data_dir/checkpoint 模块级硬编码。
- `214-218` [🟡] JSON 序列化只处理 ndarray，`auc_ci` 元组自动变 list——可用但隐性。
- [🟡] 无 `__main__` guard（脚本级执行，可接受）。

### 20. scripts/eval_ecgfounder.py

- `74, 77` [🟠] data_dir/checkpoint 硬编码。
- `138-140` [🟠] 硬编码"我们模型"对比数字（0.791/0.808/0.822/68.7%/…）——**过时死数字**，若上游模型已更新会输出误导性对比 → 改为读取 `outputs/` 下真实评测结果。
- `100-107` [🟠] 线性探针按类训练：某类训练集零正样本时该列 `test_probs` 保持 0，若测试集该类有正样本，AUC 得 0 或 sklearn 报错（L111 只过滤全正/全负，不防此情形）→ 该类跳过并告警。
- [💡] 探针方法本身（冻结特征 + 逐类 LR）正确。

### 21. scripts/eval_per_source.py

- `23-24` [🟠] `FEAT_DIR`/`HEAD` 硬编码相对路径。
- `54-58` [🟡] 前缀匹配不到的 id 静默不进入任何源（但计入 overall）→ 统计 unmatched 数并告警。
- `66` [🟡] `n_bootstrap=500`（主协议 1000）——CI 略宽，注明即可。
- `20` [💡] 走 `src` 指标，口径一致。

### 22. scripts/eval_per_class.py

- `20-23` [🟠] 硬编码路径。
- `74-87` [🟡] 混淆对统计用"漏报去向"（真类 c 被预测为其它类），逻辑合理；但 `sub[:, c2].sum() >= 2` 阈值随意，建议可配置。
- `54` [💡] 阈值来自验证集，与主评测口径一致。

### 23. scripts/eval_mimic_domain.py

- `37` [🟠] `MIMIC_ROOT = D:\ecg\mimic_iv_ecg\extracted\data` 硬编码 Windows 盘符——换机必崩（脚本已注明 D 盘只读，属已知平台假设，但应参数化）。
- `62-66` [🟡] `predict_batch` 对空列表 `np.stack` 崩溃——若 MIMIC 样本全部读取失败（L110-111 只 warn），直接崩。
- `73-74` [🟡] 把 sigmoid 多标签输出归一化后当 softmax 算熵——统计学上可疑（多标签非互斥），仅作域对比分析可接受，但报告中应注明口径。
- `88-90, 118-119` [💡] 采样种子固定 42，可复现。
- `104` [🟡] "平导联"判据 `std<0.05` 在 z-score 后使用，阈值随意但两域一致。

### 24. scripts/analyze_ecg.py

- `156-159` [🔴] `--ecg xxx.hea` 分支用 `np.loadtxt` 读同名前缀文件——CinC2020 的信号文件是 **WFDB format-16 差分编码二进制**（`loader.py` 文档明确说明），`loadtxt` 读出的全是垃圾数值，随后静默生成错误的"诊断报告" → 改用 `wfdb.rdrecord` 或 `ECGLoader.load_record`。
- `129, 169` [🟡] `--threshold` 参数传入 `analyze()` 但**从未使用**（level 分级用硬编码 0.7/0.3）——死参数，删除或真正接入。
- `144` [🟡] `--index` 读旧 `processed/`（4096 版）目录，与 5k 管线不一致（模型端 padding 兜底了，但数据口径不同）。

### 25. scripts/train_modelarts.py（ModelArts 专用）

- `151-160` [🟠] `subprocess.run(..., check=False)`——pip 安装失败被吞，后续训练在缺依赖下"继续" → 检查 returncode 并失败退出。
- `87` [🟠] `manifests[0]` 任意取第一个匹配目录——多个数据目录存在时行为不确定 → 校验唯一性或显式选择。
- `91-110` [🟠] manifest 路径修复只处理 `"/"` 前缀，不处理 `"\\"`（Windows 上传场景）→ 统一 `Path(old).name`。
- `198-209` [🟡] `sync_output_to_obs` 异常被吞（`except Exception: log error`）——OBS 同步失败静默丢产物 → 失败应置非零退出码。
- `163-194` [💡] env 参数（BACKBONE/EPOCHS/LR 等）确实传入 `train_backbone.py` 命令行且该脚本参数真实生效——本组脚本中 CLI 参数生效性总体良好（无"参数定义了却没用"的普遍问题，仅 `--no-text`/`--threshold` 例外）。

---

## 修复优先级清单

**P0 —— 产出错误结果 / 泄漏红线（立即修，修后需重跑受影响的实验与缓存）**

1. `preprocess_data_5k.py:113-121` 重采样比率破坏 → 重跑 st_petersburg 源，重建 processed_5k 与其全部下游（特征缓存、CLIP、fair 评测）。
2. `train_multimodal.py:317` test=val → 用 test_loader 重评，更新 best_model 与结果。
3. `train_backbone.py:112-116`、`train_simclr.py:83-87` 对比预训练含 val/test → 对比数据集限定 train manifest（或至少文档声明并重训）。
4. `analyze_ecg.py:156-159` `.hea` 读取 → 换 wfdb/ECGLoader。

**P1 —— 指标口径与可复现性（影响结论可信度）**

5. 统一指标入口：`trainer.evaluate`、`eval_ensemble`、`eval_comprehensive` 的 macro-F1 统一为 M0.4（验证集 Youden）；`eval_comprehensive` 阈值改在验证集上定、bootstrap 加种子。
6. 所有训练脚本补 `torch.manual_seed`/`random.seed`（含 DataLoader worker 种子）。
7. `train_ecgfounder_head.py` 删除自带指标副本，改 import `classification.py`，消除漂移源。

**P2 —— 工程健壮性**

8. 参数化所有硬编码路径（`/cache/*`、`C:\Users\...`、`D:\ecg\...`、相对 CWD 的 checkpoint），默认值锚定项目根。
9. `--skip-existing`/repair/reprocess 增加 shape+provenance 校验与幂等去重。
10. 清理或接入 `configs/*.yaml` 死文件，明确参数唯一真源。
11. 评估脚本统一落盘 JSON（`eval_topk`/`eval_ensemble` 现仅打印）。
12. `train_multimodal --no-text` 语义修正；`analyze_ecg --threshold` 删除或实现。

---

## 附：泄漏红线与口径核对结论

- **对比预训练**：`train_backbone`/`train_simclr` 含 val/test 🔴；CLIP（`train_ecg_text_clip`）train-only ✅ 但红线检查仅 warning 🟠。
- **多模态配对**：`train_multimodal` 按 manifest 配对本身无跨 split 泄漏，但"测试评估=验证集"🔴；fair 版 signal-only 推理 ✅。
- **划分复用**：5k 版复用旧 manifest 划分 ✅（防泄漏的正确做法）。
- **指标口径**：`train_ecgfounder_head`/`eval_per_source`/`eval_per_class`/`train_multimodal_fair` ✅ 与 `classification.py` 一致（其中 head 是副本、其余是 import）；`trainer.evaluate`（PR 阈值）与 `eval_comprehensive`（test 定阈值+无种子）❌ 漂移。
