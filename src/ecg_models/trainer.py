"""
ECG 模型训练器 — SimCLR 对比预训练 + 多标签微调。

支持:
    - SimCLR 风格对比预训练（InfoNCE 损失）
    - 多标签微调（Asymmetric Loss）
    - 混合精度训练 (AMP)
    - 梯度裁剪
    - Cosine 学习率衰减 + Linear 预热
    - 早停 + 模型检查点
    - PhysioNet Challenge Score 评估

使用示例:
    trainer = ECGTrainer(model, device="auto")  # 自动检测 NPU/CUDA/CPU
    # 步骤1: 对比预训练
    trainer.train_contrastive(contrastive_loader, epochs=100)
    # 步骤2: 多标签微调
    trainer.train_multilabel(train_loader, val_loader, epochs=50)
    # 步骤3: 测试评估
    metrics = trainer.evaluate(test_loader)
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from src.utils.device_utils import GradScaler, autocast, detect_device, is_accelerator
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# sklearn 是可选的（仅评估时需要）
try:
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn 未安装；评估指标将返回 0")


class ECGTrainer:
    """ECG 模型训练管理器。

    参数:
        model:      PyTorch 模型（ArrhythmiaClassifier 或 backbone 单独）
        device:     计算设备 ("auto" / "npu" / "cuda" / "cpu")
        output_dir: 检查点和日志输出目录
        use_amp:    是否启用自动混合精度（支持 CUDA / Ascend NPU）
        grad_clip:  梯度裁剪最大范数（0 表示禁用）
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        output_dir: str = "outputs",
        use_amp: bool = True,
        grad_clip: float = 1.0,
    ):
        # 自动检测设备: NPU > CUDA > CPU
        self._device_str = detect_device(device)
        self.device = torch.device(self._device_str)
        self.model = model.to(self.device)

        # 加速器上启用 AMP（NPU 或 CUDA）
        self.use_amp = use_amp and is_accelerator(self.device)
        # NPU/CUDA 需要 non_blocking DMA 传输；CPU 上无意义且会警告
        self._use_non_blocking = is_accelerator(self.device)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = grad_clip
        self.scaler = GradScaler(enabled=self.use_amp)

        # 训练状态追踪
        self.current_epoch = 0
        self.best_metric = 0.0
        # 2E-O7 修复：fit 过程中在验证集上确定的最优阈值（测试集评估必须用它，
        # 禁止在测试集上现算阈值——样本内拟合乐观）
        self.val_thresholds: Optional[np.ndarray] = None

    # ================================================================
    # SimCLR 对比预训练
    # ================================================================

    def train_contrastive(
        self,
        train_loader: DataLoader,
        epochs: int = 100,
        temperature: float = 0.07,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        log_interval: int = 50,
    ) -> Dict:
        """SimCLR 风格对比预训练。

        ECGContrastiveDataset 提供两个增强视图作为正样本对，
        batch 内其他样本作为负样本，使用 InfoNCE 损失。

        参数:
            train_loader:  DataLoader，产出 (view1, view2) 对
            epochs:        训练轮数
            temperature:   InfoNCE 温度参数（越小 softmax 越尖锐）
            lr:            学习率
            weight_decay:  AdamW 权重衰减
            warmup_epochs: 学习率线性预热轮数
            log_interval:  多少步打印一次日志

        返回:
            训练历史字典 {"loss": [...]}
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = self._cosine_schedule(optimizer, epochs, warmup_epochs)
        self.model.train()

        history = {"loss": []}
        t0 = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (view1, view2) in enumerate(train_loader):
                view1 = view1.to(self.device, non_blocking=self._use_non_blocking)
                view2 = view2.to(self.device, non_blocking=self._use_non_blocking)

                with autocast(enabled=self.use_amp):
                    # 提取两个视图的特征
                    z1 = self.model(view1)  # (B, feature_dim)
                    z2 = self.model(view2)

                    # InfoNCE 对比损失
                    loss = self._info_nce_loss(z1, z2, temperature)

                # NaN保护: 跳过包含NaN的batch（AMP可能导致梯度下溢）
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"对比训练 Epoch {epoch+1}/{epochs} "
                        f"[{batch_idx}/{len(train_loader)}] loss为NaN/Inf，跳过"
                    )
                    continue

                # 反向传播
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # 检查梯度是否含 NaN/Inf（NPU 上 backward 可能产生 NaN）
                grad_nan = any(
                    torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    for p in self.model.parameters() if p.grad is not None
                )
                if grad_nan:
                    logger.warning(
                        f"对比训练 [{batch_idx}/{len(train_loader)}] "
                        f"NaN/Inf梯度，跳过optimizer step"
                    )
                    optimizer.zero_grad()
                    continue

                if self.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % log_interval == 0:
                    logger.info(
                        f"对比训练 Epoch {epoch+1}/{epochs} "
                        f"[{batch_idx}/{len(train_loader)}] loss={loss.item():.4f}"
                    )

            scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            history["loss"].append(avg_loss)

            logger.info(
                f"Epoch {epoch+1}/{epochs} 完成 | "
                f"avg_loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
            )

        logger.info(f"对比训练完成，耗时 {time.time()-t0:.0f}s")
        return history

    @staticmethod
    def _info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        """SimCLR 的 InfoNCE 对比损失。

        原理:
            1. L2 归一化特征向量
            2. 拼接两个视图的特征 → (2B, D)
            3. 计算相似度矩阵，除以温度
            4. 正样本对: z1[i] 和 z2[i]（对角线下半部分）
            5. 负样本: batch 内所有其他样本
            6. 交叉熵损失: 分类"哪个是正样本"

        注意: 温度只在相似度矩阵计算时除一次（不在 final logits 上再次除）。
        """
        # L2 归一化
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.t()) / temperature  # (2B, 2B) 相似度矩阵

        # 提取正样本对的相似度值
        sim_i_j = torch.diag(sim, z1.size(0))   # z1[i] 和 z2[i]
        sim_j_i = torch.diag(sim, -z1.size(0))  # z2[i] 和 z1[i]
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)  # (2B,)

        # 检查报告 1.4 修复：负样本掩码必须同时剔除自身与正样本对。
        # 旧版只剔主对角线 → 每个锚点的正样本对同时出现在负样本集里，
        # 训练目标自相矛盾（同一对被同时奖励和惩罚）。
        mask = torch.zeros((z.size(0), z.size(0)), dtype=torch.bool,
                           device=z.device)
        mask.fill_diagonal_(True)                       # (i,i) 与 (i+B,i+B)
        b = z1.size(0)
        mask[torch.arange(b), torch.arange(b, 2 * b)] = True   # (i, i+B)
        mask[torch.arange(b, 2 * b), torch.arange(b)] = True   # (i+B, i)
        negatives = sim[~mask].view(z.size(0), z.size(0) - 2)  # (2B, 2B-2)
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)

        labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        # 注意: 不再对 logits 除以 temperature（sim 已除过）
        return nn.functional.cross_entropy(logits, labels)

    # ================================================================
    # 多标签微调
    # ================================================================

    def train_multilabel(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        loss_fn: Optional[nn.Module] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        early_stopping_patience: int = 10,
        label_smoothing: float = 0.0,
        grad_noise: float = 0.0,
        log_interval: int = 50,
        save_best: bool = True,
    ) -> Dict:
        """多标签分类微调。

        参数:
            train_loader: 训练 DataLoader (signal, labels)
            val_loader:   验证 DataLoader（可选）
            epochs:       训练轮数
            loss_fn:      损失函数（默认 AsymmetricLoss）
            warmup_epochs: LR 预热轮数
            early_stopping_patience: 早停耐心
            label_smoothing: 标签平滑系数（0=关闭, 推荐0.05-0.1）
            grad_noise:    梯度噪声标准差（0=关闭, 推荐0.001-0.01, Neelakantan et al. 2015）
            save_best:    是否保存最佳模型

        返回:
            训练历史 {"train_loss": [...], "val_f1": [...], "val_auc": [...]}
        """
        if loss_fn is None:
            # 默认使用 BCE（数值更稳定），ASL 可在确认收敛后切换
            loss_fn = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = self._cosine_schedule(optimizer, epochs, warmup_epochs)

        history = {"train_loss": [], "val_f1": [], "val_auc": []}
        best_val_auc = 0.0
        patience_counter = 0
        best_state = None  # R6（2B 🟠-2）：best-val 权重快照，早停后回填
        best_val_thresholds = None  # 第三轮审查 3C-TRAIN-1：best 时刻的 val 阈值快照
        t0 = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (signals, labels) in enumerate(train_loader):
                signals = signals.to(self.device, non_blocking=self._use_non_blocking)
                labels = labels.to(self.device, non_blocking=self._use_non_blocking)

                # 标签平滑（均匀平滑 ε）：正样本 1→1-ε/2，负样本 0→ε/2
                # 第三轮审查 3C-TRAIN-4：注释与公式一致（旧版注释写 1→(1-ε) 与实现不符）
                if label_smoothing > 0:
                    labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

                with autocast(enabled=self.use_amp):
                    logits = self.model(signals)
                    loss = loss_fn(logits, labels)

                # NaN保护: 跳过包含NaN的batch
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Batch {batch_idx}: loss为NaN/Inf，跳过")
                    continue

                optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # 检查梯度是否含 NaN/Inf（NPU 上 backward 可能产生 NaN）
                grad_nan = any(
                    torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    for p in self.model.parameters() if p.grad is not None
                )
                if grad_nan:
                    logger.warning(
                        f"Batch {batch_idx}: NaN/Inf梯度，跳过optimizer step。"
                        f"模型权重未更新。"
                    )
                    optimizer.zero_grad()
                    continue

                if self.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                # 梯度噪声: 在 optimizer step 前添加高斯噪声
                # 隐式正则化，鼓励收敛到平坦极小值（Neelakantan et al. 2015）
                # 第三轮审查 3C-TRAIN-5：注释与实现统一——噪声标准差 = grad_noise * eta_t
                # （线性衰减，非 sqrt；与原文近似）；注意 AMP 下噪声加在缩放梯度上，
                # 等效噪声经 scaler 反缩放后量级略有偏差（grad_clip=0 时无 unscale_）
                if grad_noise > 0:
                    eta_t = 1.0 / (1 + self.current_epoch) ** 0.55
                    noise_std = grad_noise * eta_t
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad) * noise_std)

                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] "
                        f"loss={loss.item():.4f}"
                    )

            scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_loss)

            # ---- 验证 ----
            val_msg = ""
            if val_loader is not None:
                # 2E-O7 修复：fit 内部对验证集现算阈值是合法用途（验证集定阈值），
                # 并保存供后续测试集评估使用
                val_metrics = self.evaluate(val_loader, fit_thresholds=True)
                history["val_f1"].append(val_metrics["macro_f1"])
                history["val_auc"].append(val_metrics["macro_auc"])
                val_msg = (
                    f"val_f1={val_metrics['macro_f1']:.4f} "
                    f"val_auc={val_metrics['macro_auc']:.4f}"
                )

                # 早停（用 val_auc 判断，比 val_f1 更适合极端不平衡数据）
                # min_delta: 改善小于此值不重置计数器，避免微小波动拖长训练
                if val_metrics["macro_auc"] > best_val_auc + 1e-4:
                    best_val_auc = val_metrics["macro_auc"]
                    patience_counter = 0
                    # R6（2B 🟠-2）：保存 best-val 权重快照，训练结束后回填模型，
                    # 保证"报告指标用的权重" == "best_model.pt 权重"
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    # 第三轮审查 3C-TRAIN-1：同步快照 best 时刻的 val 阈值
                    # （旧版回填权重后 val_thresholds 仍是最后一轮阈值，口径失配）
                    best_val_thresholds = (
                        self.val_thresholds.copy() if self.val_thresholds is not None
                        else None)
                    if save_best:
                        self._save_checkpoint("best_model.pt", best_val_auc)
                else:
                    patience_counter += 1

            logger.info(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | {val_msg}")

            # R6（2B 🟠-1）：patience=0 语义修复——文档称"0=禁用早停"，
            # 旧版 patience=0 时第 1 个 epoch 后恒触发 break（只训 1 轮）
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                logger.info(f"早停触发于 Epoch {epoch+1}")
                break

        # R6（2B 🟠-2）：早停/正常结束后回填 best-val 权重，与磁盘检查点一致
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if best_val_thresholds is not None:
                self.val_thresholds = best_val_thresholds
            logger.info("已回填验证集最优权重与对应阈值（与 best_model.pt 一致）")

        logger.info(f"多标签训练完成，耗时 {time.time()-t0:.0f}s")
        self.best_metric = best_val_auc
        return history

    # ================================================================
    # 评估
    # ================================================================

    @torch.no_grad()
    def evaluate(self, loader: DataLoader,
                 thresholds: Optional[np.ndarray] = None,
                 fit_thresholds: bool = False) -> Dict:
        """评估多标签分类指标。

        2E-O7 修复：阈值确定口径——fit_thresholds=True 仅允许 fit 内部
        对验证集现算（验证集定阈值）；外部评估必须传验证集阈值
        （thresholds），否则用 0.5 并告警，禁止在测试集上现算。

        计算: macro AUC, macro F1, Challenge Score, mAP。

        返回:
            {"macro_auc": ..., "macro_f1": ..., "challenge_score": ..., "mAP": ...}
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn 未安装；返回空评估指标")
            return {
                "macro_auc": 0.0, "macro_f1": 0.0,
                "challenge_score": 0.0, "mAP": 0.0,
            }

        training = self.model.training
        self.model.eval()
        all_logits, all_labels = [], []

        for signals, labels in loader:
            signals = signals.to(self.device, non_blocking=self._use_non_blocking)
            logits = self.model(signals)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # NaN保护: 如果模型输出含NaN，替换为0
        nan_mask = np.isnan(logits) | np.isinf(logits)
        if nan_mask.any():
            logger.warning(f"logits包含 {nan_mask.sum()} NaN/Inf值，已替换为0")
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        num_classes = labels.shape[1]

        # Macro AUC (只对有正负样本的类别计算)
        aucs = []
        for c in range(num_classes):
            if 0 < labels[:, c].sum() < len(labels):
                aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
        macro_auc = float(np.mean(aucs)) if aucs else 0.0

        # Macro F1 + Challenge Score（逐类阈值：验证集确定或显式传入）
        from sklearn.metrics import roc_curve
        if fit_thresholds:
            # 仅 fit 内部对验证集现算（验证集定阈值）并保存
            # 4D-ORANGE-6 修复：判据改为 Youden J = TPR - FPR（与协议宣称一致；
            # 旧版 F1-max 口径与 train_ecgfounder_head/classification.youden_thresholds 脱节）
            best_thresholds = []
            preds_optimal = np.zeros_like(probs, dtype=np.float32)
            for c in range(num_classes):
                if labels[:, c].sum() == 0:
                    best_thresholds.append(0.5)
                    continue
                if labels[:, c].sum() == len(labels):
                    # 4D 定向复查（δ）：全正类 roc_curve 的 fpr=NaN → argmax
                    # 退化取 thresh[0]=inf，inf 写入检查点后在测试集恒判负
                    # （该类 F1=0）——全正类无区分性，阈值取 0.5 阻断 inf
                    # 5D 审查（🟡-3）：全正类的最优预测是全 1（precision=recall
                    # =F1=1），旧版 preds_optimal[:,c] 从不赋值（保持全 0）
                    # 把该类 F1 从 1.0 低估为 0.0，拉低 macro_f1/Challenge Score
                    best_thresholds.append(0.5)
                    preds_optimal[:, c] = 1.0
                    continue
                fpr, tpr, thresh = roc_curve(labels[:, c], probs[:, c])
                j = tpr - fpr
                best_thresh = float(thresh[int(np.argmax(j))]) if len(thresh) else 0.5
                # 复检2 修复（🟡-1）：sklearn 1.9 的 roc_curve 对任意类别返回
                # thresh[0]=inf——混合类 argmax==0（无区分度/AUC<0.5）时同样
                # 取到 inf（全正类守卫之外的第二缺口）；inf 会写进检查点并在
                # 测试集恒判负。非有限阈值一律回退 0.5
                if not np.isfinite(best_thresh):
                    best_thresh = 0.5
                best_thresholds.append(best_thresh)
                preds_optimal[:, c] = (probs[:, c] >= best_thresh).astype(np.float32)
            self.val_thresholds = np.array(best_thresholds, dtype=np.float32)
        elif thresholds is not None:
            best_thresholds = list(np.asarray(thresholds, dtype=np.float32))
            preds_optimal = (probs >= np.asarray(thresholds, dtype=np.float32)).astype(np.float32)
        else:
            logger.warning(
                "evaluate() 未传 thresholds 且非 fit_thresholds 模式：用 0.5 默认值"
                "（2E-O7：禁止在测试集上现算阈值，请传验证集阈值 trainer.val_thresholds）")
            best_thresholds = [0.5] * num_classes
            preds_optimal = (probs >= 0.5).astype(np.float32)

        f1s = []
        for c in range(num_classes):
            if labels[:, c].sum() > 0:
                f1s.append(f1_score(labels[:, c], preds_optimal[:, c], zero_division=0))
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        # Challenge Score: 用最优阈值重新计算
        # 检查报告 1.9 修复：实参错位——签名是 (labels, probs, preds_binary, thresholds)，
        # 旧版把二值预测传给了 probs、把概率传给了 preds_binary → Challenge Score 失真
        challenge = self._challenge_score(labels, probs, preds_optimal, best_thresholds)

        # mAP
        # 第三轮审查 3C-TRAIN-3：与 macro_auc 口径一致——仅对 0<sum<len 的类别
        # 计算 AP 后取平均（旧版 average='macro' 把无正样本类按 0 计入，系统性拉低）
        try:
            ap_per_class = [
                average_precision_score(labels[:, c], probs[:, c])
                for c in range(num_classes)
                if 0 < labels[:, c].sum() < len(labels)
            ]
            mAP = float(np.mean(ap_per_class)) if ap_per_class else 0.0
        except Exception:
            mAP = 0.0

        # 恢复训练状态
        if training:
            self.model.train()

        return {
            "macro_auc": macro_auc,
            "macro_f1": macro_f1,
            "challenge_score": float(challenge),
            "mAP": mAP,
        }

    @staticmethod
    def _challenge_score(labels: np.ndarray, probs: np.ndarray,
                         preds_binary: np.ndarray = None,
                         thresholds: list = None,
                         beta: float = 2.0) -> float:
        """PhysioNet 2020 Challenge 官方评分指标。

        Challenge Score = (F_beta + G_beta) / 2
        - F_beta: 多标签 F-beta (beta=2 偏向召回率)
        - G_beta: 基于排序的 NDCG 风格指标

        IDCG 只累加前 n_pos 个位置（真实正样本数），而非全部位置。
        零正样本的类别自动跳过。

        参数:
            preds_binary: 预计算的最优阈值二值预测（可选，None=用0.5）
            thresholds:   逐类最优阈值列表（保留，供外部使用）
        """
        beta2 = beta ** 2
        if preds_binary is None:
            preds_binary = (probs >= 0.5).astype(np.float32)

        tp = (preds_binary * labels).sum(axis=0)
        fp = ((1 - labels) * preds_binary).sum(axis=0)
        fn = (labels * (1 - preds_binary)).sum(axis=0)
        n_pos = labels.sum(axis=0)  # 每类正样本数

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

        # 只对有正样本的类别取平均（与 G_beta 行为一致）
        f_beta_valid = f_beta[n_pos > 0] if (n_pos > 0).any() else np.array([0.0])

        # G-beta: 基于排序的加权指标
        g_beta = 0.0
        n_classes_with_pos = 0
        for c in range(labels.shape[1]):
            n_pos = int(labels[:, c].sum())
            if n_pos == 0:
                continue  # 跳过没有正样本的类别
            n_classes_with_pos += 1

            sorted_idx = np.argsort(-probs[:, c])
            dcg = 0.0
            idcg = 0.0
            for i, idx in enumerate(sorted_idx):
                rel = labels[idx, c]
                dcg += rel / np.log2(i + 2)
            # IDCG: 前 n_pos 个理想排序位置
            for i in range(n_pos):
                idcg += 1.0 / np.log2(i + 2)
            g_beta += dcg / (idcg + 1e-8)

        if n_classes_with_pos > 0:
            g_beta /= n_classes_with_pos

        return float((np.mean(f_beta_valid) + g_beta) / 2.0)

    # ================================================================
    # 工具方法
    # ================================================================

    @staticmethod
    def _cosine_schedule(optimizer, epochs, warmup):
        """Cosine 衰减 + Linear 预热学习率调度器。

        前 warmup 个 epoch: LR 从 ~0 线性增长到初始 LR
        剩余 epoch: Cosine 衰减
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup, 1)
        )
        if warmup > 0:
            def warmup_fn(epoch):
                if epoch < warmup:
                    return float(epoch + 1) / float(max(warmup, 1))
                return 1.0

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_fn)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup],
            )
        return scheduler

    def _save_checkpoint(self, filename: str, best_metric: Optional[float] = None):
        """保存模型检查点。

        R6（2B 🟡-1）：best_metric 由调用方传入（训练中保存的检查点
        self.best_metric 尚为初值 0.0，旧版元数据恒为 0）。
        """
        path = self.output_dir / filename
        ckpt = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_metric": (best_metric if best_metric is not None
                            else self.best_metric),
        }
        # 4D-ORANGE-7 修复：val_thresholds 随检查点保存——旧版只存权重，
        # 加载后 val_thresholds=None、测试指标无法按验证集阈值复现
        if self.val_thresholds is not None:
            ckpt["val_thresholds"] = np.asarray(self.val_thresholds, dtype=np.float32)
        torch.save(ckpt, path)
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str, strict: bool = True, key_prefix: str = ""):
        """加载模型检查点。

        第三轮审查 3C-TRAIN-2：改用 safe_torch_load（weights_only 白名单口径，
        旧版裸 torch.load 绕过仓库安全加载）；支持可选 key 前缀剥离与 strict。
        第三轮审查 3C-CKPT-1：磁盘上的历史检查点（2026-08-29 前生成）的
        best_metric 元数据恒为 0.0（修复前产物）——消费方不应依赖该字段，
        如需真实值请重新训练生成。
        """
        from src.utils.safe_load import safe_torch_load
        ckpt = safe_torch_load(path, map_location=self.device)
        sd = ckpt["model_state_dict"]
        if key_prefix and any(k.startswith(key_prefix) for k in sd):
            sd = {k[len(key_prefix):]: v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=strict)
        self.current_epoch = ckpt["epoch"]
        self.best_metric = ckpt.get("best_metric", 0.0)
        # 4D-ORANGE-7：加载检查点中的验证集阈值（旧检查点无该字段则保持 None，
        # 消费方需自行从验证集重算或传入 thresholds）
        if ckpt.get("val_thresholds") is not None:
            self.val_thresholds = np.asarray(ckpt["val_thresholds"], dtype=np.float32)
        logger.info(f"检查点已加载 (epoch={self.current_epoch}, "
                    f"val_thresholds={'有' if self.val_thresholds is not None else '无'})")
