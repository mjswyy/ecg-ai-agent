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
    trainer = ECGTrainer(model, device="cuda")
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
from torch.cuda.amp import GradScaler, autocast
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
        device:     计算设备 ("cuda" / "cpu")
        output_dir: 检查点和日志输出目录
        use_amp:    是否启用自动混合精度（仅 CUDA）
        grad_clip:  梯度裁剪最大范数（0 表示禁用）
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        output_dir: str = "outputs",
        use_amp: bool = True,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # CPU 上无法使用 AMP
        self.use_amp = use_amp and self.device.type == "cuda"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = grad_clip
        self.scaler = GradScaler(enabled=self.use_amp)

        # 训练状态追踪
        self.current_epoch = 0
        self.best_metric = 0.0

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
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)

                with autocast(enabled=self.use_amp):
                    # 提取两个视图的特征
                    z1 = self.model(view1)  # (B, feature_dim)
                    z2 = self.model(view2)

                    # InfoNCE 对比损失
                    loss = self._info_nce_loss(z1, z2, temperature)

                # NaN保护: 跳过包含NaN的batch（AMP可能导致梯度下溢）
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                # 反向传播
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()

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

        # 构建 logits: 第0列是正样本，其余是负样本
        mask = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
        negatives = sim[~mask].view(z.size(0), -1)  # (2B, 2B-1)
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
            save_best:    是否保存最佳模型

        返回:
            训练历史 {"train_loss": [...], "val_f1": [...], "val_auc": [...]}
        """
        if loss_fn is None:
            from .classifiers.arrhythmia_classifier import AsymmetricLoss
            loss_fn = AsymmetricLoss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = self._cosine_schedule(optimizer, epochs, warmup_epochs)

        history = {"train_loss": [], "val_f1": [], "val_auc": []}
        best_val_f1 = 0.0
        patience_counter = 0
        t0 = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (signals, labels) in enumerate(train_loader):
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.use_amp):
                    logits = self.model(signals)
                    loss = loss_fn(logits, labels)

                # NaN保护: 跳过包含NaN的batch
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Batch {batch_idx}: loss为NaN/Inf，跳过")
                    continue

                optimizer.zero_grad()
                self.scaler.scale(loss).backward()

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
                        f"Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] "
                        f"loss={loss.item():.4f}"
                    )

            scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_loss)

            # ---- 验证 ----
            val_msg = ""
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_f1"].append(val_metrics["macro_f1"])
                history["val_auc"].append(val_metrics["macro_auc"])
                val_msg = (
                    f"val_f1={val_metrics['macro_f1']:.4f} "
                    f"val_auc={val_metrics['macro_auc']:.4f}"
                )

                # 早停
                if val_metrics["macro_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["macro_f1"]
                    patience_counter = 0
                    if save_best:
                        self._save_checkpoint("best_model.pt")
                else:
                    patience_counter += 1

            logger.info(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | {val_msg}")

            if patience_counter >= early_stopping_patience:
                logger.info(f"早停触发于 Epoch {epoch+1}")
                break

        logger.info(f"多标签训练完成，耗时 {time.time()-t0:.0f}s")
        self.best_metric = best_val_f1
        return history

    # ================================================================
    # 评估
    # ================================================================

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """评估多标签分类指标。

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
            signals = signals.to(self.device)
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

        # Macro F1 (阈值=0.5)
        preds = (probs >= 0.5).astype(np.float32)
        f1s = []
        for c in range(num_classes):
            if labels[:, c].sum() > 0:
                f1s.append(f1_score(labels[:, c], preds[:, c], zero_division=0))
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        # Challenge Score: (F_beta + G_beta) / 2
        challenge = self._challenge_score(labels, probs)

        # mAP
        try:
            mAP = float(average_precision_score(labels, probs, average="macro"))
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
    def _challenge_score(labels: np.ndarray, probs: np.ndarray, beta: float = 2.0) -> float:
        """PhysioNet 2020 Challenge 官方评分指标。

        Challenge Score = (F_beta + G_beta) / 2
        - F_beta: 多标签 F-beta (beta=2 偏向召回率)
        - G_beta: 基于排序的 NDCG 风格指标

        IDCG 只累加前 n_pos 个位置（真实正样本数），而非全部位置。
        零正样本的类别自动跳过。
        """
        beta2 = beta ** 2
        preds_binary = (probs >= 0.5).astype(np.float32)

        tp = (preds_binary * labels).sum(axis=0)
        fp = ((1 - labels) * preds_binary).sum(axis=0)
        fn = (labels * (1 - preds_binary)).sum(axis=0)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

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

        return float((np.mean(f_beta) + g_beta) / 2.0)

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
            optimizer, T_max=epochs - warmup
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

    def _save_checkpoint(self, filename: str):
        """保存模型检查点。"""
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "best_metric": self.best_metric,
            },
            path,
        )
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载模型检查点。"""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.best_metric = ckpt.get("best_metric", 0.0)
        logger.info(f"检查点已加载 (epoch={self.current_epoch})")
