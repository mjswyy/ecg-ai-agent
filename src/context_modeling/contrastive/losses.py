"""Contrastive Losses — InfoNCE and related contrastive learning objectives."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive pretraining (CLIP-style).

    检查报告 1.4 修复：旧版把温度直接作为无约束可学习参数（nn.Parameter），
    可能漂移到 0（logits 爆炸/NaN）或负值。新版采用 CLIP 式 logit_scale =
    log(1/temperature)，前向先 clamp 到 [ln(1/100), ln(100)] 再 exp——
    温度恒为正且不漂移（等价于温度约束在 [0.01, 100]）。

    Usage:
        loss_fn = InfoNCELoss(temperature=0.07)
        loss = loss_fn(ecg_embeddings, text_embeddings)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # 5E 审查（🟡-2）：temperature 必须 >0——temp<0 时 log(1/temp) 为
        # NaN 静默传播、temp=0 时 ZeroDivisionError；SupConLoss 已有守卫未同步
        if temperature <= 0:
            raise ValueError(f"InfoNCELoss temperature 必须 > 0，收到 {temperature}")
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / temperature)))

    @property
    def temperature(self) -> float:
        """当前等效温度（只读，用于日志/复现记录）。"""
        with torch.no_grad():
            scale = torch.clamp(self.logit_scale,
                                min=math.log(1 / 100), max=math.log(100)).exp()
            return float((1.0 / scale).item())

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            z1, z2: L2-normalized embeddings (B, D).

        Returns:
            Scalar loss.
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # 先 clamp logit_scale 再 exp：防止 exp 溢出产生 inf/NaN
        scale = torch.clamp(self.logit_scale,
                            min=math.log(1 / 100), max=math.log(100)).exp()

        # Similarity matrix
        logits = (z1 @ z2.T) * scale  # (B, B)

        # Labels: diagonal is positive
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits.T, labels)

        return (loss1 + loss2) / 2.0


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss — leverages label information.

    Positive pairs are samples with the same label.

    Args:
        temperature: Temperature parameter.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # 4E/4F 审查修复：💡-13 —— temperature 必须 >0，否则 sim = .../temperature
        # 产生 inf/NaN（默认 0.07 下数值稳定，此处仅做下界保护）。
        if temperature <= 0:
            raise ValueError(f"SupConLoss temperature 必须 > 0，收到 {temperature}")
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised contrastive loss.

        Args:
            features: (B, D) L2-normalized features.
            labels: (B,) class indices or (B, C) multi-hot.

        Returns:
            Scalar loss.
        """
        features = F.normalize(features, dim=-1)
        sim = features @ features.T / self.temperature  # (B, B)

        # Positive mask: same label
        if labels.dim() == 1:
            pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        else:
            # Multi-label: share at least one label
            pos_mask = (labels @ labels.T) > 0

        # Remove self
        pos_mask = pos_mask.fill_diagonal_(False)

        # Compute loss
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
        # 检查报告 1.4 修复：负样本掩码必须同时剔除自身——
        # 旧版 (~pos_mask) 含对角线（exp(sim[i,i])=exp(1/tau)≈1.6e6 主导分母 → 梯度稀释）
        neg_mask = (~pos_mask).fill_diagonal_(False)
        neg_sum = (exp_sim * neg_mask.float()).sum(dim=1)

        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        # 4E/4F 审查修复：🟠-3 —— 正样本掩码为空（batch=1 或全批标签互异）时，
        # loss[pos_sum>0] 为空张量，.mean() 返回 NaN。改为返回 0.0 标量
        # （features.sum()*0.0 保持计算图与 requires_grad，0 梯度不产生学习信号；
        #  注意不能用 loss.sum()*0.0：无正样本时 loss 为 inf，inf*0=NaN）。
        pos = pos_sum > 0
        if pos.any():
            return loss[pos].mean()
        return features.sum() * 0.0
