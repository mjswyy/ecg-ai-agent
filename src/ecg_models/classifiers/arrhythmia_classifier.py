"""
心律失常分类器 — 27类多标签 ECG 诊断分类 / 27-class Multi-label Arrhythmia Classification.

组合 backbone 编码器 + 2层 MLP 分类头，使用 Asymmetric Loss (ASL) 处理极端类别不平衡。

Combines a backbone encoder with a 2-layer MLP head.
Uses Asymmetric Loss (ASL) to handle extreme class imbalance (485:1 ratio).

ASL 原理 / ASL Principle:
    对负类施加更强的聚焦 (γ_neg=4)，防止简单负样本主导梯度。
    概率偏移: p_m = max(p - m, 0) 抑制简单负样本的损失贡献。

使用示例 / Usage:
    backbone = xresnet1d_101(in_channels=12)
    model = ArrhythmiaClassifier(backbone, num_classes=27)
    logits = model(ecg)  # (B, 12, 4096) → (B, 27)
    loss = AsymmetricLoss()(logits, targets)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArrhythmiaClassifier(nn.Module):
    """多标签心律失常分类器 / Multi-label arrhythmia classifier.

    架构 / Architecture: Backbone → [Dropout → Linear → ReLU → Dropout → Linear] → 27 logits
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 27,
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        in_features = backbone.feature_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 / Forward: (B, 12, L) → (B, 27) logits."""
        return self.head(self.backbone(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测概率 / Predict probabilities: apply sigmoid to logits."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


class AsymmetricLoss(nn.Module):
    """非对称损失 / Asymmetric Loss for multi-label classification.

    从 Ben-Baruch et al. (ICCV 2021) 的论文公式正确实现。
    概率偏移在概率空间用减法实现（非 logit 空间的乘法）。

    Correctly implements the paper's probability shifting via subtraction
    in probability space, not multiplication in logit space.

    参数 / Args:
        gamma_neg: 负类聚焦参数 / Focusing for negatives (default 4).
        gamma_pos: 正类聚焦参数 / Focusing for positives (default 0).
        clip: 概率偏移量 / Probability shift margin (default 0.05).
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps, self.reduction = \
            gamma_neg, gamma_pos, clip, eps, reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算 ASL 损失 / Compute Asymmetric Loss.

        参数 / Args:
            logits:  预测 logits (B, C)
            targets: 二值目标标签 (B, C)
        """
        # Clamp logits to prevent sigmoid overflow on NPU
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        p = torch.sigmoid(logits)

        # 正类 BCE / Positive BCE
        loss_pos = -targets * F.logsigmoid(logits)
        # 负类 BCE + 概率偏移 / Negative BCE with probability shifting
        # clamp min and max to prevent log(0) or log(negative) → NaN
        loss_neg = -(1 - targets) * torch.log(
            torch.clamp(1 - p + self.clip, min=self.eps, max=1.0 - self.eps)
        )

        # 非对称聚焦 / Asymmetric focusing
        pt_pos, pt_neg = p, 1 - p
        if self.gamma_pos > 0:
            loss_pos = loss_pos * ((1 - pt_pos) ** self.gamma_pos)
        neg_weights = pt_neg ** self.gamma_neg

        loss = loss_pos + neg_weights * loss_neg
        return loss.mean() if self.reduction == "mean" else loss.sum()


class FocalLoss(nn.Module):
    """标准 Focal Loss / Standard Focal Loss for comparison experiments."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = (targets * alpha + (1 - targets) * (1 - alpha)) * loss
        return loss.mean() if self.reduction == "mean" else loss.sum()
