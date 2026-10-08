# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: ECGTextCLIP 类未被训练脚本使用；现役 CLIP 投影 = train_ecg_text_clip.py 的 CLIPPairProjectors
"""ECG-Text CLIP — Contrastive pretraining for ECG and clinical text.

Aligns ECG and text representations in a shared embedding space using
contrastive learning (CLIP-style).

Usage:
    model = ECGTextCLIP(ecg_encoder, text_encoder, proj_dim=256)
    ecg_emb, text_emb = model(ecg_signals, clinical_texts)
    loss = InfoNCELoss()(ecg_emb, text_emb)
"""
# ============================================================
# R5 下线护栏（第二次检查报告 2C）：本模块属已废弃多模态/VQ-VAE
# 遗留栈（含 🔴 缺陷：VQ-VAE perplexity 恒 1、编解码不对称、
# fusion nhead 关键字崩溃、MetadataEncoder age=None 维度错配），
# 已彻底下线、不再维护。import 即报错，防止继续使用。
# 如需恢复请从 git 历史找回旧版文件。
# 4E/4F 审查修复：💡-11（注释-only，不改逻辑）—— encode_text 的 if/else 两分支
# 完全相同（冗余代码）。本模块为离线历史模块（import 即 raise），缺陷不可达，
# 不影响现役链路。
# ============================================================
raise RuntimeError(
    "模块已下线（R5/2C）：旧多模态/VQ-VAE 遗留栈不再维护，禁止 import。"
    "现役多模态方案见 train_multimodal_fair.py / train_ecg_text_clip.py。")



import torch
import torch.nn as nn

from .losses import InfoNCELoss


class ECGTextCLIP(nn.Module):
    """ECG-Text CLIP: Joint embedding for ECG signals and clinical text.

    Args:
        ecg_encoder: ECG backbone (from ecg_models.backbone).
        text_encoder: Text encoder (TextEncoder).
        proj_dim: Shared projection dimension.
        temperature: Initial temperature for InfoNCE.
    """

    def __init__(
        self,
        ecg_encoder: nn.Module,
        text_encoder: nn.Module,
        proj_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.text_encoder = text_encoder

        # Projection heads
        self.ecg_proj = nn.Sequential(
            nn.Linear(ecg_encoder.feature_dim, proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_encoder.output_dim, proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim * 2, proj_dim),
        )

        # 检查报告 1.4 修复：删除死参数 logit_scale（定义后从未参与 forward，
        # 且与 loss_fn 内部温度不一致）。温度由 InfoNCELoss 统一管理
        # （logit_scale = log(1/temperature)，有界可学习）。
        self.loss_fn = InfoNCELoss(temperature=temperature)

    def encode_ecg(self, ecg: torch.Tensor) -> torch.Tensor:
        """Encode ECG to normalized embedding.

        Args:
            ecg: (B, 12, L) signal.

        Returns:
            (B, proj_dim) normalized embedding.
        """
        features = self.ecg_encoder(ecg)
        return nn.functional.normalize(self.ecg_proj(features), dim=-1)

    def encode_text(self, texts) -> torch.Tensor:
        """Encode text to normalized embedding.

        Args:
            texts: List of strings or (B, ...) tokens.

        Returns:
            (B, proj_dim) normalized embedding.
        """
        if isinstance(texts, (list, tuple)):
            features = self.text_encoder(texts)
        else:
            features = self.text_encoder(texts)
        return nn.functional.normalize(self.text_proj(features), dim=-1)

    def forward(
        self,
        ecg: torch.Tensor,
        texts,
    ) -> tuple:
        """Forward pass: encode both modalities.

        Args:
            ecg: (B, 12, L) ECG signals.
            texts: List of clinical text strings.

        Returns:
            (ecg_emb, text_emb, loss) tuple.
        """
        ecg_emb = self.encode_ecg(ecg)
        text_emb = self.encode_text(texts)
        loss = self.loss_fn(ecg_emb, text_emb)
        return ecg_emb, text_emb, loss
