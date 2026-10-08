# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: 旧多模态栈（仅被废弃的 train_multimodal.py 使用）；现役公平版 = train_multimodal_fair.py
"""Multi-modal Model — Unified ECG + Text + Metadata classification model.

Combines ECG backbone, text encoder, metadata encoder, and fusion module
into a single end-to-end model for diagnosis classification.

Usage:
    model = MultimodalModel(ecg_encoder, text_encoder, fusion, num_classes=27)
    logits = model(ecg, texts, ages, sexes)
"""
# ============================================================
# R5 下线护栏（第二次检查报告 2C）：本模块属已废弃多模态/VQ-VAE
# 遗留栈（含 🔴 缺陷：VQ-VAE perplexity 恒 1、编解码不对称、
# fusion nhead 关键字崩溃、MetadataEncoder age=None 维度错配），
# 已彻底下线、不再维护。import 即报错，防止继续使用。
# 如需恢复请从 git 历史找回旧版文件。
# 4E/4F 审查修复：🟡-10（注释-only，不改逻辑）—— forward 将文本送入
# fusion→classifier，违反"推理时 signal-only"红线（历史设计）。本模块为离线历史
# 模块（import 即 raise），缺陷不可达，不影响现役链路（公平版文本仅作监督）。
# ============================================================
raise RuntimeError(
    "模块已下线（R5/2C）：旧多模态/VQ-VAE 遗留栈不再维护，禁止 import。"
    "现役多模态方案见 train_multimodal_fair.py / train_ecg_text_clip.py。")



import torch
import torch.nn as nn


class MultimodalModel(nn.Module):
    """Multi-modal ECG diagnosis model.

    Args:
        ecg_encoder: ECG backbone or SimpleECGProjector.
        text_encoder: Text encoder (TextEncoder).
        metadata_encoder: Metadata encoder (MetadataEncoder).
        fusion: Fusion module (CrossAttentionFusion/GatedFusion/LateFusion).
        num_classes: Number of output classes (27).
        dropout: Classification head dropout.
    """

    def __init__(
        self,
        ecg_encoder: nn.Module,
        text_encoder: nn.Module,
        metadata_encoder: nn.Module,
        fusion: nn.Module,
        num_classes: int = 27,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.text_encoder = text_encoder
        self.metadata_encoder = metadata_encoder
        self.fusion = fusion

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        ecg: torch.Tensor,
        texts=None,
        ages: torch.Tensor = None,
        sexes: torch.Tensor = None,
        sources=None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            ecg: (B, 12, L) ECG signals.
            texts: List of clinical text strings (or None).
            ages: (B,) normalized ages (-1.0 = unknown).
            sexes: (B,) sex indices (0=Female, 1=Male, 2=Unknown).
            sources: List of data source names.

        Returns:
            (B, num_classes) logits.
        """
        # ECG features
        ecg_feat = self.ecg_encoder(ecg)  # (B, ecg_dim) or (B, seq, ecg_dim)

        # Text features
        if texts is not None:
            text_feat = self.text_encoder(texts)  # (B, text_dim)
        else:
            text_feat = torch.zeros(
                ecg_feat.shape[0], self.text_encoder.output_dim,
                device=ecg.device,
            )

        # Metadata features
        meta_feat = self.metadata_encoder(
            age=ages, sex=sexes, source=sources,
        )  # (B, meta_dim)

        # Fusion
        fused = self.fusion(ecg_feat, text_feat, meta_feat)

        # Classify
        return self.classifier(fused)
