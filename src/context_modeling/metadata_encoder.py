# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: 旧多模态栈组件（同上）
"""
元数据编码器 — 可学习嵌入 / Learnable Embeddings for Patient Metadata.

将患者人口统计学信息 (年龄、性别、数据来源) 编码为固定维度的特征向量。
使用可学习嵌入表 + MLP 投影。

Encodes tabular patient information into fixed-size feature vectors
using learnable embedding tables + MLP projection.

使用示例 / Usage:
    encoder = MetadataEncoder(output_dim=256)
    features = encoder(age=tensor([0.62, -1.0]), sex=tensor([1, 2]))
    # → (B, 256)
"""
# ============================================================
# R5 下线护栏（第二次检查报告 2C）：本模块属已废弃多模态/VQ-VAE
# 遗留栈（含 🔴 缺陷：VQ-VAE perplexity 恒 1、编解码不对称、
# fusion nhead 关键字崩溃、MetadataEncoder age=None 维度错配），
# 已彻底下线、不再维护。import 即报错，防止继续使用。
# 如需恢复请从 git 历史找回旧版文件。
# 4E/4F 审查修复：🟡-7（注释-only，不改逻辑）—— age=None 且 source 非空时
# batch 维错配（torch.cat 维度不匹配）。本模块为离线历史模块（import 即
# raise），缺陷不可达，不影响现役链路。
# ============================================================
raise RuntimeError(
    "模块已下线（R5/2C）：旧多模态/VQ-VAE 遗留栈不再维护，禁止 import。"
    "现役多模态方案见 train_multimodal_fair.py / train_ecg_text_clip.py。")



from typing import List, Optional
import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    """元数据编码器 / Metadata Encoder.

    参数 / Args:
        age_dim, sex_dim, source_dim: 各字段的嵌入维度 / Embed dims.
        output_dim: 最终输出维度 / Final output dim.
    """

    DEFAULT_SOURCES = ["cpsc_2018", "cpsc_2018_extra", "georgia", "ptb", "ptb-xl", "st_petersburg_incart", "unknown"]
    SEX_CATEGORIES = ["Female", "Male", "Unknown"]

    def __init__(self, age_dim=16, sex_dim=8, source_dim=16, hidden_dim=128, output_dim=256, dropout=0.1,
                 known_sources: Optional[List[str]] = None):
        super().__init__()
        # 年龄 / Age: 归一化 [0,1] → 嵌入, -1.0=未知 / -1.0=unknown
        self.age_unknown_embed = nn.Parameter(torch.randn(1, age_dim) * 0.02)
        self.age_proj = nn.Sequential(nn.Linear(1, age_dim), nn.ReLU())

        # 性别 / Sex: 可学习嵌入表 (3-way)
        self.sex_embed = nn.Embedding(len(self.SEX_CATEGORIES), sex_dim)

        # 数据来源 / Source: 可学习嵌入表
        sources = known_sources or self.DEFAULT_SOURCES
        self.source_to_idx = {s: i for i, s in enumerate(sources)}
        self.source_embed = nn.Embedding(len(sources), source_dim)

        # MLP 投影 / MLP projection
        total_in = age_dim + sex_dim + source_dim
        self.mlp = nn.Sequential(nn.Linear(total_in, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int: return self._output_dim

    def forward(self, age=None, sex=None, source=None) -> torch.Tensor:
        """前向传播 / Forward: 编码元数据 → (B, output_dim).

        参数 / Args:
            age:    (B,) 归一化年龄 / Normalized ages (-1.0=未知).
            sex:    (B,) 性别索引 / Sex index (0=F, 1=M, 2=Unknown).
            source: List[str] 数据来源名 / Data source names.
        """
        features, batch_size = [], None

        # 年龄 / Age — 未知时使用专门的可学习嵌入
        if age is not None:
            if age.dim() == 1: age = age.unsqueeze(-1)
            batch_size = age.shape[0]
            age_mask = (age < 0).float().squeeze(-1)
            age_feat = self.age_proj(age)
            age_feat = (1 - age_mask.unsqueeze(-1)) * age_feat + age_mask.unsqueeze(-1) * self.age_unknown_embed.expand(age.shape[0], -1)
            features.append(age_feat)
        else: batch_size = batch_size or 1

        # 性别 / Sex
        if sex is not None:
            batch_size = batch_size or sex.shape[0]
            features.append(self.sex_embed(sex.long().clamp(0, 2)))
        else:
            dev = features[0].device if features else None
            features.append(self.sex_embed(torch.zeros(batch_size, dtype=torch.long, device=dev)))

        # 数据来源 / Source
        dev = features[0].device if features else None
        if source is not None:
            source_idx = torch.tensor([self.source_to_idx.get(s, self.source_to_idx.get("unknown", 0)) for s in source], dtype=torch.long, device=dev)
            features.append(self.source_embed(source_idx))
        else:
            features.append(self.source_embed(torch.full((batch_size,), self.source_to_idx.get("unknown", 0), dtype=torch.long, device=dev)))

        return self.mlp(torch.cat(features, dim=-1))
