"""Gated Fusion — Learnable modality gating.

Each modality (ECG, Text, Metadata) gets a learnable gate weight.
The fused output is a weighted combination of all modalities.

Usage:
    fusion = GatedFusion(ecg_dim=256, text_dim=768, meta_dim=256)
    fused = fusion(ecg_features, text_features, metadata_features)
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Gated multi-modal fusion with learnable per-modality weights.

    Args:
        ecg_dim: ECG feature dimension.
        text_dim: Text feature dimension.
        meta_dim: Metadata feature dimension.
        hidden_dim: Common projection dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        ecg_dim: int = 256,
        text_dim: int = 768,
        meta_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project to common space
        self.ecg_proj = nn.Sequential(
            nn.Linear(ecg_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim), nn.ReLU(inplace=True),
        )

        # Gate network
        gate_in = hidden_dim * 3
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # 3 gates
            nn.Softmax(dim=-1),
        )

        # Post-fusion MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self._output_dim = hidden_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        ecg_features: torch.Tensor,
        text_features: torch.Tensor,
        metadata_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Gated fusion.

        Args:
            ecg_features: (B, ecg_dim) or (B, seq, ecg_dim).
            text_features: (B, text_dim) or (B, seq, text_dim).
            metadata_features: (B, meta_dim) optional.

        Returns:
            (B, hidden_dim) fused features.
        """
        # Pool sequence dimensions if present
        if ecg_features.dim() == 3:
            ecg_features = ecg_features.mean(dim=1)
        if text_features.dim() == 3:
            text_features = text_features.mean(dim=1)

        # Project
        ecg = self.ecg_proj(ecg_features)
        text = self.text_proj(text_features)
        if metadata_features is not None:
            meta = self.meta_proj(metadata_features)
        else:
            meta = torch.zeros_like(ecg)

        # Compute gates
        concat = torch.cat([ecg, text, meta], dim=-1)
        gates = self.gate(concat)  # (B, 3)

        # Weighted combination
        fused = (gates[:, 0:1] * ecg + gates[:, 1:2] * text + gates[:, 2:3] * meta)

        return self.mlp(fused)
