"""Late Fusion — Simple concatenation-based multi-modal fusion.

Each modality is independently encoded, then concatenated and passed
through an MLP for classification.

Usage:
    fusion = LateFusion(ecg_dim=256, text_dim=768, meta_dim=256)
    fused = fusion(ecg_features, text_features, metadata_features)
"""

import torch
import torch.nn as nn


class LateFusion(nn.Module):
    """Late fusion via concatenation + MLP.

    Args:
        ecg_dim, text_dim, meta_dim: Input dimensions.
        hidden_dim: MLP hidden dimension.
        output_dim: Output dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        ecg_dim: int = 256,
        text_dim: int = 768,
        meta_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        total_in = ecg_dim + text_dim + meta_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_in, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        ecg_features: torch.Tensor,
        text_features: torch.Tensor,
        metadata_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Late fusion.

        Args:
            ecg_features: (B, ecg_dim).
            text_features: (B, text_dim).
            metadata_features: (B, meta_dim) optional.

        Returns:
            (B, output_dim) fused features.
        """
        if ecg_features.dim() == 3:
            ecg_features = ecg_features.mean(dim=1)
        if text_features.dim() == 3:
            text_features = text_features.mean(dim=1)
        if metadata_features is None:
            metadata_features = torch.zeros(
                ecg_features.shape[0], self._output_dim,
                device=ecg_features.device,
            )

        concat = torch.cat([ecg_features, text_features, metadata_features], dim=-1)
        return self.mlp(concat)
