"""InceptionTime — Lightweight Inception-based 1D CNN for time series.

Based on Fawaz et al. "InceptionTime: Finding AlexNet for Time Series
Classification" (Data Mining and Knowledge Discovery, 2020).

Key design:
    - 3 Inception modules with multi-scale kernels (9, 19, 39)
    - Bottleneck convs for dimensionality reduction
    - Global average pooling → 128-dim features
    - ~2M parameters (very lightweight)

Usage:
    backbone = InceptionTime(in_channels=12)
    features = backbone(x)  # x: (B, 12, L) → (B, 128)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """Single Inception module with multi-scale 1D convolutions.

    Uses bottleneck convs to reduce parameters before large-kernel convs.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        bottleneck_channels: int = 32,
        kernel_sizes: tuple = (9, 19, 39),
    ):
        super().__init__()

        # Branch 1: bottleneck → Conv(kernel_size)
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False),
                nn.BatchNorm1d(bottleneck_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(bottleneck_channels, n_filters, ks,
                          padding=ks // 2, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        # Branch 2: MaxPool → bottleneck
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, 1, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(inplace=True),
        )

        # Combine
        total_filters = n_filters * (len(kernel_sizes) + 1)
        self.combine = nn.Sequential(
            nn.BatchNorm1d(total_filters),
            nn.ReLU(inplace=True),
        )

        self.out_channels = total_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        outputs.append(self.pool_branch(x))
        x = torch.cat(outputs, dim=1)
        return self.combine(x)


class InceptionTime(nn.Module):
    """InceptionTime backbone for 12-lead ECG.

    Args:
        in_channels: Input channels (12 for standard ECG).
        n_filters: Filters per kernel size in Inception modules.
        bottleneck_channels: Channels for bottleneck convs.
        kernel_sizes: Kernel sizes for multi-scale branches.
        depth: Number of Inception modules (3 recommended).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 12,
        n_filters: int = 32,
        bottleneck_channels: int = 32,
        kernel_sizes: tuple = (9, 19, 39),
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initial stem (no stride to preserve temporal info)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, 7, padding=3, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(inplace=True),
        )

        # Inception modules with residual connections
        self.inception_modules = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        current_channels = n_filters

        for d in range(depth):
            module = InceptionModule(
                current_channels,
                n_filters=n_filters,
                bottleneck_channels=bottleneck_channels,
                kernel_sizes=kernel_sizes,
            )
            self.inception_modules.append(module)

            # Residual projection if channels change
            if current_channels != module.out_channels:
                proj = nn.Conv1d(current_channels, module.out_channels, 1, bias=False)
            else:
                proj = nn.Identity()
            self.residual_projections.append(proj)

            current_channels = module.out_channels

        self._feature_dim = current_channels
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ECG signal (B, 12, L).

        Returns:
            Feature tensor (B, feature_dim).
        """
        x = self.stem(x)

        for module, proj in zip(self.inception_modules, self.residual_projections):
            residual = proj(x)
            x = module(x)
            x = x + residual  # Residual connection
            x = F.relu(x)

        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        return x

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


def inception_time(in_channels: int = 12, **kwargs) -> InceptionTime:
    """Create InceptionTime backbone (~2M parameters)."""
    return InceptionTime(in_channels=in_channels, **kwargs)
