"""xResNet1D-101 — PhysioNet 2020 Challenge winning architecture.

Based on Ribeiro et al. "Automatic Diagnosis of the 12-lead ECG Using a
Deep Neural Network" (Nature Communications, 2020).

Key design decisions from the challenge-winning solution:
    - Bottleneck blocks with kernel_size=3 (not 1-3-1)
    - 4 stages with depths [3, 4, 23, 3]
    - Stem: Conv1d(kernel=15, stride=2)
    - Global average pooling → 512-dim features
    - ~15M parameters

Usage:
    backbone = xresnet1d_101(in_channels=12)
    features = backbone(x)  # x: (B, 12, L) → (B, 512)
"""

from typing import Optional, Type, Union

import torch
import torch.nn as nn

from .resnet1d import Bottleneck1D


class xResNet1D101(nn.Module):
    """xResNet1D-101: Challenge 2020 winning architecture.

    Config:
        in_channels: 12 (standard 12-lead ECG)
        base_channels: 32
        layers: [3, 4, 23, 3]
        block: Bottleneck1D (expansion=4)
        stem_kernel: 15, stem_stride: 2
        feature_dim: 512 (after GAP)
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 32,
        dropout: float = 0.1,
        stem_kernel_size: int = 15,
        stem_stride: int = 2,
    ):
        super().__init__()
        self._feature_dim = 512

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, stem_kernel_size,
                      stride=stem_stride, padding=stem_kernel_size // 2,
                      bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 1: base_channels, stride=1 (after stem stride=2)
        self.stage1 = self._make_stage(
            base_channels, base_channels, 3, stride=1, dropout=dropout
        )
        # Stage 1 output: base_channels * 4 = 128

        # Stage 2: 2x channels
        ch2 = base_channels * 2  # 64
        self.stage2 = self._make_stage(
            base_channels * 4, ch2, 4, stride=2, dropout=dropout
        )
        # Stage 2 output: ch2 * 4 = 256

        # Stage 3: 2x channels
        ch3 = base_channels * 4  # 128
        self.stage3 = self._make_stage(
            ch2 * 4, ch3, 23, stride=2, dropout=dropout
        )
        # Stage 3 output: ch3 * 4 = 512

        # Stage 4: 2x channels
        ch4 = base_channels * 8  # 256
        self.stage4 = self._make_stage(
            ch3 * 4, ch4, 3, stride=2, dropout=dropout
        )
        # Stage 4 output: ch4 * 4 = 1024

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(ch4 * 4, self._feature_dim),
        )

        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        downsample = None
        expansion = Bottleneck1D.expansion  # 4

        if stride != 1 or in_channels != out_channels * expansion:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * expansion),
            )

        layers = []
        layers.append(Bottleneck1D(
            in_channels, out_channels, stride, downsample,
            kernel_size=3, dropout=dropout,
        ))
        for _ in range(1, num_blocks):
            layers.append(Bottleneck1D(
                out_channels * expansion, out_channels,
                kernel_size=3, dropout=dropout,
            ))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-init residual
        for m in self.modules():
            if isinstance(m, Bottleneck1D):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ECG signal (B, 12, L).

        Returns:
            Feature tensor (B, 512).
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)
        x = self.feature_proj(x)
        return x

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


def xresnet1d_101(in_channels: int = 12, dropout: float = 0.1, **kwargs) -> xResNet1D101:
    """Create xResNet1D-101 backbone (~15M parameters)."""
    return xResNet1D101(in_channels=in_channels, dropout=dropout, **kwargs)
