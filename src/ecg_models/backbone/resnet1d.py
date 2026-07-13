"""1D ResNet Building Blocks.

Provides BasicBlock1D, Bottleneck1D, and a configurable ResNet1D backbone
for ECG signal processing.

Usage:
    backbone = ResNet1D([2, 2, 2, 2], in_channels=12, base_channels=32)
    # → ResNet-18 equivalent, ~5M params
    features = backbone(x)  # x: (B, 12, L) → (B, 512)
"""

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn


# ----------------------------------------------------------------
# Basic Blocks
# ----------------------------------------------------------------

class BasicBlock1D(nn.Module):
    """ResNet basic block (2 conv layers + skip connection)."""
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck1D(nn.Module):
    """ResNet bottleneck block (1×1 → 3×3 → 1×1 + skip)."""
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion,
                               1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ----------------------------------------------------------------
# ResNet1D Backbone
# ----------------------------------------------------------------

class ResNet1D(nn.Module):
    """Configurable 1D ResNet backbone for ECG signals.

    Args:
        layers: List of block counts per stage, e.g. [2,2,2,2] → ResNet-18.
        block: BasicBlock1D or Bottleneck1D.
        in_channels: Input channels (12 for standard 12-lead ECG).
        base_channels: Channels in the first stage (doubles each stage).
        in_kernel_size: Kernel size for the initial stem convolution.
        in_stride: Stride for the initial stem convolution.
        dropout: Dropout rate in blocks.
        zero_init_residual: If True, zero-initialize the last BN in each block.

    Examples:
        # ResNet-34 equivalent
        ResNet1D([3,4,6,3], BasicBlock1D, base_channels=32)

        # ResNet-101 equivalent (xResNet1D-101 uses different kernel config)
        ResNet1D([3,4,23,3], Bottleneck1D, base_channels=32)
    """

    def __init__(
        self,
        layers: List[int],
        block: Type[Union[BasicBlock1D, Bottleneck1D]] = BasicBlock1D,
        in_channels: int = 12,
        base_channels: int = 64,
        in_kernel_size: int = 15,
        in_stride: int = 2,
        dropout: float = 0.0,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        padding = in_kernel_size // 2

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, in_kernel_size,
                      stride=in_stride, padding=padding, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Stages
        self.stages = nn.ModuleList()
        channels = base_channels

        for stage_idx, num_blocks in enumerate(layers):
            stage_channels = base_channels * (2 ** stage_idx)
            stage = self._make_stage(
                block, channels, stage_channels, num_blocks,
                stride=2 if stage_idx > 0 else 1, dropout=dropout,
            )
            self.stages.append(stage)
            channels = stage_channels * block.expansion

        self._feature_dim = channels

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Init
        self._init_weights(zero_init_residual)

    def _make_stage(
        self,
        block: Type,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * block.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, dropout=dropout))
        for _ in range(1, num_blocks):
            layers.append(block(
                out_channels * block.expansion, out_channels, dropout=dropout
            ))

        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ECG signal (B, 12, L).

        Returns:
            Feature tensor (B, feature_dim).
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).squeeze(-1)
        return x

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


# ----------------------------------------------------------------
# Pre-configured models
# ----------------------------------------------------------------

def resnet1d_18(in_channels: int = 12, **kwargs) -> ResNet1D:
    """ResNet-18 equivalent for 1D signals."""
    return ResNet1D([2, 2, 2, 2], BasicBlock1D, in_channels=in_channels, **kwargs)


def resnet1d_34(in_channels: int = 12, **kwargs) -> ResNet1D:
    """ResNet-34 equivalent for 1D signals (~5M params)."""
    return ResNet1D([3, 4, 6, 3], BasicBlock1D, in_channels=in_channels, **kwargs)
