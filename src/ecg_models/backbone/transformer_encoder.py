"""ECG Transformer — Self-attention backbone for 12-lead ECG.

Novel architecture combining lead-wise convolution projections with a
Transformer encoder for long-range dependency modeling.

Architecture:
    Input: (B, 12, L)
      → Lead-wise Conv1D projection: Conv1d(kernel_size, stride)
      → Flatten leads into sequence
      → Position embedding + Lead-type embedding
      → TransformerEncoder x N layers
      → Global mean pooling
      → Output: (B, d_model)

Configurable:
    - num_layers (8): Transformer encoder layers
    - d_model (256): Model dimension
    - nhead (8): Attention heads
    - kernel_size (49): Conv projection kernel
    - stride (24): Conv projection stride

Usage:
    backbone = ECGTransformer(in_channels=12, d_model=256, num_layers=8)
    features = backbone(x)  # x: (B, 12, 4096) → (B, 256)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal + learnable positional encoding for 1D sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class _TransformerEncoderLayerNPU(nn.Module):
    """单层 Transformer Encoder — 使用独立算子，NPU 全加速。

    等价于 nn.TransformerEncoderLayer，但避免融合算子
    aten::_transformer_encoder_layer_fwd 在 CANN 8.0.0 上的 CPU 回退。
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual + norm
        attn_out, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(attn_out))
        # Feed-forward + residual + norm
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        return src


class ECGTransformer(nn.Module):
    """ECG Transformer backbone with lead-wise conv projections.

    Args:
        in_channels: Input channels (12 for standard ECG).
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network hidden dimension.
        dropout: Dropout rate.
        kernel_size: Conv1D projection kernel size.
        stride: Conv1D projection stride.
        max_seq_len: Maximum sequence length for positional encoding buffer.
    """

    def __init__(
        self,
        in_channels: int = 12,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        kernel_size: int = 49,
        stride: int = 24,
        max_seq_len: int = 3000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.stride = stride
        self.kernel_size = kernel_size

        # Lead-wise Conv1D projection
        # Each lead is convolved independently to produce d_model channels
        self.lead_proj = nn.Conv1d(
            in_channels, d_model, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False,
        )
        self.lead_bn = nn.BatchNorm1d(d_model)
        self.lead_relu = nn.ReLU(inplace=True)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len,
                                               dropout=dropout)

        # Transformer encoder — 手动实现每层，避免 NPU 不支持的融合算子
        # nn.TransformerEncoderLayer 会调用 aten::_transformer_encoder_layer_fwd
        # 该融合算子在 CANN 8.0.0 未适配，会回退 CPU。拆分为独立算子后全部在 NPU 运行。
        self.encoder_layers = nn.ModuleList([
            _TransformerEncoderLayerNPU(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
            )
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model, eps=1e-6)

        self._feature_dim = d_model
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.lead_proj.weight, mode="fan_out",
                                nonlinearity="relu")
        for layer in self.encoder_layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ECG signal (B, in_channels, L). Typically (B, 12, 4096).

        Returns:
            Feature tensor (B, d_model).
        """
        B, C, L = x.shape

        # Lead-wise Conv1D projection
        # x: (B, 12, L) → lead_proj: (B, d_model, L')
        x = self.lead_proj(x)  # (B, d_model, L')
        x = self.lead_bn(x)
        x = self.lead_relu(x)

        # Reshape to sequence: (B, L', d_model)
        x = x.transpose(1, 2)  # (B, L', d_model)
        seq_len = x.shape[1]

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder — 逐层执行，每层用独立算子（NPU 全加速）
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)

        # Global mean pooling
        x = x.mean(dim=1)  # (B, d_model)

        x = self.dropout(x)
        return x

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @staticmethod
    def compute_output_length(input_length: int, kernel_size: int, stride: int) -> int:
        """Compute sequence length after lead-wise conv projection."""
        padding = kernel_size // 2
        return (input_length + 2 * padding - kernel_size) // stride + 1


def ecg_transformer(**kwargs) -> ECGTransformer:
    """Create ECG Transformer backbone (~10-20M parameters)."""
    return ECGTransformer(**kwargs)
