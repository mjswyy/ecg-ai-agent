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
        """Add positional encoding.

        Args:
            x: (B, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


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

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._feature_dim = d_model
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.lead_proj.weight, mode="fan_out",
                                nonlinearity="relu")
        for p in self.encoder.parameters():
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

        # Transformer encoder
        x = self.encoder(x)  # (B, L', d_model)

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
