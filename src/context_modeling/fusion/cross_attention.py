"""Cross-Attention Fusion — Multi-modal fusion via cross-modal attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Cross-modal attention fusion for ECG + Text + Metadata.

    Uses scaled dot-product cross-attention: ECG as query, text as key/value.
    """

    def __init__(
        self, ecg_dim=256, text_dim=768, meta_dim=256, hidden_dim=256,
        num_heads=8, dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.ecg_proj = nn.Linear(ecg_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.meta_proj = nn.Linear(meta_dim, hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self._output_dim = hidden_dim

    @property
    def output_dim(self): return self._output_dim

    def forward(self, ecg_features, text_features, metadata_features=None):
        if ecg_features.dim() == 2:
            ecg_features = ecg_features.unsqueeze(1)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)

        B = ecg_features.shape[0]
        ecg = self.ecg_proj(ecg_features)
        text = self.text_proj(text_features)

        if metadata_features is not None:
            meta = self.meta_proj(metadata_features.unsqueeze(1))
            text = torch.cat([text, meta], dim=1)

        Q = self.q_proj(ecg)
        K = self.k_proj(text)
        V = self.v_proj(text)

        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, -1, self.hidden_dim)
        out = self.out_proj(out)

        ecg = self.norm1(ecg + self.dropout(out))
        ecg = self.norm2(ecg + self.dropout(self.ffn(ecg)))
        return ecg.mean(dim=1)
