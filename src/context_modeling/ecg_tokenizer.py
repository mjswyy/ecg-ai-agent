"""
ECG 分词器 — VQ-VAE 将 ECG 离散化为视觉 token / VQ-VAE ECG Tokenizer.

将原始 12 导联 ECG 转化为离散 token 序列，支持:
    1. ECGTokenizer (VQ-VAE): 编码→量化→解码，Codebook K=1024, dim=256
    2. SimpleECGProjector (轻量版): Conv1D + Pooling → 固定维度特征向量

Converts raw 12-lead ECG into discrete tokens via Vector Quantized VAE.
Also provides a lightweight Conv projector for v1 prototyping.

使用示例 / Usage:
    # VQ-VAE
    vq = ECGTokenizer(input_length=4096)
    tokens, features, vq_loss, ppl = vq.encode(ecg)
    recon = vq.decode(tokens)

    # 简化版 / Simplified
    proj = SimpleECGProjector(input_channels=12, output_dim=256)
    emb = proj(ecg)  # (B, 12, 4096) → (B, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """向量量化层 / Vector Quantization (VQ-VAE).

    将连续潜在向量映射到最近的 codebook 条目。
    使用直通估计器 (straight-through estimator) 进行梯度传播。
    """

    def __init__(self, num_embeddings=1024, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        """量化潜在向量 / Quantize latent vectors.

        参数 / Args: z (B, D, L) — 潜在向量
        返回 / Returns: z_q (量化后), indices (codebook索引), loss, perplexity
        """
        B, D, L = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, D)
        distances = z_flat.pow(2).sum(1, keepdim=True) - 2 * z_flat @ self.codebook.weight.t() + self.codebook.weight.pow(2).sum(1)
        indices = distances.argmin(dim=1)
        z_q = self.codebook(indices).view(B, L, D).permute(0, 2, 1)
        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()  # 直通估计器 / Straight-through
        perplexity = torch.exp(-torch.sum(F.one_hot(indices, self.codebook.num_embeddings).float() * torch.log(F.one_hot(indices, self.codebook.num_embeddings).float() + 1e-10)) / B)
        return z_q, indices.view(B, L), loss, perplexity


class ECGTokenizer(nn.Module):
    """VQ-VAE ECG 分词器 / VQ-VAE tokenizer for 12-lead ECG.

    参数 / Args:
        input_channels: 输入通道 (12) / Input channels.
        input_length:   信号长度 (4096) / Signal length.
        hidden_dims:    编码器各阶段隐藏维度 / Hidden dims per encoder stage.
        codebook_size:  VQ codebook 大小 (1024) / Codebook entries.
        codebook_dim:   每个 codebook 条目的维度 (256) / Embedding dim.
    """

    def __init__(self, input_channels=12, input_length=4096, hidden_dims=(64, 128, 256),
                 codebook_size=1024, codebook_dim=256, commitment_cost=0.25):
        super().__init__()
        # 编码器 / Encoder: Conv1D × 3 下采样
        encoder_layers = []
        in_dim = input_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Conv1d(in_dim, h_dim, 4, 2, 1), nn.BatchNorm1d(h_dim), nn.ReLU(inplace=True)])
            in_dim = h_dim
        encoder_layers.append(nn.Conv1d(in_dim, codebook_dim, 3, 1, 1))
        self.encoder = nn.Sequential(*encoder_layers)

        # VQ 层 / VQ layer
        self.vq = VectorQuantizer(codebook_size, codebook_dim, commitment_cost)

        # 解码器 / Decoder: ConvTranspose1D × 3 上采样
        self._enc_seq_len = input_length // (2 ** len(hidden_dims))
        decoder_layers = []
        rev_dims = list(reversed(hidden_dims)) + [input_channels]
        in_dim = codebook_dim
        for h_dim in rev_dims:
            decoder_layers.extend([nn.ConvTranspose1d(in_dim, h_dim, 4, 2, 1), nn.BatchNorm1d(h_dim),
                                   nn.ReLU(inplace=True) if h_dim != input_channels else nn.Tanh()])
            in_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x): z = self.encoder(x); return self.vq(z)  # → (tokens, features, loss, ppl)
    def decode(self, indices): return self.decoder(self.vq.codebook(indices).permute(0, 2, 1))

    def forward(self, x):
        recon, indices, vq_loss, ppl = self.encode(x); recon = self.decode(indices)
        if recon.shape[2] > x.shape[2]:   recon = recon[:, :, :x.shape[2]]
        elif recon.shape[2] < x.shape[2]: recon = F.pad(recon, (0, x.shape[2] - recon.shape[2]))
        return recon, indices, vq_loss, ppl

    @property
    def seq_len(self) -> int: return self._enc_seq_len


class SimpleECGProjector(nn.Module):
    """简化的 ECG 投影器（轻量替代 VQ-VAE）/ Lightweight Conv projector.

    参数 / Args:
        input_channels: 输入通道 (12)
        output_dim:     输出特征维度 (512)
    """

    def __init__(self, input_channels=12, output_dim=512):
        super().__init__()
        self.output_dim = output_dim
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, 2, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(256, output_dim)

    def forward(self, x):
        """(B, 12, L) → (B, output_dim)"""
        return self.proj(self.conv(x).squeeze(-1))
