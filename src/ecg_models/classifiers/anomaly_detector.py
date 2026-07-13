"""
ECG 异常检测器 — 基于 VAE 的无监督异常检测 / VAE-based Unsupervised Anomaly Detection.

训练 VAE 仅在正常 ECG (窦性心律) 上学习紧凑的潜在表示。
异常 ECG 会产生高重建误差，从而被识别为异常。

Trains a Variational Autoencoder on normal ECGs (Sinus Rhythm).
Anomalies are detected via reconstruction error — higher = more anomalous.

使用示例 / Usage:
    detector = ECGAnomalyDetector(backbone, latent_dim=128)
    detector.fit_threshold(normal_loader)      # 在正常ECG上校准阈值
    score = detector.anomaly_score(ecg)         # → 重建误差
    is_abnormal = detector.is_anomaly(ecg)     # → True/False (需先fit_threshold!)
"""

import logging
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VAEEncoder(nn.Module):
    """编码器 / Encoder: backbone 特征 → (μ, log σ²)."""
    def __init__(self, feature_dim, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mu, self.logvar = nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)

    def forward(self, x): h = self.net(x); return self.mu(h), self.logvar(h)


class VAEDecoder(nn.Module):
    """解码器 / Decoder: 潜在向量 → 重建特征."""
    def __init__(self, latent_dim, feature_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, feature_dim))
    def forward(self, z): return self.net(z)


class ECGAnomalyDetector(nn.Module):
    """基于 VAE 的 ECG 异常检测器 / VAE-based ECG anomaly detector.

    参数 / Args:
        backbone:        ECG 骨干网络 / ECG backbone for feature extraction.
        latent_dim:      VAE 潜在空间维度 / Latent dimension.
        encoder_hidden:  编码器/解码器 MLP 隐藏层维度 / Hidden dim.
        beta:            KL 散度权重 (β-VAE) / KL divergence weight.
    """

    def __init__(self, backbone: nn.Module, latent_dim=128, encoder_hidden=256, beta=1.0):
        super().__init__()
        self.backbone, self.feature_dim = backbone, backbone.feature_dim
        self.latent_dim, self.beta = latent_dim, beta
        self.encoder = VAEEncoder(self.feature_dim, latent_dim, encoder_hidden)
        self.decoder = VAEDecoder(latent_dim, self.feature_dim, encoder_hidden)
        self.register_buffer("threshold", torch.tensor(0.0))
        self._fitted = False
        self._init_weights()

    def _init_weights(self):
        for m in [self.encoder, self.decoder]:
            for mod in m.modules():
                if isinstance(mod, nn.Linear): nn.init.xavier_uniform_(mod.weight); nn.init.zeros_(mod.bias)

    def reparameterize(self, mu, logvar):
        """重参数化技巧 / Reparameterization trick: z = μ + σ·ε."""
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        return mu

    def forward(self, x):
        """前向传播 / Forward: (B,12,L) → (recon, mu, logvar, z, features)."""
        features = self.backbone(x)
        mu, logvar = self.encoder(features)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z, features

    def loss_fn(self, recon, features, mu, logvar):
        """VAE 损失 = MSE重建 + β * KL散度 / Reconstruction + β * KL divergence."""
        recon_loss = F.mse_loss(recon, features)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """计算异常分数 (每样本重建MSE) / Per-sample reconstruction MSE."""
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            mu, logvar = self.encoder(features)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return F.mse_loss(recon, features, reduction="none").mean(dim=1)

    def is_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        """二值异常判断 / Binary anomaly prediction.

        必须先调用 fit_threshold() 校准阈值！
        Must call fit_threshold() before using this!
        """
        if not self._fitted:
            raise RuntimeError("必须先调用 fit_threshold() 校准异常阈值! / Must call fit_threshold() first!")
        return self.anomaly_score(x) > self.threshold

    def fit_threshold(self, loader, percentile=95.0, device=None):
        """在正常 ECG 分布上设置异常阈值 / Set anomaly threshold from normal distributions.

        参数 / Args:
            loader:     正常 ECG 的 DataLoader / DataLoader of normal ECGs.
            percentile: 阈值百分位数 / Threshold percentile (95 = top 5% flagged).
            device:     计算设备 / Compute device.
        """
        self.eval()
        scores = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                if device: x = x.to(device)
                scores.append(self.anomaly_score(x).cpu().numpy())
        self.threshold = torch.tensor(float(np.percentile(np.concatenate(scores), percentile)))
        self._fitted = True
        logger.info(f"异常阈值设为 {self.threshold.item():.4f} (百分位={percentile}, n={len(np.concatenate(scores))})")
